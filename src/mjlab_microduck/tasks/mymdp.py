"""microduck 任务的 MDP 函数"""

import math
import os
from dataclasses import dataclass as _dataclass

import numpy as np
import torch
from typing import TYPE_CHECKING, Optional
import mujoco

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardManager as _RewardManager
from mjlab.entity import Entity
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp import observations as _velocity_obs
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers import CommandTermCfg, RewardTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.lab_api.math import matrix_from_quat, wrap_to_pi, quat_apply, quat_apply_inverse, quat_from_angle_axis
from rsl_rl.algorithms.ppo import PPO as _PPO

# ---------------------------------------------------------------------------
# Patch 1: RewardManager.compute —— 在 reward 进入
# PPO buffer 之前清洗 NaN。mjlab 在【重置环境之前】计算 reward，所以任何
# 作用在 NaN 物理状态上的 reward term 都会返回 NaN。该 NaN 会传播：
# NaN reward → NaN advantage → NaN loss → NaN gradient →
# NaN/负 std → 下一个 mini-batch 在 torch.normal 处崩溃。
# ---------------------------------------------------------------------------
_orig_reward_compute = _RewardManager.compute

def _nan_safe_reward_compute(self, dt: float) -> torch.Tensor:
    result = _orig_reward_compute(self, dt)
    # _episode_sums 在 compute() 内部更新，nan_to_num 来不及起作用。
    # 就地清洗，使各 term 的 metric 不显示 NaN。
    for key in self._episode_sums:
        torch.nan_to_num_(self._episode_sums[key], nan=0.0)
    return torch.nan_to_num(result, nan=0.0)

_RewardManager.compute = _nan_safe_reward_compute

# ---------------------------------------------------------------------------
# Patch 2: PPO.compute_returns —— 在归一化之前清洗 advantage。
# 在 curriculum 突变时（例如 reward 权重 ×2.5），value function 严重失准：
# 所有 TD error 平移相同量，std(advantages) → 极小，
# 于是 (A − mean) / (std + 1e-8) → 巨大。这会让 std 的梯度爆炸，
# 优化器随后把 std 压到零以下。归一化前把 NaN/Inf advantage 清零
# 可让它们保持在安全范围内。
# ---------------------------------------------------------------------------
_orig_compute_returns = _PPO.compute_returns

def _safe_compute_returns(self, obs) -> None:
    _orig_compute_returns(self, obs)
    st = self.storage
    torch.nan_to_num_(st.advantages, nan=0.0, posinf=0.0, neginf=0.0)
    torch.nan_to_num_(st.returns,    nan=0.0, posinf=0.0, neginf=0.0)

_PPO.compute_returns = _safe_compute_returns

# Patch 3（ActorCritic._update_distribution 的 std-clamp）已在 mjlab 的
# 1.3.0 migration: rsl_rl 5.0.1 refactored the policy (no ActorCritic class; the
# 迁移中移除（distribution 现在位于 rsl_rl.modules.distribution）。它只是
# 防止 std 变负/NaN 的防御性创可贴（microban 没有它也正常运行）。如果
# 1.3.0 下 std 爆炸再现，针对新的 GaussianDistribution 恢复该补丁。

print("[mdp] Patches 1-2 active: NaN-safe reward/advantage")

# ---------------------------------------------------------------------------
# Patch 4: exporter_utils.get_base_metadata —— 新的 microduck 模型有
# passive 关节（jaw 连杆通过 equality 约束闭合），属于 articulation 的一部分
# 但没有 XML actuator。上游 exporter
# 遍历 robot.joint_names（16）并索引 joint_name_to_ctrl_id（13），
# 在 passive_* 上因 KeyError 崩溃。把这些 passive 关节从
# 导出的 metadata 中过滤掉，使策略与 13 维 action 空间保持一致。
# ---------------------------------------------------------------------------
from mjlab.rl import exporter_utils as _exporter_utils  # noqa: E402
from mjlab.envs.mdp.actions import JointPositionAction as _JointAction  # noqa: E402

def _get_base_metadata_no_passive(env, run_path):
    robot = env.scene["robot"]
    joint_action = env.action_manager.get_term("joint_pos")
    assert isinstance(joint_action, _JointAction)
    full_names = list(robot.joint_names)
    keep_idx = [i for i, n in enumerate(full_names) if not n.startswith("passive_")]
    joint_names = [full_names[i] for i in keep_idx]
    joint_name_to_ctrl_id = {a.target.split("/")[-1]: a.id for a in robot.spec.actuators}
    ctrl_ids = [joint_name_to_ctrl_id[n] for n in joint_names]
    stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0]
    damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]
    default_jp = robot.data.default_joint_pos[0].cpu().tolist()
    return {
        "run_path": run_path,
        "joint_names": joint_names,
        "joint_stiffness": stiffness.tolist(),
        "joint_damping": damping.tolist(),
        "default_joint_pos": [default_jp[i] for i in keep_idx],
        "command_names": list(env.command_manager.active_terms),
        "observation_names": env.observation_manager.active_terms["actor"],
        "action_scale": joint_action._scale[0].cpu().tolist()
        if isinstance(joint_action._scale, torch.Tensor)
        else joint_action._scale,
    }

_exporter_utils.get_base_metadata = _get_base_metadata_no_passive
# 同时修补 velocity 任务 exporter 中已导入的引用。
try:
    from mjlab.tasks.velocity.rl import exporter as _vel_exporter  # noqa: E402
    if hasattr(_vel_exporter, "get_base_metadata"):
        _vel_exporter.get_base_metadata = _get_base_metadata_no_passive
except Exception:
    pass

print("[mdp] Patch 4 active: ONNX export filters passive_* joints")

# Patch 5: warm start。MjlabOnPolicyRunner.load 会从 checkpoint 恢复
# env.common_step_counter（rsl_rl 恢复 iteration），使【续训】运行保持
# 其 curriculum。而【warm start】是把另一个任务的权重加载进新任务——
# 此时恢复的计数器（例如一次 3750 iter 行走的 90 000 步）会让
# 每个基于步数的 curriculum 直接跳到最终阶段（2026-09-09 见过：
# fell_over 被禁用，所有保护成本在第 1 次迭代就已是满权重）。
# 设置 MICRODUCK_WARM_START=1 时，加载后计数器从 0 重启（权重、
# normalizer 和优化器保留）。由 hf_jobs.py 转发进 HF Jobs。
WARM_START_ENV = "MICRODUCK_WARM_START"

try:
    from mjlab.rl.runner import MjlabOnPolicyRunner as _MjlabRunner  # noqa: E402

    _orig_runner_load = _MjlabRunner.load

    def _load_with_warm_start(self, path, *args, **kwargs):
        infos = _orig_runner_load(self, path, *args, **kwargs)
        if os.environ.get(WARM_START_ENV, "") not in ("", "0"):
            restored = self.env.unwrapped.common_step_counter
            self.env.unwrapped.common_step_counter = 0
            self.current_learning_iteration = 0
            print(
                f"[mdp] Patch 5: WARM START from {path} — common_step_counter "
                f"{restored} → 0, iteration → 0 (curricula restart; weights/normalizer/optimizer kept)"
            )
        return infos

    _MjlabRunner.load = _load_with_warm_start
    print("[mdp] Patch 5 active: MICRODUCK_WARM_START=1 restarts curricula after checkpoint load")
except Exception as _e:  # pragma: no cover
    print(f"[mdp] Patch 5 NOT applied ({_e!r})")

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# 匹配 4 个颈部/头部受驱关节的名称模式。供 head_pose
# tracking reward 以及 UniformPoseCommand 的 asset 挂钩使用。
# _NECK_JOINT_PATTERNS = [r".*neck_pitch.*", r".*head_pitch.*", r".*head_yaw.*", r".*head_roll.*"]
_NECK_JOINT_PATTERNS = [r"neck_joint_.*"]



def _servo_joint_ids(env: "ManagerBasedRlEnv", asset: Entity) -> list:
    """伺服（非 ``passive_``）关节在实体内的索引，带缓存。

    本模块中所有基于关节索引的 reward/event 参数（``joint_indices``、
    ``target_overrides``、qpos 列运算）都是按标准的 13 伺服布局写的。在带有额外
    非驱动关节（backlash 铰链、滚轮、jaw 连杆，全部命名为 ``passive_*``）的模型上，
    实体关节数组更宽且交错，直接用原始索引会选错关节。通过该列表索引可获得
    仅含伺服的视图；在普通模型上它就是恒等映射。
    """
    cache = env.__dict__.setdefault("_servo_joint_ids_cache", {})
    key = id(asset)
    ids = cache.get(key)
    if ids is None:
        # ids, _ = asset.find_joints(r"^(?!passive_).*")
        ids, _ = asset.find_joints(r"^(?!.*_backlash).*")
        cache[key] = ids
    return ids


def _env_origin_z(env: "ManagerBasedRlEnv", env_ids: torch.Tensor) -> torch.Tensor:
    """每个环境原点的地形高度。所有写【绝对】trunk z 的 spawn 函数都必须加上它：
    在粗糙地形上（slope 金字塔把平台抬到远高于 z=0），0.05-0.09 m 的绝对 z 会把
    机器人放进地形内部 → 接触爆炸 → returns/value/std 爆炸。这正是第一次
    rough+backlash velstand 运行失败的原因（wy8gcaus, 2026-09-10）：一直稳定，
    直到 iter 700 打开 prone spawn，然后 value loss 0.6 → 42、entropy 5 → 33，
    与 prone_prob 同步。平地运行从未发现（原点 z = 0）。"""
    return env.scene.env_origins[env_ids.long(), 2]


def _servo_joint_pos(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.joint_pos[:, _servo_joint_ids(env, asset)]


def _servo_joint_vel(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.joint_vel[:, _servo_joint_ids(env, asset)]


def _servo_default_joint_pos(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.default_joint_pos[:, _servo_joint_ids(env, asset)]


def reset_with_forward_velocity(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    velocity_range: tuple[float, float] = (0.3, 0.8),
    fraction_stages: list[dict] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """让一部分重置环境以随机前向速度做 warm start。

    机器人生成时已经在身体朝前方向上运动，因此它首先发现"高速滑行"是什么感觉。
    该比例在训练中逐渐降低，迫使它逐步学会从静止自己获得速度。

    Args:
        velocity_range: 前向速度范围 (min, max)，单位 m/s。
        fraction_stages: {"step": int, "fraction": float} 字典列表，按 step 排序。
            使用当前训练 step 对应的 fraction。
            示例：[{"step":0,"fraction":0.8}, {"step":2000*24,"fraction":0.0}]
        asset_cfg: 机器人实体配置。
    """
    if fraction_stages is None:
        fraction_stages = [{"step": 0, "fraction": 0.8}]

    # 根据训练 step 确定当前比例
    step = env.common_step_counter
    fraction = fraction_stages[0]["fraction"]
    for stage in fraction_stages:
        if step >= stage["step"]:
            fraction = stage["fraction"]

    if len(env_ids) == 0 or fraction <= 0.0:
        return

    n_warmstart = max(1, int(len(env_ids) * fraction))
    perm = torch.randperm(len(env_ids), device=env.device)[:n_warmstart]
    warmstart_ids = env_ids[perm]

    lo, hi = velocity_range
    vx = lo + torch.rand(n_warmstart, device=env.device) * (hi - lo)

    # 只用 yaw 构造水平前向方向 —— 忽略 pitch/roll。
    # 重要：从 qpos 读取四元数，而不是 root_link_quat_w。
    # root_link_quat_w 读取的 xquat 要求 sim.forward() 是最新的。
    # reset_base 向 qpos 写入新 yaw 后，xquat 仍是旧的（上一 episode）。
    # qpos 由 write_root_pose 立即更新，因此总是最新的。
    asset: Entity = env.scene[asset_cfg.name]
    qpos_q_adr = asset.data.indexing.free_joint_q_adr[3:7]  # quat indices in qpos
    q = asset.data.data.qpos[warmstart_ids][:, qpos_q_adr]  # (n, 4) [w, x, y, z]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    forward_world = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)

    velocities = torch.zeros(n_warmstart, 6, device=env.device)
    velocities[:, :3] = vx.unsqueeze(-1) * forward_world

    asset.write_root_link_velocity_to_sim(velocities, env_ids=warmstart_ids)

    # # 让轮子旋转以匹配前向速度 —— 防止瞬时无滑移制动。
    # # 轮子半径 = 0.0175 m（实测）。
    # # 前进时全部 4 个轮子以 +ω 旋转（由 test_wheel_direction.py 验证）。
    # _WHEEL_RADIUS = 0.0175
    # all_wheel_ids, _ = asset.find_joints(r"^passive_.*")

    # if all_wheel_ids:
    #     joint_pos = asset.data.joint_pos[warmstart_ids].clone()
    #     joint_vel = asset.data.joint_vel[warmstart_ids].clone()
    #     omega = vx / _WHEEL_RADIUS  # (n,) rad/s, positive = forward
    #     joint_vel[:, all_wheel_ids] = omega.unsqueeze(-1).expand(-1, len(all_wheel_ids))
    #     asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=warmstart_ids)


def reset_action_history(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """
    为正在重置的环境重置缓存的动作历史。
    这对 action rate 和加速度 penalty 项至关重要。

    该函数应在 post_reset 回调中或在 episode 终止时调用。

    Args:
        env: 环境
        env_ids: 正在重置的环境索引
        asset_cfg: 资产配置
    """
    if len(env_ids) == 0:
        return

    asset: Entity = env.scene[asset_cfg.name]

    # 重置腿部 action rate 缓存
    if hasattr(env, '_prev_leg_actions'):
        # 设为当前动作（若还没有动作则为 0）
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            leg_joint_indices = list(range(0, 5)) + list(range(8, 13))
            env._prev_leg_actions[env_ids] = env.action_manager.action[env_ids][:, leg_joint_indices]
        else:
            env._prev_leg_actions[env_ids] = 0.0

    # 重置颈部 action rate 缓存
    if hasattr(env, '_prev_neck_actions'):
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            neck_joint_indices = list(range(5, 8))
            env._prev_neck_actions[env_ids] = env.action_manager.action[env_ids][:, neck_joint_indices]
        else:
            env._prev_neck_actions[env_ids] = 0.0

    # 重置腿部 action 加速度缓存
    if hasattr(env, '_prev_leg_actions_for_acc'):
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            leg_joint_indices = list(range(0, 5)) + list(range(8, 13))
            current_action = env.action_manager.action[env_ids][:, leg_joint_indices]
            env._prev_leg_actions_for_acc[env_ids] = current_action
            env._prev_prev_leg_actions_for_acc[env_ids] = current_action
        else:
            env._prev_leg_actions_for_acc[env_ids] = 0.0
            env._prev_prev_leg_actions_for_acc[env_ids] = 0.0

    # 重置颈部 action 加速度缓存
    if hasattr(env, '_prev_neck_actions_for_acc'):
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            neck_joint_indices = list(range(5, 8))
            current_action = env.action_manager.action[env_ids][:, neck_joint_indices]
            env._prev_neck_actions_for_acc[env_ids] = current_action
            env._prev_prev_neck_actions_for_acc[env_ids] = current_action
        else:
            env._prev_neck_actions_for_acc[env_ids] = 0.0
            env._prev_prev_neck_actions_for_acc[env_ids] = 0.0

    # 重置用于关节加速度的关节速度缓存
    if hasattr(asset.data, '_prev_joint_vel'):
        # 获取待重置环境当前的关节速度
        joint_vel = asset.data.joint_vel[env_ids, :][:, asset_cfg.joint_ids]
        asset.data._prev_joint_vel[env_ids] = joint_vel

    # 重置接触频率跟踪
    if hasattr(env, '_contact_change_count'):
        env._contact_change_count[env_ids] = 0.0
    if hasattr(env, '_contact_change_timer'):
        env._contact_change_timer[env_ids] = 0.0
    if hasattr(env, '_prev_contacts_for_freq'):
        if "feet_ground_contact" in env.scene.sensors:
            contacts = env.scene.sensors["feet_ground_contact"].data.found[env_ids, :2]
            env._prev_contacts_for_freq[env_ids] = contacts

    # 重置脚部力平滑度跟踪
    if hasattr(env, '_prev_foot_forces'):
        if "feet_ground_contact" in env.scene.sensors:
            forces = env.scene.sensors["feet_ground_contact"].data.found[env_ids, :2].squeeze(-1)
            env._prev_foot_forces[env_ids] = forces

    # 重置执行器力矩变化率跟踪
    if hasattr(env, '_prev_actuator_forces'):
        env._prev_actuator_forces[env_ids] = asset.data.actuator_force[env_ids].clone()


def joint_accelerations_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    用 L2 平方范数惩罚关节加速度。
    关节加速度由关节速度的有限差分计算。

    Args:
        env: 环境
        asset_cfg: 资产配置

    Returns:
        形状为 (num_envs,) 的 penalty 张量 —— 各关节加速度平方和
    """
    asset: Entity = env.scene[asset_cfg.name]

    # 获取当前关节速度
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # 获取上一时刻关节速度（存在 asset data 中）
    # 注意：这里假设环境保存了上一时刻的关节速度
    if not hasattr(asset.data, '_prev_joint_vel'):
        # 首次调用时初始化
        asset.data._prev_joint_vel = joint_vel.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # 用有限差分计算关节加速度
    dt = env.step_dt
    joint_acc = (joint_vel - asset.data._prev_joint_vel) / dt

    # 保存当前速度供下一步使用
    asset.data._prev_joint_vel = joint_vel.clone()

    # 返回 L2 平方范数
    return torch.sum(torch.square(joint_acc), dim=1)


def leg_action_rate_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    惩罚腿部动作的变化率 (action_t - action_{t-1})。
    腿部关节是索引 0-4 和 9-13（共 10 个关节）。

    Args:
        env: 环境
        asset_cfg: 资产配置

    Returns:
        形状为 (num_envs,) 的 penalty 张量
    """
    # 获取腿部关节索引
    leg_joint_indices = list(range(0, 5)) + list(range(8, 13))

    # 只获取腿部关节的当前和上一时刻动作
    # 动作存在 env 中（假设该动作可用）
    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    # 获取 joint position action
    actions = env.action_manager.action
    if actions.shape[1] < 13:
        return torch.zeros(env.num_envs, device=env.device)

    leg_actions = actions[:, leg_joint_indices]

    if not hasattr(env, '_prev_leg_actions'):
        env._prev_leg_actions = leg_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_rate = leg_actions - env._prev_leg_actions
    env._prev_leg_actions = leg_actions.clone()

    return torch.sum(torch.square(action_rate), dim=1)


def neck_action_rate_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    惩罚颈部动作的变化率 (action_t - action_{t-1})。
    颈部关节是索引 5-8（共 4 个关节）。

    Args:
        env: 环境
        asset_cfg: 资产配置

    Returns:
        形状为 (num_envs,) 的 penalty 张量
    """
    # 获取颈部关节索引
    neck_joint_indices = list(range(5, 8))

    # 只获取颈部关节的当前和上一时刻动作
    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 13:
        return torch.zeros(env.num_envs, device=env.device)

    neck_actions = actions[:, neck_joint_indices]

    if not hasattr(env, '_prev_neck_actions'):
        env._prev_neck_actions = neck_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_rate = neck_actions - env._prev_neck_actions
    env._prev_neck_actions = neck_actions.clone()

    return torch.sum(torch.square(action_rate), dim=1)


def leg_action_acceleration_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    惩罚腿部动作加速度 (action_t - 2*action_{t-1} + action_{t-2})。
    腿部关节是索引 0-4 和 9-13（共 10 个关节）。

    Args:
        env: 环境
        asset_cfg: 资产配置

    Returns:
        形状为 (num_envs,) 的 penalty 张量
    """
    # 获取腿部关节索引
    leg_joint_indices = list(range(0, 5)) + list(range(8, 13))

    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 13:
        return torch.zeros(env.num_envs, device=env.device)

    leg_actions = actions[:, leg_joint_indices]

    if not hasattr(env, '_prev_leg_actions_for_acc'):
        env._prev_leg_actions_for_acc = leg_actions.clone()
        env._prev_prev_leg_actions_for_acc = leg_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_acc = leg_actions - 2 * env._prev_leg_actions_for_acc + env._prev_prev_leg_actions_for_acc

    env._prev_prev_leg_actions_for_acc = env._prev_leg_actions_for_acc.clone()
    env._prev_leg_actions_for_acc = leg_actions.clone()

    return torch.sum(torch.square(action_acc), dim=1)


def neck_action_acceleration_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    惩罚颈部动作加速度 (action_t - 2*action_{t-1} + action_{t-2})。
    颈部关节是索引 5-8（共 4 个关节）。

    Args:
        env: 环境
        asset_cfg: 资产配置

    Returns:
        形状为 (num_envs,) 的 penalty 张量
    """
    # 获取颈部关节索引
    neck_joint_indices = list(range(5, 8))

    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 13:
        return torch.zeros(env.num_envs, device=env.device)

    neck_actions = actions[:, neck_joint_indices]

    if not hasattr(env, '_prev_neck_actions_for_acc'):
        env._prev_neck_actions_for_acc = neck_actions.clone()
        env._prev_prev_neck_actions_for_acc = neck_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_acc = neck_actions - 2 * env._prev_neck_actions_for_acc + env._prev_prev_neck_actions_for_acc

    env._prev_prev_neck_actions_for_acc = env._prev_neck_actions_for_acc.clone()
    env._prev_neck_actions_for_acc = neck_actions.clone()

    return torch.sum(torch.square(action_acc), dim=1)


def _fallen_mask(
    env: ManagerBasedRlEnv,
    asset,
    gate_z_below: float,
    gate_tilt_above_deg: float,
) -> torch.Tensor:
    """每环境的 float mask：机器人被视为【跌倒】处为 1.0 —— trunk 高度
    低于 `gate_z_below` 或倾角超过 `gate_tilt_above_deg`。用于给恢复 reward 加门控，
    使它们只在真正跌倒时起作用，在干净行走时贡献恰好为零（无行走税 / 反弹刷分）。"""
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    quat = asset.data.root_link_quat_w
    # cos(tilt) = R22 = 1 - 2(qx² + qy²)
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = (z < gate_z_below) | (cos_tilt < math.cos(math.radians(gate_tilt_above_deg)))
    return fallen.float()


def feet_air_time_upright(
    env: ManagerBasedRlEnv,
    gate_tilt_above_deg: float = 40.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **air_time_kwargs,
) -> torch.Tensor:
    """velocity 模板的 feet_air_time，在【跌倒】时（tilt > gate）清零。

    velstand：躺在躯干上的机器人仍能通过 air-time 窗口有节奏地抬脚 —— 即观察到的
    "躺着抖腿"漏洞。Air time 只在直立时才有意义。
    """
    from mjlab.tasks.velocity.mdp import feet_air_time as _template_air_time
    reward = _template_air_time(env, **air_time_kwargs)
    asset: Entity = env.scene[asset_cfg.name]
    upright = 1.0 - _fallen_mask(env, asset, 0.0, gate_tilt_above_deg)
    return reward * upright


def upright_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """基于势能的直立塑形：每步 Δcos(tilt)。

    为朝直立的【进展】付奖励，对朝跌倒的进展收费，而对保持任何姿态恰好付【零】——
    因此没有任何状态能刷它（它取代的带门控 state-reward 在三次 velstand 运行中
    被从坐姿、平躺和 head-tripod 倾斜刷分）。基于势能的塑形是策略不变的
    （Ng 等人）：它加速恢复的学习而不产生新的最优解。完整的 prone→stand 恢复
    累计 Δ≈+1（× 权重）；跌倒过程在下落时付出同样的量。
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    cos_tilt = torch.nan_to_num(
        1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), nan=1.0
    )
    if not hasattr(env, "_upright_potential_prev"):
        env._upright_potential_prev = cos_tilt.clone()
    # 刚重置的环境：不受上一 episode 姿态产生的虚假 delta 影响。
    fresh = env.episode_length_buf <= 1
    env._upright_potential_prev[fresh] = cos_tilt[fresh]
    delta = cos_tilt - env._upright_potential_prev
    env._upright_potential_prev = cos_tilt.clone()
    return delta


def height_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ceiling: float = 0.115,
) -> torch.Tensor:
    """基于势能的高度塑形：每步 Δ min(trunk z, ceiling)。

    这是 ``upright_progress`` 在 z 轴上的配套项（velstand 下蹲终点经验）：恢复的最后一步
    —— 从深蹲伸直膝盖 —— 主要是在中等倾角下的【高度】变化，正是 Gaussian 的
    upright/pose reward 平坦、Δcos(tilt) 极小的地方。上升付奖励，下落收费，保持付零，
    因此步态起伏净收益为零，也没法刷分。封顶在 ``ceiling``（略低于完全站立的
    trunk z ≈ 0.117），使跳到站姿高度以上不再多付。
    """
    asset: Entity = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    pot = torch.clamp(z, max=ceiling)
    if not hasattr(env, "_height_potential_prev"):
        env._height_potential_prev = pot.clone()
    fresh = env.episode_length_buf <= 1
    env._height_potential_prev[fresh] = pot[fresh]
    delta = pot - env._height_potential_prev
    env._height_potential_prev = pot.clone()
    return delta


def fallen_state_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gate_tilt_above_deg: float = 40.0,
    release_tilt_below_deg: float | None = None,
    release_z_above: float | None = None,
) -> torch.Tensor:
    """跌倒时为 1.0（权重设为负）：对一直躺着施加平坦的每步税。没有它，静躺是
    ~0/step，而尝试恢复要付 action-rate/力矩 penalty —— 等待 fallen_too_long 回收
    反而是理性策略。（对坏状态施加惩罚是安全的；被刷分的是对坏状态加持的【正】reward。）

    设置 ``release_*`` 后，该税带【滞回】（velstand 下蹲终点经验）：跌倒触发它，
    并持续收费直到机器人真正站起来（tilt < release_tilt 且 z > release_z），而不是
    仅仅低于触发门控。没有它，刚好卡在 40° 门控下方的下蹲是零成本休息态 —— 恢复学会
    停在那里而不是完成站立。只在真正跌倒时触发，所以步态周期的倾角抖动不会被收税。"""
    asset: Entity = env.scene[asset_cfg.name]
    fallen = _fallen_mask(env, asset, 0.0, gate_tilt_above_deg).bool()
    if release_tilt_below_deg is None:
        return fallen.float()
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    up = cos_tilt > math.cos(math.radians(release_tilt_below_deg))
    if release_z_above is not None:
        up &= z > release_z_above
    if not hasattr(env, "_fallen_tax_armed"):
        env._fallen_tax_armed = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    fresh = env.episode_length_buf <= 1
    env._fallen_tax_armed[fresh] = False
    env._fallen_tax_armed |= fallen
    env._fallen_tax_armed &= ~up
    return env._fallen_tax_armed.float()


def recovery_success(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    fallen_tilt_deg: float = 40.0,
    min_fallen_s: float = 0.5,
    up_tilt_deg: float = 25.0,
    up_z: float = 0.105,
) -> torch.Tensor:
    """对【完成】恢复的一次性奖励：当某个环境已跌倒（tilt > fallen_tilt 持续
    ≥ min_fallen_s）后变为真正直立（tilt < up_tilt 且 trunk z > up_z）的那一帧触发。
    滞回：只有再次跌倒才会重新触发，因此在门控附近振荡得不到奖励。它提供了密集门控项
    所缺乏的稀疏但强力的终点梯度。
    """
    asset: Entity = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = cos_tilt < math.cos(math.radians(fallen_tilt_deg))
    up = (cos_tilt > math.cos(math.radians(up_tilt_deg))) & (z > up_z)
    if not hasattr(env, "_recovery_fallen_s"):
        env._recovery_fallen_s = torch.zeros(env.num_envs, device=env.device)
        env._recovery_armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fresh = env.episode_length_buf <= 1
    env._recovery_fallen_s[fresh] = 0.0
    env._recovery_armed[fresh] = False
    env._recovery_fallen_s = torch.where(
        fallen, env._recovery_fallen_s + env.step_dt, torch.zeros_like(env._recovery_fallen_s)
    )
    env._recovery_armed |= env._recovery_fallen_s >= min_fallen_s
    fired = env._recovery_armed & up
    env._recovery_armed &= ~fired
    return fired.float()


# ── VelStand / 保护性跌倒：伺服保护成本 ────────────────────────
# 什么会损坏 XL330（塑料齿轮组，约 288:1）：(1) 肢体被撞击
# 快速反驱 —— 即使断电，转子惯量经 288² 反射也让齿轮承受尖峰，所以接触时的
# 关节【加速度】是损伤代理指标；(2) 堵转 —— 肢体被身体压住而伺服
# 以最大力矩持续推会打齿 / 过热；(3) 外壳本身撞到
# 地面。下面三个成本都返回 >= 0（mjlab 风格）→ 用负权重，
# 并且在干净行走时约为 0，因此它们只对跌倒和糟糕的恢复
# 计费。


def servo_stall_penalty(
    env: ManagerBasedRlEnv,
    torque_thresh: float = 0.4,
    vel_thresh: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """正顶着某个不动物体的伺服数量。

    当 |actuator torque| > ``torque_thresh``（N·m）且 |joint velocity| < ``vel_thresh``
    （rad/s）时，伺服处于"堵转"。BAM 的电压有界力矩在训练 vin 范围内饱和于
    vin·kt/R ≈ 0.85–1.07 N·m；准静态站立时膝盖只承受几个 1e-2 N·m，所以 0.4 只在
    伺服真正在对抗卡死/堵塞时触发。返回每步计数（>= 0）——用负权重。
    """
    asset: Entity = env.scene[asset_cfg.name]
    torque = torch.nan_to_num(asset.data.actuator_force, nan=0.0)
    vel = torch.nan_to_num(_servo_joint_vel(env, asset), nan=0.0)
    stalled = (torque.abs() > torque_thresh) & (vel.abs() < vel_thresh)
    return stalled.float().sum(dim=1)


def servo_acc_spike_penalty(
    env: ManagerBasedRlEnv,
    acc_thresh: float = 300.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """各伺服的 ReLU(|关节加速度| - acc_thresh) 之和，加速度按控制频率对关节速度做
    有限差分。

    撞击反驱才是给齿轮组加载的原因（反射转子惯量）；一个被推倒的 CPU 基线在【所有】
    策略（保持 / 部分瘫软 / 断电）下，伺服关节的加速度峰值都达 450–550 rad/s²，
    因此策略的杠杆在于它如何、在哪里落地。状态存在 env 上；重置那一步会被清零，
    使新 episode 的首帧不会计到上一 episode 的速度。返回 >= 0（超过阈值的 rad/s²）
    ——用负权重。
    """
    asset: Entity = env.scene[asset_cfg.name]
    vel = torch.nan_to_num(_servo_joint_vel(env, asset), nan=0.0)
    prev = getattr(env, "_servo_acc_prev_vel", None)
    if prev is None or prev.shape != vel.shape:
        prev = vel.detach().clone()
    acc = (vel - prev) / env.step_dt
    if hasattr(env, "episode_length_buf"):
        reset_mask = (env.episode_length_buf <= 1).unsqueeze(1)
        acc = torch.where(reset_mask, torch.zeros_like(acc), acc)
    env._servo_acc_prev_vel = vel.detach().clone()
    return torch.clamp(acc.abs() - acc_thresh, min=0.0).sum(dim=1)


def _fallen_scale(env: ManagerBasedRlEnv, asset: Entity, fallen_scale: float, gate_tilt_above_deg: float) -> torch.Tensor:
    fallen = _fallen_mask(env, asset, 0.0, gate_tilt_above_deg).bool()
    return torch.where(fallen, torch.full_like(fallen, fallen_scale, dtype=torch.float), torch.ones(fallen.shape, device=env.device))


def action_rate_l2_fallen_scaled(
    env: ManagerBasedRlEnv,
    fallen_scale: float = 0.1,
    gate_tilt_above_deg: float = 40.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """mjlab 的 action_rate_l2，在 tilt > gate 时乘以 ``fallen_scale``。

    protective-fall velstand 第 1 次运行（2026-09-09）的教训：从已部署的 walk warm start
    会把 action_rate 从第 0 步就钉在最终权重（-1.0，是整个 reward 栈里最大的成本，
    -1.5..-2/step）。相对于平坦的 -0.5 fallen 税，任何起身尝试都比为躺着代价更高 ——
    attempt-tax 冻结：800 iter 内 0 次恢复，跌倒时 action rate 只有直立时的 1/5。
    walk 需要完整的平滑税；跌倒的机器人需要被允许尝试。返回 >= 0 → 负权重，
    与 action_rate_l2 相同。
    """
    asset: Entity = env.scene[asset_cfg.name]
    base = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return base * _fallen_scale(env, asset, fallen_scale, gate_tilt_above_deg)


def joint_torque_rate_l2_fallen_scaled(
    env: ManagerBasedRlEnv,
    fallen_scale: float = 0.1,
    gate_tilt_above_deg: float = 40.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """跌倒时（tilt > gate）的 joint_torque_rate_l2 × ``fallen_scale``（见 action_rate_l2_fallen_scaled）。"""
    asset: Entity = env.scene[asset_cfg.name]
    return joint_torque_rate_l2(env, asset_cfg) * _fallen_scale(env, asset, fallen_scale, gate_tilt_above_deg)


def body_upright_linear(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gate_z_below: float | None = None,
    gate_tilt_above_deg: float = 40.0,
) -> torch.Tensor:
    """机身直立度的线性 reward —— 在每个倾角都提供梯度。

    完全直立时返回 +1，水平（俯卧/仰卧）时为 0，倒置时为 -1。与 flat_orientation
    （Gaussian）不同，它在各处梯度非零，因此机器人即使从俯卧开始也总有朝直立旋转的信号。

    计算为机身局部 Z 轴在世界系下的 z 分量，对四元数 [w, x, y, z] 等于
    R[2,2] = 1 - 2*(qx² + qy²)。
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4): [w, x, y, z]
    qx = quat[:, 1]
    qy = quat[:, 2]
    reward = 1.0 - 2.0 * (qx * qx + qy * qy)
    if gate_z_below is not None:
        # 恢复门控变体（velstand）：只在跌倒时生效，恰好
        # 在干净行走时为零，因此不会稀释跟踪 reward。
        reward = reward * _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg)
    return reward


def body_upright_gaussian(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.1,
) -> torch.Tensor:
    """倾角大小的 Gaussian reward —— 强烈拉向完全竖直。

    补充 ``body_upright_linear``（即 ``cos(tilt)``，其梯度 ``sin(tilt)`` 在目标处
    【消失】）。该 Gaussian 的梯度在接近竖直时非零，随远离而衰减，因此在线性版本最弱
    的区域产生强微分拉力。

    用 ``2*(qx² + qy²) = 1 - cos(tilt) ≈ tilt²/2`` 作为倾角平方的代理，施加
    ``exp(-tilt²/std²)``。默认 std=0.1 rad ≈ 5.7°。
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)  # ≈ 1 − cos(tilt); small-angle: tilt²/2
    return torch.exp(-tilt_sq / (std * std))


def upright_gaussian_at_height(
    env: ManagerBasedRlEnv,
    std: float,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """``body_upright_gaussian`` 乘以 trunk z 上的 smoothstep。

    当 ``z >= height_high`` 时给出完整的 Gaussian 直立 reward，``z <= height_low``
    时为零，中间平滑过渡。当直立激励只应在目标站立高度施加时用这个 —— 否则策略
    会找到"低蹲且竖直"的局部最优，一直不站起来也能拿直立 reward。
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)
    upright_g = torch.exp(-tilt_sq / (std * std))
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    t = torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return upright_g * smooth


def body_ang_vel_at_height(
    env: ManagerBasedRlEnv,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    tilt_full_deg: float | None = None,
    tilt_zero_deg: float = 45.0,
) -> torch.Tensor:
    """由 trunk z（可选再加 tilt）门控的 trunk ``sum(ω_xy²)`` 惩罚。

    高度门控的到站阻尼器：低于 ``height_low`` 为零（地面恢复 —— 翻身/翻滚需要大幅
    trunk 旋转，必须保持自由），高于 ``height_high`` 为满值。与 mjlab 的
    body_angular_velocity_penalty 公式相同（世界系 ω_xy，z 轴旋转自由），但返回门控后的
    【正】成本；用负权重。

    ``tilt_full_deg``（可选但强烈推荐）：额外按 tilt 门控 —— 仅当 tilt ≤ tilt_full_deg
    时满成本，≥ tilt_zero_deg 时为零，中间 smoothstep。教训（2026-07 那次破坏前恢复的
    运行）：只有高度门控时，弯腰起身最后的伸直（tilt 60°→0 发生在 z 门控【内部】）
    本身就是大幅 trunk 旋转 —— 对它收税会在终点前筑起 reward 墙，策略反而停在门控下方
    弯腰。加了 tilt 门控后，接近竖直是自由的；只有竖直【附近】的残余晃动
    （过冲→倾斜→重试振荡）被阻尼。
    """
    asset = env.scene[asset_cfg.name]
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
    cost = torch.sum(torch.square(ang_vel[:, :2]), dim=1)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    t = torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    gate = t * t * (3.0 - 2.0 * t)
    if tilt_full_deg is not None:
        quat = asset.data.root_link_quat_w
        cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        tilt_deg = torch.rad2deg(torch.acos(cos_tilt.clamp(-1.0, 1.0)))
        s = torch.clamp(
            (tilt_zero_deg - tilt_deg) / max(tilt_zero_deg - tilt_full_deg, 1e-6),
            0.0,
            1.0,
        )
        gate = gate * (s * s * (3.0 - 2.0 * s))
    return cost * gate


def standing_composite_score(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    joint_indices: list,
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """平滑的乘性目标态分数（三个 Gaussian 的乘积）。

    返回 ``height_score * upright_score * pose_score``，各 ∈ [0, 1]。因为因子【相乘】，
    任一项的不足都会拉垮整个 reward —— 策略无法靠 3 项中 2 项完美就拿到 80%。
    梯度处处非零，因此分数在起身过程中也有效（不像二元奖励只在目标态才有效）。

    用于打破纳什均衡式的折中（例如满足加法 reward 部分和的"高度正确但 trunk 倾斜"盆地）。
    """
    asset = env.scene[asset_cfg.name]

    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    height_score = torch.exp(-((z - target_height) / height_std) ** 2)

    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)
    upright_score = torch.exp(-tilt_sq / (upright_std * upright_std))

    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    pose_err_sq = ((joint_pos - target) ** 2).mean(dim=-1)
    pose_score = torch.exp(-pose_err_sq / (pose_std * pose_std))

    return height_score * upright_score * pose_score


def standing_success_bonus(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_tol: float,
    upright_threshold: float,
    pose_tol: float,
    joint_indices: list,
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """二元奖励：当且仅当高度、直立度和姿态都在容差内时为 1.0。

    创建一个梯度型的 pose/upright/height reward 自身无法完全匹配的离散目标态吸引子。
    周围的折中（倾斜 trunk 以平衡头部前移的 CoM、停在离目标 z 差 1cm 处等）能拿到部分
    梯度积分但【零】奖励 —— 奖励只在真正目标态可得，因此当其余 reward 把策略带到
    接近目标时，它改变了策略的相对偏好。
    """
    asset = env.scene[asset_cfg.name]

    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    height_ok = (z - target_height).abs() <= height_tol

    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    upright = 1.0 - 2.0 * (qx * qx + qy * qy)
    upright_ok = upright >= upright_threshold

    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    pose_err = (joint_pos - target).abs().max(dim=-1).values  # tightest joint
    pose_ok = pose_err <= pose_tol

    return (height_ok & upright_ok & pose_ok).float()


def com_upward_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    max_height: float = 0.08,
    gate_z_below: float | None = None,
    gate_tilt_above_deg: float = 40.0,
    max_vz: float | None = None,
) -> torch.Tensor:
    """奖励向上的 CoM 速度以激励动态起身动作。

    由高度门控：仅在 CoM 低于 `max_height`（站立目标）时生效。一旦站起，reward 归零，
    因此机器人没有动机一直蹲着刷向上速度 reward。

    ``max_vz``（可选）：限制被奖励的速度。不封顶时 reward 与 vz 成正比，对爆发式起跳
    每步付【更多】—— 这是暴力起身激励。封顶后，任何 ≥ max_vz 的起身都拿同样值，
    于是刚好达到上限的最温和起身成为最优（|a_z| 惩罚随后挑出平滑的那个）。引导特性
    保留：任何向上运动仍然立刻得到奖励。
    """
    asset: Entity = env.scene[asset_cfg.name]
    # nan_to_num：接触不稳定时 MuJoCo 可能产生 NaN；按 z=0 处理
    com_z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    below_target = (com_z < max_height).float()
    reward = torch.clamp(vz, min=0.0, max=max_vz) * below_target
    if gate_z_below is not None:
        # 恢复门控（velstand）：没有门控，步态中 trunk 穿过 max_height 时
        # 会为下蹲-起身付奖励 → 反弹激励。
        reward = reward * _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg)
    return reward


def fallen_too_long(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gate_z_below: float = 0.10,
    gate_tilt_above_deg: float = 40.0,
    max_duration_s: float = 5.0,
) -> torch.Tensor:
    """终止连续【跌倒】达 `max_duration_s` 的环境。

    对混合行走与跌倒恢复的环境（velstand）：fell_over 终止被 curriculum 禁用，使策略
    能尝试恢复，但没有兜底的话，失败的恢复会在整段 20 s episode 里刷恢复 reward，
    饿死行走数据（审计：行走占比约 25%）。这给每次跌倒一个公平的恢复窗口，然后回收环境。
    """
    asset: Entity = env.scene[asset_cfg.name]
    fallen = _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg).bool()
    if not hasattr(env, "_fallen_timer_s"):
        env._fallen_timer_s = torch.zeros(env.num_envs, device=env.device)
    # 刚重置的环境从干净的计时器开始。
    env._fallen_timer_s[env.episode_length_buf <= 1] = 0.0
    env._fallen_timer_s = torch.where(
        fallen, env._fallen_timer_s + env.step_dt, torch.zeros_like(env._fallen_timer_s)
    )
    return env._fallen_timer_s >= max_duration_s


def robot_state_is_nan(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    sensor_names: tuple[str, ...] = (),
) -> torch.Tensor:
    """终止 MuJoCo 产生 NaN 关节位置的环境。

    MuJoCo 的接触求解器在极端穿透或冲量下（例如机器人高速落地）可能溢出为 NaN。
    NaN 仿真状态会传播到观测，破坏策略网络权重。

    立即终止可在级联扩散前重置环境：
    - 返回给 runner 的观测来自有效的重置状态。
    - 避免后续步出现 NaN reward。

    注意：这个终止步的 reward 仍可能因仿真而是 NaN；mjlab 在重置【之前】计算 reward
    （见 manager_based_rl_env.py 的 step()）。我们的自定义 reward 函数内部用 nan_to_num
    防 NaN，但标准 mjlab reward 此处仍可能是 NaN。一个 NaN reward 可以容忍，因为
    done=True 阻止它经 GAE 反向传播。

    覆盖【全部】物理状态，不只是 joint_pos：接触发散时爆炸的往往是基础的 FREE-JOINT
    （位置/朝向/速度）或被动 ROUES（轮子），而非受驱关节。这些量喂给 critic 的 obs 项
    （base_lin_vel、base_ang_vel、projected_gravity、wheel_vel）；不监控它们，环境就不会
    重置，NaN 到达 obs → rsl_rl 的 check_nan 会杀掉整个训练。我们检测非有限值
    （NaN 和 inf，inf 在 projected_gravity 归一化下游会变成 NaN）。
    """
    asset: Entity = env.scene[asset_cfg.name]
    d = asset.data
    bad = ~torch.isfinite(d.joint_pos).all(dim=1)
    bad |= ~torch.isfinite(d.joint_vel).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_pos_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_quat_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_lin_vel_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_ang_vel_w).all(dim=1)

    # 接触力可能比 qpos/qvel 早一步爆炸：MuJoCo 求解一个
    # 退化的接触可能在积分状态仍为有限值时，把力变成 inf/NaN 冲量。
    # 那个力喂给仅供 critic 用的 `foot_contact_forces` obs
    # (sign(F)*log1p(|F|))，而上面的状态检查【不覆盖】它 —— 所以环境
    # 没有被重置，NaN 到达 runner 的 check_nan，杀掉了
    # 整次运行（2026-08-21 崩溃，Velocity2-Rough-Backlash 带 hfield 斜坡）。
    for name in sensor_names:
        if name not in env.scene.sensors:
            continue
        force = getattr(env.scene.sensors[name].data, "force", None)
        if force is not None:
            bad |= ~torch.isfinite(force).flatten(start_dim=1).all(dim=1)
    return bad


def root_height_below(
    env: ManagerBasedRlEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """当 trunk 在世界 z 上低于 ``min_height`` 时终止。

    roller_slope 用它表示"掉进虚空"：地形在斜坡底部有一块出口平台，因此正常
    下坡绝不会低于最低出口平台的高度。把 min_height 选在该高度之下 => 终止只在
    机器人脱离实体、坠入虚空时触发。与斜坡的具体几何（长度/坡度）无关。
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] < min_height


def descent_speed_reward(
    env: ManagerBasedRlEnv,
    cap: float = 0.8,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """奖励朝斜坡【下方】的前进速度（世界 +x）。

    斜坡沿 +x 下降，因此世界 x 方向的线速度衡量下降进度。封顶在 ``cap`` m/s：
    鼓励顺势滑行而不推动自己越滑越快。机器人后退/上坡时（vx < 0）为零。
    没有这个 reward，最优解是静止直立（机器人"刹车"而不是滑行）。NaN 安全。
    """
    asset: Entity = env.scene[asset_cfg.name]
    vx = torch.nan_to_num(
        asset.data.root_link_lin_vel_w[:, 0], nan=0.0, posinf=0.0, neginf=0.0
    )
    return torch.clamp(vx, min=0.0, max=cap)


def reset_rolling_entry(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    speed_range: tuple = (0.25, 0.45),
    wheel_radius: float = 0.0175,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """以纯滚动（无滑移）出发（给轮子一个初速度）。

    为每个 env 采样一个前进速度 v；设基础【线速度】（世界 x）= v，同时 4 个被动轮
    的【旋转速度】= v / r，因此 ω·r = v => 接触处零滑移。避免旧的"只推基础"的
    突兀感（基础在动、轮子不动 = 第一步剧烈打滑）。要在 reset_base【之后】执行
    （它负责摆放基座；不要再给它 velocity_range）。
    """
    asset: Entity = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = int(env_ids.shape[0])
    lo, hi = speed_range
    v = torch.rand(n, device=env.device) * (hi - lo) + lo  # (n,) vitesse avant

    # 基础速度（世界系）：仅 +x。
    root_vel = torch.zeros(n, 6, device=env.device)
    root_vel[:, 0] = v
    asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

    # 4 个被动轮的转速 = v / r（正 = 前进，见 wheel_speed）。
    wheel_ids = []
    for name in ("passive_LF_?wheel", "passive_LR_?wheel", "passive_RF_?wheel", "passive_RR_?wheel"):
        ids, _ = asset.find_joints(name)
        wheel_ids.append(ids[0])
    wheel_ids_t = torch.tensor(wheel_ids, device=env.device)
    omega = (v / wheel_radius).unsqueeze(1).repeat(1, len(wheel_ids))  # (n, 4)
    asset.write_joint_velocity_to_sim(omega, joint_ids=wheel_ids_t, env_ids=env_ids)


def wheel_glide_reward(
    env: ManagerBasedRlEnv,
    cap_speed: float = 0.35,
    wheel_radius: float = 0.0175,
) -> torch.Tensor:
    """奖励轮子【向前滚动】（滑行），带封顶。

    与 descent_speed（【基座】速度，可以靠"跑"/推动达到）不同，这里奖励被动轮的
    旋转 = 真正的滚动滑行。与任何指令无关（斜坡任务指令为零：滑行来自重力）。
    封顶在 ``cap_speed``（滚动速度的 m/s）-> 【没有】加速超过它的激励；轮子倒转
    （上坡）时为零。NaN 安全。
    """
    asset: Entity = env.scene["robot"]
    lf, _ = asset.find_joints("passive_LF_?wheel")
    lr, _ = asset.find_joints("passive_LR_?wheel")
    rf, _ = asset.find_joints("passive_RF_?wheel")
    rr, _ = asset.find_joints("passive_RR_?wheel")
    vel = asset.data.joint_vel
    # 前进时 4 个轮子正向旋转（见 wheel_speed_reward）。
    omega = (vel[:, lf[0]] + vel[:, lr[0]] + vel[:, rf[0]] + vel[:, rr[0]]) / 4.0
    speed = torch.nan_to_num(omega * wheel_radius, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(speed, min=0.0, max=cap_speed)


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    """
    存活 reward（未被终止）

    Args:
        env: 环境

    Returns:
        形状为 (num_envs,) 的 reward 张量 —— 所有环境均为 1
    """
    return torch.ones(env.num_envs, device=env.device)


def com_height_target(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    target_height_min: float = 0.1,
    target_height_max: float = 0.15,
) -> torch.Tensor:
    """
    让质心保持在目标高度范围内的 reward。
    在范围内返回正 reward，范围外返回负惩罚。

    Args:
        env: 环境
        asset_cfg: 资产配置
        target_height_min: CoM 的最小目标高度（米）
        target_height_max: CoM 的最大目标高度（米）

    Returns:
        形状为 (num_envs,) 的 reward 张量
    """
    asset: Entity = env.scene[asset_cfg.name]

    # 相对地形生成原点的高度（世界 z 减去地形 z）。
    # 平地上 env_origins[:, 2] 为 0，因此这里无条件安全。
    # nan_to_num：接触不稳定时 MuJoCo 可能产生 NaN；按 z=0 处理
    # 使惩罚有限（很小，因为 0 接近目标范围）。
    com_height = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )

    # 在范围内给奖励，范围外给惩罚
    # 用随距范围距离二次增长的平滑惩罚
    below_min = com_height < target_height_min
    above_max = com_height > target_height_max
    in_range = ~(below_min | above_max)

    # 计算出范围外的惩罚
    penalty_below = torch.square(com_height - target_height_min) * below_min.float()
    penalty_above = torch.square(com_height - target_height_max) * above_max.float()

    # Reward：范围内 +1，范围外为 -距离的平方
    reward = in_range.float() - (penalty_below + penalty_above)

    return reward


def crouch_height_target(
    phase: torch.Tensor,
    height_low: float,
    height_high: float,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
) -> torch.Tensor:
    """沿相位 [0,1) 的"梯形" trunk 高度目标。

    phase ∈ [0, hold_lo)      : 下降   height_high -> height_low
    phase ∈ [hold_lo, hold_hi): 平台  height_low  （下蹲滑行）
    phase ∈ [hold_hi, 1.0)    : 上升   height_low  -> height_high

    Args:
        phase: (B,) 每个环境的相位，范围 [0, 1)。
        height_low: 下蹲时的 trunk 高度（m）。
        height_high: 站立时的 trunk 高度（m）。
        hold_lo, hold_hi: 低位平台在相位中的边界比例。
    Returns:
        (B,) 目标高度，单位米。
    """
    descend = phase < hold_lo
    hold = (phase >= hold_lo) & (phase < hold_hi)

    frac_d = phase / hold_lo
    t_descend = height_high + (height_low - height_high) * frac_d

    t_hold = torch.full_like(phase, height_low)

    frac_r = (phase - hold_hi) / (1.0 - hold_hi)
    t_rise = height_low + (height_high - height_low) * frac_r

    return torch.where(descend, t_descend, torch.where(hold, t_hold, t_rise))


def crouch_glide_reward_from_values(
    com_height: torch.Tensor,
    cmd_cos: torch.Tensor,
    cmd_sin: torch.Tensor,
    height_low: float,
    height_high: float,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
    std: float = 0.02,
) -> torch.Tensor:
    """对高度目标跟踪的 Gaussian reward（纯函数）。

    从 [cos, sin] 解码相位，再把测得高度与梯形目标比较。返回
    exp(-((h - target)/std)^2) ∈ (0, 1]。
    """
    phase = (torch.atan2(cmd_sin, cmd_cos) / (2 * torch.pi)) % 1.0
    target = crouch_height_target(phase, height_low, height_high, hold_lo, hold_hi)
    return torch.exp(-((com_height - target) / std) ** 2)


def crouch_glide_height_by_phase(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    height_low: float = 0.075,
    height_high: float = 0.11,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """主 reward：沿相位跟踪 trunk 高度目标。

    CoM 高度按 `com_height_target` 计算（世界 z 减地形原点，nan->0）。相位来自
    GroundPick 指令。
    """
    asset: Entity = env.scene[asset_cfg.name]
    com_height = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    cmd = env.command_manager.get_command(command_name)
    return crouch_glide_reward_from_values(
        com_height, cmd[:, 0], cmd[:, 1],
        height_low, height_high, hold_lo, hold_hi, std,
    )


def forward_speed_reward(
    env: ManagerBasedRlEnv,
    vel_ref: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """奖励 trunk 的前进速度（保持冲量 / 不刹车）。

    与指令无关（指令携带的是相位，不是速度）。tanh(clamp(vx, 0)/vel_ref) → 饱和于
    ~1，绝不奖励后退。
    """
    asset: Entity = env.scene[asset_cfg.name]
    vx = asset.data.root_link_lin_vel_b[:, 0]
    return torch.tanh(torch.clamp(vx, min=0.0) / vel_ref)


def crouch_pose_blend(
    phase: torch.Tensor,
    descent_end: float,
    hold_end: float,
    rise_end: float,
) -> torch.Tensor:
    """沿相位 [0,1) 混合 0..1 —— 0 = 站姿，1 = 下蹲姿态。

    [0, descent_end)      : 0 -> 1  （下蹲）
    [descent_end, hold_end): 1      （低位 / 下蹲）
    [hold_end, rise_end)  : 1 -> 0  （起身）
    [rise_end, 1.0)       : 0       （高位 / 站立，休息）
    """
    b = torch.zeros_like(phase)
    descend = phase < descent_end
    b = torch.where(descend, phase / descent_end, b)
    low = (phase >= descent_end) & (phase < hold_end)
    b = torch.where(low, torch.ones_like(phase), b)
    rise = (phase >= hold_end) & (phase < rise_end)
    b = torch.where(rise, 1.0 - (phase - hold_end) / (rise_end - hold_end), b)
    return b


def _crouch_pose_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    crouch_pose: dict,
    descent_end: float,
    hold_end: float,
    rise_end: float,
    stand_pose: Optional[dict] = None,
):
    """相位插值下蹲姿态的 (cur, target) 关节张量。

    目标按每个关节在 STAND <-> crouch_pose 之间用 4 段混合 b(phase) ∈ [0,1]
    （0 = 站立，1 = 下蹲）插值。STAND 取给定的 `stand_pose`，否则取模型默认值
    （HOME）。关节【按名称】解析，因此 roller 机器人上被动轮的交错永远不会错位索引。
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
    blend = crouch_pose_blend(phase, descent_end, hold_end, rise_end)   # (B,) 0..1

    names = list(crouch_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]
    default = asset.data.default_joint_pos[:, ids]                     # (B,k)

    stand = default.clone()                                            # source pose
    if stand_pose:
        for j, n in enumerate(names):
            if n in stand_pose:
                stand[:, j] = stand_pose[n]
    crouch = torch.tensor(
        [crouch_pose[n] for n in names], device=env.device, dtype=default.dtype
    ).unsqueeze(0)                                                     # (1,k)

    target = stand + blend.unsqueeze(-1) * (crouch - stand)           # (B,k)
    cur = asset.data.joint_pos[:, ids]                                # (B,k)
    return cur, target


def crouch_glide_pose_by_phase(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    crouch_pose: Optional[dict] = None,
    stand_pose: Optional[dict] = None,
    std: float = 0.4,
    descent_end: float = 0.10,
    hold_end: float = 0.50,
    rise_end: float = 0.60,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """对相位插值关节姿态（站姿 <-> 下蹲）的 Gaussian 匹配。

    指令式 reward：告诉机器人在每个相位应处于的确切关节构型。起身（目标 =
    stand_pose）与下蹲（目标 = crouch_pose）被完全同样地奖励 —— 构造上对称。
    """
    cur, target = _crouch_pose_error(
        env, asset_cfg, command_name, crouch_pose or {},
        descent_end, hold_end, rise_end, stand_pose,
    )
    return torch.exp(-((cur - target) / std) ** 2).mean(dim=-1)


def crouch_glide_pose_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    crouch_pose: Optional[dict] = None,
    stand_pose: Optional[dict] = None,
    descent_end: float = 0.10,
    hold_end: float = 0.50,
    rise_end: float = 0.60,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """朝相位插值下蹲姿态的 L1 引导（负惩罚）。

    处处梯度恒定 —— 即使上面的 Gaussian 在远离目标时已饱和到 ~0，也能给策略
    指向目标姿态的方向。
    """
    cur, target = _crouch_pose_error(
        env, asset_cfg, command_name, crouch_pose or {},
        descent_end, hold_end, rise_end, stand_pose,
    )
    return -(cur - target).abs().mean(dim=-1)


def crouch_forward_lean(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pitch: float = 0.08,
    std: float = 0.1,
    descent_end: float = 0.10,
    hold_end: float = 0.50,
    rise_end: float = 0.60,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",)),
) -> torch.Tensor:
    """下蹲时 trunk 略微【前倾】（由 crouch 混合值门控）。

    对抗髋部快速屈曲引起的向后翻转。pitch 代理 = projected_gravity_b[:,0]
    （正 = 向前，已验证）。门控（混合值）在下降+低位时为 1，站立时为 0 →
    【只】在下蹲时施加偏置。target_pitch 小 = "只一点点"。
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
    gate = crouch_pose_blend(phase, descent_end, hold_end, rise_end)
    lean = asset.data.projected_gravity_b[:, 0]
    return gate * torch.exp(-((lean - target_pitch) ** 2) / std ** 2)


def neck_joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    惩罚颈部关节速度以保持头部稳定。
    颈部关节是索引 5-8（共 4 个关节）。

    Args:
        env: 环境
        asset_cfg: 资产配置

    Returns:
        形状为 (num_envs,) 的 penalty 张量
    """
    asset: Entity = env.scene[asset_cfg.name]

    # 获取颈部关节索引（neck_pitch、head_pitch、head_yaw、head_roll）。
    # 伺服视图：passive_* 关节（backlash、轮子）不会使索引偏移。
    neck_joint_indices = list(range(5, 8))
    joint_vel = _servo_joint_vel(env, asset)
    neck_joint_vel = joint_vel[:, neck_joint_indices]

    # 返回颈部关节速度的 L2 平方范数
    return torch.sum(torch.square(neck_joint_vel), dim=1)


def leg_joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    惩罚腿部关节速度以鼓励更平滑、更低动态的动作。
    腿部关节是索引 0-4 和 8-13（共 10 个关节）。

    Args:
        env: 环境
        asset_cfg: 资产配置

    Returns:
        形状为 (num_envs,) 的 penalty 张量
    """
    asset: Entity = env.scene[asset_cfg.name]

    # 获取腿部关节索引（左髋-踝：0-4，右髋-踝：9-13）。
    # 伺服视图：passive_* 关节（backlash、轮子）不会使索引偏移。
    leg_joint_indices = list(range(0, 5)) + list(range(8, 13))
    joint_vel = _servo_joint_vel(env, asset)
    leg_joint_vel = joint_vel[:, leg_joint_indices]

    # 返回腿部关节速度的 L2 平方范数
    return torch.sum(torch.square(leg_joint_vel), dim=1)

# _NECK_JOINT_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*(neck|head).*",))
# _HIP_PITCH_KNEE_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*(hip_pitch|knee).*",))
# _ROLLER_FEET_SITE_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
_NECK_JOINT_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!.*_backlash).*neck_joint_.*",))
_HIP_PITCH_KNEE_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!.*_backlash).*(leg_joint_03|leg_joint_04).*",))
_ROLLER_FEET_SITE_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))


def feet_flat_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROLLER_FEET_SITE_CFG,
    sensor_name: str | None = None,
) -> torch.Tensor:
    """惩罚脚 site 不与地面平行。

    脚 site 坐标系在放平时 Z+ 朝上。我们把一个单位重力向量（朝下）投影到每个脚
    site 的局部坐标系。放平时，重力在 site 系中映射为 [0,0,-1]（xy=0，penalty=0）。
    任何倾斜都会让 Z 偏离世界上方，产生非零 xy 分量。

    最大值约每脚 2.0（脚完全侧翻），总计约 4.0。

    给出 ``sensor_name`` 时，每脚的 penalty 由该脚自身的接地【门控】：离地（摆动）脚
    可以自由倾斜，只有支撑脚被要求保持平放（使其轮子保持抓地）。没有这个门控，
    penalty 会惩罚一步所需的Recovery抬脚 —— 它的最小化方式是让两只脚都平贴地面，
    即那种擦地滑行。假设 site 顺序（左、右）与传感器槽位顺序（ankle_l_v1、
    ankle_r_v1）一致 —— 本模型中都是左在前。

    Bug 记录：必须【逐环境】用 dim=-1 归一化重力。torch.norm() 不带 dim 会在所有
    env × 3 维上算一个标量，使向量幅值约 1/sqrt(num_envs) → penalty 小约 num_envs 倍。
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse
    import torch.nn.functional as F

    asset: Entity = env.scene[asset_cfg.name]
    gravity_w_n = F.normalize(asset.data.gravity_vec_w, dim=-1)  # (B, 3), unit vector per env

    foot_quats = asset.data.site_quat_w[:, asset_cfg.site_ids, :]  # (B, N_feet, 4)
    per_foot = torch.zeros(env.num_envs, foot_quats.shape[1], device=env.device)
    for i in range(foot_quats.shape[1]):
        proj = quat_apply_inverse(foot_quats[:, i, :], gravity_w_n)  # (B, 3)
        per_foot[:, i] = torch.sum(torch.square(proj[:, :2]), dim=1)  # xy² only

    if sensor_name is not None:
        from mjlab.sensor import ContactSensor
        sensor: ContactSensor = env.scene[sensor_name]
        contact_time = sensor.data.current_contact_time  # (B, N_feet)
        assert contact_time is not None
        per_foot = per_foot * (contact_time > 0.0).float()

    return per_foot.sum(dim=1)


def feet_tiptoe_alignment(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROLLER_FEET_SITE_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """奖励每个脚 site 的局部 x 轴朝下 —— 踮脚姿态。

    放平时脚 site 的 x 大致朝前（水平）。把脚向前俯（脚跟抬、脚尖下）会让 x 转向
    世界 -Z。我们奖励脚 x 轴的 z 分量为 -1（完全朝下）。

    每脚：对齐度 ∈ [-1, 1]，两脚求和 ∈ [-2, 2]。

    由 |vel_cmd_xy| > command_threshold 门控，使策略不必在静止时踮脚 —— 只在行走时。
    配套的 feet_flat_penalty 在本任务中【不使用】；两者会互相冲突。
    """
    asset: Entity = env.scene[asset_cfg.name]
    quats = asset.data.site_quat_w[:, asset_cfg.site_ids, :]  # (B, N, 4) [w, x, y, z]
    w, qx, qy, qz = quats[:, :, 0], quats[:, :, 1], quats[:, :, 2], quats[:, :, 3]
    x_axis_z = 2.0 * (qx * qz - w * qy)  # (B, N) — z-component of local x-axis in world
    alignment = (-x_axis_z).sum(dim=-1)  # +1 per foot when pointing straight down

    cmd = env.command_manager.get_command(command_name)
    cmd_mag = torch.linalg.norm(cmd[:, :2], dim=1)
    active = (cmd_mag > command_threshold).float()
    return alignment * active


def hip_pitch_knee_vel_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _HIP_PITCH_KNEE_CFG,
) -> torch.Tensor:
    """惩罚 hip_pitch 和 knee 的关节速度（L2 平方）。

    行走需要这些矢状面关节快速振荡。滑冰则用 hip_roll 侧向发力并以极小的矢状面运动
    滑行。本项惩罚那种振荡，而不妨碍静态平衡调整。
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def neck_joint_pos_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _NECK_JOINT_CFG,
    pattern: str = r".*(neck|head).*",
) -> torch.Tensor:
    """惩罚颈/头关节位置相对默认值的偏差（L2 平方）。

    每次调用都用 find_joints()，以避免同一个 SceneEntityCfg 单例在不同关节布局的机器人
    之间复用时索引缓存过期（例如 walk 机器人 vs rollers 机器人，被动轮会让颈部索引偏移）。

    ``pattern`` 选择被计入的关节（默认：整个颈部 + 头部）。spin 任务传入一个
    【排除】`head_yaw` 的模式，让头部充当启动旋转的飞轮。
    """
    asset: Entity = env.scene[asset_cfg.name]
    # 排除 passive_* 关节（backlash 铰链的名字也含 "neck"/"head"）。
    if not pattern.startswith(r"^(?!passive_)"):
        pattern = r"^(?!passive_)" + pattern.lstrip("^")
    joint_ids, _ = asset.find_joints(pattern)
    error = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return torch.sum(torch.square(error), dim=1)


def joint_torques_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    惩罚执行器力（力矩）以鼓励节能动作。

    Args:
        env: 环境
        asset_cfg: 资产配置

    Returns:
        形状为 (num_envs,) 的 penalty 张量 —— 执行器力平方和
    """
    asset: Entity = env.scene[asset_cfg.name]

    # 获取执行器力（执行器空间中的标量驱动量）
    actuator_forces = asset.data.actuator_force

    # 返回 L2 平方范数
    return torch.sum(torch.square(actuator_forces), dim=1)


def joint_torque_rate_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """惩罚执行器力矩的变化率（齿轮箱冲击的代理指标）。

    机器人撞击地面、执行器抵抗冲量时会出现突然的力矩尖峰。惩罚该变化率
    鼓励柔和落地和保护齿轮箱的平滑力过渡。

    返回与上一步力矩差的平方和。
    """
    asset: Entity = env.scene[asset_cfg.name]
    current = asset.data.actuator_force  # (num_envs, num_actuators)

    if not hasattr(env, '_prev_actuator_forces'):
        env._prev_actuator_forces = current.clone()
        return torch.zeros(env.num_envs, device=env.device)

    rate = current - env._prev_actuator_forces
    env._prev_actuator_forces = current.clone()
    return torch.sum(torch.square(rate), dim=1)


def feet_grounded_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """脚接触地面时的正 reward（0、+0.5 或 +1.0）。

    使用接触传感器的 `found` 字段。对有 2 个 primary 脚 geom 的 feet_ground_contact
    传感器，`found` 形状为 (num_envs, 2)，每脚一个二值接触标志。我们求和并归一化到
    [0, 1]。
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found  # (num_envs, num_feet) or (num_envs, 1)
    if found.dim() > 1:
        found = found.sum(dim=-1)  # collapse foot dimension
    return torch.clamp(found, 0.0, 2.0) / 2.0


def body_impact_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 1.0,
) -> torch.Tensor:
    """惩罚受保护部位上超过阈值的地形接触力。

    用于阻止跌倒时把 trunk 外壳或头猛撞到地面。传感器应覆盖相关 body 或子树，
    并用 reduce='netforce'。低于阈值的力免费；超过后惩罚线性增长。

    Args:
        sensor_name: 一个 fields=("force",)、reduce="netforce" 的 ContactSensorCfg 名称。
        threshold: 低于该接触力（N）不施加惩罚。

    Returns:
        penalty 张量 (num_envs,) —— 每步超过阈值的 N 值。
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    sensor = env.scene.sensors[sensor_name]
    forces = sensor.data.force  # (num_envs, N_bodies, 3)
    total_force = forces.sum(dim=1)  # sum over bodies in the subtree
    force_mag = torch.norm(total_force, dim=1)
    return torch.clamp(force_mag - threshold, min=0.0)


# def wheel_speed_reward(
#     env: ManagerBasedRlEnv,
#     command_name: str,
#     wheel_radius: float = 0.0175,
#     vel_scale: float = 0.5,
#     bidirectional: bool = False,
# ) -> torch.Tensor:
#     """奖励与指令推动成正比的轮子旋转。

#     前进时全部 4 个轮子正向旋转（目视验证）。在 vel_scale m/s 等效处用 tanh 饱和，
#     防止失控。

#     - ``bidirectional=False``（默认）：仅前进 —— cmd_x > 0 时奖励正转，否则静默
#       （cmd_x < 0 由制动 reward 处理）。
#     - ``bidirectional=True``：奖励朝【指令方向】的轮子旋转 —— cmd_x > 0 时前进，
#       cmd_x < 0 时后退，幅值为 |cmd_x|。让 cmd_x < 0 表示"后退"而非"刹车"。
#     """
#     cmd_x = env.command_manager.get_command(command_name)[:, 0]  # (B,)

#     asset: Entity = env.scene["robot"]
#     lf_ids, _ = asset.find_joints("passive_LF_?wheel")
#     lr_ids, _ = asset.find_joints("passive_LR_?wheel")
#     rf_ids, _ = asset.find_joints("passive_RF_?wheel")
#     rr_ids, _ = asset.find_joints("passive_RR_?wheel")

#     vel = asset.data.joint_vel
#     # 前进时全部 4 个轮子正向旋转（由 test_wheel_direction.py 验证）
#     forward_omega = (vel[:, lf_ids[0]] + vel[:, lr_ids[0]] + vel[:, rf_ids[0]] + vel[:, rr_ids[0]]) / 4.0

#     omega_scale = vel_scale / wheel_radius
#     if bidirectional:
#         # 旋转方向与指令符号一致（+ 前进，- 后退）
#         aligned = torch.sign(cmd_x) * forward_omega
#         return torch.abs(cmd_x) * torch.tanh(torch.clamp(aligned, min=0.0) / omega_scale)
#     return torch.clamp(cmd_x, min=0.0) * torch.tanh(torch.clamp(forward_omega, min=0.0) / omega_scale)


def coasting_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    vel_std: float = 0.3,
    stillness_std: float = 5.0,
    # asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(r".*(hip|knee|ankle).*",)),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(r".*leg_joint_.*",)),

) -> torch.Tensor:
    """奖励滑行：在目标速度时腿部关节速度低。

    返回 exp(-vel_error / vel_std²) × exp(-sum(joint_vel²) / stillness_std²)。
    两个因子必须同时高 —— 机器人因【既】处于目标速度【又】保持腿不动（滑翔）而获奖，
    单独满足其一不行。

    滑行良好时的典型值：~0.7–1.0。当原地高速踩踏时，joint_vel 项会把 reward 压向 0。
    """
    cmd = env.command_manager.get_command(command_name)
    vel_b = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    vel_error = torch.sum(torch.square(cmd[:, :2] - vel_b), dim=1)
    at_speed = torch.exp(-vel_error / vel_std ** 2)

    asset: Entity = env.scene[asset_cfg.name]
    joint_vel_sq = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    stillness = torch.exp(-joint_vel_sq / stillness_std ** 2)

    return at_speed * stillness


def braking_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    vel_std: float = 0.3,
) -> torch.Tensor:
    """奖励在 cmd_x < 0 时停下来（指令刹车）。

    返回 clamp(-cmd_x, 0) * exp(-fwd_vel² / vel_std²)。
    - cmd_x ≥ 0（滑行或推动）时静默。
    - cmd_x = -1 且 vel = 0：reward = 1.0（完全停住）。
    - cmd_x = -1 且 vel = vel_std：reward ≈ 0.37（强梯度）。
    vel_std=0.3 m/s 让梯度在步行速度下仍有意义。
    """
    cmd = env.command_manager.get_command(command_name)
    cmd_x = cmd[:, 0]
    braking_strength = torch.clamp(-cmd_x, min=0.0)
    fwd_vel = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    stopped = torch.exp(-(fwd_vel.clamp(min=0.0) ** 2) / (vel_std ** 2))
    return braking_strength * stopped


def contact_frequency_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    max_contact_changes_per_sec: float = 4.0,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """
    惩罚过高的接触变化频率以鼓励更慢的迈步。
    跟踪每秒接触状态变化的次数，超过阈值时惩罚。

    Args:
        env: 环境
        sensor_name: 接触传感器的名称
        max_contact_changes_per_sec: 每秒允许的最大接触变化次数
        command_threshold: 施加惩罚所需的最小指令幅值

    Returns:
        形状为 (num_envs,) 的 penalty 张量 —— 超阈值时为负
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    # 检查指令是否超过阈值
    if "twist" in env.command_manager._terms:
        cmd = env.command_manager.get_command("twist")
        cmd_vel = cmd[:, :3]
        cmd_norm = torch.linalg.norm(cmd_vel, dim=1)
        active_mask = cmd_norm > command_threshold
    else:
        active_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    sensor = env.scene.sensors[sensor_name]
    contacts = sensor.data.found[:, :2]  # (num_envs, 2)

    # 必要时初始化跟踪量
    if not hasattr(env, '_contact_change_count'):
        env._contact_change_count = torch.zeros(env.num_envs, device=env.device)
        env._contact_change_timer = torch.zeros(env.num_envs, device=env.device)
        env._prev_contacts_for_freq = contacts.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # 检测任意接触变化（任一脚）
    contact_changed = torch.any(contacts != env._prev_contacts_for_freq, dim=1)

    # 递增变化计数器
    env._contact_change_count += contact_changed.float()

    # 更新计时器
    env._contact_change_timer += env.step_dt

    # 计算当前频率（每秒变化次数）
    # 避免除零
    freq = env._contact_change_count / torch.clamp(env._contact_change_timer, min=0.01)

    # 每 1 秒重置计数器和计时器
    reset_mask = env._contact_change_timer >= 1.0
    env._contact_change_count[reset_mask] = 0.0
    env._contact_change_timer[reset_mask] = 0.0

    # 频率超过最大值时惩罚
    # 对超阈值的频率使用二次惩罚
    excess_freq = torch.clamp(freq - max_contact_changes_per_sec, min=0.0)
    penalty = -torch.square(excess_freq)

    # 更新上一时刻的接触状态
    env._prev_contacts_for_freq = contacts.clone()

    # 施加指令阈值掩码
    penalty = penalty * active_mask.float()

    return penalty


# ==============================================================================
# Ground Pick 相关 Reward
# ==============================================================================

# def mouth_ground_proximity(
#     env: ManagerBasedRlEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
#     std: float = 0.03,
#     target_height: float = 0.0,
#     command_name: str = "twist",
# ) -> torch.Tensor:
#     """奖励嘴尖接近地面，由接近相位加权。

#     ground pick 任务的指令是 [cos(2π*phase), sin(2π*phase), 0]。
#     接近相位是前半周期（sin > 0，phase ∈ [0, 0.5]），由 max(0, sin(2π*phase))
#     平滑加权。

#     Args:
#         std: mouth_tip 高度上的 Gaussian std（m）。0.03 m 给出强梯度。
#         target_height: mouth_tip 的目标 z 高度（m）。0 = 地面高度。
#     """
#     asset = env.scene[asset_cfg.name]
#     mouth_z = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2]  # (num_envs,)
#     proximity = torch.exp(-((mouth_z - target_height) / std) ** 2)

#     # 接近权重：max(0, sin(2π*phase)) —— 在 phase=0.25 处峰值为 1，在 0 和 0.5 处为零
#     cmd = env.command_manager.get_command(command_name)
#     approach_weight = torch.clamp(cmd[:, 1], min=0.0)

#     return approach_weight * proximity


# def mouth_perpendicular_to_ground(
#     env: ManagerBasedRlEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
#     command_name: str = "twist",
# ) -> torch.Tensor:
#     """奖励接近相位期间嘴尖 x 轴竖直（朝下）。

#     完全垂直接触给出 alignment=1；水平为 0；朝上为 -1。
#     由 max(0, sin(2π*phase)) 加权，使其只在下降期间生效。
#     """
#     asset = env.scene[asset_cfg.name]
#     # site_quat_w: (num_envs, num_sites, 4)，格式为 [w, x, y, z]
#     q = asset.data.site_quat_w[:, asset_cfg.site_ids[0], :]  # (num_envs, 4)
#     w, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
#     # site x 轴在世界系下的 z 分量（旋转矩阵第一列）
#     x_axis_z = 2.0 * (qx * qz - w * qy)
#     # 与 [0, 0, -1] 点乘：1 = 完全朝下，-1 = 朝上
#     alignment = -x_axis_z

#     cmd = env.command_manager.get_command(command_name)
#     approach_weight = torch.clamp(cmd[:, 1], min=0.0)

#     return approach_weight * alignment


def sit_grounded(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: Optional[str] = None,
    sin_threshold: float = 0.7,
    min_progress_frac: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    upright_cos_threshold: float = 0.5,
) -> torch.Tensor:
    """【直立时】trunk 与地面接触的正 reward。

    额外由 trunk 机身系 +Z 轴大致指向世界上方来门控（cosine >=
    ``upright_cos_threshold``，默认 0.5 → 接受最多 60° 倾斜）。没有这个门控，
    策略可以通过侧翻或脸朝前扑获得接触奖励 —— trunk 在这些奇怪姿态下撞到地面，
    sit_grounded 触发，策略收敛到与真正坐姿竞争的"跌倒"模式。

    给出 ``command_name`` 时，reward 被门控到相位指令的坐下窗口。否则常开，
    可选通过 ``min_progress_frac`` 限制到 episode 后段。
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found.dim() > 1:
        found = found.sum(dim=-1)
    has_contact = (found > 0).float()

    # 直立检查：trunk 机身的 +Z（世界系，旋转矩阵第三列，
    # 由 trunk 四元数导出）点乘世界上方 = trunk 的 body-up · world-up。
    # 等价地：对单位四元数 (w, x, y, z) 为 1 - 2*(qx² + qy²)。
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4) = (w, x, y, z)
    qx, qy = quat[:, 1], quat[:, 2]
    upright_cos = 1.0 - 2.0 * (qx * qx + qy * qy)
    is_upright = (upright_cos >= upright_cos_threshold).float()

    contact_upright = has_contact * is_upright

    if command_name is None:
        if min_progress_frac > 0.0:
            progress = env.episode_length_buf.float() / float(env.max_episode_length)
            late_enough = (progress >= min_progress_frac).float()
            return late_enough * contact_upright
        return contact_upright
    cmd = env.command_manager.get_command(command_name)
    in_sit_window = (cmd[:, 1] > sin_threshold).float()
    return in_sit_window * contact_upright


def sit_stability(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: Optional[str] = None,
    ang_vel_std: float = 0.5,
    sin_threshold: float = 0.7,
    min_progress_frac: float = 0.0,
) -> torch.Tensor:
    """低机身角速度的奖励。

    设置 ``command_name`` 时按相位门控（相位指令的坐下窗口）。
    否则常开，可选通过 ``min_progress_frac`` 限制到 episode 后段。鼓励稳定的静止姿态。
    """
    asset = env.scene[asset_cfg.name]
    ang_vel_norm = asset.data.root_link_ang_vel_w.norm(dim=-1)
    stillness = torch.exp(-((ang_vel_norm / ang_vel_std) ** 2))
    if command_name is None:
        if min_progress_frac > 0.0:
            progress = env.episode_length_buf.float() / float(env.max_episode_length)
            late_enough = (progress >= min_progress_frac).float()
            return late_enough * stillness
        return stillness
    cmd = env.command_manager.get_command(command_name)
    in_sit_window = (cmd[:, 1] > sin_threshold).float()
    return in_sit_window * stillness


def joint_deviation_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """关节位置相对默认值（HOME）偏差的 L1 惩罚。

    返回所选关节上 |joint_pos - default| 之和。与 Gaussian `pose` reward（对任何
    小偏差都在 1.0 附近饱和）不同，它在所有偏差幅度上给出【线性】梯度 —— 适合作为
    对关节子集（如 hip_yaw / hip_roll）的聚焦惩罚，防止它们漂移到宽站姿，即使其他
    关节都接近 HOME。
    """
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    err = asset.data.joint_pos[:, jnt_ids] - asset.data.default_joint_pos[:, jnt_ids]
    return torch.sum(torch.abs(err), dim=-1)


def joint_pos_limit_proximity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    margin: float = 0.15,
) -> torch.Tensor:
    """关节位置进入其【硬】限位附近 ``margin``（rad）带的 L1 惩罚。

    基础的 ``joint_pos_limits`` reward 只在越过【软】限位后触发（全局
    ``soft_joint_pos_limit_factor`` = 0.9 → 大约是范围最后 7.5%）且只按超出的弧度值，
    因此对卡在限位上的关节几乎无用。本项直接读取【硬】限位，让每个 reward 自己设定
    较宽的 margin，并可限定到特定关节。

    典型场景：低 kp 位置伺服 + 宽 ctrlrange 时，策略可以"免费"（无指令侧成本）把
    指令打到远超关节限位处，让关节停在硬挡块上 —— 例如 hip_yaw 被顶到 ±limit，
    使脚滑动/枢转。过冲是【有意为之】（低 kp 伺服就是这样到达目标的），因此威慑必须
    位于 qpos 侧，并在到达挡块前很久就开始咬合。

    对每个带硬限位 ``[lo, hi]`` 的所选关节：:

        soft_lo = lo + margin,  soft_hi = hi - margin
        penalty = relu(soft_lo - q) + relu(q - soft_hi)

    各关节求和：内部为零，朝每个挡块线性攀升。
    """
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, jnt_ids]
    hard = asset.data.joint_pos_limits[:, jnt_ids]  # (num_envs, num_sel_joints, 2)
    soft_lo = hard[..., 0] + margin
    soft_hi = hard[..., 1] - margin
    below = (soft_lo - q).clip(min=0.0)
    above = (q - soft_hi).clip(min=0.0)
    return torch.sum(below + above, dim=-1)


def phase_height_track(
    env: ManagerBasedRlEnv,
    command_name: str,
    stand_z: float,
    sit_z: float,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """奖励 trunk_z 跟踪站立与坐姿高度之间的正弦插值目标。

    sitstand 任务用它代替坐姿的关节角匹配 —— 奖励【最终状态】（低 trunk）而不规定
    机器人【如何】到达。策略可自由找到任何运动策略（深蹲、头部支撑下降等）。

    指令（来自 GroundPickPhaseCommand）：cmd[:, 1] = sin(2π·phase)。
    phase 0.25 时 sin = +1（坐姿峰值）→ 目标 = sit_z。
    phase 0.75 时 sin = -1（站姿峰值）→ 目标 = stand_z。
    过渡时 sin = 0 → 目标 = 中点。
    """
    cmd = env.command_manager.get_command(command_name)
    sin_phase = cmd[:, 1]
    target_z = (stand_z + sit_z) * 0.5 - (stand_z - sit_z) * 0.5 * sin_phase
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return torch.exp(-((z - target_z) / std) ** 2)


def pose_target_match(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    joint_indices: Optional[list] = None,
    target_overrides: Optional[dict] = None,
) -> torch.Tensor:
    """关节位置相对目标姿态的常开 Gaussian。

    ``phase_pose_match`` 的非相位版本：适用于没有循环指令给 reward 加权、且目标姿态
    整段 episode 恒定的情景任务（如 sit 环境）。

    Args:
        std: 每个关节的 Gaussian std（rad）。
        joint_indices: 可选的待评估关节子集。
        target_overrides: ``{joint_index: angle_rad}``。未列出的关节默认取
            ``asset.data.default_joint_pos``（home/站立姿态）。
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)


def interpolated_pose_target_match(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    joint_indices: Optional[list] = None,
    source_overrides: Optional[dict] = None,
    target_overrides: Optional[dict] = None,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """关节位置相对时间插值目标姿态的 Gaussian。

    跟踪一个在 episode 中、在进度比例 ``ramp_start_frac`` 与 ``ramp_end_frac`` 之间
    从源姿态线性插值到目标姿态的目标。斜坡之前/之后，目标分别钳制为源/最终目标。

    要点是强制平滑下降：过早跳到最终目标会让机器人相对当前插值目标【偏离】，
    在失配期间持续损失 pose reward。

    Args:
        std: 每个关节的 Gaussian std（rad）。
        joint_indices: 可选的待评估关节子集。
        source_overrides: 定义源姿态（斜坡起点）的 ``{joint_index: angle_rad}``。
            ``None`` = 默认/HOME 姿态。
        target_overrides: 同上，定义目标姿态（斜坡终点）。
        ramp_start_frac, ramp_end_frac: [0, 1] 中的 episode 进度窗口，目标在此期间
            从源移动到目标。
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    source = _servo_default_joint_pos(env, asset).clone()
    target = _servo_default_joint_pos(env, asset).clone()
    if source_overrides:
        for idx, val in source_overrides.items():
            source[:, idx] = val
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0).unsqueeze(-1)
    interp = source * (1.0 - tau) + target * tau

    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        interp = interp[:, joint_indices]
    return torch.exp(-((joint_pos - interp) / std) ** 2).mean(dim=-1)


def interpolated_pose_l1_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_indices: Optional[list] = None,
    source_overrides: Optional[dict] = None,
    target_overrides: Optional[dict] = None,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """到时间插值目标姿态的 L1 距离（负值 —— 用作惩罚）。

    与 ``interpolated_pose_target_match`` 相同的插值调度，但返回
    ``-mean(|joint_pos - interp|)`` 而非 Gaussian。L1 梯度处处恒定 —— 当 Gaussian
    变体在远离目标处饱和为零、使策略没有梯度去发现目标方向时，它可作为引导信号。
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    source = _servo_default_joint_pos(env, asset).clone()
    target = _servo_default_joint_pos(env, asset).clone()
    if source_overrides:
        for idx, val in source_overrides.items():
            source[:, idx] = val
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0).unsqueeze(-1)
    interp = source * (1.0 - tau) + target * tau

    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        interp = interp[:, joint_indices]
    return -torch.abs(joint_pos - interp).mean(dim=-1)


def interpolated_height_l1_penalty(
    env: ManagerBasedRlEnv,
    start_height: float,
    end_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """到时间插值目标高度的 L1 距离（负值 —— 惩罚）。

    与 ``interpolated_pose_l1_penalty`` 作用相同，但作用于 trunk z。无论当前 z 偏离
    多远，都提供朝目标高度的恒定梯度，补充 Gaussian 的 ``interpolated_height_target``。
    """
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0)
    target_z = start_height * (1.0 - tau) + end_height * tau

    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return -torch.abs(z - target_z)


def interpolated_height_target(
    env: ManagerBasedRlEnv,
    start_height: float,
    end_height: float,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """trunk z 相对时间插值目标高度的 Gaussian。

    ``interpolated_pose_target_match`` 的配套项 —— 同样的时间插值逻辑作用于 trunk 高度。
    """
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0)
    target_z = start_height * (1.0 - tau) + end_height * tau

    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return torch.exp(-((z - target_z) / std) ** 2)


def bilateral_symmetry_penalty(
    env: ManagerBasedRlEnv,
    left_indices: list,
    right_indices: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """左右腿不对称的 L1 惩罚。

    对双侧对称的机器人，腿的 HOME 和任何对称目标（FOLD、SIT）在每个匹配的关节对上
    满足 ``q_left + q_right == 0``（因为左右关节使用镜像的符号约定）。本项惩罚对该
    约束的偏离。

    当 pose-target reward 的 ``mean()`` 让策略可以用"一条腿正确"蒙混过关时有用
    （你免费拿到约一半 reward，而修正另一条腿的梯度太弱，逃不出那个局部极小）。
    这里的惩罚无论幅度都有恒定的 L1 梯度，因此任何不对称都要付代价，唯一零点就是
    完全对称的构型。

    返回 ``-sum_i |q[left_i] + q[right_i]|``，在 N 对上取平均。
    """
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos
    left = pos[:, left_indices]
    right = pos[:, right_indices]
    return -torch.abs(left + right).mean(dim=-1)


def _multistage_target_pose(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    waypoints,
) -> torch.Tensor:
    """计算跨 N 个 waypoint 的时间插值关节目标。

    waypoints: 有序字典列表 {"frac": [0,1] 中的 float,
                               "overrides": dict[int,float] | None}。
    第一个 waypoint 应有 frac=0.0（通常是 HOME，overrides=None）。
    后续 waypoint 定义里程碑。两个 waypoint 之间目标线性插值。第一个之前/最后一个
    之后则钳制。

    返回 (num_envs, num_joints) 的目标关节角张量。
    """
    asset = env.scene[asset_cfg.name]
    default = _servo_default_joint_pos(env, asset)

    def build_pose(overrides):
        pose = default.clone()
        if overrides:
            for idx, val in overrides.items():
                pose[:, idx] = val
        return pose

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    # 找出当前处在哪个段（在 envs 上广播）。
    out = build_pose(waypoints[0]["overrides"])
    for i in range(1, len(waypoints)):
        f0 = waypoints[i - 1]["frac"]
        f1 = waypoints[i]["frac"]
        span = max(f1 - f0, 1e-6)
        tau = ((progress - f0) / span).clamp(0.0, 1.0).unsqueeze(-1)
        prev_pose = build_pose(waypoints[i - 1]["overrides"])
        next_pose = build_pose(waypoints[i]["overrides"])
        seg = prev_pose * (1.0 - tau) + next_pose * tau
        # 当进度在 [f0, f1] 内或已越过时，取该段的值。
        mask = (progress >= f0).float().unsqueeze(-1)
        out = torch.where(mask > 0, seg, out)
    return out


def _multistage_target_height(
    env: ManagerBasedRlEnv,
    waypoints,
) -> torch.Tensor:
    """与 _multistage_target_pose 逻辑相同，但作用于 trunk z 高度。

    waypoints: [{"frac": float, "height": float}, ...]。
    """
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    out = torch.full_like(progress, waypoints[0]["height"])
    for i in range(1, len(waypoints)):
        f0 = waypoints[i - 1]["frac"]
        f1 = waypoints[i]["frac"]
        span = max(f1 - f0, 1e-6)
        tau = ((progress - f0) / span).clamp(0.0, 1.0)
        seg = waypoints[i - 1]["height"] * (1.0 - tau) + waypoints[i]["height"] * tau
        mask = (progress >= f0).float()
        out = torch.where(mask > 0, seg, out)
    return out


def multistage_pose_target_match(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """interpolated_pose_target_match 的多 waypoint 版本。

    waypoints: [{"frac": 0.0, "overrides": None},
                {"frac": 0.4, "overrides": FOLD_OVERRIDES},
                {"frac": 0.7, "overrides": SIT_OVERRIDES}]

    用它强制一条经过一个或多个中间姿态的 curriculum 式轨迹（如 站 → 折叠 → 坐）。
    与单阶段版本有相同的逐关节 Gaussian 语义。
    """
    asset = env.scene[asset_cfg.name]
    target = _multistage_target_pose(env, asset_cfg, waypoints)
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)


def multistage_pose_l1_penalty(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """multistage_pose_target_match 的 L1 配套项。"""
    asset = env.scene[asset_cfg.name]
    target = _multistage_target_pose(env, asset_cfg, waypoints)
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def multistage_height_target(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.03,
) -> torch.Tensor:
    """trunk z 上的多 waypoint Gaussian。"""
    target_z = _multistage_target_height(env, waypoints)
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return torch.exp(-((z - target_z) / std) ** 2)


def multistage_height_l1_penalty(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """multistage_height_target 的 L1 配套项。"""
    target_z = _multistage_target_height(env, waypoints)
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return -torch.abs(z - target_z)


def pose_target_match(
    env: ManagerBasedRlEnv,
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """对单个固定目标的 Gaussian 姿态匹配。

    target = 应用逐索引 overrides 后的 ``default_joint_pos``。没有 waypoint，也没有
    episode 进度插值 —— 从 t=0 到 episode 结束都奖励同一个目标。
    """
    asset = env.scene[asset_cfg.name]
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)


def pose_l1_penalty(
    env: ManagerBasedRlEnv,
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """``pose_target_match`` 的 L1 配套项（朝目标的恒定梯度）。"""
    asset = env.scene[asset_cfg.name]
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def height_target_gaussian(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.02,
) -> torch.Tensor:
    """trunk z 相对单个固定目标的 Gaussian。"""
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return torch.exp(-((z - target_height) / std) ** 2)


def height_l1_penalty(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """``height_target_gaussian`` 的 L1 配套项。"""
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return -torch.abs(z - target_height)


def trunk_vertical_accel_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """与 trunk 的 ``|a_z|``（v_z 的有限差分）成正比的惩罚。

    既捕捉硬撞击（落地时大的减速尖峰），又鼓励平滑的准静态下降（匀速 → a_z ≈ 0）。
    静止时 a_z 为零，所以坐着的机器人不付成本。

    状态保存在 env 的 ``_prev_trunk_vz`` 中；episode 重置时加速度清零，避免上一 episode
    最终状态的瞬态泄漏到新 episode。
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    prev = getattr(env, "_prev_trunk_vz", None)
    if prev is None or prev.shape[0] != vz.shape[0]:
        prev = vz.detach().clone()
    a_z = (vz - prev) / env.step_dt
    # 在重置步把 a_z 清零，抑制跨 episode 的瞬态。
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        a_z = torch.where(reset_mask, torch.zeros_like(a_z), a_z)
    env._prev_trunk_vz = vz.detach().clone()
    return -torch.abs(a_z)


def trunk_downward_velocity_penalty(
    env: ManagerBasedRlEnv,
    max_down_vel: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """对超过 ``max_down_vel`` 的向下 trunk 速度的惩罚。

    限制下降【速度】，这是单靠 ``trunk_vertical_accel_penalty`` 做不到的：快速匀速下落
    全程 a_z ≈ 0，只在底部付一次撞击尖峰 —— 相对"更早到达目标姿态"是便宜的。本项让
    过快下降的每一步都付出 reward，因此刚好低于上限的最温和下降成为最优。静止时以及
    任何慢于上限的运动（包括所有向上运动）均为零。
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return -torch.clamp(-vz - max_down_vel, min=0.0)


def seated_stillness(
    env: ManagerBasedRlEnv,
    height_full: float = 0.06,
    height_zero: float = 0.08,
    vel_std: float = 0.05,
    tilt_full_deg: float = 25.0,
    tilt_zero_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """奖励直立【坐着】时 trunk 的静止：|v| Gaussian，由 z 和 tilt 门控。

    exp(-(|v|/vel_std)²) · smoothstep(z) · smoothstep(tilt)。z 门控在 ``height_full``
    以下为满值，``height_zero`` 以上为零（下降期间不生效）。tilt 门控在
    ``tilt_full_deg`` 以下为满值，``tilt_zero_deg`` 以上为零 —— 没有它，"仰面躺着不动"
    和"直立坐着不动"得分一样（背部着地时 trunk 在坐姿 z 带内且完全静止），这正是第 2 次
    运行收敛到的漏洞。它使"在坐姿高度直立安静休息"成为唯一被奖励的休息方式。
    """
    asset = env.scene[asset_cfg.name]
    v = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    t = torch.clamp((height_zero - z) / max(height_zero - height_full, 1e-6), 0.0, 1.0)
    z_gate = t * t * (3.0 - 2.0 * t)
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    cos_full = math.cos(math.radians(tilt_full_deg))
    cos_zero = math.cos(math.radians(tilt_zero_deg))
    u = torch.clamp((cos_tilt - cos_zero) / max(cos_full - cos_zero, 1e-6), 0.0, 1.0)
    tilt_gate = u * u * (3.0 - 2.0 * u)
    return torch.exp(-((v / vel_std) ** 2)) * z_gate * tilt_gate


def upright_while_tall(
    env: ManagerBasedRlEnv,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """由 trunk z 上的 smoothstep 加权的线性直立 reward。

    返回 ``body_upright_linear * smoothstep((z - low)/(high - low))``，使直立激励在
    机器人仍站得高时为满值，一旦进入较低的坐姿构型就淡出为零（此时臀部着地的朝向可以
    接受）。防止策略学会在还很高的时候向后倾倒（否则会通过受控跌落刷下降 reward）。
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    upright = 1.0 - 2.0 * (qx * qx + qy * qy)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    t = torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return upright * smooth


def phase_pose_blend(
    phase: torch.Tensor,
    descent_end: float,
    hold_end: float,
    rise_end: float,
) -> torch.Tensor:
    """沿相位 [0,1) 混合 0..1 —— 0 = 站姿（STAND），1 = 倒下姿态（DOWN）。

    [0, descent_end)       : 0 -> 1  （下降）
    [descent_end, hold_end): 1       （低位）
    [hold_end, rise_end)   : 1 -> 0  （起身）
    [rise_end, 1.0)        : 0       （高位 / 休息）
    """
    b = torch.zeros_like(phase)
    descend = phase < descent_end
    b = torch.where(descend, phase / descent_end, b)
    low = (phase >= descent_end) & (phase < hold_end)
    b = torch.where(low, torch.ones_like(phase), b)
    rise = (phase >= hold_end) & (phase < rise_end)
    b = torch.where(rise, 1.0 - (phase - hold_end) / (rise_end - hold_end), b)
    return b


# def kick_pose_target(
#     phase: torch.Tensor,
#     stand: torch.Tensor,
#     back: torch.Tensor,
#     forward: torch.Tensor,
#     windup_end: float,
#     kick_end: float,
#     return_end: float,
# ) -> torch.Tensor:
#     """一个 4 关键帧踢球动作的插值关节目标。

#     phase (B,) ∈ [0,1)。stand/back/forward (k,) 或 (1,k)。返回 (B,k)。

#     [0, windup_end)        STAND   -> BACK     （蓄力）
#     [windup_end, kick_end) BACK    -> FORWARD  （快速击球）
#     [kick_end, return_end) FORWARD -> STAND    （回收）
#     [return_end, 1.0)      STAND              （休息）
#     """
#     p = phase.unsqueeze(-1)  # (B,1)

#     def interp(a, b, s):
#         return a + s * (b - a)

#     s1 = (p / windup_end).clamp(0.0, 1.0)
#     s2 = ((p - windup_end) / (kick_end - windup_end)).clamp(0.0, 1.0)
#     s3 = ((p - kick_end) / (return_end - kick_end)).clamp(0.0, 1.0)

#     seg1 = interp(stand, back, s1)
#     seg2 = interp(back, forward, s2)
#     seg3 = interp(forward, stand, s3)  # à s3=1 (phase>=return_end) => STAND

#     out = seg1
#     out = torch.where(p >= windup_end, seg2, out)
#     out = torch.where(p >= kick_end, seg3, out)
#     return out


# def _kick_pose_error(
#     env: ManagerBasedRlEnv,
#     asset_cfg: SceneEntityCfg,
#     command_name: str,
#     stand_pose: dict,
#     back_pose: dict,
#     forward_pose: dict,
#     windup_end: float,
#     kick_end: float,
#     return_end: float,
#     joint_names: Optional[list] = None,
# ):
#     """踢球动作的 (cur, target)，关节【按名称】解析。

#     3 个姿态共享相同的键（13 个关节）。名称顺序由 `stand_pose` 给出（或由提供的
#     `joint_names` 给出 —— 它是键的子集，例如一侧用右腿+颈部、另一侧用左腿，以便对
#     动作腿和支撑腿施加不同的 std）。
#     """
#     if not stand_pose:
#         raise ValueError("_kick_pose_error requires a non-empty stand_pose dict")
#     asset: Entity = env.scene[asset_cfg.name]
#     names = list(joint_names) if joint_names is not None else list(stand_pose.keys())
#     ids = [int(asset.find_joints([n])[0][0]) for n in names]

#     def vec(d):
#         return torch.tensor([d[n] for n in names], device=env.device,
#                             dtype=asset.data.joint_pos.dtype)

#     stand_v, back_v, fwd_v = vec(stand_pose), vec(back_pose), vec(forward_pose)

#     cmd = env.command_manager.get_command(command_name)
#     phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
#     target = kick_pose_target(phase, stand_v, back_v, fwd_v,
#                               windup_end, kick_end, return_end)          # (B,k)
#     cur = asset.data.joint_pos[:, ids]                                   # (B,k)
#     return cur, target


# def kick_pose_track(
#     env: ManagerBasedRlEnv,
#     command_name: str = "twist",
#     stand_pose: Optional[dict] = None,
#     back_pose: Optional[dict] = None,
#     forward_pose: Optional[dict] = None,
#     std: float = 0.4,
#     windup_end: float = 0.35,
#     kick_end: float = 0.45,
#     return_end: float = 0.75,
#     asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
#     joint_names: Optional[list] = None,
# ) -> torch.Tensor:
#     """对踢球插值目标的关节姿态 Gaussian。

#     指令式且对称的 reward：每个相位施加确切的关节构型。【按名称】解析。
#     `joint_names` 把评估限制在一个子集（例如右腿+颈部紧跟踪，左支撑腿松跟踪，
#     让它保持平衡）。
#     """
#     cur, target = _kick_pose_error(
#         env, asset_cfg, command_name, stand_pose or {}, back_pose or {},
#         forward_pose or {}, windup_end, kick_end, return_end, joint_names,
#     )
#     return torch.exp(-((cur - target) / std) ** 2).mean(dim=-1)


# def kick_pose_track_l1(
#     env: ManagerBasedRlEnv,
#     command_name: str = "twist",
#     stand_pose: Optional[dict] = None,
#     back_pose: Optional[dict] = None,
#     forward_pose: Optional[dict] = None,
#     windup_end: float = 0.35,
#     kick_end: float = 0.45,
#     return_end: float = 0.75,
#     asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
#     joint_names: Optional[list] = None,
# ) -> torch.Tensor:
#     """朝插值目标的 L1 引导（梯度恒定，惩罚 <= 0）。"""
#     cur, target = _kick_pose_error(
#         env, asset_cfg, command_name, stand_pose or {}, back_pose or {},
#         forward_pose or {}, windup_end, kick_end, return_end, joint_names,
#     )
#     return -(cur - target).abs().mean(dim=-1)


# def kick_engagement(
#     phase: torch.Tensor,
#     windup_end: float,
#     return_end: float,
# ) -> torch.Tensor:
#     """动作【介入】门控 ∈ [0,1]（纯函数）—— 用于给单脚平衡 reward 加权，
#     使其只在非 STAND 休息时生效。

#     [0, windup_end)        : 0 -> 1  （蓄力期间上升）
#     [windup_end, return_end): 1      （击球阶段 = 期望单脚支撑）
#     [return_end, 1.0)      : 0       （STAND 休息，双脚支撑，CoM 居中 OK）
#     """
#     g = torch.zeros_like(phase)
#     ramp = phase < windup_end
#     g = torch.where(ramp, phase / windup_end, g)
#     hold = (phase >= windup_end) & (phase < return_end)
#     g = torch.where(hold, torch.ones_like(phase), g)
#     return g


def com_over_support_foot(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "twist",
    std: float = 0.04,
    windup_end: float = 0.35,
    return_end: float = 0.75,
) -> torch.Tensor:
    """Gaussian reward：CoM 的水平投影接近支撑脚，由击球相位（kick_engagement）门控。

    学习向支撑脚的侧向重心转移。没有它，从双脚支撑姿态抬腿得到的单脚动作会保持 CoM
    居中在两脚之间 → 另一脚一抬就侧翻摔倒。STAND 休息时门控为 0（双脚支撑，允许 CoM 居中）。

    `asset_cfg` 应指向支撑脚的 site（例如 site_names=["left_foot"]）。
    `std` 单位为米（CoM↔脚的容差半径，约脚的大小）。
    """
    asset: Entity = env.scene[asset_cfg.name]
    com_xy = asset.data.root_com_pos_w[:, :2]
    foot_id = asset_cfg.site_ids[0]
    foot_xy = asset.data.site_pos_w[:, foot_id, :2]
    dist2 = ((com_xy - foot_xy) ** 2).sum(dim=-1)
    reward = torch.exp(-dist2 / (std ** 2))

    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
    gate = kick_engagement(phase, windup_end, return_end)
    return gate * reward


def _phase_pose_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    target_pose: dict,
    descent_end: float,
    hold_end: float,
    rise_end: float,
    source_pose: Optional[dict] = None,
):
    """相位插值姿态的 (cur, target)，【按名称】解析。

    目标 = source + blend(phase)·(target_pose - source)，source = STAND
    （若提供 `source_pose` 则用之，否则取模型 DEFAULT/HOME）。blend ∈ [0,1]
    （0 = STAND，1 = target_pose），由 `phase_pose_blend` 给出。
    """
    if not target_pose:
        raise ValueError("_phase_pose_error requires a non-empty target_pose dict")

    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
    blend = phase_pose_blend(phase, descent_end, hold_end, rise_end)     # (B,)

    names = list(target_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]
    default = asset.data.default_joint_pos[:, ids]                       # (B,k)

    source = default.clone()
    if source_pose:
        for j, n in enumerate(names):
            if n in source_pose:
                source[:, j] = source_pose[n]
    target_vec = torch.tensor(
        [target_pose[n] for n in names], device=env.device, dtype=default.dtype
    ).unsqueeze(0)                                                       # (1,k)

    target = source + blend.unsqueeze(-1) * (target_vec - source)        # (B,k)
    cur = asset.data.joint_pos[:, ids]                                   # (B,k)
    return cur, target


def phase_pose_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pose: Optional[dict] = None,
    source_pose: Optional[dict] = None,
    std: float = 0.3,
    descent_end: float = 0.15,
    hold_end: float = 0.50,
    rise_end: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """对 STAND<->DOWN 插值目标的关节姿态 Gaussian。

    指令式 reward：指示每个相位应处的确切关节构型。起身（目标 → STAND）与下降
    （目标 → DOWN）被完全同样地奖励 —— 构造上对称。【按名称】解析。
    """
    cur, target = _phase_pose_error(
        env, asset_cfg, command_name, target_pose or {},
        descent_end, hold_end, rise_end, source_pose,
    )
    return torch.exp(-((cur - target) / std) ** 2).mean(dim=-1)


def phase_pose_track_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pose: Optional[dict] = None,
    source_pose: Optional[dict] = None,
    descent_end: float = 0.15,
    hold_end: float = 0.50,
    rise_end: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """朝插值目标的 L1 引导（负惩罚）。

    处处梯度恒定 —— 即使上面的 Gaussian 在远离目标处已饱和到 ~0，也能给出方向。
    """
    cur, target = _phase_pose_error(
        env, asset_cfg, command_name, target_pose or {},
        descent_end, hold_end, rise_end, source_pose,
    )
    return -(cur - target).abs().mean(dim=-1)


def phase_pose_match(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    command_name: str = "twist",
    joint_indices: Optional[list] = None,
    target_overrides: Optional[dict] = None,
    phase: str = "approach",
) -> torch.Tensor:
    """匹配目标姿态的 reward，由相位循环指令加权。

    用于相位条件任务（如坐/站）的通用辅助函数。指令把相位编码为
    [cos(2π·phase), sin(2π·phase), 0]：
      - "approach" 权重 = max(0, sin(2π·phase)) —— 在 phase 0.25 处峰值。
      - "return"   权重 = max(0, -sin(2π·phase)) —— 在 phase 0.75 处峰值。

    Args:
        std: 每个关节的 Gaussian std（rad）。
        joint_indices: 可选的待评估关节子集（其余忽略）。
        target_overrides: {joint_index: angle_rad}。未列出的关节默认取
            asset.data.default_joint_pos（home/站立姿态）。
        phase: "approach" 或 "return"。
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    pose_reward = torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)

    cmd = env.command_manager.get_command(command_name)
    if phase == "approach":
        weight = torch.clamp(cmd[:, 1], min=0.0)
    else:
        weight = torch.clamp(-cmd[:, 1], min=0.0)
    return weight * pose_reward


def ground_pick_return_pose(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    command_name: str = "twist",
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """奖励 ground pick 后回到站立姿态，由回程相位加权。

    回程相位是后半周期（sin < 0，phase ∈ [0.5, 1.0]），由 max(0, -sin(2π*phase))
    平滑加权。

    Args:
        std: 每个关节的 Gaussian std（rad）。
        joint_indices: 待评估的关节子集。用于对腿部关节与颈/头关节施加不同的 std
            （调用本 reward 两次）。
    """
    asset = env.scene[asset_cfg.name]
    joint_pos  = _servo_joint_pos(env, asset)        # (num_envs, n_servo_joints)
    default_pos = _servo_default_joint_pos(env, asset)

    if joint_indices is not None:
        joint_pos   = joint_pos[:, joint_indices]
        default_pos = default_pos[:, joint_indices]

    pose_reward = torch.exp(-((joint_pos - default_pos) / std) ** 2).mean(dim=-1)

    # 回程权重：max(0, -sin(2π*phase)) —— 在 phase=0.75 处峰值为 1，在 0.5 和 1 处为零
    cmd = env.command_manager.get_command(command_name)
    return_weight = torch.clamp(-cmd[:, 1], min=0.0)

    return return_weight * pose_reward


def ground_pick_return_upright(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.4,
    command_name: str = "twist",
) -> torch.Tensor:
    """奖励 trunk 竖直度，由【回程】相位加权（起身辅助）。

    与 ``ground_pick_return_pose`` 相同的回程加权（``max(0, -sin(2π·phase))``），
    因此只在起身时奖励直立，绝不干扰下降时的前倾。竖直度 =
    ``exp(-tilt²/std²)``，用与 ``body_upright_gaussian`` 相同的倾角代理
    （``2*(qx²+qy²) ≈ 1-cos(tilt)``）。较宽的 std（0.4 rad ≈ 23°）即使从相当倾斜的
    下蹲姿态也有梯度。
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)  # qx² + qy²
    upright = torch.exp(-tilt_sq / (std * std))
    cmd = env.command_manager.get_command(command_name)
    return_weight = torch.clamp(-cmd[:, 1], min=0.0)
    return return_weight * upright


# --------------------------------------------------------------------------- #
# Ground-pick：分段相位门控（下降/平台/上升/休息时长相互独立，#
# 取代 max(0,±sin) 的正弦加权）。                                  #
#   down-gate  = phase_pose_blend(phase, descent_end, hold_end, rise_end)  #
#               0（高位）-> 1（下降）-> 1（低位平台）-> 0（上升/休息）   #
#   up-gate    = phase_rise_gate(phase, hold_end, rise_end)              #
#               上升前为 0 -> 0..1（上升）-> 1（站立休息）              #
# --------------------------------------------------------------------------- #
def phase_rise_gate(
    phase: torch.Tensor, hold_end: float, rise_end: float
) -> torch.Tensor:
    """回程的上升门控：hold_end 之前为 0，[hold_end, rise_end) 上 0->1，之后为 1（站立休息）。"""
    g = torch.zeros_like(phase)
    rising = (phase >= hold_end) & (phase < rise_end)
    g = torch.where(rising, (phase - hold_end) / (rise_end - hold_end), g)
    g = torch.where(phase >= rise_end, torch.ones_like(phase), g)
    return g


def _gp_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0


def mouth_ground_proximity_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    std: float = 0.10,
    target_height: float = 0.0,
    command_name: str = "twist",
    descent_end: float = 0.25,
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """由分段 down-gate（下降+平台）门控的 mouth_ground_proximity。"""
    asset = env.scene[asset_cfg.name]
    mouth_z = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2]
    proximity = torch.exp(-((mouth_z - target_height) / std) ** 2)
    gate = phase_pose_blend(_gp_phase(env, command_name), descent_end, hold_end, rise_end)
    return gate * proximity


def mouth_perpendicular_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    command_name: str = "twist",
    descent_end: float = 0.25,
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """由分段 down-gate 门控的 mouth_perpendicular_to_ground。"""
    asset = env.scene[asset_cfg.name]
    q = asset.data.site_quat_w[:, asset_cfg.site_ids[0], :]
    w, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x_axis_z = 2.0 * (qx * qz - w * qy)
    alignment = -x_axis_z  # 1 = bouche pointe droit vers le bas
    gate = phase_pose_blend(_gp_phase(env, command_name), descent_end, hold_end, rise_end)
    return gate * alignment


def ground_pick_return_pose_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    command_name: str = "twist",
    joint_indices: Optional[list] = None,
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """由分段 up-gate（上升+休息）门控的 ground_pick_return_pose。"""
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    default_pos = _servo_default_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        default_pos = default_pos[:, joint_indices]
    pose_reward = torch.exp(-((joint_pos - default_pos) / std) ** 2).mean(dim=-1)
    gate = phase_rise_gate(_gp_phase(env, command_name), hold_end, rise_end)
    return gate * pose_reward


def ground_pick_return_upright_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.4,
    command_name: str = "twist",
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """由分段 up-gate 门控的 ground_pick_return_upright。"""
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    upright = torch.exp(-tilt_sq / (std * std))
    gate = phase_rise_gate(_gp_phase(env, command_name), hold_end, rise_end)
    return gate * upright


def neck_vel_descent_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    joint_indices: Optional[list] = None,
    hold_end: float = 0.35,
) -> torch.Tensor:
    """惩罚下降+平台期间颈部关节的速度（抑制头部猛低）。

    成本 = 给定关节上的 mean(joint_vel²)，phase < hold_end（下降+低位平台）时门控为 1，
    之后（上升+休息）为 0 -> 【不】妨碍颈部抬起。返回正成本；用负权重。
    """
    asset = env.scene[asset_cfg.name]
    vel = _servo_joint_vel(env, asset)
    if joint_indices is not None:
        vel = vel[:, joint_indices]
    cost = (vel ** 2).mean(dim=-1)
    phase = _gp_phase(env, command_name)
    gate = (phase < hold_end).to(vel.dtype)  # descente + palier bas uniquement
    return gate * cost


def sample_mouth_payload(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    min_kg: float = 0.01,
    max_kg: float = 0.04,
) -> None:
    """重置事件：为每环境采样一个"叼在嘴里"的物体质量（kg），存到
    env._mouth_payload_kg。由 apply_mouth_payload_force 使用。"""
    buf = getattr(env, "_mouth_payload_kg", None)
    if buf is None:
        buf = torch.zeros(env.num_envs, device=env.device)
        env._mouth_payload_kg = buf
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    buf[env_ids] = torch.rand(len(env_ids), device=env.device) * (max_kg - min_kg) + min_kg


def apply_mouth_payload_force(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["jaw_soft"], site_names=["mouth_tip"]
    ),
    command_name: str = "twist",
    hold_end: float = 0.35,
    ramp: float = 0.05,
    gravity: float = 9.81,
) -> torch.Tensor:
    """逐 step 的 hook（用作权重为 0 的 reward）：把嘴里物体的【重量】作为垂直外力
    施加到 mouth_tip，由上升相位门控（phase >= hold_end，在'抓取'时刻快速斜坡上升）。

    模拟起身时嘴尖的一个点质量：施加 m·g 力到身体 CoM，再加力矩
    (p_mouth - p_com) × F，这等价于施加在 mouth_tip（对颈部是合适的力臂）。返回
    0（这不是真正的奖励 —— 只是施加 hook）。"""
    asset: Entity = env.scene[asset_cfg.name]
    payload = getattr(env, "_mouth_payload_kg", None)
    if payload is None:
        return torch.zeros(env.num_envs, device=env.device)
    phase = _gp_phase(env, command_name)
    gate = ((phase - hold_end) / ramp).clamp(0.0, 1.0)  # 0 avant grab -> 1 après
    fz = -(gate * payload) * gravity                     # (N,) force verticale (bas)

    bid = int(asset_cfg.body_ids[0])
    sid = int(asset_cfg.site_ids[0])
    p_mouth = asset.data.site_pos_w[:, sid, :]           # (N,3)
    p_com = asset.data.body_com_pos_w[:, bid, :]         # (N,3)
    F = torch.zeros((env.num_envs, 3), device=env.device, dtype=p_mouth.dtype)
    F[:, 2] = fz
    tau = torch.cross(p_mouth - p_com, F, dim=-1)        # applique F au mouth_tip
    asset.write_external_wrench_to_sim(
        forces=F.unsqueeze(1), torques=tau.unsqueeze(1), body_ids=[bid],
    )
    return torch.zeros(env.num_envs, device=env.device)


# ==============================================================================
# Domain Randomization 事件
# ==============================================================================


def randomize_delayed_actuator_gains(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    operation: str = "scale",
):
    """每 episode 随机化固件 PD 增益（非累积）。

    在标准 BAM 执行器（``bam.mjlab.BamActuator``）下，增益通过 set_gains/reset_gains
    按 env 缩放（执行器拥有 ``kp_scale``/``kd_scale``），因此我们从不动 MuJoCo 模型 ——
    没有累积风险。采样到的逐关节因子被平均为每 env 一个标量（执行器对其所有关节施加
    同一个缩放），与之前的行为一致。非 BAM 执行器被跳过（例如 roller 的 XmlActuator，
    它不暴露 set_gains）。

    Args:
        env: 环境
        env_ids: 要随机化的环境 ID（None = 全部）
        kp_range: kp 随机化的 (min, max)
        kd_range: kd 随机化的 (min, max)
        asset_cfg: 资产配置
        operation: 未使用（为 cfg 兼容保留；缩放总是施加）
    """
    del operation
    from bam.mjlab import BamActuator

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    for actuator in asset.actuators:
        if not isinstance(actuator, BamActuator):
            continue
        n_joints = len(actuator.ctrl_ids)
        kp_samples = torch.rand(len(env_ids), n_joints, device=env.device) * (kp_range[1] - kp_range[0]) + kp_range[0]
        kd_samples = torch.rand(len(env_ids), n_joints, device=env.device) * (kd_range[1] - kd_range[0]) + kd_range[0]
        # 先恢复标称值（防止累积），再施加新的缩放。
        actuator.reset_gains(env_ids)
        actuator.set_gains(
            env_ids,
            kp_scale=kp_samples.mean(dim=1, keepdim=True),
            kd_scale=kd_samples.mean(dim=1, keepdim=True),
        )


@requires_model_fields("dof_frictionloss", "dof_damping")
def expand_bam_friction_fields(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
):
    """空操作的 startup 事件，唯一目的是上面的装饰器。

    bam 的 BamActuator（mjlab_frictionloss 分支）每步把每环境的摩擦预算写入 MuJoCo 的
    dof_frictionloss/dof_damping，这要求这些模型字段按 world 展开。mjlab 只展开事件函数
    通过 requires_model_fields 声明的字段，因此每个用 BAM 执行器的环境都必须注册这个事件。
    """


def randomize_bam_friction(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """BAM 执行器的每 episode 关节摩擦随机化（非累积）。

    BAM 下 MuJoCo 的 dof_frictionloss 被清零（BAM 在 compute() 里计算摩擦），因此
    自带的 dr.dof_frictionloss 是空操作。本函数改为在 ``scale_range`` 中采样每 env 一个
    标量，施加到 FrictionDRBamActuator 的 ``friction_scale``，它乘以 BAM 的速度无关摩擦
    预算（Coulomb + Stribeck + load）。先恢复标称值（1.0）以避免累积。对没有
    friction_scale 钩子的执行器是空操作。
    """
    from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    lo, hi = scale_range
    for actuator in asset.actuators:
        if isinstance(actuator, FrictionDRBamActuator):
            actuator.reset_friction_scale(env_ids)
            samples = torch.rand(len(env_ids), 1, device=env.device) * (hi - lo) + lo
            actuator.set_friction_scale(env_ids, samples)


def randomize_mass_and_inertia(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """用相同的缩放因子同时随机化 body 的质量和惯量。

    这保持物理一致性 —— 质量和惯量必须一起缩放，避免产生导致仿真不稳定的无效惯量张量。

    Args:
        env: 环境
        env_ids: 要随机化的环境 ID
        scale_range: 同时应用于质量和惯量的缩放因子 (min, max)
        asset_cfg: 指定要随机化哪些 body 的资产配置
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    # 获取 body 索引
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))[body_ids]
    body_indices = asset.indexing.body_ids[body_ids]

    # 每环境采样【一个】随机缩放因子（同时用于质量和惯量）
    num_envs = len(env_ids)
    num_bodies = len(body_indices)
    scales = torch.rand(num_envs, num_bodies, device=env.device) * (scale_range[1] - scale_range[0]) + scale_range[0]

    # 首次调用时保存原始值
    if not hasattr(env, '_original_mass_inertia'):
        env._original_mass_inertia = {
            'mass': env.sim.model.body_mass[0, body_indices].clone(),
            'inertia': env.sim.model.body_inertia[0, body_indices].clone(),
        }

    # 先恢复原始值（防止累积）
    original = env._original_mass_inertia
    env.sim.model.body_mass[env_ids[:, None], body_indices] = original['mass'].unsqueeze(0).expand(num_envs, -1)
    env.sim.model.body_inertia[env_ids[:, None], body_indices] = original['inertia'].unsqueeze(0).expand(num_envs, -1, -1)

    # 对质量和惯量施加相同的缩放
    env.sim.model.body_mass[env_ids[:, None], body_indices] *= scales
    env.sim.model.body_inertia[env_ids[:, None], body_indices] *= scales.unsqueeze(-1)  # Scale all 3 inertia components


def standing_envs_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    standing_stages: list[dict],
) -> torch.Tensor:
    """根据训练进度更新站立环境的相对数量。

    Args:
        env: RL 环境
        env_ids: 环境 ID（未使用，但 curriculum 接口要求）
        command_name: 速度指令项的名称
        standing_stages: 含 'step' 和 'rel_standing_envs' 键的字典列表
            示例：[
                {"step": 0, "rel_standing_envs": 0.02},
                {"step": 1000, "rel_standing_envs": 0.1},
                {"step": 2000, "rel_standing_envs": 0.2},
            ]

    Returns:
        当前的 rel_standing_envs 值，以张量形式返回
    """
    del env_ids  # Unused

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    from typing import cast

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"

    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    # 根据当前 step 更新 rel_standing_envs
    for stage in standing_stages:
        if env.common_step_counter > stage["step"]:
            cfg.rel_standing_envs = stage["rel_standing_envs"]

    return torch.tensor([cfg.rel_standing_envs])


def velocity_tracking_std_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    std_stages: list[dict],
) -> torch.Tensor:
    """根据训练进度更新速度跟踪 std 参数。

    从宽松的 std（容易拿 reward）开始学基本行走，然后逐渐收紧以提高速度跟踪精度。

    Args:
        env: RL 环境
        env_ids: 环境 ID（未使用，但 curriculum 接口要求）
        reward_name: reward 项名称（例如 "track_linear_velocity"）
        std_stages: 含 'step' 和 'std' 键的字典列表
            示例：[
                {"step": 0, "std": 0.5},      # 先宽松 —— 学走路
                {"step": 250, "std": 0.3},     # 中等 —— 改进步态
                {"step": 500, "std": 0.2},     # 严格 —— 精确跟踪
            ]

    Returns:
        当前 std 值，以张量形式返回
    """
    del env_ids  # Unused

    # 获取 reward 项配置
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)

    # 根据当前 step 更新 std
    current_std = std_stages[0]["std"]  # Default to first stage

    for stage in std_stages:
        if env.common_step_counter > stage["step"]:
            current_std = stage["std"]

    # 更新 reward 项的 std 参数
    reward_term_cfg.params["std"] = current_std

    return torch.tensor([current_std])


def push_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    push_stages: list[dict],
) -> torch.Tensor:
    """根据训练进度更新推力速度范围。

    从无推力/小推力开始学干净行走，然后逐渐增大以建立鲁棒性，而不扰乱早期学习。

    Args:
        env: RL 环境
        env_ids: 环境 ID（未使用，但 curriculum 接口要求）
        event_name: 推力事件项的名称（例如 "push_robot"）
        push_stages: 含 'step' 和 'velocity_range' 键的字典列表
            示例：[
                {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"step": 250, "velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
                {"step": 500, "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
            ]

    Returns:
        当前最大推力幅值，以张量形式返回
    """
    del env_ids  # Unused

    # 注意：必须更新活的 EventManager 的 term_cfg，而不是 env.cfg.events ——
    # EventManager.__init__ 做了 deepcopy(cfg)，所以修改 env.cfg.events 是空操作。
    event_cfg = env.event_manager.get_term_cfg(event_name)

    # 根据当前 step 更新 velocity_range
    current_range = push_stages[0]["velocity_range"]  # Default to first stage

    for stage in push_stages:
        if env.common_step_counter > stage["step"]:
            current_range = stage["velocity_range"]

    # 更新事件配置的 velocity_range 参数
    event_cfg.params["velocity_range"] = current_range

    # 返回最大幅值用于日志
    max_push = max(abs(current_range["x"][0]), abs(current_range["x"][1]))
    return torch.tensor([max_push])


# def wheel_friction_curriculum(
#     env: ManagerBasedRlEnv,
#     env_ids: torch.Tensor,
#     event_name: str,
#     ranges_stages: list[dict],
# ) -> torch.Tensor:
#     """根据训练 step 阶段更新轮子摩擦。"""
#     del env_ids  # Unused

#     current_ranges = ranges_stages[0]["ranges"]
#     for stage in ranges_stages:
#         if env.common_step_counter > stage["step"]:
#             current_ranges = stage["ranges"]

#     env.event_manager.get_term_cfg(event_name).params["ranges"] = current_ranges
#     return torch.tensor([current_ranges[0]])


def reward_weight(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    weight_stages: list[dict],
) -> torch.Tensor:
    """分阶段的 reward 权重 curriculum。

    mjlab 1.3.0 移除了内置的 ``mdp.reward_weight`` 辅助函数，microduck 于是提供自己的版本。
    ``weight_stages`` 是 ``{"step": int, "weight": float}`` 的字典列表；应用 step 已经
    过去的最新阶段的权重。修改活的 RewardManager 的 term cfg（不是 env.cfg，它在 manager
    初始化时已被 deepcopy）。
    """
    del env_ids
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    for stage in weight_stages:
        if env.common_step_counter > stage["step"]:
            term_cfg.weight = stage["weight"]
    return torch.tensor([term_cfg.weight])


def com_range_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    range_stages: list[dict],
) -> torch.Tensor:
    """根据训练进度更新 CoM 随机化范围。

    逐渐增大 CoM 偏移范围，让机器人先用小的 CoM 不确定性学走路，再逐步加大。

    Args:
        env: RL 环境
        env_ids: 环境 ID（未使用）
        event_name: CoM 随机化事件的名称（例如 "randomize_com"）
        range_stages: 含 'step' 和 'range' 键的字典列表（range 单位为米）
            示例：[
                {"step": 0,          "range": 0.003},
                {"step": 1000 * 24,  "range": 0.005},
                {"step": 2000 * 24,  "range": 0.008},
            ]

    Returns:
        当前 range 值，以张量形式返回（用于日志）
    """
    del env_ids

    # 注意：必须更新活的 EventManager 的 term_cfg，而不是 env.cfg.events ——
    # EventManager.__init__ 做了 deepcopy(cfg)，所以修改 env.cfg.events 是空操作。
    event_cfg = env.event_manager.get_term_cfg(event_name)

    current_range = range_stages[0]["range"]
    for stage in range_stages:
        if env.common_step_counter > stage["step"]:
            current_range = stage["range"]

    event_cfg.params["ranges"] = (-current_range, current_range)
    return torch.tensor([current_range])


def slope_move_masks(distance: "torch.Tensor", size_x: float):
    """斜坡 curriculum 的升级/降级掩码。

    move_up   ：走过了超过 40% 的地块 → 说明已冲下斜坡，把它变陡。与终止条件
                terrain_edge_reached 对齐（~3.8 m，size_x=8.0 上默认
                threshold_fraction=0.95），它在半程阈值（4.0 m）之前就结束 episode ——
                没有这个对齐，成功穿越者永远不会被升级。
    move_down ：几乎没有前进（< 地块的 20%）→ 早期跌倒/卡住，把斜坡变缓。
    """
    move_up = distance > size_x * 0.4
    move_down = (distance < size_x * 0.2) & (~move_up)
    return move_up, move_down


def terrain_levels_slope(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """roller_slope 的坡度 curriculum（无速度指令）。

    基于从生成原点以来走过的 x 距离进展。
    """
    asset = env.scene["robot"]
    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    distance = (
        asset.data.root_link_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    )
    move_up, move_down = slope_move_masks(distance, terrain_generator.size[0])
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def velocity_command_ranges_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[dict],
    update_lin_vel_y: bool = True,
    update_ang_vel_z: bool = True,
    forward_only: bool = False,
) -> torch.Tensor:
    """根据训练进度更新速度指令范围。

    逐渐增大指令速度范围，让机器人逐步学会更快的速度。从小范围开始以稳定学习，
    再扩展到更具挑战性的速度。

    Args:
        env: RL 环境
        env_ids: 环境 ID（未使用，但 curriculum 接口要求）
        command_name: 速度指令项的名称（例如 "twist"）
        velocity_stages: 含 'step'、'lin_vel_range' 和 'ang_vel_range' 键的字典列表
            示例：[
                {"step": 0, "lin_vel_range": 0.3, "ang_vel_range": 1.5},
                {"step": 500 * 24, "lin_vel_range": 0.4, "ang_vel_range": 1.75},
                {"step": 1000 * 24, "lin_vel_range": 0.5, "ang_vel_range": 2.0},
            ]

    Returns:
        当前最大线速度，以张量形式返回
    """
    del env_ids  # Unused

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    from typing import cast

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"

    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    # 根据当前 step 更新速度范围
    current_lin_vel = velocity_stages[0]["lin_vel_range"]
    current_ang_vel = velocity_stages[0]["ang_vel_range"]

    for stage in velocity_stages:
        if env.common_step_counter > stage["step"]:
            current_lin_vel = stage["lin_vel_range"]
            current_ang_vel = stage["ang_vel_range"]

    # 更新指令范围
    if forward_only:
        cfg.ranges.lin_vel_x = (0.0, current_lin_vel)
    else:
        cfg.ranges.lin_vel_x = (-current_lin_vel, current_lin_vel)
    if update_lin_vel_y:
        cfg.ranges.lin_vel_y = (-current_lin_vel, current_lin_vel)
    if update_ang_vel_z:
        cfg.ranges.ang_vel_z = (-current_ang_vel, current_ang_vel)

    return torch.tensor([current_lin_vel])


def projected_gravity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """机身系下的 projected gravity 向量。

    返回投影到机器人机身系中的重力向量，表示纯朝向而不含线加速度。
    这比原始加速度计更简单，只依赖朝向。

    Returns:
        torch.Tensor: 机身系下的 projected gravity (num_envs, 3)
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def _imu_misalignment_quat(env: ManagerBasedRlEnv, max_angle_rad: float) -> torch.Tensor:
    """每环境恒定的 IMU 安装错位旋转（采样一次）。

    模拟每台机器人上 IMU 固定小的安装/校准误差。首次使用时惰性采样并缓存 ——
    整轮运行中每 env 恒定（类似 startup 随机化），因此是【每机器人系统偏置】，而非
    逐 step 噪声。取代旧的 randomize_imu_orientation 事件（它写 site_quat，在 mjlab 1.3.0
    下不按 env 展开，而且 projected_gravity / base_ang_vel 观测本来也不读它）。

    返回 (num_envs, 4) 的单位四元数 (w, x, y, z)。
    """
    q = getattr(env, "_imu_misalign_quat", None)
    if q is None:
        n = env.num_envs
        axis = torch.randn(n, 3, device=env.device)
        axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
        angle = torch.rand(n, device=env.device) * max_angle_rad  # [0, max]
        q = quat_from_angle_axis(angle, axis)
        env._imu_misalign_quat = q
    return q


def projected_gravity_imu_misaligned(
    env: ManagerBasedRlEnv,
    max_angle_deg: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """带每环境恒定 IMU 安装错位的 projected_gravity。"""
    asset: Entity = env.scene[asset_cfg.name]
    q = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(q, asset.data.projected_gravity_b)


def base_ang_vel_imu_misaligned(
    env: ManagerBasedRlEnv,
    max_angle_deg: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """带与重力【相同】的每环境 IMU 错位的机身角速度。"""
    asset: Entity = env.scene[asset_cfg.name]
    q = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(q, asset.data.root_link_ang_vel_b)


def raw_accelerometer(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """原始加速度计读数（含重力 + 线加速度）。

    返回归一化的原始加速度计读数，模拟真实 IMU 的测量值。这不同于只反映朝向的纯
    projected_gravity。从 MuJoCo 的加速度计传感器 "imu_accel" 读取。

    Returns:
        torch.Tensor: 归一化的原始加速度计读数 (num_envs, 3)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # 访问模型以找到传感器地址
    # 加速度计传感器是 robot.xml 中的第 5 个传感器（索引 4）
    # 传感器顺序：framequat, gyro, gyro, velocimeter, accelerometer, subtreeangmom
    mj_model = asset.data.model

    # 从模型数组获取传感器地址（sensor_adr 是 torch 张量）
    sensor_adr_array = mj_model.sensor_adr  # This is a TorchArray/tensor
    sensor_id = 4  # imu_accel is the 5th sensor (0-indexed)
    sensor_adr = int(sensor_adr_array[sensor_id].item())  # Convert to Python int

    # 读取加速度计数据（传感器测得的比力）
    # 形状：(num_envs, 3)
    accel_raw = asset.data.data.sensordata[:, sensor_adr:sensor_adr+3]

    # MuJoCo 加速度计测量比力（与真实传感器一样）
    # 取负以匹配约定：直立静止时应指向下方
    accel_negated = -accel_raw

    # 归一化为单位向量
    accel_norm = torch.norm(accel_negated, dim=-1, keepdim=True)
    accel_normalized = torch.where(
        accel_norm > 0.1,
        accel_negated / accel_norm,
        asset.data.projected_gravity_b  # Fallback to projected gravity
    )

    return accel_normalized

def randomize_imu_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    max_angle_deg: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """用小角度随机化 IMU 传感器的安装朝向。

    模拟真实机器人上轻微的安装误差或校准偏差。通过绕随机轴旋转最多 max_angle_deg
    来随机化 IMU 朝向。

    Args:
        env: 环境
        env_ids: 要随机化的环境 ID
        max_angle_deg: 最大旋转角度，单位度（默认 2.0°）
        asset_cfg: 资产配置
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)
    
    asset: Entity = env.scene[asset_cfg.name]

    # IMU site 是 robot.xml 中的第一个 site（索引 0）
    # site 顺序：imu (0), left_foot (1), right_foot (2)
    site_id = 0
    
    # 首次调用时保存原始朝向
    if not hasattr(env, '_original_imu_quat'):
        env._original_imu_quat = env.sim.model.site_quat[0, site_id].clone()
    
    # 为每个环境生成随机旋转
    num_envs = len(env_ids)
    max_angle_rad = max_angle_deg * torch.pi / 180.0
    
    # 每个轴的随机旋转角度 [-max_angle, +max_angle]
    angles = (torch.rand(num_envs, 3, device=env.device) * 2 - 1) * max_angle_rad
    
    # 把欧拉角转成四元数（小角度近似以提高效率）
    # 对小角度：quat ≈ [1, θx/2, θy/2, θz/2]
    half_angles = angles / 2.0
    quats_delta = torch.zeros(num_envs, 4, device=env.device)
    quats_delta[:, 0] = 1.0  # w component
    quats_delta[:, 1:] = half_angles  # x, y, z components
    
    # 归一化四元数
    quats_delta = quats_delta / torch.norm(quats_delta, dim=1, keepdim=True)
    
    # 获取原始四元数并施加增量旋转
    original_quat = env._original_imu_quat.unsqueeze(0).expand(num_envs, -1)
    
    # 四元数乘法：q_new = q_delta * q_original
    # q1 * q2 = [w1*w2 - dot(v1,v2), w1*v2 + w2*v1 + cross(v1,v2)]
    w1, x1, y1, z1 = quats_delta[:, 0], quats_delta[:, 1], quats_delta[:, 2], quats_delta[:, 3]
    w2, x2, y2, z2 = original_quat[:, 0], original_quat[:, 1], original_quat[:, 2], original_quat[:, 3]
    
    new_quat = torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
    ], dim=1)
    
    # 应用到选定的环境
    env.sim.model.site_quat[env_ids, site_id] = new_quat


def standing_phase(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """站立任务的简单基于时间的相位。

    返回一个随时间从 0 循环到 1 的标量相位值。这让策略即使站立时也有时间推进感。

    Args:
        env: RL 环境
        asset_cfg: 未使用，但为 API 一致性保留

    Returns:
        相位值 [0, 1]，形状为 (num_envs, 1) 的张量
    """
    # 简单的时间相位，每 2 秒循环一次
    # 这给策略一个随时间变化的信号
    phase_period = 2.0  # seconds
    time = env.episode_length_buf * env.step_dt
    phase = (time % phase_period) / phase_period

    return phase.unsqueeze(-1)  # Shape: (num_envs, 1)


def air_time_adaptive(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.01,    # below this: no reward (standing)
    running_threshold: float = 0.5,     # above this: use running air-time window
    walk_threshold_min: float = 0.10,
    walk_threshold_max: float = 0.25,
    run_threshold_min: float = 0.05,
    run_threshold_max: float = 0.25,
) -> torch.Tensor:
    """带行走/跑步各自摆动时间窗口的 air-time reward。

    - command < command_threshold  → 0（站立，无奖励）
    - command_threshold–running_threshold → 行走窗口 [walk_min, walk_max]
    - command > running_threshold  → 跑步窗口 [run_min,  run_max]

    这让行走步态保持刻意的 100–250 ms 摆动，而跑步可用更快的 50–250 ms 节奏。
    """
    sensor = env.scene.sensors[sensor_name]
    current_air_time = sensor.data.current_air_time  # (num_envs, num_feet)
    assert current_air_time is not None

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])

    is_walking = ((total_speed >= command_threshold) & (total_speed < running_threshold)).float()  # (num_envs,)
    is_running = (total_speed >= running_threshold).float()

    # 每环境的阈值在脚上广播
    tmin = (is_walking * walk_threshold_min + is_running * run_threshold_min).unsqueeze(1)
    tmax = (is_walking * walk_threshold_max + is_running * run_threshold_max).unsqueeze(1)

    in_range = (current_air_time > tmin) & (current_air_time < tmax)
    reward = torch.sum(in_range.float(), dim=1)  # sum over feet

    # 站立时奖励为零
    active = (total_speed >= command_threshold).float()
    return reward * active


def stillness_at_zero_command(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    vel_std: float = 0.1,
) -> torch.Tensor:
    """奖励指令接近零时保持静止。

    当 command < threshold 时返回 exp(-body_vel² / vel_std²)，否则为 0。
    它随身体速度单调递减 —— 动得越快奖励总是越少。没有可跨越的阈值来'逃离'它，
    这与基于门控的迈步惩罚不同。
    """
    asset: Entity = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    body_vel = torch.norm(asset.data.root_link_vel_w[:, :2], dim=1)
    stillness = torch.exp(-body_vel ** 2 / vel_std ** 2)

    return is_standing_cmd * stillness


def joint_vel_l2_when_standing(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """只在指令接近零时惩罚腿部关节速度。

    针对站立抖动问题：策略在站立时绕 home 姿态做快速振荡修正。由指令门控，
    因此完全不影响行走步态。
    """
    asset: Entity = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    leg_indices = list(range(0, 5)) + list(range(8, 13))
    joint_vel = asset.data.joint_vel[:, leg_indices]
    vel_sq = torch.sum(joint_vel ** 2, dim=-1)

    return is_standing_cmd * vel_sq


def foot_step_penalty_when_standing(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    body_vel_threshold: float = 0.2,
    air_time_threshold: float = 0.05,
) -> torch.Tensor:
    """在零指令且身体未被推动时惩罚迈步。

    与 air_time reward 的对称对应项：
    - air_time 在 command > threshold 时给迈步【正】奖励（行走）
    - 本项在 command < threshold 时给迈步【负】奖励（站立）

    身体速度门控防止惩罚被推后的恢复步：如果机器人已经在快速移动（被推），不施加
    惩罚，使它仍能迈步稳住自己。

    返回 [0, 1] 中的值（配置中用负权重）。
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors["feet_ground_contact"]

    # 最近是否有脚抬起？（上次完成的 air phase > 阈值）
    air_time = contact_sensor.data.last_air_time[:, :2]  # (num_envs, 2)
    any_foot_stepped = (air_time > air_time_threshold).any(dim=1).float()

    # 是否处于站立模式？（指令接近零）
    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing = (total_speed < command_threshold).float()

    # 身体是否静止？（未被推动）
    body_vel = torch.norm(asset.data.root_link_vel_w[:, :2], dim=1)
    is_still = (body_vel < body_vel_threshold).float()

    return any_foot_stepped * is_standing * is_still


def recovery_stepping_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    velocity_threshold: float = 0.3,
    air_time_threshold: float = 0.05,
) -> torch.Tensor:
    """只在零指令且机器人高速（被推后恢复）时奖励脚的 air time。

    这鼓励机器人被推时迈步恢复平衡，但在正常行走时（command > threshold）不触发。

    Args:
        env: RL 环境
        asset_cfg: 资产配置（未使用，为 API 一致性保留）
        command_name: command manager 中速度指令的名称
        command_threshold: 低于该速度视为站立模式
        velocity_threshold: 激活迈步奖励的线速度阈值（m/s）
        air_time_threshold: 计为一次迈步的最小 air time（秒）

    Returns:
        形状为 (num_envs,) 的 reward 张量
    """
    asset: Entity = env.scene[asset_cfg.name]

    # 只对站立环境触发（指令接近零）
    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    # 获取基座线速度的幅值
    base_lin_vel = asset.data.root_link_vel_w[:, :3]  # (num_envs, 3)
    vel_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)  # Only XY plane

    # 只在速度高（被推）时奖励迈步
    should_step = vel_magnitude > velocity_threshold

    # 从接触传感器获取脚的 air time
    contact_sensor = env.scene.sensors["feet_ground_contact"]
    air_time = contact_sensor.data.last_air_time[:, :2]  # (num_envs, 2) - left and right foot

    # 任一脚最近在空中则奖励
    foot_in_air = (air_time > air_time_threshold).any(dim=1)  # (num_envs,)

    # 只在：站立指令 AND 高身体速度 AND 脚迈步 时给奖励
    reward = is_standing_cmd * should_step.float() * foot_in_air.float()

    return reward


def adaptive_pose_weight(
    env: ManagerBasedRlEnv,
    base_pose_reward: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    velocity_threshold: float = 0.3,
    min_weight: float = 0.3,
) -> torch.Tensor:
    """当机器人高速（被推后恢复）时降低姿态跟踪权重。

    这让机器人在迈恢复步时可以偏离站姿，而在静止站立时保持严格的姿态跟踪。

    Args:
        env: RL 环境
        base_pose_reward: 原始 pose reward（加权前）
        asset_cfg: 资产配置（未使用，为 API 一致性保留）
        velocity_threshold: 开始降权的线速度阈值（m/s）
        min_weight: 高速时的最小权重乘子 (0-1)

    Returns:
        形状为 (num_envs,) 的加权 reward 张量
    """
    asset: Entity = env.scene[asset_cfg.name]

    # 获取基座线速度的幅值
    base_lin_vel = asset.data.root_link_vel_w[:, :3]  # (num_envs, 3)
    vel_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)  # Only XY plane

    # 计算权重：静止时 1.0，高速时为 min_weight
    # 用类似 sigmoid 的函数做平滑过渡
    weight = min_weight + (1.0 - min_weight) * torch.exp(
        -((vel_magnitude - velocity_threshold) / velocity_threshold).clamp(min=0.0) ** 2
    )

    return base_pose_reward * weight


def randomize_base_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    max_pitch_deg: float = 10.0,
    max_roll_deg: float = 5.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """在 episode 开始时随机化基座朝向以强制产生反应式行为。

    在每个 episode 开始时给机器人基座朝向加上随机 pitch 和 roll。这防止策略记住单一
    初始状态，迫使它用反馈去适应不同朝向。

    Args:
        env: 环境
        env_ids: 要随机化的环境 ID
        max_pitch_deg: 最大俯仰角，单位度（前/后倾斜）
        max_roll_deg: 最大横滚角，单位度（左右倾斜）
        asset_cfg: 资产配置
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    num_envs = len(env_ids)

    # 生成随机的 pitch 和 roll 角度
    max_pitch_rad = max_pitch_deg * torch.pi / 180.0
    max_roll_rad = max_roll_deg * torch.pi / 180.0

    pitch = (torch.rand(num_envs, device=env.device) * 2 - 1) * max_pitch_rad
    roll = (torch.rand(num_envs, device=env.device) * 2 - 1) * max_roll_rad
    yaw = torch.zeros(num_envs, device=env.device)  # Keep yaw at 0

    # 把欧拉角 (roll, pitch, yaw) 转成四元数
    # 使用标准航空顺序 (ZYX)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    quat_w = cr * cp * cy + sr * sp * sy
    quat_x = sr * cp * cy - cr * sp * sy
    quat_y = cr * sp * cy + sr * cp * sy
    quat_z = cr * cp * sy - sr * sp * cy

    new_quat = torch.stack([quat_w, quat_x, quat_y, quat_z], dim=1)

    # 归一化四元数
    new_quat = new_quat / torch.norm(new_quat, dim=1, keepdim=True)

    # 获取 root 位置索引（freejoint 从 qpos 索引 0 开始）
    # Freejoint: [x, y, z, qw, qx, qy, qz]
    root_quat_idx = 3  # Quaternion starts at index 3

    # 把随机化后的朝向应用到选定的环境
    env.sim.data.qpos[env_ids, root_quat_idx:root_quat_idx+4] = new_quat


def set_face_down_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """把机器人设为俯卧（腹部朝下）姿态用于起身训练。

    绕 pitch 轴 (Y) 把机器人前倾 90°，使前部/腹部朝向地面、腿朝上。再叠加随机 yaw。

    四元数推导：
        quat_pitch90 = [s, 0, s, 0]   其中 s = sqrt(2)/2（绕 Y 转 90°）
        quat_yaw     = [cy, 0, 0, sy]
        combined     = quat_yaw * quat_pitch90 = [s*cy, -s*sy, s*cy, s*sy]
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0 ** -0.5  # sqrt(2)/2

    new_quat = torch.stack(
        [
            s * cy,   # w
            -s * sy,  # x
            s * cy,   # y
            s * sy,   # z
        ],
        dim=1,
    )

    # Freejoint qpos: [x, y, z, qw, qx, qy, qz, ...]
    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0


def set_random_prone_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    face_down_prob: float = 0.5,
    side_prob: float = 0.0,
):
    """随机初始化每个 env 为躺倒姿态，带随机 yaw。

    前 ``side_prob`` 比例的环境侧躺（±90° roll，左右各 50%）；其余按
    ``face_down_prob`` 分成俯卧/仰卧。

    俯卧：  +90° pitch → quat = [s*cy, -s*sy,  s*cy,  s*sy]
    仰卧：  -90° pitch → quat = [s*cy,  s*sy, -s*cy,  s*sy]
    侧躺：  ±90° roll  → quat = yaw ⊗ roll = [s*cy, ±s*cy, ±s*sy, s*sy]

    侧躺生成于 2026-09 加入（protective-fall velstand 第 1 次运行）：一次无头评估显示
    79% 的被推跌倒以【侧躺】结束，20% 仰卧，0% 俯卧 —— 主导的跌倒状态此前完全没有
    反向课程数据。

    Args:
        face_down_prob: 非侧躺环境中俯卧（相对仰卧）的概率。
        side_prob: 放在侧躺的环境比例。
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0 ** -0.5  # sqrt(2)/2

    face_down = torch.stack([ s * cy, -s * sy,  s * cy,  s * sy], dim=1)
    face_up   = torch.stack([ s * cy,  s * sy, -s * cy,  s * sy], dim=1)
    sign = torch.where(torch.rand(num, device=env.device) < 0.5, 1.0, -1.0)
    on_side  = torch.stack([ s * cy, sign * s * cy, sign * s * sy, s * sy], dim=1)

    u = torch.rand(num, device=env.device)
    side_mask = u < side_prob
    mask = torch.rand(num, device=env.device) < face_down_prob  # True → face-down
    new_quat = torch.where(mask.unsqueeze(1), face_down, face_up)
    new_quat = torch.where(side_mask.unsqueeze(1), on_side, new_quat)

    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0


def set_random_ground_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    face_down_prob: float = 0.4,
    face_up_prob: float = 0.4,
    sitting_prob: float = 0.2,
    standing_prob: float = 0.0,
    prone_z_min: float = 0.20,
    prone_z_max: float = 0.25,
    sitting_z_min: float = 0.07,
    sitting_z_max: float = 0.09,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
    sitting_joint_overrides: Optional[dict] = None,
    sitting_joint_noise_std: float = 0.0,
    sitting_tilt_max: float = 0.0,
    face_up_roll_max: float = 0.0,
):
    """重置到随机地面状态：俯卧、仰卧、坐姿或站立。

    比 ``set_random_prone_orientation`` 更宽 —— 供 stand-up 环境使用，使策略学会从
    任何合理的姿态恢复，包括坐姿关键帧（sit 策略的休息态）和已站立姿态（从而也学会
    【保持】站立，而不只是起身）。

    模式（概率会被归一化，不必和为 1.0）：
      - 俯卧（腹部贴地）：+90° pitch，随机 yaw，z ∈ [prone_z_min, prone_z_max]。
      - 仰卧（背部贴地）：-90° pitch，随机 yaw，z ∈ [prone_z_min, prone_z_max]。
      - 坐姿：直立（±sitting_tilt_max），随机 yaw，z 低，关节设为
        ``sitting_joint_overrides``。
      - 站立：直立（±sitting_tilt_max），随机 yaw，z ∈
        [standing_z_min, standing_z_max]，关节保持 HOME（``reset_robot_joints`` 设的值）。

    Args:
        sitting_joint_overrides: ``{qpos_joint_index: angle_rad}``，写入被采样到坐姿桶
            环境的 ``qpos[7+idx]``。``None`` 表示关节保持 ``reset_robot_joints`` 已设的值。
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    total = face_down_prob + face_up_prob + sitting_prob + standing_prob
    p_fd  = face_down_prob / total
    p_fu  = (face_down_prob + face_up_prob) / total
    p_sit = (face_down_prob + face_up_prob + sitting_prob) / total

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0 ** -0.5  # sqrt(2)/2

    face_down = torch.stack([ s * cy, -s * sy,  s * cy,  s * sy], dim=1)
    face_up   = torch.stack([ s * cy,  s * sy, -s * cy,  s * sy], dim=1)
    # 直立坐姿：默认只有 yaw，外加可选的 ±sitting_tilt_max
    # pitch/roll 噪声，使策略不过拟合完美直立的起点。
    if sitting_tilt_max > 0.0:
        pitch = (torch.rand(num, device=env.device) * 2 - 1) * sitting_tilt_max
        roll  = (torch.rand(num, device=env.device) * 2 - 1) * sitting_tilt_max
        cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll  * 0.5); sr = torch.sin(roll  * 0.5)
        # ZYX 内禀欧拉角 → 四元数 (yaw * pitch * roll)。
        sit_w = cr * cp * cy + sr * sp * sy
        sit_x = sr * cp * cy - cr * sp * sy
        sit_y = cr * sp * cy + sr * cp * sy
        sit_z = cr * cp * sy - sr * sp * cy
        sitting = torch.stack([sit_w, sit_x, sit_y, sit_z], dim=1)
    else:
        sitting = torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), sy], dim=1)

    u = torch.rand(num, device=env.device)
    is_fd    = u < p_fd
    is_fu    = (u >= p_fd) & (u < p_fu)
    is_sit   = (u >= p_fu) & (u < p_sit)
    is_stand = u >= p_sit

    # 仰卧的部分横滚噪声：绕身体长轴旋转仰卧姿态，
    # 身体【长轴】均匀 ±face_up_roll_max 旋转。为什么（2026-07，后向恢复靠
    # 运气才成功）：仰卧和俯卧之间的 reward 地形是【平坦】的 ——
    # upright_linear（cos tilt）在整个翻转过程中 ≈0，高度也不
    # 变 —— 所以从背部翻下来只有靠随后的前起身路径才付奖励，
    # 这是一种长时程依赖，从完全平的仰卧起点靠噪声探索很少能
    # 发现。加了 roll 噪声后，一部分仰卧生成起点接近侧躺（翻转到一半）：策略
    # 从简单起点学会完成翻转，再泛化回平躺
    # 仰卧 —— 一个内置的反向课程。均匀采样保证每个
    # 难度都有代表（±90° 时平躺 |roll|<15° 约占 17%），因此不需要
    # annealing 调度，而且多样的跌倒后姿态本来就是部署时的真实
    # DR。
    # 
    if face_up_roll_max > 0.0:
        theta = (torch.rand(num, device=env.device) * 2 - 1) * face_up_roll_max
        ct = torch.cos(theta * 0.5)
        st = torch.sin(theta * 0.5)
        # 注：log-roll = 绕身体【长轴】旋转，即身体 z（站立时
        # trunk z 朝上 → 躺下时水平）。不是身体
        # x —— 仰卧时身体 x 指向天空，绕 x 翻转只会像已有的 yaw 噪声一样
        # 让机器人原地打转。
        # 机身系旋转 → 右乘：q_fu ⊗ [ct, 0, 0, st]。
        w, x, y, z = face_up[:, 0], face_up[:, 1], face_up[:, 2], face_up[:, 3]
        face_up = torch.stack(
            [
                w * ct - z * st,
                x * ct + y * st,
                y * ct - x * st,
                w * st + z * ct,
            ],
            dim=1,
        )

    # 坐姿和站立共享相同的直立朝向（单位四元数 + 可选
    # ±sitting_tilt_max）；区别只在 trunk 高度和关节姿态。
    new_quat = face_down.clone()
    new_quat[is_fu]    = face_up[is_fu]
    new_quat[is_sit]   = sitting[is_sit]
    new_quat[is_stand] = sitting[is_stand]

    # 每环境随机 z：俯卧/仰卧用 prone 高度，坐姿低，站立约站高。
    z_prone = torch.rand(num, device=env.device) * (prone_z_max - prone_z_min) + prone_z_min
    z_sit   = torch.rand(num, device=env.device) * (sitting_z_max - sitting_z_min) + sitting_z_min
    z_stand = torch.rand(num, device=env.device) * (standing_z_max - standing_z_min) + standing_z_min
    new_z = z_prone.clone()
    new_z = torch.where(is_sit, z_sit, new_z)
    new_z = torch.where(is_stand, z_stand, new_z)

    env.sim.data.qpos[env_ids, 2]   = new_z + _env_origin_z(env, env_ids)
    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6]  = 0.0

    # 坐姿桶的关节 overrides（例如膝盖/脚踝弯到关键帧）。
    # override 的键是【伺服】索引（13 关节布局）；转换为实体
    # 关节索引，使带交错 passive_* 关节（backlash）的模型也能
    # 写到预期关节。qpos 列 = 7 + 实体关节索引
    #（robot free joint 在前，所有铰链都是 1 自由度）。
    asset: Entity = env.scene[asset_cfg.name]
    servo_ids = _servo_joint_ids(env, asset)
    if sitting_joint_overrides:
        sit_env_ids = env_ids[is_sit]
        if len(sit_env_ids) > 0:
            for jnt_idx, angle in sitting_joint_overrides.items():
                env.sim.data.qpos[sit_env_ids, 7 + servo_ids[jnt_idx]] = angle

    # 坐姿环境的关节噪声：对每个受驱关节加 Gaussian 噪声
    # 使策略看到一系列合理"坐姿"起点的分布，而不是
    # 单一的标准姿态。捕捉真实迁移场景：当起身策略从 sit 策略接管时，
    # 机器人的关节角不会恰好匹配 SIT 关键帧。
    # 
    if sitting_joint_noise_std > 0.0:
        sit_env_ids = env_ids[is_sit]
        if len(sit_env_ids) > 0:
            # 仅伺服关节：passive_* 关节（backlash 铰链）范围极小，
            # 重置时必须保持在 0。
            n_sit = len(sit_env_ids)
            cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
            noise = torch.randn(n_sit, len(cols), device=env.device) * sitting_joint_noise_std
            env.sim.data.qpos[sit_env_ids.unsqueeze(1).long(), cols.unsqueeze(0)] += noise


# 深蹲锚定姿态（velstand 第 5 次运行）：恢复中途的"卡住"盆地 ——
# 膝盖折在身体下、trunk 前倾、脚平放。取值来自把 HOME 的锯齿模式
#（髋前 / 膝后 / 踝前，符号约定依照 SIT 关键帧的折叠方向）
# 延伸到深屈曲，落在 ±1.57 关节限位内。
# hip_yaw/hip_roll/neck 保持在 HOME。
_CROUCH_ANCHOR_BY_NAME = {
    "left_hip_pitch": -1.15,
    "left_knee": 1.25,
    "left_ankle": 1.05,
    "right_hip_pitch": 1.15,
    "right_knee": -1.25,
    "right_ankle": -1.05,
}


def set_random_crouch_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    depth_min: float = 0.35,
    depth_max: float = 1.0,
    pitch_max_deg: float = 55.0,
    joint_noise: float = 0.12,
    z_stand: float = 0.115,
    z_deep: float = 0.06,
):
    """把选定环境重置到随机的恢复中途下蹲姿态。

    针对恢复最后一英里的反向课程（velstand 第 5 次运行经验）：从俯卧开始的 episode
    把大部分跌倒预算花在【到达】深蹲上，到达后不久就被回收，因此 crouch→stand
    这一英里几乎没有 on-policy 数据 —— 策略收敛到停在那里。在【整段】这一英里上
    播撒重置（站立与深蹲锚点间深度 λ ∈ [depth_min, depth_max]，trunk pitch 和 z
    随 λ 缩放）使前沿从 episode 第 0 步起就密集。
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    asset: Entity = env.scene[asset_cfg.name]

    lam = torch.rand(num, device=env.device) * (depth_max - depth_min) + depth_min

    # 关节：在腿部 pitch 链上把 HOME → 锚点线性插值，对
    # 仅伺服关节加均匀噪声（passive_* backlash 铰链范围只有 ±1° —— 给它们
    # 加噪声会让生成时它们被钉在限位之外）。
    joints = asset.data.default_joint_pos[env_ids].clone()
    for name, anchor in _CROUCH_ANCHOR_BY_NAME.items():
        ids, _ = asset.find_joints(f"^{name}$")
        j = ids[0]
        joints[:, j] = joints[:, j] + lam * (anchor - joints[:, j])
    noise_mask = torch.zeros(joints.shape[1], device=joints.device)
    noise_mask[_servo_joint_ids(env, asset)] = 1.0
    joints += (torch.rand_like(joints) * 2 - 1) * joint_noise * noise_mask

    # 基座朝向：前倾随深度缩放（卡住的盆地是两种跌倒方向的
    # 前向下蹲），随机 yaw，小 roll 噪声。
    pitch = lam * math.radians(pitch_max_deg) \
        + (torch.rand(num, device=env.device) * 2 - 1) * math.radians(10.0)
    pitch = torch.clamp(pitch, min=math.radians(5.0))
    roll = (torch.rand(num, device=env.device) * 2 - 1) * math.radians(8.0)
    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5); sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5); sr = torch.sin(roll * 0.5)
    # ZYX 内禀欧拉角 → 四元数 (yaw * pitch * roll)，与
    # set_random_ground_state 的坐姿分支相同。
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    # trunk 高度随深度缩放，加一点向上的余量以干净地落定。
    z = z_stand + lam * (z_deep - z_stand) \
        + torch.rand(num, device=env.device) * 0.01

    env.sim.data.qpos[env_ids, 2] = z + _env_origin_z(env, env_ids)
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qpos[env_ids, 7:] = joints
    env.sim.data.qvel[env_ids, :] = 0.0


def randomize_servo_joints_uniform(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    range_frac: float = 0.8,
):
    """对 ``env_ids`` 把每个伺服关节在其限位的中央 ``range_frac`` 范围内均匀采样；
    关节速度清零。使用实体写入 API，因此在 passive 关节交错的 backlash/roller
    模型上也正确。"""
    asset: Entity = env.scene[asset_cfg.name]
    ids = torch.as_tensor(_servo_joint_ids(env, asset), device=env.device, dtype=torch.long)
    lim = asset.data.joint_pos_limits[env_ids][:, ids]  # (n, S, 2)
    lo, hi = lim[..., 0], lim[..., 1]
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * range_frac
    sample = mid + (torch.rand(len(env_ids), len(ids), device=env.device) * 2.0 - 1.0) * half
    pos = asset.data.joint_pos[env_ids].clone()
    pos[:, ids] = sample
    asset.write_joint_position_to_sim(pos, env_ids=env_ids)
    asset.write_joint_velocity_to_sim(torch.zeros_like(pos), env_ids=env_ids)


def maybe_set_random_prone_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    prone_prob: float = 0.0,
    face_down_prob: float = 0.5,
    prone_z_min: float = 0.20,
    prone_z_max: float = 0.25,
    crouch_prob: float = 0.0,
    side_prob: float = 0.0,
    joint_random_prob: float = 0.0,
    joint_range_frac: float = 0.8,
    joint_random_extra_z: float = 0.06,
    so3_prob: float = 0.0,
    so3_extra_z: float = 0.04,
    sit_prob: float = 0.0,
    sit_params: Optional[dict] = None,
):
    """重置事件，以 `prone_prob` 概率把朝向覆盖为俯卧。

    ``so3_prob``（2026-10，velstand 起身覆盖率）：俯卧切片中得到一个【均匀】随机
    朝向（而非精确 ±90°）的比例，总是伴随随机的伺服关节（含头部）以及
    ``joint_random_extra_z + so3_extra_z`` 的额外净空 —— 机器人下落几厘米，落在
    该姿态带来的任意休息态。覆盖 ±90° HOME 关节生成从未出现的"被放下/倒在奇怪
    地方"的起点。

    ``sit_prob``：一个独占切片（在俯卧和下蹲之后）重置到 ``set_random_ground_state``
    的坐姿桶，使用 ``sit_params``（关节 overrides、噪声、倾斜、z 带）。

    ``joint_random_extra_z``：关节随机化环境的额外生成净空。随机折叠的腿会伸到
    trunk 下方约 10 cm；在粗糙地形上生成时会穿透台阶/斜坡（实测 800-1200 N 尖峰，
    而 HOME 关节约 110 N）。这点额外下落只花约 70 N。

    ``side_prob``（俯卧切片中侧躺的比例）透传给 set_random_prone_orientation。

    ``joint_random_prob``：俯卧切片中【伺服】关节被在各自限位中央 ``joint_range_frac``
    范围内重新均匀采样的比例（速度清零）。类跌倒后生成（2026-09-10，protective fall
    第 4 次运行）：真实跌倒会让腿停在步态当时的位置 —— hip roll 顶在限位、头 yaw 偏
    1.7 rad、膝盖任意位置 —— 而策略在被推后仰卧落地只恢复 58-77%，从 HOME 姿态仰卧
    生成则 100%。其余比例保持 HOME 姿态（daemon 的 from-init 起身场景）。

    以 `prone_prob` 概率，把 reset_base 已设好的直立朝向替换为俯卧朝向；否则保持直立。
    在被覆盖的环境中，`face_down_prob` 选择俯卧（腹部）还是仰卧（背部）。

    还会把被覆盖环境的 z 抬到 [prone_z_min, prone_z_max]，使头/颈净空充足 ——
    vel 环境的重置 z（~0.125）在 90° pitch 时会让头穿过地面。

    在 prone_prob=2/3、face_down_prob=0.5 时得到直立/俯卧/仰卧平衡的 33/33/33 划分，
    这是与正常直立起点一起学跌倒恢复的标准混合。

    ``crouch_prob`` > 0 时，额外一个独占切片通过 ``set_random_crouch_state`` 重置到
    随机恢复中途下蹲（恢复最后一英里的反向课程 —— 见其 docstring）。
    """
    if prone_prob <= 0.0 and crouch_prob <= 0.0 and sit_prob <= 0.0:
        return
    # env_ids=None 表示"所有环境"（初始全局重置传的是 None ——
    # 旧的早退会在那里静默跳过俯卧初始化）。
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    env_ids_t = env_ids.to(env.device, dtype=torch.long) if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=env.device, dtype=torch.long)
    # 一次抽样把环境划分为互斥的 俯卧 / 下蹲 / 未动 三个切片。
    u = torch.rand(len(env_ids_t), device=env.device)
    selected = env_ids_t[u < prone_prob]
    crouch_selected = env_ids_t[(u >= prone_prob) & (u < prone_prob + crouch_prob)]
    sit_selected = env_ids_t[(u >= prone_prob + crouch_prob) & (u < prone_prob + crouch_prob + sit_prob)]
    if len(selected) > 0:
        set_random_prone_orientation(
            env, selected, asset_cfg=asset_cfg, face_down_prob=face_down_prob, side_prob=side_prob
        )
        # 覆盖 z，使俯卧身体落定时有头/颈净空。
        z = torch.rand(len(selected), device=env.device) * (prone_z_max - prone_z_min) + prone_z_min
        env.sim.data.qpos[selected, 2] = z + _env_origin_z(env, selected)
        # so3_prob == 0 时不抽样，使 RNG 流与 so3 之前的代码一致（v7 配方）
        so3 = (torch.rand(len(selected), device=env.device) < so3_prob) if so3_prob > 0.0 else torch.zeros(len(selected), dtype=torch.bool, device=env.device)
        if so3.any():
            ids = selected[so3]
            env.sim.data.qpos[ids, 3:7] = _uniform_random_quat(len(ids), env.device)
            env.sim.data.qpos[ids, 2] += so3_extra_z
        if joint_random_prob > 0.0 or so3.any():
            jr = selected[(torch.rand(len(selected), device=env.device) < joint_random_prob) | so3]
            if len(jr) > 0:
                randomize_servo_joints_uniform(env, jr, asset_cfg=asset_cfg, range_frac=joint_range_frac)
                env.sim.data.qpos[jr, 2] += joint_random_extra_z
    if len(crouch_selected) > 0:
        set_random_crouch_state(env, crouch_selected, asset_cfg=asset_cfg)
    if len(sit_selected) > 0:
        set_random_ground_state(
            env, sit_selected, asset_cfg=asset_cfg,
            face_down_prob=0.0, face_up_prob=0.0, sitting_prob=1.0, standing_prob=0.0, **(sit_params or {}),
        )


def _uniform_random_quat(n: int, device) -> torch.Tensor:
    """均匀随机单位四元数 (w, x, y, z) —— Shoemake 方法。"""
    a, b, c = torch.rand(n, 3, device=device).unbind(1)
    return torch.stack([
        torch.sqrt(1 - a) * torch.sin(2 * math.pi * b), torch.sqrt(1 - a) * torch.cos(2 * math.pi * b),
        torch.sqrt(a) * torch.sin(2 * math.pi * c), torch.sqrt(a) * torch.cos(2 * math.pi * c),
    ], dim=1)


def event_param_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    param_stages: list[dict],
) -> torch.Tensor:
    """在调度的 step 修改某个事件项的 params。

    与 termination_param_curriculum 类似，但用于事件。通过 get_term_cfg 使用活的
    EventManager 项 cfg，因为 env.cfg.events 是 deepcopy。param_stages:
    {step: int, params: dict} 的列表。在最新匹配的阶段浅合并到活事件项的 params。
    """
    del env_ids
    event_cfg = env.event_manager.get_term_cfg(event_name)
    current = param_stages[0]["params"]
    for stage in param_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["params"]
    event_cfg.params.update(current)
    first_val = next(iter(current.values()))
    return torch.tensor(float(first_val) if isinstance(first_val, (int, float)) else 0.0)


def face_down_prob_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    prob_stages: list[dict],
) -> torch.Tensor:
    """在训练中逐步提高某重置事件的 face_down_prob。

    Args:
        event_name: 使用 set_random_prone_orientation 的事件项名称
        prob_stages: {step: int, prob: float} 的列表。prob 越高 = 越多
            俯卧重置（任务越简单）；随训练推向 0.5。
    """
    del env_ids

    # 注意：必须更新活的 EventManager 的 term_cfg，而不是 env.cfg.events ——
    # EventManager.__init__ 做了 deepcopy(cfg)，所以修改 env.cfg.events 是空操作。
    event_cfg = env.event_manager.get_term_cfg(event_name)

    current_prob = prob_stages[0]["prob"]
    for stage in prob_stages:
        if env.common_step_counter > stage["step"]:
            current_prob = stage["prob"]

    event_cfg.params["face_down_prob"] = current_prob
    return torch.tensor([current_prob])


class VelocityCommandCommandOnly(UniformVelocityCommand):
    """类似 UniformVelocityCommand，但只画指令箭头（不画实际速度箭头）。"""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        # 原地转向练习：对一部分环境，把线速度置零
        # 并强制一个有意义（远离零）的 yaw 指令。独立均匀
        # 采样几乎不会产生"lin≈0, |ang| 大"（约占样本 2%），所以
        # 原地打转基本没被训练 → 真机转向缓慢/不稳定。
        # 镜像基础的 rel_forward_envs 机制。
        p = getattr(self.cfg, "rel_turn_in_place_envs", 0.0)
        if p <= 0.0:
            return
        r = torch.empty(len(env_ids), device=self.device)
        turn_ids = env_ids[r.uniform_(0.0, 1.0) < p]
        if len(turn_ids) == 0:
            return
        self.vel_command_b[turn_ids, 0] = 0.0
        self.vel_command_b[turn_ids, 1] = 0.0
        lo, hi = self.cfg.ranges.ang_vel_z
        maxr = max(abs(lo), abs(hi))
        rr = torch.empty(len(turn_ids), device=self.device)
        sign = torch.where(rr.uniform_(0.0, 1.0) < 0.5, -1.0, 1.0)
        lo_frac = getattr(self.cfg, "turn_in_place_min_frac", 0.4)
        mag = torch.empty(len(turn_ids), device=self.device).uniform_(lo_frac * maxr, maxr)
        self.vel_command_b[turn_ids, 2] = sign * mag
        # 这些环境必须真的转向 —— 取消其站立标记（否则指令会被
        # 清零）并刷新世界系参考副本。
        self.is_standing_env[turn_ids] = False
        self.vel_command_w[turn_ids] = self.vel_command_b[turn_ids]

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        batch = visualizer.env_idx
        if batch >= self.num_envs:
            return

        cmds = self.command.cpu().numpy()
        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        base_quat_w = self.robot.data.root_link_quat_w
        base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()

        base_pos_w = base_pos_ws[batch]
        base_mat_w = base_mat_ws[batch]
        cmd = cmds[batch]

        if np.linalg.norm(base_pos_w) < 1e-6:
            return

        def local_to_world(vec: np.ndarray) -> np.ndarray:
            return base_pos_w + base_mat_w @ vec

        scale = self.cfg.viz.scale * 2.0
        z_offset = self.cfg.viz.z_offset

        # 指令线速度箭头（蓝色）。
        cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
        cmd_lin_to = local_to_world(
            (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
        )
        visualizer.add_arrow(cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015)


@_dataclass(kw_only=True)
class VelocityCommandCommandOnlyCfg(UniformVelocityCommandCfg):
    # 每次重采样时被指令原地转向的环境比例（lin=0，|ang| 强制到
    # [0.4·max, max]）。0 = 禁用（仅基础均匀采样）。
    rel_turn_in_place_envs: float = 0.0
    # 强制原地转向 |wz| 的下界，取范围最大值的比例。
    # 0.4 (historical) means in-place yaw below 0.4 rad/s is never trained (standing
    # 环境把整个指令清零；均匀采样永远不会给出 lin == 0）。
    turn_in_place_min_frac: float = 0.4

    def build(self, env: ManagerBasedRlEnv) -> "VelocityCommandCommandOnly":
        return VelocityCommandCommandOnly(self, env)


class RelativeHeadingVelocityCommand(VelocityCommandCommandOnly):
    """速度指令，其中 cmd[2] 是机器人机身系下的朝向误差。

    cmd[0] = lin_vel_x  （油门：0=滑行，+推动，-刹车）
    cmd[1] = lin_vel_y  （未使用，0）
    cmd[2] = heading_error  （+ = 目标在右/顺时针，- = 在左/逆时针）
             0 → 直走，±max = 目标在右/左 max_angle rad

    训练时：每次 episode 重置采样一个随机的世界系朝向。每步
    cmd[2] = clamp(wrap(current_yaw - target_yaw), ±max_angle)。
    机器人朝向目标逆时针（左）偏时为正 → 需要右转。

    推理时：用户直接喂 cmd[2]。保持 cmd[2] = 常数即给出比例朝向修正 = 近似恒定转速。

    cfg 中设置 heading_command=False、rel_heading_envs=0.0（我们内部处理朝向）。
    cfg 中的 ang_vel_z 范围用作 cmd[2] 的裁剪上限。
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # 每环境采样的目标朝向，世界系（rad）
        self._target_heading_w = torch.zeros(self.num_envs, device=self.device)
        # cmd[2] 的裁剪上限：用 cfg 里的 ang_vel_z[1]（正的上界）
        ang_rng = cfg.ranges.ang_vel_z
        self._heading_max = float(ang_rng[1]) if ang_rng else 1.0

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        n = len(env_ids)
        # 在 [-π, π] 上均匀采样随机的世界系目标朝向
        self._target_heading_w[env_ids] = (
            torch.rand(n, device=self.device) * 2.0 * math.pi - math.pi
        )
        # ang_vel 槽位置零；_update_command 每步会填充它
        self.vel_command_b[env_ids, 2] = 0.0

    def _update_command(self) -> None:
        # 不要调用 super()._update_command() —— 它会跑朝向
        # 比例控制器并用 yaw 速率覆盖 cmd[2]。
        # 改为每步从头重新计算朝向误差。
        quat = self.robot.data.root_link_quat_w  # (N, 4) [w, x, y, z]
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        current_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        # 正 = 目标在机器人逆时针（左）方向 → 左转。标准约定。
        delta = self._target_heading_w - current_yaw
        heading_error = torch.atan2(torch.sin(delta), torch.cos(delta))
        self.vel_command_b[:, 2] = heading_error.clamp(-self._heading_max, self._heading_max)

    def _update_metrics(self) -> None:
        pass  # No velocity tracking metrics for heading command


class RelativeHeadingVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> "RelativeHeadingVelocityCommand":
        return RelativeHeadingVelocityCommand(self, env)


def heading_tracking_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float = 0.5,
) -> torch.Tensor:
    """当 cmd[2] 编码朝向误差时，奖励减小朝向误差。

    返回 exp(-cmd[2]² / std²)。
    - 误差 = 0（朝向正确）：reward = 1.0。
    - 误差 = std：reward ≈ 0.37（强梯度）。
    - std=0.5 时误差 = 1.0 rad：reward ≈ 0.018（几乎为零）。

    std=0.5 rad（≈28°）在预期范围内给出有意义的梯度。
    """
    cmd = env.command_manager.get_command(command_name)
    heading_error = cmd[:, 2]
    return torch.exp(-(heading_error ** 2) / (std ** 2))


def skating_air_time_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.4,
    vel_gate_ref: float = 0.0,
) -> torch.Tensor:
    """只在推动时（cmd_x > 0）奖励脚的 air time。

    鼓励机器人在滑冰划动的恢复阶段抬起每只脚，而不是拖着它。按 cmd_x 缩放，
    使激励随推动强度增长。

    当 ``vel_gate_ref`` > 0 时，reward 还要乘一个前向速度门控，使抬脚但不推动身体
    （原地踢踏）得不到奖励。``threshold_min`` 设定计为一次摆动的最短时长 —— 提高它
    可禁止疯狂的高频抖动。
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None

    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    reward = reward * torch.clamp(cmd_x, min=0.0)
    gate = _forward_progress_gate(env, vel_gate_ref)
    if gate is not None:
        reward = reward * gate
    return reward


def _forward_progress_gate(env: ManagerBasedRlEnv, v_ref: float) -> torch.Tensor | None:
    """身体前向速度的 0→1 斜坡：静止时为 0，达到/超过 v_ref 时为 1。

    用于给步态塑形 reward 加门控，使不推动身体的迈步（如原地踢踏）得不到奖励 ——
    步态【形式】的 reward 只在该步态真正完成其【职责】（向前移动）时才支付。
    禁用时（v_ref <= 0）返回 None。"""
    if v_ref <= 0.0:
        return None
    v_fwd = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    return (v_fwd.clamp(min=0.0) / v_ref).clamp(max=1.0)


def single_support_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    vel_gate_ref: float = 0.0,
    double_penalty: float = 0.25,
) -> torch.Tensor:
    """奖励单脚支撑（一次滑冰划动），轻度抑制擦地滑行（swizzle）。

    真正的滑冰是一次【划动】：一只脚蹬地、另一只摆动，即左右交替的单脚支撑。
    对称的 swizzle 让两只脚全程贴地却仍能转轮子，所以单靠 wheel_speed 会收敛到它。

    每步，按接触的脚数：
      - 恰好 1 只脚着地（划动）→ + clamp(cmd_x,0) · gate
      - 2 只脚着地（双脚支撑）→ − double_penalty · clamp(cmd_x,0)
      - 0 只脚着地（腾空/跳跃）→ 0

    【正】的单脚支撑 reward 由前向速度（``vel_gate_ref``）门控，因此原地迈步（无推进）
    得不到奖励 —— 杀死踢踏漏洞。双脚支撑惩罚很小且【不】门控：重心转移/蹬地时短暂
    双脚支撑是正常的滑冰，所以我们只是轻微抑制【永久】双脚支撑（swizzle）而非禁止。
    真正的反 swizzle 信号是 skating_air_time —— swizzle 从不抬脚。
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None

    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)  # (num_envs,)
    single = (n_contact == 1).float()
    double = (n_contact >= 2).float()

    cmd_x = torch.clamp(env.command_manager.get_command(command_name)[:, 0], min=0.0)
    single_r = single * cmd_x
    gate = _forward_progress_gate(env, vel_gate_ref)
    if gate is not None:
        single_r = single_r * gate
    return single_r - double_penalty * double * cmd_x


def glide_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    vel_ref: float = 0.2,
    stillness_std: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=(r".*(hip|knee|ankle).*",)
    ),
) -> torch.Tensor:
    """奖励划动的【滑翔】阶段：单脚贴地、腿部安静地滑行。

    没有别的 reward 奖励滑翔 —— skating_air_time 为每次摆动付奖励，所以策略会最大化
    摆动【频率】（疯狂踢腿）。本项奖励保持单脚并滑行，给策略放慢并专注完成每次划动的
    理由：

        reward = single_support · forward_gate · stillness · (cmd_x >= 0)

    - single_support：恰好【一只】脚接触。【必需】—— 这是对之前坏的滑翔的修复，
      它漏掉了这一项，让双脚 swizzle 滑行能刷 reward 并使步态退化。
    - forward_gate = clamp(v_fwd,0,vel_ref)/vel_ref → 不向前时为零。
    - stillness = exp(-Σ leg_joint_vel² / stillness_std²) → 只在腿安静时高；
      踢腿（快速关节运动）约得 0，所以只有真正的滑翔才有奖励。
    - 只在推动/滑行时生效（cmd_x >= 0）；刹车时静默。
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None
    single = (torch.sum((contact_time > 0.0).float(), dim=1) == 1).float()

    forward_gate = _forward_progress_gate(env, vel_ref)
    if forward_gate is None:
        forward_gate = torch.ones(env.num_envs, device=env.device)

    asset: Entity = env.scene[asset_cfg.name]
    joint_vel_sq = torch.sum(
        torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1
    )
    stillness = torch.exp(-joint_vel_sq / stillness_std ** 2)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    active = (cmd_x >= 0.0).float()
    return single * forward_gate * stillness * active


def leg_symmetry_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_bases: tuple = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"),
) -> torch.Tensor:
    """奖励左右腿镜像 —— swizzle 的定义性对称特征。

    机器人使用镜像的左右符号约定，因此双侧对称构型在每个匹配关节对上满足
    q_left + q_right ≈ 0。返回 ``-mean_pairs |q_left + q_right|``（L1，梯度恒定）；
    配合【正】权重使用，从而惩罚不对称、偏好对称的 swizzle。左右索引对按名称解析一次
    并缓存在 env 上。
    """
    asset: Entity = env.scene[asset_cfg.name]
    if not hasattr(env, "_leg_sym_ids"):
        left, right = [], []
        for base in joint_bases:
            li, _ = asset.find_joints([f"left_{base}"])
            ri, _ = asset.find_joints([f"right_{base}"])
            left.append(li[0])
            right.append(ri[0])
        env._leg_sym_ids = (
            torch.tensor(left, device=env.device),
            torch.tensor(right, device=env.device),
        )
    lids, rids = env._leg_sym_ids
    q = asset.data.joint_pos
    return -torch.abs(q[:, lids] + q[:, rids]).mean(dim=-1)


def grounded_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
) -> torch.Tensor:
    """奖励【两只】脚都接触 —— 经典 swizzle 始终贴地（不抬脚）。

    是 single_support_reward 的镜像，但奖励双脚支撑（n_contact >= 2），按 |cmd_x|
    缩放，以便在【任一】方向塑造推动阶段（前进或后退 —— swizzle 环境把 cmd_x < 0
    当作"后退"）。
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None
    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)
    grounded = (n_contact >= 2).float()
    cmd_x = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    return grounded * cmd_x


def gait_symmetry_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """惩罚左右脚使用不均衡（一只脚干了大部分活）。

    对称性增强关闭时，没有什么阻止策略学到主要靠一条腿蹬的不对称步态 —— 它会跑偏、
    失稳（尤其在起滑时）。本项在整段 episode 中累积每只脚的摆动时间，惩罚归一化的
    不平衡 |L - R| / (L + R)：
      - 均衡交替步态 → ~0（无惩罚）
      - 一只脚摆动明显更多 → ~1（最大惩罚）
    只惩罚【累积】不平衡 —— 真实步态中瞬时的单脚支撑不对称（当前一只脚在摆）是正常的。
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    air = sensor.data.current_air_time  # (N, num_feet)
    assert air is not None

    if not hasattr(env, "_swing_accum") or env._swing_accum.shape[0] != env.num_envs:
        env._swing_accum = torch.zeros(env.num_envs, air.shape[1], device=env.device)
    reset = env.episode_length_buf <= 1
    env._swing_accum[reset] = 0.0
    env._swing_accum += (air > 0.0).float() * env.step_dt

    L = env._swing_accum[:, 0]
    R = env._swing_accum[:, 1]
    return torch.abs(L - R) / (L + R + 1e-3)


def heading_hold_reward(
    env: ManagerBasedRlEnv,
    std: float = 0.4,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """奖励保持【生成时的朝向】（直走）—— 基于角度的纠正式方案。

    奖励 yaw【角】保持在重置时捕获的朝向附近：
        reward = exp(-wrap(yaw - yaw_spawn)² / std²)

    这是直走的正确方式（对比惩罚 yaw【速率】—— 那只是告诉策略"永不转向" → 它无法
    转回来纠正、只能开环漂移）。这里漂移会降低 reward，而策略可以自由地把 yaw 转回来
    恢复它。

    生成朝向在重置后头几步（episode_length_buf <= 1）、机器人仍接近生成姿态时逐 env
    捕获。读取 root_link_quat_w，它在计算 reward 时（物理步之后）是最新的。朝向不变：
    参考是每个 env 自己的随机生成 yaw，因此与重置时的整圈 yaw 随机化兼容。
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4) [w, x, y, z]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    if not hasattr(env, "_heading_ref") or env._heading_ref.shape[0] != env.num_envs:
        env._heading_ref = yaw.clone()
    just_reset = env.episode_length_buf <= 1
    env._heading_ref = torch.where(just_reset, yaw, env._heading_ref)

    err = yaw - env._heading_ref
    err = torch.atan2(torch.sin(err), torch.cos(err))  # wrap to [-π, π]
    return torch.exp(-(err ** 2) / std ** 2)


def action_over_limit_penalty(
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
    overshoot: float = 0.3,
) -> torch.Tensor:
    """惩罚把关节目标指令到超过其硬限位（+ 过冲）。

    策略侧的威慑，防止把关节过度驱动撞到机械挡块：例如 hip_roll 限位 ±0.38 rad，
    但 ctrlrange 是 ±10 rad，低 kp 伺服可以被指令远超挡块，用最大力矩猛撞它 ——
    这是只在仿真里有效的脆弱技巧，无法迁移。

    读取指令目标（raw_action · scale + offset），只惩罚【超出】(hard_limit + overshoot)
    的部分：

        penalty = Σ relu(target - (hi + overshoot)) + relu((lo - overshoot) - target)

    与 qpos 限位惩罚不同，它在【指令】而非关节位置上触发 —— 因此关节仍可达其完整范围
    （指令 ≈ 限位），不会损失可用幅度。因为它约束策略的【输出】，学到的行为被烘焙进
    网络，部署时【不需要】任何环境侧的动作裁剪（裁剪只存在于仿真 → 不匹配）。
    ``overshoot`` 给低 kp 伺服留出在负载下到达近限位目标的余量；只惩罚远超其的狂暴
    过驱。
    """
    term = env.action_manager.get_term(action_name)
    target = term.raw_action * term.scale + term.offset  # (B, action_dim) abs targets
    jnt_ids = term.target_ids
    hard = env.scene["robot"].data.joint_pos_limits[:, jnt_ids]  # (B, action_dim, 2)
    lo = hard[..., 0] - overshoot
    hi = hard[..., 1] + overshoot
    over = (target - hi).clip(min=0.0) + (lo - target).clip(min=0.0)
    return torch.sum(over, dim=-1)


def forward_lean_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    target_pitch: float = 0.08,
    std: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",)),
) -> torch.Tensor:
    """奖励推动时略微前倾，以抵消滑冰划动带来的向后力矩。

    用 projected_gravity_b 的 x 分量作为 pitch 代理：
      forward_lean = -gravity_b[:, 0]  （前倾时为正）

    只在 cmd_x > 0 时触发。在 target_pitch rad 的前倾处达到峰值。
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    forward_lean = asset.data.projected_gravity_b[:, 0]
    push = torch.clamp(cmd_x, min=0.0)
    return push * torch.exp(-((forward_lean - target_pitch) ** 2) / (std ** 2))


class GroundPickPhaseCommand(UniformVelocityCommand):
    """用于 ground pick / sit-stand 任务的相位编码指令。

    用循环相位信号取代速度指令：
        command = [cos(2π*phase), sin(2π*phase), 0]

    Phase ∈ [0, 0.5]：接近（下降）。
    Phase ∈ [0.5, 1.0]：回程（上升回来）。

    相位在 episode 重置时逐环境随机化以解相关。周期默认 4s；通过 cfg.period 覆盖
    （sitstand 用 8s 以实现更慢、更柔和的坐下）。
    """

    PERIOD: float = 4.0  # default; cfg.period overrides

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._gp_phase = torch.zeros(self.num_envs, device=self.device)
        self._period = float(getattr(cfg, "period", self.PERIOD))
        # 为 False 时，每个 episode 从相位 0（站立）开始，而不是
        # 随机相位。匹配运行时：按钮从站立在相位 0 启动循环。
        # 默认 True 保持历史上 ground_pick 的行为
        #（随机相位以解相关环境）。
        self._randomize_phase = bool(getattr(cfg, "randomize_phase", True))

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def compute(self, dt: float) -> None:
        self._gp_phase = (self._gp_phase + dt / self._period) % 1.0
        self.vel_command_b[:, 0] = torch.cos(2 * torch.pi * self._gp_phase)
        self.vel_command_b[:, 1] = torch.sin(2 * torch.pi * self._gp_phase)
        self.vel_command_b[:, 2] = 0.0

    def reset(self, env_ids: torch.Tensor | None) -> dict:
        if env_ids is not None and len(env_ids) > 0:
            if self._randomize_phase:
                self._gp_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
            else:
                self._gp_phase[env_ids] = 0.0
        return {}

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        pass  # Phase is continuous; no resampling needed

    def _update_command(self) -> None:
        pass  # Updated in compute()

    def _update_metrics(self) -> None:
        pass  # No velocity tracking metrics for ground pick


from dataclasses import dataclass as _dataclass

@_dataclass(kw_only=True)
class GroundPickPhaseCommandCfg(UniformVelocityCommandCfg):
    class_type: type = GroundPickPhaseCommand
    period: float = 4.0  # cycle length in seconds; sitstand uses 8.0
    randomize_phase: bool = True  # False -> each episode starts at phase 0 (standing)

    def build(self, env: ManagerBasedRlEnv) -> "GroundPickPhaseCommand":
        return GroundPickPhaseCommand(self, env)


# --------------------------------------------------------------------------- #
# 统一姿态指令机制                                                            #
# --------------------------------------------------------------------------- #
#
# 背景：我们废弃了旧的 NeckOffsetJointPositionAction +
# 扰动随机化方案（头/身体运动是策略应对的外部扰动）。
# 那训练出的是微弱、间接的信号 —— 事后分析见
# `project_neck_offset_decoupling.md`。
# 
#
# 替代方案：头和身体姿态现在是【指令】—— 直接、密集的策略
# 输入，配有跟踪 reward。部署时，运行时往这些槽位喂
# 用户请求的任意姿态；训练时，从逐维范围均匀采样
# （从第 0 步就保持非零，使输入神经元保持活跃）
# 并通过 curriculum 渐增。
#
# 布局，为运行时 obs 兼容在所有 microduck 策略中统一：
#   指令向量 (13D) = [vx, vy, vtheta,           ← "twist"（速度）
#                      neck_pitch, head_pitch,   ← "head_pose"（增量）
#                      head_yaw, head_roll,
#                      body_x, body_y, body_z,   ← "body_pose"（增量）
#                      body_roll, body_pitch, body_yaw]
# 策略 obs 总维度变为 61D（51 - 3 + 13）。
# --------------------------------------------------------------------------- #


from dataclasses import dataclass, field


class UniformPoseCommand(CommandTerm):
    """通用 N 维均匀姿态指令。

    在 cfg.ranges[i] = (lo, hi) 中对每维独立均匀采样，并在两次重采样之间保持该值。
    没有 metric、没有调试可视化 —— 保持轻量，因为我们有很多这样的指令。
    """

    cfg: "UniformPoseCommandCfg"

    def __init__(self, cfg: "UniformPoseCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.dim = len(cfg.ranges)
        self._command = torch.zeros(self.num_envs, self.dim, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        r = torch.empty(n, device=self.device)
        for i, (lo, hi) in enumerate(self.cfg.ranges):
            self._command[env_ids, i] = r.uniform_(lo, hi)
        # 显式的零指令桶。均匀采样几乎永远不会
        # 产生全零指令，因此部署空闲场景（"保持
        # 标称姿态"）否则会缺席训练（body-control
        # 第 1 次运行的教训：策略只有在有指令时
        # 才静止站立）。
        if self.cfg.zero_command_prob > 0.0:
            zero_mask = torch.rand(n, device=self.device) < self.cfg.zero_command_prob
            self._command[env_ids[zero_mask]] = 0.0


@dataclass(kw_only=True)
class UniformPoseCommandCfg(CommandTermCfg):
    """逐维均匀范围；构建一个 UniformPoseCommand。"""
    # 每维的 (lo, hi) 元组。长度定义指令的维度。
    ranges: tuple[tuple[float, float], ...] = ()
    # 一次重采样产生精确全零指令的概率。
    zero_command_prob: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> "UniformPoseCommand":
        return UniformPoseCommand(self, env)


class StandingGatedPoseCommand(UniformPoseCommand):
    """只在 twist 指令处于站立状态时才【生效】的 UniformPoseCommand。

    身体控制是部署时的站立功能：行走时运行时发送零身体指令。采样值存在 ``_raw``；
    暴露的指令是 twist 项的 ``_raw × is_standing_env``，每步重新施加（twist 项可以在
    自己的重采样时把环境在站立/行走之间切换）。要求 twist 指令在本指令【之前】注册
    （CommandManager 按插入顺序计算；基础 velocity cfg 先插入 twist）。

    ``axis_zero_prob[i]``：重采样时把某轴清零的逐轴概率 —— 叠加在 ``zero_command_prob``
    （全零）之上。VelStand 用它让一部分倾斜指令携带 z == 0（那些帧由 stand 专家教，
    它不会下蹲；见 distill.py 的 body routing）。
    """

    cfg: "StandingGatedPoseCommandCfg"

    def __init__(self, cfg: "StandingGatedPoseCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._raw = torch.zeros_like(self._command)

    def _gate(self) -> None:
        twist = self._env.command_manager.get_term(self.cfg.twist_command_name)
        standing = getattr(twist, "is_standing_env", None)
        if standing is None:
            self._command[:] = 0.0
        else:
            self._command[:] = self._raw * standing.float().unsqueeze(1)

    def _update_command(self) -> None:
        self._gate()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        super()._resample_command(env_ids)  # writes self._command[env_ids] (incl. zero bucket)
        raw = self._command[env_ids].clone()
        for i, p in enumerate(self.cfg.axis_zero_prob):
            if p > 0.0:
                raw[torch.rand(n, device=self.device) < p, i] = 0.0
        self._raw[env_ids] = raw
        self._gate()  # resets don't run _update_command: keep the walking-zero invariant


@dataclass(kw_only=True)
class StandingGatedPoseCommandCfg(UniformPoseCommandCfg):
    twist_command_name: str = "twist"
    axis_zero_prob: tuple[float, ...] = ()

    def build(self, env: ManagerBasedRlEnv) -> "StandingGatedPoseCommand":
        return StandingGatedPoseCommand(self, env)


from mjlab.tasks.velocity.mdp.rewards import variable_posture as _variable_posture  # noqa: E402
from mjlab.utils.lab_api.string import resolve_matching_names_values as _resolve_names_values  # noqa: E402


class variable_posture_body_relaxed(_variable_posture):
    """mjlab 的 ``variable_posture`` + 第 4 种状态：带【有效】身体指令的站立环境
    （``body_active_axes`` 中任一非零）使用 ``std_body``。

    原因（velstand 身体控制，2026-10 实测）：站立 std 刻意设得很紧（hip_roll 0.05
    维持 5° 站姿），但指令 10° 的 trunk roll 本身就是 hip_roll 变化 —— alpha_stand
    跟踪它时 pose 从 0.92 掉到 0.07/step，而 2 cm 下蹲几乎吃掉全部 z 跟踪余量。
    行走和零指令站立看到的完全是旧项。"""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
        _, _, std_body = _resolve_names_values(data=cfg.params["std_body"], list_of_strings=joint_names)
        self.std_body = torch.tensor(std_body, device=env.device, dtype=torch.float32)

    def __call__(self, env, std_standing, std_walking, std_running, asset_cfg, command_name,
                 walking_threshold: float = 0.5, running_threshold: float = 1.5,
                 std_body=None, body_command_name: str = "body_pose",
                 body_active_axes: tuple[int, ...] = (2, 3, 4)) -> torch.Tensor:
        del std_standing, std_walking, std_running, std_body
        asset: Entity = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        standing = speed < walking_threshold
        walking = (speed >= walking_threshold) & (speed < running_threshold)
        running = speed >= running_threshold
        body = env.command_manager.get_command(body_command_name)[:, list(body_active_axes)]
        body_active = standing & (body.abs().amax(dim=1) > 0.0)
        std = (self.std_standing * (standing & ~body_active).float().unsqueeze(1)
               + self.std_body * body_active.float().unsqueeze(1)
               + self.std_walking * walking.float().unsqueeze(1)
               + self.std_running * running.float().unsqueeze(1))
        err2 = torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - self.default_joint_pos[:, asset_cfg.joint_ids])
        return torch.exp(-torch.mean(err2 / (std ** 2), dim=1))


def track_yaw_rate_fine(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """仅偏航速率的 Gaussian，exp(-(wz_cmd - wz)²/std²)，【附加】在原有的
    track_angular_velocity 旁边（后者保持不变）。

    原因（velstand 偏航死区，2026-10-01）：反事实打分（给策略喂更大的 yaw 指令使其
    真的迈步+转向，在真实指令处算 reward）显示原地转向几乎不比站着强（0.6 rad/s 时
    +0.33/step）—— 原项 std 0.71 在小误差处太松，还把迈步产生的 roll/pitch 速率折进
    yaw 误差。用它替换成尖锐 yaw 项（run sape62zb）反而更【差】（1.0 rad/s：0.42 →
    0.16）：std-0.35 的 Gaussian 在策略 0.6 rad/s 的误差处几乎没有梯度。附加一个精细项
    保留了原项的远误差梯度并给小误差定价：0.6 rad/s 处 margin +0.33 → +0.97，
    1.0 处 +0.62 → +1.05。"""
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    return torch.exp(-torch.square(cmd[:, 2] - asset.data.root_link_ang_vel_b[:, 2]) / std**2)


def upright_body_cmd_relative(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str = "body_pose",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """mjlab 平地的 ``upright``（exp(-|g_b,xy|²/std²)），但相对【指令的】roll/pitch
    而非水平来测量：目标 g_b = (sin θ, -sin φ·cos θ)，其中身体指令 roll φ = cmd[3]、
    pitch θ = cmd[4]（ZYX，与 body_pose_tracking_locomotion 约定相同）。零指令时与
    ``upright`` 完全相同，因此行走和普通站立看到的正是旧项；指令倾斜不再被 reward 栈中
    最强的姿态项收税（2.0，std²=0.05 → 10° 倾斜要花 0.9/step，约等于跟踪该倾斜的
    全部收益）。"""
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1) if asset_cfg.body_ids else asset.data.root_link_quat_w
    g_b = quat_apply_inverse(quat, asset.data.gravity_vec_w)
    cmd = env.command_manager.get_command(command_name)
    roll, pitch = cmd[:, 3], cmd[:, 4]
    target = torch.stack([torch.sin(pitch), -torch.sin(roll) * torch.cos(pitch)], dim=1)
    err = torch.sum(torch.square(g_b[:, :2] - target), dim=1)
    return torch.exp(-err / std**2)


def zero_command_padding(
    env: ManagerBasedRlEnv,
    dim: int,
) -> torch.Tensor:
    """宽度为 `dim` 的常量零 obs 项。

    供不主动跟踪 head/body 指令（如 sitstand、ground_pick）但仍需要统一 61D obs
    形状的环境使用，使运行时能用相同的 buffer 布局喂所有策略。
    """
    return torch.zeros(env.num_envs, dim, device=env.device)


def head_pose_tracking(
    env: ManagerBasedRlEnv,
    command_name: str = "head_pose",
    std: float = 0.5,
    fine_std: float | None = None,
    fine_weight: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """匹配指令颈部/头部增量的逐关节 Gaussian reward。

    对 4 个颈/头关节取 exp(-(err/std)^2) 的均值。结果为 (N,)，范围 [0, 1]。用均值
    形式（而非平方和）使只有一个关节偏离时梯度仍存活 —— 平方和里单个大误差会杀死
    整个 reward。

    `std` 是逐关节容差：err=std 时逐关节 reward 为 1/e（约 0.37）。std 应按指令范围
    的量级选取，使 curriculum 放宽时梯度不消失。

    `fine_std`（可选）混入第二个更窄的 Gaussian：
    (1-fine_weight)·exp(-(err/std)²) + fine_weight·exp(-(err/fine_std)²)。
    理由：单个宽 std（0.5 rad ≈ 29°）让小误差几乎免费 —— 沉重头部 10° 的重力下垂只
    花约 0.03 reward，于是策略任其下垂。窄分量（约 0.1 rad）给这些小误差定价，宽分量
    在 curriculum 放宽时在远指令处保持梯度。

    cmd 形状 (N, 4) = 相对默认关节位置的增量，顺序
    [neck_pitch, head_pitch, head_yaw, head_roll]。

    在 backlash 模型上测得的角度是 qpos[servo] + qpos[backlash] —— 即【输出】连杆，
    也是编码器 obs（joint_pos_rel_backlash）报告的值。只测伺服会让头部在齿隙游隙内
    无成本下垂，同时又因补偿它而惩罚策略（伺服上偏 = 伺服侧"误差"）。在没有
    passive_*_backlash 关节的模型上 mask 为 0，退化为只用伺服。
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 4)

    if not hasattr(env, "_head_pose_neck_ids"):
        ids, names = asset.find_joints_by_actuator_names(_NECK_JOINT_PATTERNS)
        env._head_pose_neck_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        name_to_id = {n: i for i, n in enumerate(asset.joint_names)}
        bl = [name_to_id.get(f"passive_{n}_backlash") for n in names]
        env._head_pose_bl_ids = torch.tensor(
            [0 if b is None else b for b in bl], device=env.device, dtype=torch.long
        )
        env._head_pose_bl_mask = torch.tensor(
            [0.0 if b is None else 1.0 for b in bl], device=env.device
        )

    neck_ids = env._head_pose_neck_ids
    joint_pos = asset.data.joint_pos
    measured = (
        joint_pos[:, neck_ids]
        + joint_pos[:, env._head_pose_bl_ids] * env._head_pose_bl_mask
    )
    actual = measured - asset.data.default_joint_pos[:, neck_ids]
    err = actual - cmd
    per_joint = torch.exp(-(err / std) ** 2)
    if fine_std is not None:
        per_joint = (1.0 - fine_weight) * per_joint + fine_weight * torch.exp(
            -(err / fine_std) ** 2
        )
    return per_joint.mean(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 传感器派生 critic 观测的 NaN 安全包装。
#
# `robot_state_is_nan` 覆盖关节 + root 状态，所以所有从
# 它们派生的 obs 都被它触发的重置保护。下面三项【不】被保护：
# 它们读取传感器数据（射线高度、接触 air-time、接触力），
# 在积分后的机器人状态仍干净时，MuJoCo 也可能返回非有限值。
# 它们仅供 critic，因此清洗一步对策略没有代价，而放任其值通过
# 会经 rsl_rl 的 check_nan 杀死整次运行。这里清洗不会掩盖真实的物理
# 爆炸 —— 那些仍然通过 nan_state 终止，并表现为
# wandb 中的 Episode_Termination/nan_state。
# 
# ─────────────────────────────────────────────────────────────────────────────


def _finite(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def foot_contact_forces_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """NaN 安全的 `foot_contact_forces`（见上面的说明）。"""
    return _finite(_velocity_obs.foot_contact_forces(env, sensor_name))


def foot_height_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """NaN 安全的 `foot_height`（见上面的说明）。"""
    return _finite(_velocity_obs.foot_height(env, sensor_name))


def foot_air_time_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """NaN 安全的 `foot_air_time`（见上面的说明）。"""
    return _finite(_velocity_obs.foot_air_time(env, sensor_name))


def head_pose_bias_penalty(
    env: ManagerBasedRlEnv,
    command_name: str = "head_pose",
    tau_s: float = 1.0,
    gate_height_low: float | None = None,
    gate_height_high: float = 0.11,
    gate_tilt_full_deg: float = 20.0,
    gate_tilt_zero_deg: float = 45.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """惩罚时间平均（DC）的颈/头跟踪误差：-mean(|EMA(err)|)。

    是 ``head_pose_tracking`` 的配套项，后者评估【瞬时】误差。为什么单独设 DC 项
    而不只是收紧那个 Gaussian 的 std：行走时那个占机器人质量 38% 的头部必然晃动，
    瞬时紧容差项是对行走的永久税，任何策略都无法规避 —— 实测约 0.77/step，而 air_time
    reward 约 1.01/step，这正是让 2026-08-20 的 velocity 运行彻底放弃迈步的原因
    （wandb 5yay13u4）。而稳态下垂是【可规避】的：策略可以把颈部指令上偏以抵消重力
    下垂。在 ``tau_s`` 上取平均让振荡相互抵消，只对偏置定价。

    特意用 L1（而非 Gaussian）：在大偏置处梯度保持恒定，而紧 Gaussian 在那里平坦且
    死掉。

    在 backlash 模型上测得的角度透过游隙读取，与 head_pose_tracking 和编码器 obs 一致。

    ``gate_height_low``（可选）：恢复环境（standup / velstand）的直立门控，形状与语义
    同 body_ang_vel_at_height —— 低于 gate_height_low 或倾斜超过 gate_tilt_zero_deg 时
    为零，高于 gate_height_high 且低于 gate_tilt_full_deg 时为满值。该门控乘在【喂给 EMA
    的误差】上（不只是输出）：跌倒/起身期间 EMA 看到零并衰减，于是到达直立时偏置时钟
    从 ~0 开始，而不是在终点把整段地面阶段的累计误差一起收费 —— 那会是恢复完成前的
    reward 墙，正是已退役的 head_impact_penalty 的失败模式。输出也被门控，因此新一次
    跌倒会立即停止收费。
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 4)

    if not hasattr(env, "_head_pose_neck_ids"):
        # 与 head_pose_tracking 共享 id 缓存（两者都可能先运行）。
        head_pose_tracking(env, command_name=command_name, asset_cfg=asset_cfg)

    neck_ids = env._head_pose_neck_ids
    joint_pos = asset.data.joint_pos
    measured = (
        joint_pos[:, neck_ids]
        + joint_pos[:, env._head_pose_bl_ids] * env._head_pose_bl_mask
    )
    err = (measured - asset.data.default_joint_pos[:, neck_ids]) - cmd

    if gate_height_low is not None:
        z = torch.nan_to_num(
            asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2],
            nan=0.0,
        )
        t = torch.clamp(
            (z - gate_height_low) / max(gate_height_high - gate_height_low, 1e-6),
            0.0, 1.0,
        )
        gate = t * t * (3.0 - 2.0 * t)
        quat = asset.data.root_link_quat_w
        cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        tilt_deg = torch.rad2deg(torch.acos(cos_tilt.clamp(-1.0, 1.0)))
        st = torch.clamp(
            (gate_tilt_zero_deg - tilt_deg)
            / max(gate_tilt_zero_deg - gate_tilt_full_deg, 1e-6),
            0.0, 1.0,
        )
        gate = gate * (st * st * (3.0 - 2.0 * st))
        err = err * gate.unsqueeze(-1)
    else:
        gate = None

    if not hasattr(env, "_head_bias_ema"):
        env._head_bias_ema = torch.zeros_like(err)
    # 刚重置的环境：丢弃上一 episode 累积的偏置。
    fresh = env.episode_length_buf <= 1
    env._head_bias_ema[fresh] = 0.0

    alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-6))
    env._head_bias_ema = (1.0 - alpha) * env._head_bias_ema + alpha * err
    out = -env._head_bias_ema.abs().mean(dim=-1)
    if gate is not None:
        out = out * gate
    return out


def body_pose_tracking_6d(
    env: ManagerBasedRlEnv,
    command_name: str = "body_pose",
    nominal_height: float = 0.095,
    xy_std: float = 0.02,
    z_std: float = 0.01,
    angle_std: float = math.radians(8),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """跟踪指令身体姿态的 6 个逐轴 Gaussian reward 的均值。

    cmd 形状 (N, 6) = [x, y, z, roll, pitch, yaw]，全部为相对标称站姿的增量
    （xy 增量相对生成原点，z 增量相对 nominal_height，角度增量相对直立 = 0）。
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 6)
    dx, dy, dz = cmd[:, 0], cmd[:, 1], cmd[:, 2]
    droll, dpitch, dyaw = cmd[:, 3], cmd[:, 4], cmd[:, 5]

    # 相对环境生成原点的位置。用 nan_to_num 因为 MuJoCo 可能
    # 在接触不稳定时产生 NaN，我们不想污染 reward。
    pos_w = asset.data.root_link_pos_w
    origin = env.scene.terrain.env_origins
    rel = torch.nan_to_num(pos_w - origin, nan=0.0)
    x_err = rel[:, 0] - dx
    y_err = rel[:, 1] - dy
    z_err = rel[:, 2] - (nominal_height + dz)

    # 从四元数得到 ZYX 欧拉角。
    quat = asset.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll  = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw   = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    roll_err  = roll  - droll
    pitch_err = pitch - dpitch
    yaw_err   = wrap_to_pi(yaw - dyaw)

    r_x = torch.exp(-(x_err / xy_std) ** 2)
    r_y = torch.exp(-(y_err / xy_std) ** 2)
    r_z = torch.exp(-(z_err / z_std) ** 2)
    r_r = torch.exp(-(roll_err  / angle_std) ** 2)
    r_p = torch.exp(-(pitch_err / angle_std) ** 2)
    r_w = torch.exp(-(yaw_err   / angle_std) ** 2)

    return (r_x + r_y + r_z + r_r + r_p + r_w) / 6.0


def termination_param_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    term_name: str,
    param_stages: list[dict],
) -> torch.Tensor:
    """在调度的 step 修改某个终止项的 params。

    TerminationManager 保存自己的 cfg 字典 deepcopy，因此必须直接编辑活的 term_cfgs
    列表 —— env.cfg.terminations 是空操作。适合在训练后期禁用某终止项（例如在第 N 次
    迭代把 bad_orientation 的 limit_angle 设为 pi，让机器人可以跌倒而不结束 episode，
    从而学习恢复）。

    param_stages: {step: int, params: dict} 的列表。dict 在最新匹配的阶段浅合并到活的
    term_cfg.params。
    """
    del env_ids
    tm = env.termination_manager
    if term_name not in tm._term_names:
        # 该 term 已被移除（例如 play 模式完全禁用 fell_over）。
        return torch.tensor(0.0)
    idx = tm._term_names.index(term_name)
    term_cfg = tm._term_cfgs[idx]

    current = param_stages[0]["params"]
    for stage in param_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["params"]
    term_cfg.params.update(current)

    first_val = next(iter(current.values()))
    return torch.tensor(float(first_val) if isinstance(first_val, (int, float)) else 0.0)


def body_pose_tracking_locomotion(
    env: ManagerBasedRlEnv,
    command_name: str = "body_pose",
    nominal_height: float = 0.105,
    xy_std: float = 0.02,
    z_std: float = 0.03,
    angle_std: float = math.radians(30),
    axis_weights: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    vel_gate_command_name: str | None = None,
    vel_gate_std: float = 0.1,
    standing_gate_command_name: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
) -> torch.Tensor:
    """ locomotion 感知的 6D 身体姿态跟踪。

    与 body_pose_tracking_6d 形状相同（6D cmd，6 个 Gaussian 的均值），但 x/y/yaw 是
    【相对脚部支撑多边形】测量的，而非生成原点。这使 reward 在机器人行走（或站立）时
    仍有意义：

      x, y  : trunk 位置 − 脚部质心，旋转到 trunk 机身系。
              dx = +0.02 表示"trunk 前倾于脚质心 2 cm。"
      z     : trunk 世界高度（− nominal_height）—— locomotion 中性。
      roll  : trunk 世界 roll —— locomotion 中性。
      pitch : trunk 世界 pitch —— locomotion 中性。
      yaw   : trunk 世界 yaw − circular-mean(脚 site 的 yaw)。dyaw = +0.3 rad
              表示"trunk 相对脚朝向扭转 17°。"

    body_pose_tracking_6d 把 x/y/yaw 相对生成原点/世界 yaw 测量，机器人一平移或转向
    梯度就死掉。本版本无论机器人在世界何处都保持有意义。
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 6)
    dx, dy, dz = cmd[:, 0], cmd[:, 1], cmd[:, 2]
    droll, dpitch, dyaw = cmd[:, 3], cmd[:, 4], cmd[:, 5]

    pos_w = asset.data.root_link_pos_w
    quat = asset.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    trunk_yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    roll  = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))

    # 世界系下的脚部质心。
    foot_pos = asset.data.site_pos_w[:, feet_cfg.site_ids]   # (N, 2, 3)
    foot_quat = asset.data.site_quat_w[:, feet_cfg.site_ids] # (N, 2, 4)
    feet_centroid = foot_pos.mean(dim=1)                     # (N, 3)

    # 相对脚质心的机身系 trunk xy（把世界 Δxy 旋转 −yaw）。
    dx_w = pos_w[:, 0] - feet_centroid[:, 0]
    dy_w = pos_w[:, 1] - feet_centroid[:, 1]
    cos_y = torch.cos(trunk_yaw)
    sin_y = torch.sin(trunk_yaw)
    x_body =  cos_y * dx_w + sin_y * dy_w
    y_body = -sin_y * dx_w + cos_y * dy_w

    # 相对生成原点地形高度的 z（仍在世界系）。
    origin = env.scene.terrain.env_origins
    z_world = torch.nan_to_num(pos_w[:, 2] - origin[:, 2], nan=0.0)

    # 脚的 yaw → circular mean。注意：这依赖 site 朝向
    # 与脚指向方向一致；若 site 坐标系被旋转，该
    # yaw 参考可能有偏移（每 env 恒定，因此 dyaw=0 仍映射
    # 为"与脚对齐"）。
    fqw, fqx, fqy, fqz = foot_quat[..., 0], foot_quat[..., 1], foot_quat[..., 2], foot_quat[..., 3]
    foot_yaws = torch.atan2(2.0 * (fqw * fqz + fqx * fqy), 1.0 - 2.0 * (fqy * fqy + fqz * fqz))  # (N, 2)
    mean_foot_yaw = torch.atan2(torch.sin(foot_yaws).mean(dim=1), torch.cos(foot_yaws).mean(dim=1))

    x_err     = x_body - dx
    y_err     = y_body - dy
    z_err     = z_world - (nominal_height + dz)
    roll_err  = roll  - droll
    pitch_err = pitch - dpitch
    yaw_err   = wrap_to_pi(trunk_yaw - mean_foot_yaw - dyaw)

    r_x = torch.exp(-(x_err / xy_std) ** 2)
    r_y = torch.exp(-(y_err / xy_std) ** 2)
    r_z = torch.exp(-(z_err / z_std) ** 2)
    r_r = torch.exp(-(roll_err  / angle_std) ** 2)
    r_p = torch.exp(-(pitch_err / angle_std) ** 2)
    r_w = torch.exp(-(yaw_err   / angle_std) ** 2)

    # 逐轴加权均值。传 axis_weights=(0,0,1,1,1,1) 可禁用 xy
    # 跟踪 —— 当 xy 倾斜在该机器人上与 pitch/roll 机械耦合时有用，
    # 那会让独立的 xy 指令成为噪声源而非
    # 可学习目标。
    wx, wy, wz, wr, wp, wyaw = axis_weights
    total_w = wx + wy + wz + wr + wp + wyaw
    reward = (wx*r_x + wy*r_y + wz*r_z + wr*r_r + wp*r_p + wyaw*r_w) / max(total_w, 1e-6)

    # 可选门控：设置 vel_gate_command_name 时，用一个
    # 速度指令幅值上的 Gaussian 缩放 reward。vel_gate_std ≈ 0.1 时，
    # 指令速度为 0 门控约 1，到 |vel_cmd|≥0.3 衰减到约 exp(-9)≈0
    # —— 身体跟踪只在机器人
    # 机器人本应静止站立时。避免跟踪与行走的
    # 冲突 —— 之前那次运行因而两者都没学好。
    if vel_gate_command_name is not None:
        # 只对指令的【线】速度（xy）门控 —— 原地转向仍
        # 保持身体姿态有意义，但向前/侧方行走则不然。
        vel_cmd = env.command_manager.get_command(vel_gate_command_name)  # (N, 3)
        vel_mag = torch.linalg.vector_norm(vel_cmd[:, :2], dim=-1)
        gate = torch.exp(-(vel_mag / vel_gate_std) ** 2)
        reward = reward * gate
    # 硬站立门控（与 StandingGatedPoseCommand 配对）：只在
    # twist 指令为站立环境的环境中付费 —— 行走和原地转向环境恰好得 0，
    # 因此身体跟踪绝不会给步态收税。
    if standing_gate_command_name is not None:
        standing = getattr(env.command_manager.get_term(standing_gate_command_name), "is_standing_env", None)
        reward = reward * (standing.float() if standing is not None else 0.0)

    return reward


def pose_command_range_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    range_stages: list[dict],
) -> torch.Tensor:
    """在训练中渐增 UniformPoseCommand 的逐维范围。

    range_stages: {step: int, ranges: tuple[(lo, hi), ...]} 的列表。
    第一阶段在其 step 之前生效；最新的已过阶段胜出。总是使用活的 CommandManager
    项 cfg（【不是】env.cfg.commands）使更新生效 —— CommandManager 保存自己的项引用，
    每次重采样读取 `term.cfg.ranges`。
    """
    del env_ids

    term = env.command_manager.get_term(command_name)
    assert term is not None, f"Command term '{command_name}' not found"
    cfg = term.cfg  # type: ignore[assignment]

    current = range_stages[0]["ranges"]
    for stage in range_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["ranges"]

    cfg.ranges = tuple(current)
    # 返回最大绝对范围作为标量，便于 wandb 观察。
    max_abs = max((max(abs(lo), abs(hi)) for lo, hi in current), default=0.0)
    return torch.tensor(max_abs)


# ─────────────────────────────────────────────────────────────────────────────
# 从 mjlab_microban 移植的步态塑形惩罚（microban velocity 配方）。
# ─────────────────────────────────────────────────────────────────────────────
def no_stepping_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """当指令速度低于阈值时惩罚脚在空中。

    阻止机器人在应静止站立时原地踏步。返回每环境在空中的脚数量（配合负权重）。
    从 mjlab_microban 移植。
    """
    command = env.command_manager.get_command(command_name)  # (N, 3)
    cmd_speed = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
    below_threshold = cmd_speed < command_threshold

    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found  # (N, num_feet) or (N, num_feet, num_slots)
    if found.dim() == 3:
        found = found.any(dim=-1)  # (N, num_feet)
    in_air = ~found.bool()

    return in_air.float().sum(dim=-1) * below_threshold.float()


def feet_distance_penalty(
    env: ManagerBasedRlEnv,
    min_dist: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """惩罚两只脚在水平面上靠得太近。

    每环境返回 ``clamp(min_dist - d, min=0)``（配合负权重），其中 ``d`` 是两个脚
    site 的水平 (xy) 距离。从 mjlab_microban 移植。尚未接入 velocity —— 留待后用。
    """
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # (N, 2, 2)
    dist = torch.norm(foot_pos_xy[:, 0] - foot_pos_xy[:, 1], dim=-1)  # (N,)
    return torch.clamp(min_dist - dist, min=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 非累积 domain randomization（先恢复标称值再施加）。
#
# 自带的 mdp.randomize_field 配 operation="add"/"scale" + mode="reset"
# 读取【当前】模型值并对它施加操作，没有恢复到
# 标称值 —— 所以每次 episode 重置扰动都会【叠加】在上一次的
# 基础上，参数在训练中随机游走偏离标称。对
# body_ipos（CoM）这就是长期存在的 microduck 不稳定源：CoM
# 在数百次重置中漂移数厘米离开中心 → 机器人逐渐
# 失衡 → 跌倒更多 → 早期峰值后 reward/episode 长度崩溃。
# 这些函数仿照 randomize_mass_and_inertia：缓存一次标称值，
# 每次采样前恢复它，再施加一个新采样的扰动 ——
# 因此每个 episode 重新采样但绝不累积。
# ─────────────────────────────────────────────────────────────────────────────
def randomize_com(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    ranges: tuple[float, float],
    field: str = "body_ipos",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """每 episode 随机化 body CoM (body_ipos) 且【不累积】。

    是有 bug 的 mdp.randomize_field(add, body_ipos, reset) 的直接替代品。
    ``ranges`` 是 (lo, hi)，作用于所有 3 个 CoM 轴；com_range curriculum 更新同一
    个 ``ranges`` 参数。声明 ``field`` 使事件能以 ``domain_randomization=True`` 运行
    （mjlab 读取 params["field"] 来按 env 展开该模型字段）。
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))[body_ids]
    body_indices = asset.indexing.body_ids[body_ids]

    mf = getattr(env.sim.model, field)
    # 用 (field, body 集合) 作为缓存键：多个 randomize_com 事件可以共享
    # 同一个字段（例如 trunk + head 都随机化 body_ipos）且【不能】
    # 在单个 _original_body_ipos 属性上冲突 —— 它们的 body 数量不同。
    _bidx = body_indices.tolist() if hasattr(body_indices, "tolist") else list(body_indices)
    cache_attr = f"_original_{field}_" + "_".join(str(int(i)) for i in _bidx)
    # 首次调用时缓存标称值（此时 model[0] 仍是标称的）。
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, mf[0, body_indices].clone())
    nominal = getattr(env, cache_attr)

    num_envs = len(env_ids)
    num_bodies = len(body_indices)

    # 先恢复标称值（防止累积），再加一个新采样偏移。
    mf[env_ids[:, None], body_indices] = nominal.unsqueeze(0).expand(num_envs, -1, -1)
    lo, hi = ranges
    offsets = torch.rand(num_envs, num_bodies, 3, device=env.device) * (hi - lo) + lo
    mf[env_ids[:, None], body_indices] += offsets
    return torch.tensor(float(hi))


def randomize_dof_field_scaled(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    field: str,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """每 episode 缩放逐 dof 的模型字段（如 dof_frictionloss/dof_damping）
    【不累积】：先恢复标称值，再施加新的缩放。

    ``field`` 同时作为 domain_randomization 的字段名。注意：在 BAM 执行器下，
    dof_frictionloss 和 dof_damping 在 edit_spec 中被清零（BAM 自己建模摩擦），
    所以缩放它们是空操作 —— 这些只对 XML position 执行器有意义。保持正确是为了在
    重新启用时避开累积陷阱。
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = list(range(len(asset.indexing.joint_ids)))[joint_ids]
    dof_indices = asset.indexing.joint_v_adr[joint_ids]

    mf = getattr(env.sim.model, field)
    cache_attr = f"_original_{field}"
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, mf[0, dof_indices].clone())
    nominal = getattr(env, cache_attr)

    num_envs = len(env_ids)
    num_dofs = len(dof_indices)

    mf[env_ids[:, None], dof_indices] = nominal.unsqueeze(0).expand(num_envs, -1)
    lo, hi = scale_range
    scales = torch.rand(num_envs, num_dofs, device=env.device) * (hi - lo) + lo
    mf[env_ids[:, None], dof_indices] *= scales
    return torch.tensor(float(hi))


# =============================================================================
# BallKick 任务 —— 球重置事件、踢球 reward、仅 critic 的球观测
# =============================================================================


def _ball_kick_dir(env: ManagerBasedRlEnv) -> torch.Tensor:
    """每环境世界系的踢球方向（XY 单位向量），惰性分配。

    由 ``reset_ball_in_front_of_foot`` 在 episode 重置时设为机器人前向。整段
    episode 冻结，使策略不能在踢球后通过转向重新定义"前方"。
    """
    if not hasattr(env, "_ball_kick_dir_w"):
        env._ball_kick_dir_w = torch.zeros(env.num_envs, 2, device=env.device)
        env._ball_kick_dir_w[:, 0] = 1.0
    return env._ball_kick_dir_w


def reset_ball_in_front_of_foot(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    offset: tuple = (0.09, -0.042),
    noise_xy: float = 0.015,
    ball_radius: float = 0.035,
    asset_name: str = "ball",
):
    """把球放在（右）脚前方；存储踢球方向。

    ``offset`` 是在机器人 yaw 系下球心的标称位置：HOME 时右脚中心在 (0, -0.042)，
    脚尖在 x≈0.034，所以 (0.08, -0.042) 把一个半径 35mm 的球放在脚尖前约 1cm。
    ``noise_xy``（逐轴均匀 ±）是放置 DR：策略【看不见】球，因此正是它迫使摆腿动作
    能适应真实世界的放置误差。

    直接从 qpos 读取机器人 root（root_link_pos_w 要等下一次 forward() 才更新）；
    必须登记在 reset_base / set_ground_state【之后】（事件按字典插入顺序运行），
    使机器人姿态是最终的。
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device)
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]

    root = env.sim.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]
    qw, qx, qy, qz = root[:, 3], root[:, 4], root[:, 5], root[:, 6]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    n = len(env_ids)
    off = torch.tensor(offset, device=env.device, dtype=torch.float).repeat(n, 1)
    off += (torch.rand(n, 2, device=env.device) * 2.0 - 1.0) * noise_xy

    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0] = root[:, 0] + cos_y * off[:, 0] - sin_y * off[:, 1]
    pose[:, 1] = root[:, 1] + sin_y * off[:, 0] + cos_y * off[:, 1]
    pose[:, 2] = env.scene.terrain.env_origins[env_ids, 2] + ball_radius
    pose[:, 3] = 1.0  # identity quat
    ball.write_root_link_pose_to_sim(pose, env_ids)
    ball.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=env.device), env_ids
    )

    kick_dir = _ball_kick_dir(env)
    kick_dir[env_ids, 0] = cos_y
    kick_dir[env_ids, 1] = sin_y


def ball_forward_velocity(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
    max_speed: float = 5.0,
) -> torch.Tensor:
    """球沿每环境踢球方向的 XY 速度，钳制在 [0, max]。

    直到 ``max_speed`` 都密集且速度线性：球每多滚一步，每一点前向球速都多付奖励，
    因此探索性的轻推就能自举出踢球，无需峰值检测机制。球向后/侧向运动得 0 而非惩罚
    —— 打偏不应把策略吓到完全不敢碰球。

    当 ``max_speed`` 设为一个【目标】速度（而非一个大上限）时，配合
    ``ball_speed_overshoot_penalty``：光让 reward 在目标处饱和【不能】消除"越用力越好"
    —— 更用力的踢会让球在上限处多待几步，滚动时间积分仍随击球速度增长。正是过冲
    惩罚使目标成为真正的最优。
    """
    ball: Entity = env.scene[asset_name]
    vel_xy = ball.data.root_link_lin_vel_w[:, :2]
    fwd = (vel_xy * _ball_kick_dir(env)).sum(dim=1)
    return torch.nan_to_num(fwd, nan=0.0).clamp(0.0, max_speed)


def ball_speed_overshoot_penalty(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
    target_speed: float = 1.0,
    max_penalty: float = 5.0,
) -> torch.Tensor:
    """球超过 ``target_speed`` 的前向速度（线性，≥ 0）。

    是 ``ball_forward_velocity`` 针对目标速度踢球的配套项：低于目标时为 0（封顶的
    线性 reward 提供向上梯度）；高于目标时，每超 1 m/s 在其持续的每步线性收费。
    保持本项 |权重| 【低于】封顶 reward 的权重，使合成地形在目标处达峰、过冲侧斜率
    更缓 —— 稍微用力过猛必须仍比完全不踢便宜。
    """
    ball: Entity = env.scene[asset_name]
    vel_xy = ball.data.root_link_lin_vel_w[:, :2]
    fwd = (vel_xy * _ball_kick_dir(env)).sum(dim=1)
    over = torch.nan_to_num(fwd, nan=0.0) - target_speed
    return over.clamp(0.0, max_penalty)


def single_foot_grounded_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """二元 reward：感知到的脚接触地面时为 1。

    是 ``feet_grounded_reward`` 的单脚版本 —— 用于在踢球时钉住【支撑】脚（防跳）：
    摆右腿免费，抬左脚每步都要花掉这个 reward。
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    found = env.scene.sensors[sensor_name].data.found
    if found.dim() > 1:
        found = found.sum(dim=-1)
    return torch.clamp(found, 0.0, 1.0)


def ball_pos_in_base(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
) -> torch.Tensor:
    """球相对机器人 root 的位置，在机器人基座坐标系下。

    仅 critic 观测（非对称 actor-critic）：部署的策略没有球体感知，因此 actor 必须对
    球保持盲目 —— critic 仍可用它预测踢球收益。
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]
    rel = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    rot = matrix_from_quat(robot.data.root_link_quat_w)
    return torch.bmm(rot.transpose(1, 2), rel.unsqueeze(-1)).squeeze(-1)


def ball_vel_in_base(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
) -> torch.Tensor:
    """球在机器人基座坐标系下的线速度。仅 critic（见上）。"""
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]
    rot = matrix_from_quat(robot.data.root_link_quat_w)
    vel = ball.data.root_link_lin_vel_w
    return torch.bmm(rot.transpose(1, 2), vel.unsqueeze(-1)).squeeze(-1)


# --------------------------------------------------------------------------- #
# SPIN 任务 —— 在滚轮上原地快速旋转                                            #
# --------------------------------------------------------------------------- #
# 相位包络：按钮槽位的指令携带一个相位，它驱动一个梯形【目标偏航速率】（而不是像下蹲那样的姿态）。
#   [0, accel_end)        0.5 s   0 -> rate_max    （启动）
#   [accel_end, hold_end) 1.6 s   rate_max         （稳态）
#   [hold_end, brake_end) 0.5 s   rate_max -> 0    （刹车）
#   [brake_end, 1.0)      1.4 s   0                （站立休息）
# 
# 一个周期内包络下的面积 = 2.1 * SPIN_RATE_MAX rad。在 3.0 rad/s 时：
# 2.1 * 3.0 = 6.3 rad ~ 1 tour (et non ~2, comme avec l'ancienne cible 6.0).
SPIN_PERIOD = 4.0
SPIN_RATE_MAX = 3.0
SPIN_ACCEL_END = 0.125
SPIN_HOLD_END = 0.525
SPIN_BRAKE_END = 0.650


def spin_rate_by_phase(
    phase: torch.Tensor,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """沿相位的目标偏航速率（rad/s，正 = 逆时针）。"""
    w = torch.zeros_like(phase)
    accel = phase < accel_end
    w = torch.where(accel, rate_max * phase / accel_end, w)
    hold = (phase >= accel_end) & (phase < hold_end)
    w = torch.where(hold, torch.full_like(phase, rate_max), w)
    brake = (phase >= hold_end) & (phase < brake_end)
    w = torch.where(
        brake, rate_max * (1.0 - (phase - hold_end) / (brake_end - hold_end)), w
    )
    return w


def spin_gate_by_phase(
    phase: torch.Tensor,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """[0,1] 中的塑形门控 = 归一化包络。

    在整个休息段为 0：启动提示（双腿剪动、轮子差速）只在启动+稳态期间施加，
    因此机器人在把控制权交还给 roller 策略前会回到中立站姿。
    """
    return spin_rate_by_phase(phase, rate_max, accel_end, hold_end, brake_end) / rate_max


def spin_phase_from_command(cmd: torch.Tensor) -> torch.Tensor:
    """从槽位指令 [cos(2πφ), sin(2πφ), 0] 中恢复 [0,1) 相位。"""
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0


def _spin_target_rate(
    env: ManagerBasedRlEnv,
    command_name: str,
    rate_max: float,
    accel_end: float,
    hold_end: float,
    brake_end: float,
) -> torch.Tensor:
    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    return spin_rate_by_phase(phase, rate_max, accel_end, hold_end, brake_end)


def _spin_gate(
    env: ManagerBasedRlEnv,
    command_name: str,
    rate_max: float,
    accel_end: float,
    hold_end: float,
    brake_end: float,
) -> torch.Tensor:
    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    return spin_gate_by_phase(phase, rate_max, accel_end, hold_end, brake_end)


def spin_rate_reward_from_values(
    omega_z: torch.Tensor, omega_target: torch.Tensor, std: float
) -> torch.Tensor:
    """偏航速率误差上的 Gaussian（纯函数，可测试）。"""
    return torch.exp(-(((omega_z - omega_target) / std) ** 2))


def spin_rate_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    std: float = 1.5,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """spin 的主目标：跟踪目标偏航速率 ω*(φ)。

    ω_z 取机身系（这是 IMU 陀螺仪看到的，也就是策略观测到的）。反向旋转比不动
    受更重的惩罚，因为 Gaussian 以正目标为中心。
    """
    asset: Entity = env.scene[asset_cfg.name]
    omega_z = asset.data.root_link_ang_vel_b[:, 2]
    target = _spin_target_rate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return spin_rate_reward_from_values(omega_z, target, std)


def spin_rate_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 引导：即使 `spin_rate_track` 的 Gaussian 在远离目标处饱和，仍有朝目标的恒定梯度。
    配合【正】权重使用（返回值已经是负的）。"""
    asset: Entity = env.scene[asset_cfg.name]
    omega_z = asset.data.root_link_ang_vel_b[:, 2]
    target = _spin_target_rate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return -torch.abs(omega_z - target)


SPIN_LAUNCH_DRIFT_SCALE = 0.2  # atténuation du coût de dérive pendant le lancement


def spin_stay_in_place(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    launch_scale: float = SPIN_LAUNCH_DRIFT_SCALE,
    accel_end: float = SPIN_ACCEL_END,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """trunk 的 ‖v_xy‖² 成本：【原地】旋转，并消掉进入时的冲量。

    没有参考状态（不像从重置开始测量的漂移），因此在 episode 的 5 个周期内都有效。
    配合【负】权重使用。

    【启动期间衰减】：在 `[0, accel_end)` 上机器人必须向地面蹬以获得角动量，
    进入状态给它最多 0.3 m/s 的速度，它应当【转化】为旋转。此时按全价收平移税
    直接与目标相悖。因此只在这一段把成本乘以 `launch_scale`，其余段（稳态、刹车、
    休息）为全价 —— 那里"原地"才是真正的判据。

    与其他 spin 提示不同，本项【不】被 `spin_gate_by_phase` 关掉：在休息段我们恰恰
    希望它保持满值，因为那时机器人应当静止。
    """
    asset: Entity = env.scene[asset_cfg.name]
    v_xy = asset.data.root_link_lin_vel_b[:, :2]
    cost = torch.sum(torch.square(v_xy), dim=1)

    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    scale = torch.where(
        phase < accel_end,
        torch.full_like(cost, launch_scale),
        torch.ones_like(cost),
    )
    return cost * scale


# 在 rollers 模型上实测的半轮距（HOME 姿态，site left_foot/right_foot）：
# 0.0499 m, contre 0.03 m estimé au spec. Conséquence mécanique de SPIN_RATE_MAX
# (A1) : 期望差速 = 2*SPIN_RATE_MAX*半轮距/r，r = 0.0175 m。
# 在旧目标 6.0 rad/s 时：2*6.0*0.0499/0.0175 = 34.2 rad/s（作为上限保留）。
# 34.0, soit +71% par rapport aux 20.0 estimés au spec -> seuil de 30% dépassé).
# 在新目标 3.0 rad/s 时：2*3.0*0.0499/0.0175 = 17.1 rad/s。若把 34.0 留在
# 这里会把该项钳制在 tanh(17.1/34) = 0.47 的自身最大值，从而
# 那会削弱我们正想强化的塑形（见 spin_stay_in_place）。
SPIN_WHEEL_OMEGA_SCALE = 17.0  # rad/s ; recalibré sur la demi-voie mesurée et SPIN_RATE_MAX = 3.0


def spin_wheel_differential_from_values(
    diff: torch.Tensor, gate: torch.Tensor, omega_scale: float
) -> torch.Tensor:
    """纯函数：轮子差速的 tanh，由门控承载，钳制 ≥ 0。"""
    return gate * torch.tanh(torch.clamp(diff, min=0.0) / omega_scale)


def spin_wheel_differential(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    omega_scale: float = SPIN_WHEEL_OMEGA_SCALE,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """奖励【滚动】旋转（而非打滑旋转）。

    对逆时针 spin，左履带后退、右履带前进；由于 4 个轮子在前进时正向旋转，
    这给出 ω_D − ω_G > 0。tanh 在 `omega_scale` 处饱和以避免轮子速度军备竞赛。
    """
    asset: Entity = env.scene["robot"]
    lf_ids, _ = asset.find_joints("passive_LF_?wheel")
    lr_ids, _ = asset.find_joints("passive_LR_?wheel")
    rf_ids, _ = asset.find_joints("passive_RF_?wheel")
    rr_ids, _ = asset.find_joints("passive_RR_?wheel")

    vel = asset.data.joint_vel
    omega_left = (vel[:, lf_ids[0]] + vel[:, lr_ids[0]]) / 2.0
    omega_right = (vel[:, rf_ids[0]] + vel[:, rr_ids[0]]) / 2.0
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return spin_wheel_differential_from_values(
        omega_right - omega_left, gate, omega_scale
    )


def spin_grounded(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """spin 期间两只滑刀都着地 —— 防止"我跳起来再拧"。

    是 swizzle 的 `grounded_reward` 的变体，不能在这里直接复用：它按 cmd_x 加权，
    而在相位指令上 cmd_x 是 cos(2πφ)。
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None
    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)
    grounded = (n_contact >= 2).float()
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return grounded * gate


def leg_antisymmetry(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_bases: tuple = ("hip_pitch", "knee"),
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """在 spin 期间引导双腿【剪动】（一条向前 / 一条向后）。

    机器人使用左右【镜像】符号约定：对称姿态满足 q_G + q_D ≈ 0（见
    `leg_symmetry_reward`），因此剪动满足 q_G ≈ q_D。我们返回
    `gate(φ) · (−mean|q_G − q_D|)` —— 配合【正】权重使用，并由 curriculum 递减：
    引导逐渐消退，让策略打磨自己的动作。
    """
    asset: Entity = env.scene[asset_cfg.name]
    left, right = [], []
    for base in joint_bases:
        li, _ = asset.find_joints([f"left_{base}"])
        ri, _ = asset.find_joints([f"right_{base}"])
        left.append(li[0])
        right.append(ri[0])
    lids = torch.tensor(left, device=env.device)
    rids = torch.tensor(right, device=env.device)

    q = asset.data.joint_pos
    scissor = -torch.abs(q[:, lids] - q[:, rids]).mean(dim=-1)
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return gate * scissor


# =============================================================================
# Backlash 模型 —— 透过齿隙的编码器关节观测
# =============================================================================
# backlash 模型（robot_groundcontact_backlash.xml）在每个伺服关节上串联一个
# 非驱动的 ``passive_<joint>_backlash`` 铰链。连杆
# 角度是 qpos[servo] + qpos[backlash]，而真实编码器位于游隙的
# 【输出】侧 —— 它读到的是两者之和。这些 obs 在 backlash 任务中取代 joint_pos_rel /
# joint_vel_rel（见 tasks/backlash.py），使策略看到
# 与运行时将喂给它的完全一致的值。期望 asset_cfg 的正则
# 只选择伺服关节（通常的 ``^(?!passive_).*``）。


def _backlash_encoder_ids(
    env: "ManagerBasedRlEnv",
    asset: Entity,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(main_ids, backlash_ids, mask) —— 按 (实体, 关节选择) 缓存。

    在存在匹配的 passive_<name>_backlash 关节处 mask 为 1.0，因此同样的 obs 函数
    在没有 backlash 关节的模型上无需改动即可运行。
    """
    key = (asset_cfg.name, str(asset_cfg.joint_ids))
    cache = env.__dict__.setdefault("_backlash_encoder_cache", {})
    hit = cache.get(key)
    if hit is not None:
        return hit

    names = asset.joint_names
    jnt_ids = asset_cfg.joint_ids
    if isinstance(jnt_ids, slice):
        main_ids = list(range(len(names)))[jnt_ids]
    else:
        main_ids = [int(i) for i in jnt_ids]
    name_to_id = {n: i for i, n in enumerate(names)}
    bl_ids, mask = [], []
    for i in main_ids:
        bl = name_to_id.get(f"passive_{names[i]}_backlash")
        bl_ids.append(0 if bl is None else bl)
        mask.append(0.0 if bl is None else 1.0)

    device = asset.data.joint_pos.device
    out = (
        torch.tensor(main_ids, dtype=torch.long, device=device),
        torch.tensor(bl_ids, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
    )
    cache[key] = out
    return out


def joint_pos_rel_backlash(
    env: "ManagerBasedRlEnv",
    biased: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """编码器透过齿隙铰链读取的 joint_pos_rel。

    返回 (qpos[servo] + qpos[backlash]) - default[servo]。biased=True 时，逐环境的
    编码器校准偏置施加到伺服读数上（每个伺服一个编码器 → 每关节一个偏置；齿隙求和项
    保持原始值）。
    """
    asset: Entity = env.scene[asset_cfg.name]
    main_ids, bl_ids, mask = _backlash_encoder_ids(env, asset, asset_cfg)
    joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
    pos = joint_pos[:, main_ids] + asset.data.joint_pos[:, bl_ids] * mask
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    return pos - default_joint_pos[:, main_ids]


def joint_vel_rel_backlash(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """编码器透过齿隙铰链读取的 joint_vel_rel。

    固件从编码器位置推导 present_velocity，因此它也看到齿隙运动：
    qvel[servo] + qvel[backlash]。
    """
    asset: Entity = env.scene[asset_cfg.name]
    main_ids, bl_ids, mask = _backlash_encoder_ids(env, asset, asset_cfg)
    vel = asset.data.joint_vel[:, main_ids] + asset.data.joint_vel[:, bl_ids] * mask
    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None
    return vel - default_joint_vel[:, main_ids]


# ─────────────────────────────────────────────────────────────────────────────
# 坐↔站姿态指令 + 姿态条件 reward（sitstand 环境）。
#
# 一个策略，两个方向：指令是承载在 twist 槽位里的单个坐/站标志
#（cmd = [sit_flag, 0, 0]，因此"站"是全零
# 指令 —— 与其他所有策略的部署空闲态相同）。下面所有任务 reward
# 都从活动指令中、逐环境选择它们的目标（SIT 关键帧 + SIT_Z 或 HOME + STAND_Z），
# 因此同一套 reward 栈驱动下降、坐姿休息、起身和站立休息。使用 _servo_* 辅助函数
# → 兼容 backlash 模型。
# 
# ─────────────────────────────────────────────────────────────────────────────


class SitStandCommand(UniformVelocityCommand):
    """姿态指令：cmd = [sit_flag, 0, 0]，带停留时间重采样和一个
    【渐变（slewed）】的内部目标混合。

    sit_flag ∈ {0.0, 1.0}。由 command manager 按 cfg 的 resampling_time_range
    （在每个姿态停留的时间）以及 episode 重置时重采样。cfg.sit_prob 是一次重采样
    指令 SIT 的概率；结合重置状态混合，这训练全部四种（起始状态 × 指令）组合，
    包括"保持你正在做的事"。

    ``alpha``（0 = STAND 目标，1 = SIT 目标）以恒定速率朝标志变化（在 cfg.ramp_s
    秒内完成完整转换），它是 posture_* reward 跟踪的对象。【防崩溃机制】：用二值目标时，
    早到会为节省的每一步支付完整的目标态头奖，而线性限速惩罚只积分成有界的多余距离
    成本 —— 瞬间跌落以约 7× 胜过 1 秒下降。用渐变目标时，【领先】于斜坡在
    height/composite 栈上得分 ~0（z 远离指令高度），因此跟踪缓慢设定点【就是】argmax；
    限速仍作为过冲/反弹的后盾。OBS 保持原始二值标志（部署：运行时写 0/1；对翻转训练出
    的响应是约 ramp_s 的滑行）。

    episode 重置时，alpha 从机器人【实际】trunk 高度初始化，而非从标志 —— 坐姿生成
    不应被一个从站立初始化的斜坡向上拖（反之亦然）。
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._sit_prob = float(getattr(cfg, "sit_prob", 0.5))
        self._ramp_s = float(getattr(cfg, "ramp_s", 2.0))
        self._sit_z = float(getattr(cfg, "sit_z", 0.060))
        self._stand_z = float(getattr(cfg, "stand_z", 0.115))
        self._env_ref = env
        self._alpha = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    @property
    def alpha(self) -> torch.Tensor:
        """渐变目标混合：0 = STAND 目标，1 = SIT 目标。"""
        return self._alpha

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        sit = (torch.rand(n, device=self.device) < self._sit_prob).float()
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids, 0] = sit

    def _alpha_from_height(self) -> torch.Tensor:
        z = torch.nan_to_num(
            self.robot.data.root_link_pos_w[:, 2]
            - self._env_ref.scene.terrain.env_origins[:, 2],
            nan=self._stand_z,
        )
        return torch.clamp(
            (self._stand_z - z) / max(self._stand_z - self._sit_z, 1e-6), 0.0, 1.0
        )

    def compute(self, dt: float) -> None:
        super().compute(dt)
        # episode 开始时从【实际】trunk 高度重新初始化混合量。
        # 在这里做（而非 reset()）是因为 command manager 的重置【早于】
        # set_ground_state 事件传送机器人，重置时读到的是
        # 传送前的高度。在 episode 的第一次 compute 时，生成
        # 状态已就位。
        fresh = self._env_ref.episode_length_buf <= 1
        if fresh.any():
            self._alpha = torch.where(fresh, self._alpha_from_height(), self._alpha)
        # 目标混合量朝指令标志的恒定速率渐变。
        step = dt / max(self._ramp_s, 1e-6)
        delta = self.vel_command_b[:, 0] - self._alpha
        self._alpha += torch.clamp(delta, -step, step)

    def _update_command(self) -> None:
        pass  # No heading controller / standing-env machinery.

    def _update_metrics(self) -> None:
        pass  # No velocity-tracking metrics for a posture flag.


@_dataclass(kw_only=True)
class SitStandCommandCfg(UniformVelocityCommandCfg):
    class_type: type = SitStandCommand
    # 一次重采样指令 SIT（相对 STAND）的概率。
    sit_prob: float = 0.5
    # 内部目标混合量完整走完 STAND↔SIT 的秒数。
    ramp_s: float = 2.0
    # 休息高度，用于从生成状态初始化混合量。
    sit_z: float = 0.060
    stand_z: float = 0.115

    def build(self, env: ManagerBasedRlEnv) -> "SitStandCommand":
        return SitStandCommand(self, env)


def _posture_blend(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """姿态 reward 的目标混合量 ∈ [0, 1]（0 = STAND，1 = SIT）。

    当项暴露 SitStandCommand 的渐变 ``alpha``（移动设定点）时使用它；否则回退到原始
    二值标志。
    """
    term = env.command_manager.get_term(command_name)
    alpha = getattr(term, "alpha", None)
    if alpha is not None:
        return alpha
    return env.command_manager.get_command(command_name)[:, 0]


def _posture_targets(
    env: ManagerBasedRlEnv,
    asset: Entity,
    command_name: str,
    sit_overrides: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """指令姿态的 (目标混合量, 逐环境关节目标)。

    STAND 目标 = default_joint_pos (HOME)；SIT 目标 = 应用关键帧 overrides 的 HOME；
    【渐变】混合量在两者之间插值，因此斜坡中途被奖励的姿态与下降的高度同步折叠。
    """
    blend = _posture_blend(env, command_name)
    stand_target = _servo_default_joint_pos(env, asset)
    sit_target = stand_target.clone()
    for idx, val in sit_overrides.items():
        sit_target[:, idx] = val
    target = stand_target + blend.unsqueeze(-1) * (sit_target - stand_target)
    return blend, target


def _posture_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """每环境的 (渐变目标 trunk z, 实际 trunk z)。"""
    blend = _posture_blend(env, command_name)
    target_z = stand_z + blend * (sit_z - stand_z)
    asset = env.scene["robot"]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return target_z, z


def posture_pose_match(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_overrides: dict,
    joint_indices: list,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """对指令姿态目标姿态的 Gaussian 匹配。"""
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    return torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)


def posture_pose_l1(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_overrides: dict,
    joint_indices: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """``posture_pose_match`` 的 L1 配套项（朝目标的恒定梯度）。"""
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def posture_height_gaussian(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """对指令姿态目标高度的 trunk z Gaussian。"""
    del asset_cfg  # trunk z read via _posture_height
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    return torch.exp(-((z - target_z) / std) ** 2)


def posture_height_l1(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """``posture_height_gaussian`` 的 L1 配套项 —— 转换驱动器。

    当机器人停在【错误】姿态时，它每步收取恒定成本（约 |Δz| = 55 mm），这使
    "忽略指令"在两个方向上都成为净负策略。
    """
    del asset_cfg
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    return -torch.abs(z - target_z)


def posture_composite(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_overrides: dict,
    joint_indices: list,
    sit_z: float,
    stand_z: float,
    height_std: float = 0.03,
    upright_std: float = 0.40,
    pose_std: float = 0.40,
    head_std: float | None = None,
    head_command_name: str = "head_pose",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """对指令姿态的乘性目标分数（高度·直立·姿态[·头部]）。

    是 ``standing_composite_score`` 的姿态条件版本：任一因子的不足都会拉垮整项，
    因此部分和的折中（平板、瘫倒、倾斜）永远得不到奖励。两个休息态都要求 trunk 直立，
    因此直立因子与姿态无关。

    ``head_std``（可选）：增加第四个因子，针对 ``head_pose`` 指令评估颈/头关节
    （误差约定同 head_pose_tracking）。没有它，目标态对头部是盲的：训练出的策略休息时
    头垂到地上 —— trunk 直立、腿在姿态、z 在目标全都保持，而头挂着，只花很轻的跟踪项。
    有了该因子，"到达"【要求】头处于指令姿态，因此头部辅助在转换途中仍是自由的
    （那里 composite 本来 ≈0），但必须收回才能拿到目标奖励。
    """
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)

    height_score = torch.exp(-((z - target_z) / height_std) ** 2)

    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    upright_score = torch.exp(-tilt_sq / (upright_std * upright_std))

    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    pose_err_sq = ((joint_pos - target[:, joint_indices]) ** 2).mean(dim=-1)
    pose_score = torch.exp(-pose_err_sq / (pose_std * pose_std))

    score = height_score * upright_score * pose_score

    if head_std is not None:
        if not hasattr(env, "_head_pose_neck_ids"):
            ids, _ = asset.find_joints_by_actuator_names(_NECK_JOINT_PATTERNS)
            env._head_pose_neck_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        neck_ids = env._head_pose_neck_ids
        head_cmd = env.command_manager.get_command(head_command_name)
        actual = asset.data.joint_pos[:, neck_ids] - asset.data.default_joint_pos[:, neck_ids]
        head_err_sq = ((actual - head_cmd) ** 2).mean(dim=-1)
        score = score * torch.exp(-head_err_sq / (head_std * head_std))

    return score


def posture_stillness(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
    band_full: float = 0.012,
    band_zero: float = 0.03,
    vel_std: float = 0.05,
    tilt_full_deg: float = 25.0,
    tilt_zero_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """奖励在【处于】指令姿态且直立时 trunk 的静止。

    把 ``seated_stillness`` 推广到两个休息态：exp(-(|v|/std)²)，由
    |z − 指令 z| 上的 smoothstep 门控（``band_full`` 内满值，``band_zero`` 外为零
    → 转换期间不生效）以及 trunk 倾斜门控（倾斜休息 —— 背/脸/侧 —— 得不到奖励）。
    还要求目标斜坡【完成】（|flag − alpha| 小），使静止在转换中途永不付费。使
    "在指令高度直立安静休息"成为 reward 栈的峰值。
    """
    asset = env.scene[asset_cfg.name]
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    v = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)

    flag = env.command_manager.get_command(command_name)[:, 0]
    blend = _posture_blend(env, command_name)
    ramp_done = ((flag - blend).abs() < 0.02).float()

    err = torch.abs(z - target_z)
    t = torch.clamp((band_zero - err) / max(band_zero - band_full, 1e-6), 0.0, 1.0)
    z_gate = t * t * (3.0 - 2.0 * t)

    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    cos_full = math.cos(math.radians(tilt_full_deg))
    cos_zero = math.cos(math.radians(tilt_zero_deg))
    u = torch.clamp((cos_tilt - cos_zero) / max(cos_full - cos_zero, 1e-6), 0.0, 1.0)
    tilt_gate = u * u * (3.0 - 2.0 * u)

    return torch.exp(-((v / vel_std) ** 2)) * z_gate * tilt_gate * ramp_done


def posture_rise_bootstrap(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_height: float,
    max_vz: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """向上 vz 奖励，只在指令 STAND 且 z < max_height 时生效。

    standup 环境的教训：只看目的地的 reward 在零运动时梯度为零，所以"保持坐着并吃掉
    L1"是一个局部最优 —— 为起身【动作】本身付费使任何尝试立刻为正。在
    ``max_height`` 以上关闭（设为略【高于】站立目标，使最后几厘米仍有收益；恰好在
    STAND_Z 门控会让策略停在差一点处）。指令 SIT 时为零，因此绝不会干扰下降。
    ``max_vz`` 限制被奖励的速度（任何 ≥ 上限的起身拿同样值，因此爆发式起跳无法比温和
    起身赚更多）。
    """
    asset = env.scene[asset_cfg.name]
    sit = env.command_manager.get_command(command_name)[:, 0]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return torch.clamp(vz, min=0.0, max=max_vz) * (z < max_height).float() * (1.0 - sit)


def trunk_upward_velocity_penalty(
    env: ManagerBasedRlEnv,
    max_up_vel: float = 0.08,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """对超过 ``max_up_vel`` 的向上 trunk 速度的惩罚。

    ``trunk_downward_velocity_penalty`` 针对起身的镜像：为过快（暴力）站起的每一步收费，
    使爆发式起身无法靠到达站立的 reward 摊销。静止、任何慢于上限的起身、以及所有向下
    运动时为 0。在起身被发现【之后】通过 curriculum 引入（attempt-tax 教训）。
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return -torch.clamp(vz - max_up_vel, min=0.0)


# ── 坐姿跌倒加固（sitstand，2026-10）────────────────────────────────────────
# 真机报告：坐着时突然的头部动作会把机器人向【后】掀翻；
# 它保持倾斜（约 30°）卡在坐姿，颈部伺服过载。Sim
# 电池测试（claude_experiments/sitstand_head_bench.py）复现了它：快速头部
# 摆动 / 俯仰+偏航跳变会掀翻 30-45% 的坐姿机器人，无一恢复，
# 颈部 |τ| 达平静值的 5-8 倍。训练中从未有 (a) 快速头部指令 ——
# 头部指令每几秒才重采样一次 —— 也没有 (b) 任何后倾的坐姿
# 状态可供恢复。下面三项补上两者，并直接给过载定价。
# 


def _seated_mask(env: ManagerBasedRlEnv, command_name: str, min_alpha: float) -> torch.Tensor:
    """姿态目标（接近）完全 SIT 的环境：标志打开且渐变完成。"""
    sit = env.command_manager.get_command(command_name)[:, 0] > 0.5
    return sit & (_posture_blend(env, command_name) >= min_alpha)


def seated_head_command_jerk(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    prob: float,
    flip_frac: float = 0.5,
    posture_command_name: str = "twist",
    head_command_name: str = "head_pose",
    min_alpha: float = 0.98,
) -> None:
    """区间事件：为【坐姿】环境施加突变的头部指令变化。

    每次触发时，每个被选中的环境（坐姿，概率为 ``prob``）要么把当前头部指令符号
    翻转（其中 ``flip_frac`` —— 快速摇杆抖动，一步内全幅摆动），要么在其活动范围内
    重新均匀采样一个新值（突变跳跃）。短间隔下这产生真机从手柄得到的亚秒级头部运动，
    而停留时间重采样的指令从未训练到它。``prob`` 由 event_param_curriculum 渐增
    （坐姿存在之前为 0）。修改 UniformPoseCommand 的保持值；它自己的重采样计时器不受影响。
    """
    if prob <= 0.0 or env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    seated = _seated_mask(env, posture_command_name, min_alpha)[env_ids]
    pick = seated & (torch.rand(len(env_ids), device=env.device) < prob)
    ids = env_ids[pick]
    if len(ids) == 0:
        return
    term = env.command_manager.get_term(head_command_name)
    flip = torch.rand(len(ids), device=env.device) < flip_frac
    fresh = torch.empty(len(ids), term.dim, device=env.device)
    for i, (lo, hi) in enumerate(term.cfg.ranges):
        fresh[:, i].uniform_(lo, hi)
    cur = term._command[ids]
    term._command[ids] = torch.where(flip.unsqueeze(1), -cur, fresh)


def seated_backward_tip(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    ang_vel_range: tuple[float, float],
    prob: float = 1.0,
    lateral_frac: float = 0.3,
    posture_command_name: str = "twist",
    min_alpha: float = 0.98,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """区间事件：为【坐姿】环境施加向后的俯仰冲击。

    绕 trunk 横轴沿抬头（向后）方向施加 |ω| ∈ ang_vel_range —— trunk_base +x 是前方，
    +ω_y 是低头俯仰，所以冲击是沿机身 y 的 −ω（世界系）。``lateral_frac`` 混入一个随机
    ± roll 分量（向后-对角跌倒）。生成物理上一致的后倾坐姿 —— 这是任何生成都覆盖不到的
    恢复前沿（传送到坐姿高度的后倾姿态会穿透地板）。
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    seated = _seated_mask(env, posture_command_name, min_alpha)[env_ids]
    pick = seated & (torch.rand(len(env_ids), device=env.device) < prob)
    ids = env_ids[pick]
    if len(ids) == 0 or ang_vel_range[1] <= 0.0:
        return
    asset: Entity = env.scene[asset_cfg.name]
    n = len(ids)
    mag = torch.empty(n, device=env.device).uniform_(*ang_vel_range)
    roll = (torch.rand(n, device=env.device) * 2 - 1) * lateral_frac
    omega_b = torch.stack([roll * mag, -mag, torch.zeros_like(mag)], dim=1)
    omega_w = quat_apply(asset.data.root_link_quat_w[ids], omega_b)
    vel = asset.data.root_link_vel_w[ids].clone()
    vel[:, 3:6] += omega_w
    asset.write_root_link_velocity_to_sim(vel, env_ids=ids)


def neck_torque_when_tilted(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    tilt_start_deg: float = 12.0,
    tilt_full_deg: float = 25.0,
    torque_scale: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """成本（≥ 0 → 负权重）：mean (|τ_neck| / torque_scale)² × gate。

    过载症状：在坐姿中后倾时，颈部把沉重头部以近堵转力矩悬臂对抗重力。Gate =
    姿态混合量（SIT 目标）×【后】倾（projected gravity x < 0；trunk +x 是前方）×
    倾斜斜坡（低于 ``tilt_start_deg`` 为 0，高于 ``tilt_full_deg`` 为满值）。直立头部控制
    （坐姿约 3-5°）、站立、起身（混合量 → 0）以及头部辅助的前倾下降都不付费；一旦后倾，
    颈部用力就要付成本，于是恢复到直立或放松头部都获胜。``torque_scale`` ≈ 平静与过载的
    边界（坐姿平静 |τ| ≈ 0.02 Nm，后倾过载 ≈ 0.15 Nm）。
    """
    asset: Entity = env.scene[asset_cfg.name]
    if not hasattr(env, "_neck_actuator_ids"):
        ids, _ = asset.find_actuators(_NECK_JOINT_PATTERNS, preserve_order=True)
        env._neck_actuator_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    tau = torch.nan_to_num(asset.data.actuator_force[:, env._neck_actuator_ids], nan=0.0)
    cost = torch.square(tau / torque_scale).mean(dim=1)
    g = torch.nan_to_num(asset.data.projected_gravity_b, nan=0.0)
    tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)))
    gate = torch.clamp((tilt - tilt_start_deg) / max(tilt_full_deg - tilt_start_deg, 1e-6), 0.0, 1.0)
    gate = gate * (g[:, 0] < 0.0).float() * _posture_blend(env, command_name)
    return cost * gate
# ==============================================================================
# Roulade（前滚翻）任务 —— 情景式动态技巧
# ==============================================================================
#
# 第三次尝试 roulade。前两次的教训：
#   • origin/roulade（相位时钟 + 时间窗口 reward 阶段）：卡在
#     俯卧约 90° —— 时间窗口就是时间上的关键帧，是可扎营的局部
#     最优（正是 sit/standup 的教训）。还把 -ω_y 积分成前向
#     进展，但按本代码库自己的约定（俯卧 = +90° pitch
#     = 绕 +y 旋转，见 set_random_ground_state）这是【错误符号】——
#     进展 reward 为向后旋转付费。
#   • origin/roulade 后续提交（关键帧模仿）：同一类 waypoint 扎营
#     家族，按 feedback-episodic-pose-landing 的建议弃用。
#
# 本设计改用已验证的情景式配方：
#   • 单一密集进展信号：按【增量】支付迄今最大累计
#     前向旋转（基于势能 —— 扎营策略每步赚 0，一次完整
#     翻滚无论路径或速度恰好赚到 2π）。
#   • 落地 reward（乘性复合、直立、高度、起身速度）由
#     【翻滚完成】门控（最大旋转 ≥ 阈值）—— 基于状态的门控，
#     而非时钟。"什么都不做"不赚钱；在生成点站立不赚钱；
#     只有翻滚才打开"站立吸引子年金"。
#   • 通过中途翻滚生成做反向课程（修复后向恢复的仰卧部分翻转技巧）：
#     一部分 episode 从已翻 50°–185° 开始，抱团，可选带前向角动量，
#     且旋转累加器初始化为生成角度，使进展
#     记账保持一致。
# 
#
# 第 1 次运行的教训（2026-08）：用不受支撑约束的旋转计数和无上限的
# 付费速率时，最优策略是暴力的弹道式猛甩（"霹雳舞"）——
# 同样是 2π，但更快完成、拿到更多折现年金。无法迁移。修复：
#   • 支撑门控：累加器只在机器人某个 geom 接触地形时才积分
#     （robot_ground_contact 传感器）—— 真正的 roulade 从不
#     离地；腾空旋转现在不赚钱，也无法打开
#     完成门控。
#   • 头部锁存：落地年金额外要求在累加处于第一象限窗口期间发生过
#     头部-地面接触 —— "翻过头"是硬性要求，而非一个 0.5 权重的建议。
#   • 付费速率上限：进展增量封顶在 max_paid_rate；旋转
#     快于上限【丧失】多余部分（而非延后），因此速度不
#     再能赚钱。一个显式的超速惩罚为此兜底。
# 
#
# env 对象上的每环境状态（惰性创建，由
# reset_roulade_state 重置）：
#   env._roulade_accum      —— 仅支撑时的前向俯仰速率积分 (rad)
#   env._roulade_max        —— 本 episode 迄今 max(accum)（进展前沿）
#   env._roulade_paid       —— roulade_progress 已支付的前沿
#   env._roulade_head_latch —— 第一象限期间头部触地后置 True

# 前滚符号：俯卧是 +90° pitch = 绕机身 +y 旋转
#（set_random_ground_state 约定），因此前滚 = 【正】机身系
# ω_y。已实证验证（见 claude_experiments smoke test）：正的
# 绕 +y 的 qvel 让机器人低头/向前并推高 accum。
_ROULADE_FWD_SIGN = 1.0

# 累加器更新读取的传感器名称（必须与 env cfg 一致）。
_ROULADE_SUPPORT_SENSOR = "robot_ground_contact"
_ROULADE_HEAD_SENSOR = "head_ground_contact"

# 头部锁存窗口：累加在该窗口内时的头部-地面接触
# 把该 episode 标记为真正翻过头。在真正的 roulade 中，
# 头部在机身旋转约 60–120° 时着地；窗口在其周围放宽。
_HEAD_LATCH_LO = math.radians(20.0)
_HEAD_LATCH_HI = math.radians(170.0)

# 头顶轴，在 jaw_soft 的【局部】系中（2026-08-13 实测：
# 机器人以 HOME 落定时，世界上方在 jaw_soft 系中的表示）。
# 锁存要求该轴在接触时指向【下方】—— "头的平顶
# 着地"，而不是脸或外壳侧面（第 5 次运行修复：
# 第 4 次运行的策略翻过肩膀，那仍会碰到 jaw_soft）。
_HEAD_TOP_AXIS = (0.882, 0.0, 0.471)
# dot(top_axis_world, -z) 阈值。实测标志点（trunk 倾斜 110°）：
# 被动脸着地（颈部 HOME）读 +0.6，完全收下巴（neck_pitch −1、
# head_pitch +1）读 −0.99 —— 0.3 接受部分收下巴，同时远离
# 任何脸/侧面接触。
_HEAD_TOP_DOWN_MIN = 0.3

# 累加器上的矢状面平坦度门控（第 5 次运行）：在干净前滚中
# 身体的【侧向】轴全程保持水平 —— 其世界 z
# 分量对任意纯俯仰量都是 2(q_y·q_z + q_w·q_x) ≈ 0，而
# 若改为翻过肩膀则趋向 ±1。侧向轴在水平 ~30° 内给
# 完整旋转积分，超过 ~60° 为零：侧翻不算旋转，不赚进展，
# 也永远不会打开落地门。
# 
_FLAT_FULL = 0.5    # |lateral_axis_z| = sin(30°): full credit below
_FLAT_ZERO = 0.866  # sin(60°): zero credit above


def _lateral_axis_z(quat: torch.Tensor) -> torch.Tensor:
    """机身侧向 (y) 轴的世界 z 分量。0 = 平坦/矢状面。"""
    return 2.0 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1])


def _head_top_down(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    """头顶轴指向地板处为 True（与 -z 点乘 > min）。"""
    if not hasattr(env, "_roulade_head_body_id"):
        ids, _ = asset.find_bodies("jaw_soft")
        env._roulade_head_body_id = ids[0]
    q = asset.data.body_link_quat_w[:, env._roulade_head_body_id]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    a, b, c = _HEAD_TOP_AXIS
    # R(q) @ axis_local 的 z 分量
    axis_world_z = (
        2.0 * (x * z - w * y) * a + 2.0 * (y * z + w * x) * b + (1.0 - 2.0 * (x * x + y * y)) * c
    )
    return axis_world_z < -_HEAD_TOP_DOWN_MIN


def _sensor_any_contact(env: ManagerBasedRlEnv, name: str) -> torch.Tensor | None:
    if name not in env.scene.sensors:
        return None
    found = env.scene.sensors[name].data.found
    return (found.view(found.shape[0], -1) > 0).any(dim=-1)


def _roulade_state(env: ManagerBasedRlEnv) -> tuple:
    if not hasattr(env, "_roulade_accum"):
        z = torch.zeros(env.num_envs, device=env.device)
        env._roulade_accum = z.clone()
        env._roulade_max = z.clone()
        env._roulade_paid = z.clone()
        env._roulade_head_latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._roulade_last_update_step = -1
    return env._roulade_accum, env._roulade_max, env._roulade_paid


def _update_roulade_accum(env: ManagerBasedRlEnv, asset: Entity) -> None:
    """把前向俯仰速率积分到每环境旋转累加器中。

    带 step 保护，使同一控制步中读取累加器的多个 reward 项不会重复积分。前沿
    (max) 只向前移动；向后摇摆（蓄力）既不付费也不退费。

    支撑门控（第 1 次运行修复）：只在机器人接触地形时才积分旋转 —— roulade 是有支撑的
    动作；弹道翻转累积不到任何量，因此既拿不到报酬也打不开完成门。

    当累加处于第一象限窗口期间头部触地时，还会锁存 env._roulade_head_latch ——
    落地年金要求它，使"翻过头"成为任务的硬性要求。
    """
    _roulade_state(env)
    step = int(env.common_step_counter)
    if step != env._roulade_last_update_step:
        omega_fwd = _ROULADE_FWD_SIGN * asset.data.root_link_ang_vel_b[:, 1]
        delta = torch.nan_to_num(omega_fwd, nan=0.0) * env.step_dt
        supported = _sensor_any_contact(env, _ROULADE_SUPPORT_SENSOR)
        if supported is not None:
            delta = delta * supported.float()
        # 矢状面平坦度门控（第 5 次运行）：侧/肩翻不算。
        y_z = torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w), nan=1.0).abs()
        t = torch.clamp((_FLAT_ZERO - y_z) / (_FLAT_ZERO - _FLAT_FULL), 0.0, 1.0)
        delta = delta * (t * t * (3.0 - 2.0 * t))
        env._roulade_accum = env._roulade_accum + delta
        env._roulade_max = torch.maximum(env._roulade_max, env._roulade_accum)

        head_contact = _sensor_any_contact(env, _ROULADE_HEAD_SENSOR)
        if head_contact is not None:
            in_window = (env._roulade_accum > _HEAD_LATCH_LO) & (
                env._roulade_accum < _HEAD_LATCH_HI
            )
            # 第 5 次运行：接触必须是头部的【平顶】（顶轴
            # 指向地板）—— 脸/外壳侧面接触不锁存。
            env._roulade_head_latch = env._roulade_head_latch | (
                head_contact & in_window & _head_top_down(env, asset)
            )
        env._roulade_last_update_step = step


def _roulade_completion_gate(
    env: ManagerBasedRlEnv,
    gate_lo: float,
    gate_hi: float,
    require_head: bool = False,
) -> torch.Tensor:
    """进展前沿上的 smoothstep：低于 gate_lo rad 为 0，高于 gate_hi 为 1。

    是旧相位时钟落地窗口的基于状态的替代 —— 只有（在【支撑】状态下）真正旋转才能打开，
    因此翻滚前站立和弹道翻转都拿不到。require_head=True 时门控还要求头部锁存 ——
    episode 必须翻过头才能解锁落地年金。
    """
    _, max_accum, _ = _roulade_state(env)
    t = torch.clamp((max_accum - gate_lo) / max(gate_hi - gate_lo, 1e-6), 0.0, 1.0)
    gate = t * t * (3.0 - 2.0 * t)
    if require_head:
        gate = gate * env._roulade_head_latch.float()
    return gate


def reset_roulade_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    standing_prob: float = 0.5,
    midroll_prob: float = 0.5,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
    standing_tilt_max: float = 0.0,
    forward_vel_range: tuple = (0.0, 0.0),
    midroll_pitch_min: float = math.radians(50.0),
    midroll_pitch_max: float = math.radians(185.0),
    midroll_z_min: float = 0.05,
    midroll_z_max: float = 0.10,
    midroll_omega_range: tuple = (0.0, 0.0),
    tuck_overrides: Optional[dict] = None,
    tuck_factor_range: tuple = (0.3, 1.0),
    joint_noise_std: float = 0.0,
):
    """重置到站立起点或翻滚中途状态（反向课程）。

    站立桶：直立（±standing_tilt_max pitch/roll 噪声），随机 yaw，HOME 关节
    （由 reset_robot_joints 保留），z ∈ [standing_z_min, _max]。``forward_vel_range``
    是冲量钩子：逐环境前向基座速度（机身 x，通过生成 yaw 映射到世界）均匀采样 —— 0
    表示静止起翻，后续可放宽以训练从行走起翻。

    翻滚中途桶：俯仰 ``midroll_pitch_min..max`` 进入翻滚（90° = 在头上，180° = 在背上），
    随机 yaw，腿按 ``tuck_factor_range`` 中的逐环境因子在 HOME→抱团间插值，
    z ∈ [midroll_z_min, _max]，可选来自 ``midroll_omega_range`` 的前向角动量。旋转累加器
    初始化为生成俯仰角，使进展记账（以及完成门）保持一致：170° 生成只对剩余的约
    190° 付费。
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    accum, max_accum, paid = _roulade_state(env)

    total = standing_prob + midroll_prob
    is_mid = torch.rand(num, device=env.device) < (midroll_prob / max(total, 1e-6))

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    # 每桶的俯仰：站立用小噪声，否则用翻滚中途角度。
    pitch = (torch.rand(num, device=env.device) * 2 - 1) * standing_tilt_max
    mid_pitch = (
        torch.rand(num, device=env.device) * (midroll_pitch_max - midroll_pitch_min)
        + midroll_pitch_min
    )
    pitch = torch.where(is_mid, mid_pitch, pitch)
    roll = (torch.rand(num, device=env.device) * 2 - 1) * max(standing_tilt_max, math.radians(5.0))

    cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5); sr = torch.sin(roll * 0.5)
    # ZYX 内禀欧拉角 → 四元数 (yaw * pitch * roll)，与
    # set_random_ground_state 相同。
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    z_stand = torch.rand(num, device=env.device) * (standing_z_max - standing_z_min) + standing_z_min
    z_mid = torch.rand(num, device=env.device) * (midroll_z_max - midroll_z_min) + midroll_z_min
    new_z = torch.where(is_mid, z_mid, z_stand)

    env.sim.data.qpos[env_ids, 2] = new_z + _env_origin_z(env, env_ids)
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0

    servo_ids = _servo_joint_ids(env, asset)

    # 翻滚中途关节：在被 override 的关节上 HOME → 抱团插值，对所有
    # 伺服关节加噪声（passive_* backlash 铰链必须保持 0）。
    mid_env_ids = env_ids[is_mid]
    if len(mid_env_ids) > 0 and tuck_overrides:
        u = (
            torch.rand(len(mid_env_ids), device=env.device)
            * (tuck_factor_range[1] - tuck_factor_range[0])
            + tuck_factor_range[0]
        )
        for jnt_idx, angle in tuck_overrides.items():
            col = 7 + servo_ids[jnt_idx]
            home = env.sim.data.qpos[mid_env_ids, col]
            env.sim.data.qpos[mid_env_ids, col] = home + u * (angle - home)
    if len(mid_env_ids) > 0 and joint_noise_std > 0.0:
        cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
        noise = torch.randn(len(mid_env_ids), len(cols), device=env.device) * joint_noise_std
        env.sim.data.qpos[mid_env_ids.unsqueeze(1), cols.unsqueeze(0)] += noise

    # 翻滚中途的前向角动量：绕机身 +y 旋转。MuJoCo free
    # joint 的 qvel[3:6] 是【机身系】下的角速度，所以 [0, ω, 0]
    # 无论生成 yaw 如何都是前滚轴（在 smoke
    # 测试中验证 —— 带 yaw 的生成仍在其自身系中笔直前滚）。
    if len(mid_env_ids) > 0 and midroll_omega_range[1] > 0.0:
        omega = (
            torch.rand(len(mid_env_ids), device=env.device)
            * (midroll_omega_range[1] - midroll_omega_range[0])
            + midroll_omega_range[0]
        )
        env.sim.data.qvel[mid_env_ids, 4] = _ROULADE_FWD_SIGN * omega

    # 冲量钩子：站立生成的前向基座速度，机身 x → 世界 xy
    # 经生成 yaw 映射。(0, 0) = 静止起翻，禁用。
    stand_env_ids = env_ids[~is_mid]
    if len(stand_env_ids) > 0 and forward_vel_range[1] > 0.0:
        vx = (
            torch.rand(len(stand_env_ids), device=env.device)
            * (forward_vel_range[1] - forward_vel_range[0])
            + forward_vel_range[0]
        )
        yaw_s = yaw[~is_mid]
        env.sim.data.qvel[stand_env_ids, 0] = vx * torch.cos(yaw_s)
        env.sim.data.qvel[stand_env_ids, 1] = vx * torch.sin(yaw_s)

    # 进展记账：站立从 0 开始，翻滚中途从生成俯仰角开始。
    spawn_angle = torch.where(is_mid, mid_pitch, torch.zeros_like(mid_pitch))
    accum[env_ids] = spawn_angle
    max_accum[env_ids] = spawn_angle
    paid[env_ids] = spawn_angle
    # 头部锁存：翻滚中途生成被视为已经过了头部阶段
    #（反向课程教的是翻滚【完成】；要求一个它们没机会
    # 挣到的锁存会让它们的落地门永远关闭）。
    # 站立生成必须靠真正翻过头来挣得它。
    env._roulade_head_latch[env_ids] = is_mid


def roulade_progress(
    env: ManagerBasedRlEnv,
    target_angle: float = 2 * math.pi,
    max_paid_rate: float = 3.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """支付进展前沿的增量，最多一整圈翻滚。

    reward = Δ(min(max_accum, target)) / (step_dt · target)，封顶在
    max_paid_rate rad/s 的付费旋转。俯卧扎营（0/step）、在前沿下方摇摆（0/step）、
    或转过 2π（钳制）都无利可图。累加器有支撑门控，因此腾空旋转也不付费。

    max_paid_rate（第 1 次运行修复）：快于上限的旋转【丧失】多余部分 —— 付费指针
    仍跳到前沿，只是只付封顶的量。因此暴力猛甩拿到的总进展 reward 比受控的 ≤cap
    翻滚【更少】，而不是同样总量但更快拿到。
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    _, max_accum, paid = _roulade_state(env)
    new_paid = torch.clamp(max_accum, max=target_angle)
    delta = torch.clamp(new_paid - torch.clamp(paid, max=target_angle), min=0.0)
    delta = torch.clamp(delta, max=max_paid_rate * env.step_dt)
    env._roulade_paid = torch.maximum(paid, new_paid)
    return delta / (env.step_dt * target_angle)


def roulade_head_pivot(
    env: ManagerBasedRlEnv,
    sensor_name: str = "head_ground_contact",
    angle_lo: float = math.radians(30.0),
    angle_hi: float = math.radians(240.0),
    rate_norm: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """奖励翻滚中途旋转时头部与地面接触。

    contact × window(accum ∈ [angle_lo, angle_hi]) × clamp(ω_fwd/rate_norm, 0, 1)
    × (0.3 + 0.7·top_down)。
    速率因子是反扎营保护：脸着地、头搁在地上的机器人 ω_fwd ≈ 0，什么都赚不到 ——
    该项只为【翻过】头部付费。top_down 因子（第 5 次运行）让这个密集塑形与锁存对齐：
    翻滚中途任何头部接触付 30%，【平顶】（收下巴）接触付满额 —— 这就是教收下巴的梯度。
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    accum, _, _ = _roulade_state(env)

    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    found = env.scene.sensors[sensor_name].data.found
    contact = (found.view(found.shape[0], -1) > 0).any(dim=-1).float()

    in_window = ((accum > angle_lo) & (accum < angle_hi)).float()
    omega_fwd = _ROULADE_FWD_SIGN * asset.data.root_link_ang_vel_b[:, 1]
    rate = torch.clamp(torch.nan_to_num(omega_fwd, nan=0.0) / rate_norm, 0.0, 1.0)
    top = 0.3 + 0.7 * _head_top_down(env, asset).float()
    return contact * in_window * rate * top


def roulade_landing_composite(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    joint_indices: list,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """standing_composite_score × 完成门。

    那笔大年金：一旦翻滚（接近）完成，每一步以 HOME 姿态站立都付费 —— 用脚落地并
    保持住主导所有部分结果。旋转到 gate_lo 之前为零，因此站立生成不能靠什么都不做刷它。
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    score = standing_composite_score(
        env,
        target_height=target_height,
        height_std=height_std,
        upright_std=upright_std,
        pose_std=pose_std,
        joint_indices=joint_indices,
        target_overrides=target_overrides,
        asset_cfg=asset_cfg,
    )
    return score * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_upright_after_roll(
    env: ManagerBasedRlEnv,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """线性 cos(tilt) × 完成门 —— 朝竖直的引导拉力。

    从【任意】朝向都有梯度（复合分数在远离目标时接近零），但只在翻滚之后：gate_lo
    之前恰好为零，因此它不能像旧的常开 upright 项那样阻碍翻转。
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    quat = asset.data.root_link_quat_w
    upright = 1.0 - 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2))
    return torch.clamp(upright, min=0.0) * _roulade_completion_gate(
        env, gate_lo, gate_hi, require_head=True
    )


def roulade_height_after_roll(
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float = 0.04,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """宽高度 Gaussian × 完成门 —— 向上拉到站立高度。"""
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    g = torch.exp(-((z - target_height) / std) ** 2)
    return g * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_landing_sharp(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float = 0.015,
    upright_std: float = 0.3,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """紧 std 的直立 × 高度 Gaussian × 完成门 —— 最后一英里。

    针对 27° 倾斜 / 1 cm 下蹲的终点盆地所做第 4 次运行修复：宽落地复合分数
    （upright_std 0.40）在该姿态得 ~0.5，于是策略停在那里。这是 standup 的两层经验 ——
    宽层负责到达，锐层负责完成。在 27° 倾斜时该项得 ~0.1（真实梯度）；竖直时付 ~1。
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2))
    upright_g = torch.exp(-tilt_sq / (upright_std * upright_std))
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    height_g = torch.exp(-((z - target_height) / height_std) ** 2)
    gate = _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)
    return upright_g * height_g * gate


def roulade_stand_tax(
    env: ManagerBasedRlEnv,
    target_height: float,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """自否定的目标以下高度 L1，只在翻滚完成后生效。

    返回 −max(0, target − z) × completion_gate —— 配合【正】权重使用（惩罚符号约定）。
    针对翻滚后瘫倒扎营的第 3 次运行修复：带门控的落地 reward 让站立优于瘫成一堆，
    但那堆本身是【免费】的 —— 只有正的带门控项时，"保持瘫倒"每步收 ≈0，是一个舒适盆地
    （standup 静态坐姿教训：盆地必须是净【负】才能迫使起身）。门控使翻滚过程本身不被
    收税，并要求头部锁存，使无翻滚 episode 不会因此被惩罚出奇怪的规避行为。
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    shortfall = torch.clamp(target_height - z, min=0.0)
    return -shortfall * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_rise_velocity(
    env: ManagerBasedRlEnv,
    max_height: float = 0.125,
    gate_lo: float = math.radians(180.0),
    gate_hi: float = math.radians(260.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """com_upward_velocity × 翻滚后期门 —— 引导退出起身。

    roulade 的后半段（仰卧 → 坐起 → 站立）就是仰卧恢复问题，而 standup 环境证明了
    那里光靠终点态 reward 在零运动时梯度为零：直接为上升 vz 付费。门控从约 180°
    （在背上）开始打开，使翻滚前的颠簸赚不到；在 max_height 以上关闭，防止靠跳跃刷分。
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    reward = torch.clamp(vz, min=0.0) * (z < max_height).float()
    return reward * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_overspeed_penalty(
    env: ManagerBasedRlEnv,
    omega_max: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """max(0, |ω_y| − omega_max)² —— 对猛甩速度的二次税。

    正数量；用负权重。补充 roulade_progress 里的付费速率上限：上限消除了快于
    ~3 rad/s 旋转的【激励】，这里增加了 omega_max 以上的显式【成本】，使"暴力"严格劣于
    "受控"，而不仅仅是不更好。受控的完整翻滚（平均约 2–3 rad/s）从不触及它。
    """
    asset: Entity = env.scene[asset_cfg.name]
    omega_y = torch.nan_to_num(asset.data.root_link_ang_vel_b[:, 1], nan=0.0)
    excess = torch.clamp(omega_y.abs() - omega_max, min=0.0)
    return excess.pow(2)


def roulade_flatness_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """(侧向轴世界 z)² —— 朝矢状面翻滚的密集梯度。

    正数量；用负权重。站立时为零，在任意深度的干净【前滚】中也为零（纯俯仰让侧向轴
    保持水平），完全翻到肩膀上时为 1。累加器的平坦度门让侧翻无利可图；该项增加了逐步
    梯度，把动作引回该平面。
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w), nan=0.0).pow(2)


def roulade_sagittal_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """偏离矢状面的旋转：机身系 ω_x² + ω_z²（正值；用负权重）。
    ω_y 是翻滚轴，保持自由。"""
    asset: Entity = env.scene[asset_cfg.name]
    omega_b = asset.data.root_link_ang_vel_b
    return torch.nan_to_num(omega_b[:, 0].pow(2) + omega_b[:, 2].pow(2), nan=0.0)


def roulade_lateral_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """机身系侧向 (y) 线速度² —— 保持翻滚笔直。"""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.nan_to_num(asset.data.root_link_lin_vel_b[:, 1].pow(2), nan=0.0)
