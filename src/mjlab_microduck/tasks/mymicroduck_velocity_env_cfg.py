"""Microduck 速度（行走）环境。

主 locomotion 任务：速度指令跟踪 + 头部姿态指令。
reward/正则化配方以 locomotion 为核心（以 lean 跟踪为主 +
gait/脚部项、通过 curriculum 逐渐增强的 action-rate 平滑），其中：

  - foot_slip 保持为 -0.1（刻意设弱——再强的话对这个机器人
    以枢轴旋转为主的转向限制太大）
  - 固定且适度的指令范围（ang ±1.0 使转向可学习），而不是
    超出机器人能力的逐步放宽 curriculum
  - 原地转向：15% 的环境设为 lin=0 + |ang| ∈ [0.4, 1.0]（2026-07 审计：
    独立均匀采样使原地打转只占约 2% 的数据 → 学不会）
  - head_pose_tracking 作为主要目标，外加一个基于 EMA 的 head_pose_bias
    惩罚，只对可规避的头部 DC 下垂定价（见下文）
  - body_pose 跟踪的基础设施保留但禁用（weight 0），以便需要它的环境
    保持 obs 槽位活跃
"""

import math
from copy import deepcopy

NUM_STEPS_PER_ENV = 24

# 被指令原地打转的环境比例（lin=0，|ang| ∈ [0.4·max, max]）
TURN_IN_PLACE_FRACTION = 0.15

# 对称性
ENABLE_SYMMETRY = False

# Domain randomization（域随机化）开关
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True  # 随机化头部组件各 body 的 CoM
ENABLE_KP_RANDOMIZATION = False # 原为 True
ENABLE_KD_RANDOMIZATION = False # 原为 True
ENABLE_MASS_INERTIA_RANDOMIZATION = True  # 行走稳定后可以开启
ENABLE_JOINT_FRICTION_RANDOMIZATION = True  # 通过 FrictionDRBamActuator.friction_scale 按环境缩放 BAM 的 friction 预算
ENABLE_JOINT_DAMPING_RANDOMIZATION = False
ENABLE_ARMATURE_RANDOMIZATION = True  # 反射转子惯量（microban 风格）。确实影响 BAM（armature 被设置而非清零）
ENABLE_VELOCITY_PUSHES = True  # 基于速度的推力，用于鲁棒性训练
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # 模拟安装误差
ENABLE_ENCODER_BIAS = True  # 每环境的关节编码器校准偏置（actor obs 看到 joint_pos + bias）
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False  # 随机化初始倾角以强制产生反应式行为

# 头部/机身姿态指令跟踪（取代旧的 neck-offset 扰动方案）。
# Head pose：颈部/头部关节相对 HOME 的 4D 增量；vel env 将其作为主要目标跟踪。
# Body pose：[x, y, z, roll, pitch, yaw] 的 6D 增量；vel env
# 采样小范围 + 极小的 reward 权重，使输入神经元保持活跃但
# 跟踪不是优先项（standup env 会提高权重）。
HEAD_POSE_CMD_RESAMPLE_S = (2.0, 5.0)
BODY_POSE_CMD_RESAMPLE_S = (2.0, 5.0)

# 观测（observation）配置
USE_PROJECTED_GRAVITY = True  # 为 True 时使用 projected gravity 而非原始加速度计

# Domain randomization 范围（按需调整）
# 已验证稳定的保守范围——如需可逐步增大
COM_RANDOMIZATION_RANGE = 0.003  # 初始 ±3mm，通过 curriculum 渐增到 ±8mm
# 头部 CoM 随机化：每个 episode 对头部组件的每个 body 施加
# （neck → neck_pitch → yaw_roll_motion → 头部横滚 body）。与上面的
# 躯干 CoM 随机化使用相同的非累积机制。头部横滚 body 在 walk 模型中名为
# bottom_head_shell，在 2026-07 的 roller 模型中名为 jaw_soft，因此二者交替。
# 注意：bearing_roll 不是头部 body——在两个模型中它都是右髋偏航连杆
# （trunk_base 的子节点）；它一直被错误地列在这里，仅为保持现有 DR 行为而保留。
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # 初始 ±3mm，通过 curriculum 渐增
# HEAD_BODY_NAMES = (
#     "neck",
#     "neck_pitch",
#     "yaw_roll_motion",
#     "(bottom_head_shell|jaw_soft)",
#     "bearing_roll",
# )
HEAD_BODY_NAMES = (
    "neck_Link_01",
    "neck_Link_02",
    "neck_Link_03",
)
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)  # ±5%，同时作用于 mass 和 inertia
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # ±15%
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10%（可增大到 0.8-1.2）
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_DAMPING_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10% 反射转子惯量（microban: dr.joint_armature，相同范围）
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)  # 每 3-6 秒施加一次推力
VELOCITY_PUSH_RANGE = (-0.3, 0.3)  # 速度变化范围，单位 m/s。原为 ±0.5——每 3-6 秒
# 施加一个大于最大行走速度（0.4）的 ADDITIVE 冲量，会训练出永久紧张的跌倒恢复步态（2026-07 审计）。
# ±0.3 既保持推力鲁棒性，又允许更平静的步态成为最优解。
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # 最大 6° 的随机轴 IMU 安装误差。注意：零均值（随机轴）——训练的是对错位*幅度*的容忍度，而非俯仰偏置。真实板子的系统性 ~5° 俯仰偏移在 runtime 源头修正（imu-pitch-offset），不在这里。
ENCODER_BIAS_RANGE = (-0.015, 0.015)  # 每关节 ±0.86° 编码器偏移（每环境恒定）
BASE_ORIENTATION_MAX_PITCH_DEG = 10.0  # episode 开始时 ±10° 前后倾斜
BASE_ORIENTATION_MAX_ROLL_DEG = 5.0  # episode 开始时 ±5° 左右倾斜

import mujoco as _mujoco
import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.mymicroduck_constants import MICRODUCK_WALK_BACKLASH_ROBOT_CFG
from mjlab_microduck.tasks import mymdp as microduck_mdp
from mjlab_microduck.tasks.mysymmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


# Microduck 专用粗糙地形：比默认的 ROUGH_TERRAINS_CFG 温和得多。
# 机器人只能把脚抬高约 1-2 cm，因此台阶高度上限为 1.5 cm。
MICRODUCK_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    sub_terrains={
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.25),
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.0, 0.015),  # 最大 1.5 cm（默认 10 cm）
            step_width=0.15,
            platform_width=2.0,
            border_width=1.0,
        ),
        # 注意：BoxInvertedPyramidStairsTerrainCfg 已移除——它把 env_origin_z 设为坑底
        # （负值），导致在 root_z = 0.12 + env_origin_z ≈ −0.10 m 处重置，
        # 把机器人放在坑底以下，从而穿过地面掉落。
        # 类似鹅卵石的不平整地面：每个格子随机高度偏移。
        # 8m 地块上 grid_width=0.12 = 66×66 = 4356 个 box/地块 → 约 26.1 万个 → OOM。
        # 0.45 m 得到 17×17 = 289 个 box/地块 → 约 1.7 万个（border = 0.35 m ✓）。
        # 必须不能整除地形尺寸（8.0 m）：0.45 × 17 = 7.65 ✓
        "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.30,
            grid_width=0.45,
            grid_height_range=(0.0, 0.010),  # 最大 1 cm
            platform_width=1.5,
        ),
        # 缓坡（heightfield 金字塔，platform 在顶部——机器人生成在
        # 平平台上，随指令重采样走下/走上/横穿斜坡）。slope_range 是坡高/水平距离：
        # 按难度 0.03→0.10 ≈ 1.7°→5.7°——小机器人，小斜坡。不用倒置版（见上面
        # inverted-pyramid 的 env_origin 注释——同类坑内生成风险）。
        # vertical_scale=0.001 把量化步长保持在 1 mm，使缓坡平滑
        # 而不是 5 mm 台阶的阶梯状。
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.03, 0.10),
            platform_width=2.0,
            vertical_scale=0.001,
        ),
    },
    add_lights=False,
)


def _soften_terrain_contacts(spec: _mujoco.MjSpec) -> None:
    """软化地形 box geom 的接触，减少边缘接触导致的 NaN 不稳定。

    Box 地形把相邻 geom 放在不同高度。高度变化处的硬边缘在脚落地时会
    造成接触法向不稳定，可能在 MuJoCo 求解器中产生脉冲式 NaN 力。

    把 solref 时间常数加倍（0.02 → 0.04 s）使接触弹簧软 2 倍——
    足以抑制不稳定，又不会明显改变宏观行走物理。应用于 "terrain" body 中的
    所有 geom，它包含 TerrainGenerator 生成的每一个 box。
    """
    body = spec.body("terrain")
    count = 0
    for geom in body.geoms:
        geom.solref = [0.04, 1.0]   # 时间常数软 2 倍（默认：0.02）
        geom.solimp = [0.85, 0.95, 0.001, 0.5, 2.0]  # 略软的阻抗
        count += 1
    print(f"[rough terrain] spec_fn: softened {count} terrain geoms (solref=0.04)")


def make_microduck_velocity_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """创建 Microduck 速度跟踪环境配置。"""

    # std_standing = {
    #     # 下半身——更紧，使机器人在站立时保持 home 姿态
    #     r".*hip_yaw.*": 0.1,
    #     r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05——保持 5° 内扣站姿（脚底平放），阻止腿部外撇
    #     r".*hip_pitch.*": 0.15,
    #     r".*knee.*": 0.15,
    #     r".*ankle.*": 0.1,
    # }

    std_standing = {
        r".*leg_joint_01.*": 0.1,
        r".*leg_joint_02.*": 0.05,
        r".*leg_joint_03.*": 0.15,
        r".*leg_joint_04.*": 0.15,
        r".*leg_joint_05.*": 0.1,
    }

    # std_walking = {
    #     # 下半身
    #     r".*hip_yaw.*": 0.3,
    #     r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05——保持 5° 内扣站姿，阻止腿部外撇成竖直
    #     r".*hip_pitch.*": 0.4,
    #     r".*knee.*": 0.4,
    #     r".*ankle.*": 0.25, # 原为 0.15
    # }
    std_walking = {
        r".*leg_joint_01.*": 0.3,
        r".*leg_joint_02.*": 0.05,
        r".*leg_joint_03.*": 0.4,
        r".*leg_joint_04.*": 0.4,
        r".*leg_joint_05.*": 0.25,
    }


    site_names = ["left_foot", "right_foot"]

    # 脚部接触传感器——左、右顺序
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            # pattern=r"^(left_foot_collision|right_foot_collision)$",  # 左脚在前，右脚在后
            pattern = r"^(lleg_link_05_collision|rleg_link_05_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # mjlab 1.3.0：foot_height obs + foot_clearance/foot_swing_height reward 现在
    # 由每脚的地形高度射线传感器驱动（原为基于 site_pos）。
    # 与 microban 的 foot_height_scan 一致。
    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in site_names),
        pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )

    foot_frictions_geom_names = (
        "lleg_link_05_collision",
        "rleg_link_05_collision",
    )

    # 基础配置
    cfg = make_velocity_env_cfg()

    # 机器人设置
    cfg.scene.entities = {"robot": MICRODUCK_WALK_BACKLASH_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, foot_height_scan_cfg)
    cfg.viewer.body_name = "trunk_base"
    cfg.decimation = 13

    # 动作配置
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # === REWARD ===
    # Pose reward 配置
    cfg.rewards["pose"].params["std_standing"] = std_standing  # command=0 时更紧
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking
    # Pose reward 只作用于腿部关节。头/颈部由指令驱动
    # （head_pose_tracking）——如果它们也纳入该 reward，会把它们拉向 HOME，
    # 而 head_pose_tracking 把它们拉向指令；当大指令下 head_pose_tracking 的
    # 梯度消失后 pose reward 占主导，策略会收敛到"忽略指令"。
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        # "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
        "robot", joint_names=(r"^(l|r)leg_joint_.*",)
        
    )
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].weight = 1.0

    # 机身相关的 reward 配置
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    # upright：刻意设强（2.0 / std²=0.05，原为 1.0 / std²=0.1）。
    # 2026-07 俯仰 vs 速度评估：策略行走时有 +2-4° 的稳定前倾（p90 ~6-8°），
    # 且有速度时约 2/3 的推力跌倒都是向前。在 weight 1.0 / std²=0.1 下
    # 4° 前倾只花 ~0.05/step——基本免费。在 2.0 / std²=0.05 下花 ~0.19/step：
    # 足以提供梯度在稳定步态中保持躯干水平，而瞬时前倾（推力恢复、加速）仍负担得起。
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    # 脚部相关配置。在 mjlab 1.3.0 中 foot_swing_height 完全由传感器驱动
    # （无 asset_cfg）；只有 foot_clearance/foot_slip 仍带有 asset_cfg，
    # 其 site_names 用于选择脚部。
    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    # 机身相关配置
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)

    # foot_slip 刻意设弱（-0.1 而非 -1.0）：对这个以枢轴旋转为主的机器人来说
    # -1.0 限制太强。
    cfg.rewards["foot_slip"].weight = -0.1
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.01

    cfg.rewards.pop("soft_landing", None)

    # 自碰撞惩罚：阻止腿部撞上躯干电池座
    # （leg、leg_2、battery_holder 上 self_collision_only 类的 geom）。
    # 有了合适的关节范围限制，策略实际上够不到机身，但这里的正向信号让它保持距离。
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )


    # air_time 窗口 [0.125, 0.300] s。注意：零指令下静止站立由 standing_envs
    # curriculum 教会（到约 iter 2000 时 → 25% 站立环境），
    # 而非通过显式的静止/不迈步 reward 项。
    cfg.rewards["air_time"].weight = 10.0 # 3.0 -> 8.0
    cfg.rewards["air_time"].params["command_threshold"] = 0.01
    cfg.rewards["air_time"].params["threshold_min"] = 0.125
    cfg.rewards["air_time"].params["threshold_max"] = 0.300

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    # 速度跟踪 reward
    cfg.rewards["track_linear_velocity"].weight = 2.0
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)
    cfg.rewards["track_angular_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)

    # 动作平滑性：阶段 0 的值；下面的 action_rate_weight curriculum
    # 到 iter 1500 将其从 -0.1 渐增到 -1.0。
    cfg.rewards["action_rate_l2"].weight = -0.1

    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.01
    cfg.rewards["foot_clearance"].params["target_height"] = 0.02  # 从 0.01 提高，以惩罚拖脚

    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.02  # 从 0.01 提高，以强制抬脚

    # 注意：没有单独的颈部 action-rate 项——共享的 action_rate_l2 对所有动作维度
    # （含颈部）求和，下面的 head_pose_tracking 为 4 个颈/头 DOF 提供了位置目标，
    # 因此颈部已被充分塑造。

    # 事件（Events）
    # BAM（mjlab_frictionloss 分支）每步写入每环境的 dof_frictionloss/dof_damping；
    # 这个空操作事件注册这些字段以便按 world 展开。
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)  # 更抓地的脚垫——从 (0.3, 1.2) 收窄
    # 终止数值不稳定（NaN 物理）的环境。
    # MuJoCo 在极端接触冲量下可能产生 NaN 关节位置。
    # 立即终止可在 NaN 传播进 obs 缓冲区并破坏网络权重之前重置到有效状态。
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    # 基于速度的推力，用于鲁棒性训练
    if ENABLE_VELOCITY_PUSHES:
        # play 模式下用更短的间隔以便更好地观察
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S

        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    # Domain randomization——在 reset 时每个 episode 重新采样。在 mjlab 1.3.0 中，
    # 自带的 dr.* 操作在 operation="add"/"scale" 时每次 reset 都从编译期默认字段读取
    # （Operation.uses_defaults=True），因此天然是非累积的——这一上游行为取代了
    # microduck 旧的自定义 restore-then-add 函数（那些函数是为了规避累积陷阱）。
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        # 随机化头部组件各 body 的 CoM（每次 reset 为每个 body 重新采样偏移）。
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        # 随机化电机 PD 增益
        # 使用处理 DelayedActuator 的自定义函数
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        # 通过 mjlab 的 pseudo_inertia 实现物理一致的 mass + inertia 随机化：
        # alpha 用 e^(2*alpha) 同时缩放 mass 和 inertia，CoM 保持不变
        # （因此不与 randomize_com 冲突）。alpha_range 由 ±5% 的 mass 缩放范围导出：
        # e^(2*alpha) ∈ [0.95, 1.05]。
        # 取代旧的自定义 randomize_mass_and_inertia——它在 mjlab 1.3.0 下是
        # 静默空操作（直接写每环境的 body_mass/body_inertia 不会被展开，会塌缩为单一共享值）。
        # Startup 模式 = 每个环境整轮运行固定（mass DR 的标准做法；无累积）。
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        # BAM 下的关节 friction DR：通过 FrictionDRBamActuator 的 friction_scale 钩子
        # 按环境缩放 BAM 的速度无关 friction 预算（Coulomb + Stribeck + load）。
        # BAM 下 MuJoCo 的 dof_frictionloss 被清零，因此自带的 dr.dof_frictionloss 是空操作
        # ——这才是 BAM 原生路径。
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_JOINT_DAMPING_RANDOMIZATION:
        # 随机化关节 damping（润滑、温度效应）。
        # 自定义非累积缩放器。注意：在 BAM 下为空操作（dof_damping
        # 在 edit_spec 中清零）；只影响 XML position actuator。
        cfg.events["randomize_joint_damping"] = EventTermCfg(
            func=microduck_mdp.randomize_dof_field_scaled,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "field": "dof_damping",  # domain_randomization=True 所必需
                "scale_range": JOINT_DAMPING_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        # 随机化反射转子惯量（armature），与 microban 完全一致
        # （dr.joint_armature，scale，±10%）。非累积（uses_defaults）。确实
        # 影响 BAM actuator——BAM 会设置 dof_armature（~0.0018），并未清零。
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # IMU 朝向随机化（安装误差）在下面的观测（OBSERVATION）层施加
    # （projected_gravity + base_ang_vel 的每环境恒定旋转）。
    # 旧的基于事件的 randomize_imu_orientation 写 site_quat，在 mjlab 1.3.0 下
    # 既不会按环境展开，也不会被这些 obs 读取——是空操作。

    # 基座朝向随机化（强制产生反应式行为）
    if ENABLE_BASE_ORIENTATION_RANDOMIZATION:
        cfg.events["randomize_base_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_base_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_pitch_deg": BASE_ORIENTATION_MAX_PITCH_DEG,
                "max_roll_deg": BASE_ORIENTATION_MAX_ROLL_DEG,
            },
        )

    # 观测（Observations）
    del cfg.observations["actor"].terms["base_lin_vel"]
    # mjlab 1.3.0 默认给两个组都加了 height_scan 项（地形射线扫描）。
    # microduck 的策略没有这种机载地形传感器，因此从两个组中都删除
    # （与 microban 一致）。
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # 只给 critic 添加 base_lin_vel（特权信息）
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )

    # 根据标志确定重力/加速度计项的名称
    gravity_term_name = "projected_gravity" if USE_PROJECTED_GRAVITY else "raw_accelerometer"

    # 标志为 False 时用 raw_accelerometer 替换 projected_gravity
    if not USE_PROJECTED_GRAVITY:
        # 删除 projected_gravity 并添加 raw_accelerometer
        del cfg.observations["actor"].terms["projected_gravity"]
        cfg.observations["actor"].terms["raw_accelerometer"] = ObservationTermCfg(
            func=microduck_mdp.raw_accelerometer,
            scale=1.0,
        )

    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1  # 原为 3（最坏 60 ms）；真实 dxl IMU 链路很快——±20 ms 包络（2026-07 审计）
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64

    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1  # 原为 3（最坏 60 ms）；真实 dxl IMU 链路很快——±20 ms 包络（2026-07 审计）
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # critic 的传感器派生项是 `nan_state` 无法保护的唯一 obs 路径
    # （它检查关节 + root 状态；这些读取射线/接触传感器数据，在状态仍干净时
    # MuJoCo 也可能返回非有限值）。这里一个 NaN 就会通过 rsl_rl 的 check_nan 毁掉整轮运行
    # ——这就是 2026-08-21 Velocity2-Rough-Backlash 崩溃的原因。仅 critic 侧，
    # 因此清洗对策略零成本。
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_height", microduck_mdp.foot_height_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    # 观测噪声配置（按需编辑这些值）
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03) # 原为 0.2
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)  # 原为 0.15
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)  # 原为 0.05
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)  # 原为 2.0

    # IMU 安装错位 DR（IMU 派生观测的每环境恒定旋转）。
    # 只作用于 ACTOR（策略看到轻微旋转的 IMU 坐标系，类似真实安装误差）；
    # critic 保持真值。
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        if USE_PROJECTED_GRAVITY:
            g = cfg.observations["actor"].terms[gravity_term_name]
            g.func = microduck_mdp.projected_gravity_imu_misaligned
            g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # joint_vel 上的 1 个控制步延迟：Dynamixel 固件通过对先前位置采样窗口
    # 做移动平均来计算 present_velocity，因此策略实际读到的值约滞后 1 个控制周期。
    # 这符合现实，并阻止策略依赖瞬时 qdot 反馈。
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # 从 joint_pos/vel obs 中排除 passive_* 关节（下颌连杆），使观测维度
    # 匹配动作维度（14）而非原始关节数（16）。
    # 先 deepcopy 每个 joint_pos/joint_vel 项——actor 和 critic 从基础模板共享
    # 相同的 term 对象/params 字典，修改一个会泄漏到另一个
    # （例如下面的 encoder-bias `biased` 标志）。
    # passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!.*_backlash$).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # Encoder-bias DR：基础模板采样每环境恒定的关节编码器偏移
    # （startup 事件 "encoder_bias"），但 joint_pos_rel 会忽略它，除非 biased=True。
    # 只把带偏置的关节位置喂给 ACTOR（真实编码器上报的值）；
    # critic 保持真实关节位置（特权信息）。
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # 指令（Commands）——deepcopy 以避免被其他 env cfg 破坏共享状态
    # （make_velocity_env_cfg() 返回的对象带有共享可变引用；
    # standup/ground_pick 环境会原地修改 commands["twist"]，把范围清零）
    command: UniformVelocityCommandCfg = deepcopy(cfg.commands["twist"])
    cfg.commands["twist"] = command
    command.rel_standing_envs = 0.02  # 从小但非零开始，由 curriculum 渐增
    command.rel_heading_envs = 0.0
    # 适度且固定的指令范围（无逐步放宽的 curriculum）：渐增到
    # lin ±0.4 / ang ±2.0 超出了机器人能力，并在 iter 1000 之后出现
    # reward/回合长度下降。ang ±1.0 是最大的变化——它使转向可学习。
    command.ranges.lin_vel_x = (-0.4, 0.4)
    command.ranges.lin_vel_y = (-0.3, 0.3)
    command.ranges.ang_vel_z = (-1.0, 1.0)
    command.viz.z_offset = 0.5
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))
    # 显式的原地转向桶（见上面的 TURN_IN_PLACE_FRACTION）。
    cfg.commands["twist"].rel_turn_in_place_envs = TURN_IN_PLACE_FRACTION

    # 头部姿态指令（相对 HOME 的 4D 增量，按关节顺序：
    #   neck_pitch, head_pitch, head_yaw, head_roll）。作为主要 reward 跟踪
    # ——见下面添加的 "head_pose_tracking"。初始范围小而非零，使输入神经元
    # 从第 0 步就保持活跃；curriculum 会逐步放宽。
    # 每个关节的最终上限反映其相对 HOME 机械可达的增量
    # （XML 限位减去 HOME 偏移，留约 10% 安全余量）：
    #   neck_pitch / head_pitch: ±1.10 rad（限位 ±π/2，HOME=±20°）
    #   head_yaw                : ±1.40 rad（限位 ±π/2，HOME=0）
    #   head_roll               : ±0.31 rad（限位 ±20°）
    # 初始范围小而非零，使输入神经元从第 0 步就保持活跃。
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        # ranges=(
        #     (-0.05, 0.05),    # neck_pitch
        #     (-0.05, 0.05),    # head_pitch
        #     (-0.07, 0.07),    # head_yaw
        #     (-0.015, 0.015),  # head_roll（更紧——机械范围小得多）
        # ),
        ranges=(
            (-0.05, 0.05),    # neck01
            (-0.05, 0.05),    # neck01
            (-0.07, 0.07),    # neck01
        ),
    )
    # 机身姿态指令（相对标称站姿的 6D 增量：[x, y, z, roll, pitch, yaw]）。
    # Vel env 保留这个槽位是为了 runtime 的 obs 形状一致；以极小权重跟踪，
    # 使输入神经元保持活跃但不主导策略。standup env 会提高权重并放宽范围。
    cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.005, 0.005),  # x (m)
            (-0.005, 0.005),  # y (m)
            (-0.005, 0.005),  # z (m)
            (-0.05, 0.05),    # roll (rad)
            (-0.05, 0.05),    # pitch (rad)
            (-0.05, 0.05),    # yaw (rad)
        ),
    )

    # 把头部 + 机身指令 obs 项追加到 policy 和 critic 两组。
    # 顺序对 runtime obs 布局很重要：[twist(3), head_pose(4), body_pose(6)]。
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "body_pose"},
        )

    # === 姿态跟踪 reward ===
    # head_pose：vel env 的主要目标——这次重写的核心。
    # std=0.5 配合逐关节高斯（见 mdp.py 的 head_pose_tracking）：在
    # 满幅 ±1.0 rad 指令下，不跟踪的策略仍有逐关节 reward
    # exp(-(1/0.5)²)=exp(-4)≈0.018——小但非零的梯度——因此 curriculum 放宽
    # 不会扼杀信号。最终 reward 是 4 个关节的均值，所以部分跟踪得部分 reward
    # （不是全有或全无）。
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=2.0,
        params={"command_name": "head_pose", "std": 0.5},
    )
    # body_pose：基础设施保留但禁用（weight 0）——obs 槽位和
    # 指令保持活跃，供提高权重的环境使用（standup）。
    cfg.rewards["body_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.body_pose_tracking_6d,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "nominal_height": 0.095,
            "xy_std": 0.05,
            "z_std": 0.02,
            "angle_std": math.radians(15),
        },
    )

    # 头部下垂修复（2026-08-20）。行走时头部约 15° 向下俯（实测：
    # run ww1g2198 的 head_pose_tracking 1.544/2.0 → 平均关节误差 14.6°）。
    # 不要通过收紧 head_pose_tracking 的 std 来修复：run 5yay13u4 试过
    # fine_std=0.1，策略到 iter 300 完全停止行走（air_time
    # 1.01 → 0.02，脚部峰值高度 15 mm → 2 mm，entropy 从 10.9 崩塌到 1.9）。
    # 瞬时紧容差对行走征税 0.77/step——占整个 air_time reward 的 76%——而且
    # 无法规避，因为 280 g 的头部（机器人质量的 38%）在迈步时必然振荡。
    # 站着不动得分更高，于是它就站着不动。
    # DC 偏置与振荡不同，是可规避的（把颈部指令偏置向上以抵消重力下垂），
    # 所以只给它定价：对误差的 1 秒 EMA 取 L1。
    # 在最优点，这对行走策略零成本。
    cfg.rewards["head_pose_bias"] = RewardTermCfg(
        func=microduck_mdp.head_pose_bias_penalty,
        weight=0.0,  # 由下面的 head_pose_bias_weight curriculum 渐增
        params={"command_name": "head_pose", "tau_s": 1.0},
    )

    # 地形（Terrain）
    if not rough:
        assert cfg.scene.terrain is not None
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    else:
        assert cfg.scene.terrain is not None
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG

        # 软化地形 box 接触：不同高度的相邻 box 会产生硬边缘，
        # 破坏接触求解器稳定性并产生 NaN 力。
        cfg.scene.spec_fn = _soften_terrain_contacts

        # velocity env 默认的 nconmax=35 对粗糙地形偏紧：当机器人跌倒、
        # 多个 body 连杆同时撞上多个 box 时，接触溢出 → 部分被静默丢弃 → 突然卸载 → NaN。
        cfg.sim.nconmax = 200   # 原为 35

        # velocity env 只用 10 次求解器迭代（默认 100），对粗糙 box 地形上的
        # 边缘接触而言太少。把迭代数增到 3 倍可显著减少接触求解失败，
        # 在 GPU 上计算开销适中（MJWarp 跨环境并行）。
        cfg.sim.mujoco.iterations = 30    # 原为 10
        cfg.sim.mujoco.ls_iterations = 50  # 原为 20

        if play:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5

    # action_rate 权重渐增：步态引导阶段用温和平滑，然后
    # 到 iter 1500 收紧到 -1.0。
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.2},
                {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.4},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.6},
                {"step": 1250 * NUM_STEPS_PER_ENV, "weight": -0.8},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -1.0},
            ],
        },
    )

    # 行走建立后逐步增加站立环境的比例
    cfg.curriculum["standing_envs"] = CurriculumTermCfg(
        func=microduck_mdp.standing_envs_curriculum,
        params={
            "command_name": "twist",
            "standing_stages": [
                {"step": 0,           "rel_standing_envs": 0.02},
                {"step": 500 * 24,    "rel_standing_envs": 0.05},
                {"step": 750 * 24,    "rel_standing_envs": 0.1},
                {"step": 1000 * 24,   "rel_standing_envs": 0.15},
                {"step": 1500 * 24,   "rel_standing_envs": 0.2},
                {"step": 2000 * 24,   "rel_standing_envs": 0.25},
            ],
        },
    )

    # 注意：没有速度指令范围 curriculum——范围是固定的（见上面的指令部分）。

    # 头部姿态指令范围 curriculum——逐关节，按每个关节相对 HOME 可达的增量缩放
    # （距 XML 限位留约 10% 余量）。与之前相同的 5 阶段形状
    # （每个关节最终上限的 5% → 15% → 35% → 65% → 100%）。
    # 颈/头俯仰最终 ±1.10 rad，head_yaw ±1.40，head_roll ±0.31。
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                # step,                ranges = ((neck_pitch), (head_pitch), (head_yaw),  (head_roll))
                {"step": 0,         "ranges": ((-0.05, 0.05),  (-0.05, 0.05),  (-0.07, 0.07))},
                {"step": 500 * 24,  "ranges": ((-0.17, 0.17),  (-0.17, 0.17),  (-0.21, 0.21))},
                {"step": 1000 * 24, "ranges": ((-0.39, 0.39),  (-0.39, 0.39),  (-0.49, 0.49))},
                {"step": 1500 * 24, "ranges": ((-0.72, 0.72),  (-0.72, 0.72),  (-0.91, 0.91))},
                {"step": 2000 * 24, "ranges": ((-1.10, 1.10),  (-1.10, 1.10),  (-1.40, 1.40))},
            ],
        },
    )

    # 机身姿态指令范围 curriculum：在 vel env 中保持小范围。Standup env
    # 会用大范围 + 高 reward 权重覆盖这个 curriculum。
    cfg.curriculum["body_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "body_pose",
            "range_stages": [
                {"step": 0, "ranges": (
                    (-0.005, 0.005),  # x (m)
                    (-0.005, 0.005),  # y (m)
                    (-0.005, 0.005),  # z (m)
                    (-0.05, 0.05),    # roll
                    (-0.05, 0.05),    # pitch
                    (-0.05, 0.05),    # yaw
                )},
            ],
        },
    )

    # CoM 随机化范围 curriculum——从小开始，逐步增大
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    # 上限 ±15 mm（2026-07 审计）：之前渐增到 ±30 mm
                    # 超出了脚部支撑多边形（脚跟只在脚踝后方 20 mm）——
                    # 随机化后的 CoM 可能完全落在支撑区之外，迫使机器人采用宽/快的
                    # 超反应式步态，并使后退平衡无法训练。性能回退的时间线与
                    # 渐增幅度吻合：0.015 → 0.02 → 0.03 时策略逐渐变差。
                    {"step": 0,          "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24,  "range": 0.01},
                    {"step": 1500 * 24,  "range": 0.015},
                ],
            },
        )

    # 头部 CoM 随机化范围 curriculum——从小开始，逐步增大
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    # 上限 ±10 mm（2026-07 审计——与躯干 CoM 同样的过度保守担忧；
                    # 头部是一个很长的力臂）。
                    {"step": 0,          "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24,  "range": 0.01},
                ],
            },
        )

    # 禁用默认 curriculum
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # head_pose_bias 渐增：iter 600 前关闭，然后到 iter 1500 从 1.0 增至 3.0。
    # 早期保持为 0，因为在步态尚未形成前，姿态精度项只会分散注意力。
    # 在 weight 3.0 下，15° 残余偏置花 0.79/step，2° 偏置花 0.10/step。
    cfg.curriculum["head_pose_bias_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_pose_bias",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 600 * NUM_STEPS_PER_ENV, "weight": 1.0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": 2.0},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": 3.0},
            ],
        },
    )

    return cfg


MicroduckRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    logger="tensorboard",
    # wandb_project="mjlab_microduck",
    experiment_name="velocity",  # 目录名
    run_name="velocity",  # 在 wandb 中追加到时间戳后：<datetime>_velocity
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
