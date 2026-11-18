---
name: 给 output.xml 添加关节限位
overview: 参照 robot_walk_backlash.xml 的关节 range，为 Microduck-urdf-20261117/output.xml 的 13 个主关节补上限位，并启用 compiler autolimits 使限位生效。
todos:
  - id: add-joint-limits
    content: 在 output.xml 启用 autolimits 并为 13 个主关节写入 range
    status: completed
  - id: verify-limits
    content: 加载 XML 校验限位生效且 backlash、自碰撞设置未变
    status: completed
    dependencies:
      - add-joint-limits
---

## 用户需求
以 `robot_walk_backlash.xml` 为参照，给 `Microduck-urdf-20261117/output.xml` 添加关节限位。

## 范围
- 仅添加/启用关节限位（`range`）。
- 不修改关节命名、backlash 关节、`self_collision_only` 设置、`chosen_actuator` class 等其他内容。

## 技术栈
- MuJoCo MJCF（XML）静态编辑，复用参考模型现有约定；不新增依赖、不引入新文件。

## 实施方法
目标文件当前 `compiler` 未开启 `autolimits`，且 13 个主关节均无 `range`（URDF 源头全为 `type="continuous"`）。仅写 `range` 而不启用限位不会生效，因此分两步：
1. 在 `compiler` 上加 `autolimits="true"`，与参考模型第 5 行一致，使所有带 `range` 的关节自动获得 `limited="true"`。
2. 按下方映射给 13 个主关节写入 `range`；backlash 关节已由 `class="backlash"` 自带 `range="-0.017453292519943295 0.017453292519943295"`，保持不变。

### 关节映射与取值（全部取自参考模型）

| 目标关节（目标文件行号） | 参考关节 | range |
|---|---|---|
| neck_joint_01（133） | neck_pitch | -1.5707963267948974 1.0471975511965967 |
| neck_joint_02（142） | head_pitch | -1.5707963267948966 1.5707963267948966 |
| neck_joint_03（151） | head_yaw | -2.9670597283903613 2.9670597283903595 |
| rleg_joint_01（170） | right_hip_yaw | -0.523598775598297 0.43633231299858416 |
| rleg_joint_02（179） | right_hip_roll | -0.3839724354387507 0.38397243543875426 |
| rleg_joint_03（188） | right_hip_pitch | -1.57079632679494 1.570796326794853 |
| rleg_joint_04（197） | right_knee | -1.5707963267949339 1.5707963267948593 |
| rleg_joint_05（207） | right_ankle | -1.5707963267949028 1.5707963267948903 |
| lleg_joint_01（224） | left_hip_yaw | -0.4363323129985824 0.5235987755982988 |
| lleg_joint_02（233） | left_hip_roll | -0.3839724354386992 0.38397243543880577 |
| lleg_joint_03（242） | left_hip_pitch | -1.570796326794949 1.5707963267948442 |
| lleg_joint_04（251） | left_knee | -1.5707963267948983 1.5707963267948948 |
| lleg_joint_05（261） | left_ankle | -1.5707963267949063 1.5707963267948868 |

颈部目标只有 3 个关节而参考模型有 4 个，按链路顺序对应并丢弃 `head_roll`；腿部为严格 5↔5 对应（依据 link 链顺序及 `link_04` 小腿自碰撞、`link_05` 脚掌地面碰撞）。

## 实施细节
- 唯一改动文件：`/home/scpv57/work/duck/myduck/microduck_rl/Microduck-urdf-20261117/output.xml`。
- `chosen_actuator` class 的 `ctrlrange="-10.0 10.0"` 保持不变；`autolimits` 不影响它。
- 验证时确认：13 个主关节确实带上限限位、backlash 关节限位未被改写、`self_collision_only` 与 `collision` geom 未被触及。

## 目录结构
- `Microduck-urdf-20261117/output.xml` [MODIFY] 给 compiler 添加 `autolimits="true"`，并为上表 13 个主关节逐一加 `range`；其余内容原样保留。
