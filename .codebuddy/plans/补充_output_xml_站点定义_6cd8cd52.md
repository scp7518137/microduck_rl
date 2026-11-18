---
name: 补充 output.xml 站点定义
overview: 参照 robot_walk_backlash.xml，在新模型 output.xml 中补齐 6 个 site（含 head_camera camera），并用分链运动学对齐按新模型几何计算各 site 的 pos/quat。
todos:
  - id: add-head-sites
    content: 在 neck_Link_03 补充 head_camera（含 camera）、tof、head_imu、mouth_tip 四个站点
    status: completed
  - id: add-foot-sites
    content: 在 lleg_link_05 与 rleg_link_05 分别补充 left_foot、right_foot 站点
    status: completed
  - id: validate-sites
    content: 校验 XML 良构性、挂载位置与坐标落点，核对左右镜像
    status: completed
    dependencies:
      - add-head-sites
      - add-foot-sites
---


## 用户需求
参照样板 `robot_walk_backlash.xml`，为已完成大部分移植的 `Microduck-urdf-20261117/output.xml` 补齐剩余站点（site）定义。站点的数值不照抄样板、不留零占位，而是按新模型几何重算；挂载位置按已确认的映射执行。

## 产品概述
新模型 `output.xml` 由 SolidWorks 导出的 `myduck.urdf` 经 `transform.py` 生成，body/joint 命名与坐标系和样板完全不同。样板中共 7 个 site，目标文件目前仅有一个占位 `imu`。本次工作是在不改动用户已完成内容的前提下，把样板中站点体系完整迁移到新模型的对应 body 上，使新模型具备与样板一致的传感器参考框架。

## 核心功能
- 在头部末端 body `neck_Link_03` 内补充 `head_camera`（含配套 `camera` 元素）、`tof`、`head_imu`、`mouth_tip` 四个站点。
- 在左足末端 body `lleg_link_05` 内补充 `left_foot` 站点。
- 在右足末端 body `rleg_link_05` 内补充 `right_foot` 站点。
- 各站点 `pos/quat` 按新模型几何计算，使站点物理位置与样板语义一致（足底接触点、相机光心、头部 IMU 等），并保持 `group="3"` 属性及样板元素结构。
- 站点写入后通过 STL 包围盒与静态解析校验落点合理性。



## Tech Stack Selection
- 目标文件为 MuJoCo MJCF（XML）配置，`output.xml` 已是标准 MJCF，无需引入新框架或依赖。
- 计算/校验使用项目现有 Python 环境：`uv run python` + `numpy`（做 Kabsch 对齐与四元数运算），无需新增依赖。
- 几何校验数据源：`Microduck-urdf-20261117/meshes/*.STL`（二进制 STL，直接解析顶点包围盒）。

## Implementation Approach
站点本质是挂在 body 下的一个坐标系；两模型 body 坐标系不同，因此不能直接复制样板的局部 `pos/quat`。做法是：在零位姿（q=0）下求出样板各站点在世界系的位姿，再把该位姿映射到新模型对应 body 的局部系。

关键决策：全局刚体对齐残差约 4.2cm（新模型整体比样板低约 0.12m，且颈部链比样板高约 3~4cm），结果不可靠；改为**分链 Kabsch 对齐**（左腿、右腿、颈部各自对齐），残差降到约 1.5cm。公式：对每条链求最优刚体变换 (R, t) 使样板关节原点匹配新模型关节原点，再以 `local_pose = inv(T_new_body) @ (R, t) @ T_template_site` 得到站点局部位姿，四元数按 MuJoCo `w x y z` 顺序输出。

性能：一次性静态编辑，无运行时开销；对齐计算仅涉及十余个 body，复杂度 O(n)，可忽略。

## 已算好的参考值（实施起点，需结合网格微调）
左腿链（rms≈0.0155）→ `lleg_link_05`：
- left_foot: pos="0.003606 0.013152 -0.043506" quat="0.747415 0.664041 -0.009972 0.017936"

右腿链（rms≈0.0155）→ `rleg_link_05`：
- right_foot: pos="0.003718 -0.013218 -0.043508" quat="0.747407 -0.664051 -0.010419 -0.017647"

颈部链（rms≈0.0146）→ `neck_Link_03`：
- head_camera: pos="-0.062498 -0.043451 -0.010199" quat="0.251372 0.239253 0.02063 -0.937627"
- tof: pos="-0.050793 -0.06277 -0.010133" quat="0.251372 0.239253 0.02063 -0.937627"
- head_imu: pos="-0.0454 -0.033321 -0.000211" quat="0.192335 -0.493825 -0.163159 -0.83218"
- mouth_tip: pos="-0.055483 -0.041828 -0.033101" quat="0.251279 0.243342 0.021727 -0.936574"

## Implementation Notes
- 站点统一保留样板格式 `<site group="3" name="..." pos="..." quat="..."/>`，插在各 body 最后一个 geom 之后、闭合 `</body>` 之前。
- `head_camera` 需同时新增配套 `<camera name="head_camera" pos="..." quat="..."/>`：pos 复用该站点 pos；朝向按同一颈部链对齐把样板 camera 朝向变换到新体系，或先用样板字面的 "0 0 -1 0" 并在校验阶段确认朝向，避免只加 site 漏掉 camera。
- STL 包围盒（body 局部）：`neck_Link_03` x∈[-0.04725,0.10276]/y∈[-0.05,0.05]/z∈[0,0.051]；`lleg_link_05` 与 `rleg_link_05` x∈[-0.0215,0.0435]/z∈[-0.056,0.004]。足部站点落在足网格内；头部站点在 x 最小侧约超 1.5cm、z 下侧约超 1cm，与样板中相机/嘴尖本就突出网格的语义一致，属可接受残差，实施时结合网格做小幅微调即可。
- 不改动用户已完成部分：defaults、sensor 块、backlash、占位 `imu`、各 body/joint/geom。
- 已知遗留问题（本计划不改）：`<sensor>` 中 `root_angmom` 引用 `body="trunk_base"`，而新模型无该 body；若要求 MuJoCo 直接加载，需另行处理。因此本计划以 XML 良构性 + 静态解析校验为主，MuJoCo 加载校验为可选。
- 左右腿站点数值符号相反（x/y 镜像、quat 的 y 分量反号），符合样板对称性，写入后核对左右镜像关系。

## Architecture Design
单文件静态配置修改，无系统级组件关系变化，省略架构图。数据流：样板站点世界位姿 → 分链 Kabsch 对齐 → 新模型 body 局部位姿 → 写入 `output.xml` → STL/静态解析校验。站点与现有 `sensor` 块通过 `objname`/`site` 名称解耦，新增站点不破坏既有传感器定义。

## Directory Structure
本次仅修改一个文件：

/workspace/Microduck-urdf-20261117/
└── output.xml  # [MODIFY] 补充 6 个站点及 1 个 camera：在 neck_Link_03 的最后一个 geom 后插入 head_camera site + camera、tof、head_imu、mouth_tip；在 lleg_link_05 与 rleg_link_05 的最后一个 geom 后分别插入 left_foot、right_foot。保留 group="3"，不改动其余已完成内容。

只读参考（不修改）：`src/mjlab_microduck/robot/microduck/robot_walk_backlash.xml`（样板）、`Microduck-urdf-20261117/urdf/myduck.urdf`、`Microduck-urdf-20261117/urdf/Microduck-整体.SLDASM.csv`、`Microduck-urdf-20261117/meshes/*.STL`。

