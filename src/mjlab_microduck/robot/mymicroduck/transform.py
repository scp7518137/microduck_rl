# import mujoco

# modle = mujoco.MjModel.from_xml_path('./urdf/myduck.urdf')

# mujoco.mj_saveLastXML('output.xml', modle)

import mujoco

m = mujoco.MjModel.from_xml_path("./robot_walk_backlash.xml") # type: ignore
print(m.body_mass[1], m.body_mass.sum())