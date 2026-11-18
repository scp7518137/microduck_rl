import mujoco

modle = mujoco.MjModel.from_xml_path('./urdf/myduck.urdf')

mujoco.mj_saveLastXML('output.xml', modle)