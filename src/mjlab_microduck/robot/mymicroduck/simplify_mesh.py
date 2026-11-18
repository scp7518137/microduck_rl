"""批量抽稀 STL，模拟 onshape-to-robot 的 simplify_stls + max_stl_size=1.0。

运行：
    cd Microduck-urdf-20261117
    uv run --with trimesh --with fast-simplification python simplify_stl.py

输出目录：meshes_simplified/（原 meshes/ 不动，方便对比）
"""
from pathlib import Path
import trimesh
from fast_simplification import simplify

SRC = Path("meshes")
DST = Path("meshes_simplified")
TARGET_FACES = 18000  # 约 0.9 MB 二进制 STL；想要更激进就调小

DST.mkdir(parents=True, exist_ok=True)

for src in sorted(SRC.glob("*.STL")) + sorted(SRC.glob("*.stl")):
    mesh = trimesh.load(src, force="mesh")
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        print(f"跳过 {src.name}: 无三角面")
        continue

    before = len(mesh.faces)
    if before <= TARGET_FACES:
        # 面数本来就够少，原样复制（保留文件名大小写）
        (DST / src.name).write_bytes(src.read_bytes())
        print(f"{src.name:28s} {before:>8,} 面  (无需简化)")
        continue

    verts, faces = simplify(
        mesh.vertices, mesh.faces,
        target_count=TARGET_FACES,
        agg=1.0,          # 0~1，越大越激进地合并顶点
    )
    out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    out.export(DST / src.name)  # 保持原文件名，output.xml 不用改引用
    print(f"{src.name:28s} {before:>8,} -> {len(faces):>8,} 面")
