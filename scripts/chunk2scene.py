import struct
import numpy as np
import os
import trimesh
from tqdm import tqdm

chunk_base = None
sdf_base = None
sdf_scene_base = None
scene_base = None
dest = None


def read_w2g(sdf_path):
    fin = open(sdf_path, 'rb')
    fin.read(28)
    world2grid = np.array(struct.unpack('f' * 4 * 4, fin.read(4 * 4 * 4))).reshape((4, 4)).astype(np.float32)
    world2grid[:3, :3] = np.eye(3)
    grid2world = np.linalg.inv(world2grid)
    fin.close()
    return grid2world[:3, 3]


def chunk_to_scene():
    all_chunks = sorted(os.listdir(chunk_base))
    for scene in tqdm(os.listdir(scene_base)):
        # if scene != "2t7WUuJeko7_room0__0__pred.ply":
        #     continue
        chunk_b_path = os.path.join(sdf_scene_base, scene.split("__")[0] + "__0__.sdf")
        base_g2w = read_w2g(chunk_b_path)
        meshes = []
        for chunk in all_chunks:
            chunk_room = "_".join(chunk.split("_")[:-1])
            if scene.split("__")[0] == chunk_room:
                mesh = trimesh.load(os.path.join(chunk_base, chunk), process=True)
                chunk_c_path = os.path.join(sdf_base, "_".join(chunk.split("_")[:-1])+"__cmp__"+chunk.split("_")[-1].split(".")[0]+".sdf")
                mesh.apply_translation(read_w2g(chunk_c_path) - base_g2w)
                meshes.append(mesh)
        if len(meshes) > 0:
            union = trimesh.util.concatenate(meshes)
            union.export(os.path.join(dest, scene.split(".")[0]+".obj"), "obj")

if __name__ == "__main__":
    import sys
    chunk_base = sys.argv[1]
    scene_base = sys.argv[2]
    sdf_base = sys.argv[3]
    sdf_scene_base = sys.argv[4]
    dest = sys.argv[5]
    os.makedirs(dest, exist_ok=True)
    chunk_to_scene()
