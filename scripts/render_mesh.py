import trimesh
import pyrender
import numpy as np
import os
import math
from PIL import Image
from tqdm import tqdm


def create_raymond_lights():
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))
    return nodes


def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0
    r = np.eye(4, 4)
    R = np.matmul(np.matmul(Rz,Ry),Rx)
    r[:3, :3] = R[:3, :3]
    return r


def read_camera(root_path, model_name, index):

    def read_matrix_file(mat_path):
        with open(mat_path, "r") as fptr:
            lines = fptr.read().splitlines()
            lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
            mat = np.asarray(lines, dtype=np.float32)
            return mat

    path_to_sdf = os.path.join(root_path, "sdf", model_name+".npz")
    path_to_pose = os.path.join(root_path, model_name, "pose",  f"{index:03d}.txt")
    path_to_intr = os.path.join(root_path, model_name, "intrinsic",  f"{index:03d}.txt")
    world2grid = np.load(path_to_sdf)['world2grid'].reshape(4, 4).astype(np.float32)
    pose = read_matrix_file(path_to_pose)
    intrinsics = read_matrix_file(path_to_intr)
    cam_W = np.matmul(np.matmul(world2grid, pose), make_rotate(math.radians(180), 0, 0))
    cam_K = intrinsics
    return cam_W, cam_K


def get_mesh_path(mesh_root, model_name, method):
    if method == 'texturefields':
        return os.path.join(mesh_root, model_name+".obj")
    if method.startswith('pifu'):
        view_idx = int(method.split("_")[1])
        return os.path.join(mesh_root, model_name + f"_{view_idx:03d}_pred.obj")
    return os.path.join(mesh_root, model_name, "model_c.obj")


def get_mesh_list(mesh_root, method):
    if method == 'pifu':
        return [x.split("_000_pred.obj")[0] for x in os.listdir(mesh_root) if x.endswith("_000_pred.obj")]
    if method == 'texturefields':
        return [x.split(".")[0] for x in os.listdir(mesh_root)]
    return list(os.listdir(mesh_root))


def render_mesh_with_camera(method, mesh_root, param_root, model_name, camera_index):
    view_dims = [240, 320]
    trimesh_obj = trimesh.load(get_mesh_path(mesh_root, model_name, method), process=True)
    mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
    extrinsic, intrinsic = read_camera(param_root, model_name, camera_index)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera_intrinsic = pyrender.IntrinsicsCamera(intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], zfar=6000)
    scene.add(camera_intrinsic, pose=extrinsic)
    for n in create_raymond_lights():
        scene.add_node(n, scene.main_camera_node)
    # pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size=view_dims[::-1])
    r = pyrender.OffscreenRenderer(view_dims[1], view_dims[0])
    color, _ = r.render(scene)
    return Image.fromarray(color)


def render_texturefields(mesh_root, param_root, dest_root):
    meshlist = get_mesh_list(mesh_root, "texturefields")
    viewlist = list(range(24))
    for mesh in meshlist:
        dest_dir = os.path.join(dest_root, mesh)
        os.makedirs(dest_dir, exist_ok=True)
        for view_idx in viewlist:
            render_mesh_with_camera("texturefields", mesh_root, param_root, mesh, view_idx).save(os.path.join(dest_dir, f"{view_idx:03}.png"))


def render_gt(mesh_root, param_root, dest_root):
    meshlist = get_mesh_list(mesh_root, "gt")
    viewlist = list(range(24))
    for mesh in meshlist:
        dest_dir = os.path.join(dest_root, mesh)
        os.makedirs(dest_dir, exist_ok=True)
        for view_idx in viewlist:
            render_mesh_with_camera("gt", mesh_root, param_root, mesh, view_idx).save(os.path.join(dest_dir, f"{view_idx:03}.png"))


def render_pifu(mesh_root, param_root, dest_root):
    for pifu_view in tqdm(range(16)):
        meshlist = get_mesh_list(mesh_root, "pifu")
        viewlist = list(range(24))
        for mesh in tqdm(meshlist):
            dest_dir = os.path.join(dest_root, mesh, f"{pifu_view:03d}")
            os.makedirs(os.path.join(dest_root, mesh), exist_ok=True)
            os.makedirs(dest_dir, exist_ok=True)
            for view_idx in viewlist:
                render_mesh_with_camera(f"pifu_{pifu_view}", mesh_root, param_root, mesh, view_idx).save(os.path.join(dest_dir, f"{view_idx:03}.png"))


if __name__ == '__main__':
    import sys
    mesh_root = sys.argv[1]
    param_root = sys.argv[2]
    dest_root = sys.argv[3]
    method = sys.argv[4]
    if method == "pifu":
        render_pifu(mesh_root, param_root, dest_root)
