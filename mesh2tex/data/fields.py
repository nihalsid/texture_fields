import os
import glob
import random
import numpy as np
import trimesh
import imageio
import struct
from mesh2tex.data.core import Field


# Make sure loading xlr works
imageio.plugins.freeimage.download()


# Basic index field
class IndexField(Field):
    def load(self, model_path, idx):
        return idx

    def check_complete(self, files):
        return True


class MeshField(Field):
    def __init__(self, folder_name, transform=None):
        self.folder_name = folder_name
        self.transform = transform

    def load(self, model_path, idx):
        folder_path = os.path.join(model_path, self.folder_name)
        file_path = os.path.join(folder_path, 'model.off')
        mesh = trimesh.load(file_path, process=False)
        if self.transform is not None:
            mesh = self.transform(mesh)

        data = {
            'vertices': np.array(mesh.vertices),
            'faces': np.array(mesh.faces),
        }

        return data

    def check_complete(self, files):
        complete = (self.folder_name in files)
        return complete

# Image field
class ImagesField(Field):
    def __init__(self, folder_name, transform=None,
                 extension='jpg', random_view=True,
                 with_camera=False,
                 imageio_kwargs=dict()):
        self.folder_name = folder_name
        self.transform = transform
        self.extension = extension
        self.random_view = random_view
        self.with_camera = with_camera
        self.imageio_kwargs = dict()

    def load(self, model_path, idx):
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
        files.sort()
        if self.random_view:
            idx_img = random.randint(0, len(files)-1)
        else:
            idx_img = 0
        filename = files[idx_img]

        image = imageio.imread(filename, **self.imageio_kwargs)
        image = np.asarray(image)

        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)
            image = np.concatenate([image, image, image], axis=2)  

        if image.shape[2] == 4:
            image = image[:, :, :3]

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255
        else:
            image = image.astype(np.float32)

        if self.transform is not None:
            image = self.transform(image)
        image = image.transpose(2, 0, 1)
        data = {
            None: image
        }

        if self.with_camera:
            camera_file = os.path.join(folder, 'cameras.npz')
            camera_dict = np.load(camera_file)
            Rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
            K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
            data['world_mat'] = Rt
            data['camera_mat'] = K

        return data

    def check_complete(self, files):
        complete = (self.folder_name in files)
        # TODO: check camera
        return complete


# 3D Fields
class PointCloudField(Field):
    def __init__(self, file_name, transform=None, with_transforms=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms

    def load(self, model_path, idx):
        file_path = os.path.join(model_path, self.file_name)

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        data = {
            None: points.T,
            'normals': normals.T,
        }

        if self.with_transforms:
            data['loc'] = pointcloud_dict['loc'].astype(np.float32)
            data['scale'] = pointcloud_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        complete = (self.file_name in files)
        return complete


class DepthImageVisualizeField(Field):
    def __init__(self, folder_name_img, folder_name_depth, transform_img=None, transform_depth=None,
                 extension_img='jpg', extension_depth='exr', random_view=True,
                 with_camera=False, camera_mode='vanilla', sdf_path=None,
                 imageio_kwargs=dict()):
        self.folder_name_img = folder_name_img
        self.folder_name_depth = folder_name_depth
        self.transform_depth = transform_depth
        self.transform_img = transform_img
        self.extension_img = extension_img
        self.extension_depth = extension_depth
        self.random_view = random_view
        self.with_camera = with_camera
        self.camera_mode = camera_mode
        self.sdf_path = sdf_path
        self.imageio_kwargs = dict()

    def load(self, model_path, idx):
        folder_img = os.path.join(model_path, self.folder_name_img)
        files_img = glob.glob(os.path.join(folder_img, '*.%s' % self.extension_img))
        files_img.sort()
        folder_depth = os.path.join(model_path, self.folder_name_depth)
        files_depth = glob.glob(os.path.join(folder_depth, '*.%s' % self.extension_depth))
        files_depth.sort()
        if self.random_view:
            idx_img = random.randint(0, len(files_img)-1)
        else:
            idx_img = 0
        
        image_all = []
        depth_all = []
        Rt = []
        K = []

        for i in range(len(files_img)):
            filename_img = files_img[i]
            filename_depth = files_depth[i]

            image = imageio.imread(filename_img, **self.imageio_kwargs)
            image = np.asarray(image)
            
            if image.shape[2] == 4:
                image = image[:,:,:3]

            depth = imageio.imread(filename_depth, **self.imageio_kwargs)

            depth = np.asarray(depth)
            if len(depth.shape) == 2:
                depth = depth.reshape((depth.shape[0], depth.shape[1], 1)).astype(np.float32)
                # chair dataset, should be parameterized
                depth = depth / 1000
            
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255
            else:
                image = image.astype(np.float32)

            if self.transform_img is not None:
                image = self.transform_img(image)

            if self.transform_depth is not None:
                depth = self.transform_depth(depth)

            image = image.transpose(2, 0, 1)
            depth = depth.transpose(2, 0, 1)
            image_all.append(image)
            depth_all.append(depth)

            if self.camera_mode == 'shapenet':
                _Rt, _K = read_camera(self.sdf_path, model_path, os.path.basename(filename_img).split(".")[0])
                Rt.append(_Rt)
                K.append(_K)
            elif self.camera_mode == 'matterport':
                _Rt, _K = read_camera_matterport(self.sdf_path, model_path, os.path.basename(filename_img).split(".")[0])
                Rt.append(_Rt)
                K.append(_K)
            else:
                camera_file = os.path.join(folder_depth, 'cameras.npz')
                camera_dict = np.load(camera_file)
                Rt.append(camera_dict['world_mat_%d' % i].astype(np.float32))
                K.append(camera_dict['camera_mat_%d' % i].astype(np.float32))

        data = {'img': np.stack(image_all), 'depth': np.stack(depth_all), 'world_mat': np.stack(Rt), 'camera_mat': np.stack(K)}

        return data


# Image field
class DepthImageField(Field):
    def __init__(self, folder_name_img, folder_name_depth, transform_img=None, transform_depth=None,
                 extension_img='jpg', extension_depth='exr', random_view=True,
                 with_camera=False, camera_mode='vanilla', sdf_path=None,
                 imageio_kwargs=dict()):
        self.folder_name_img = folder_name_img
        self.folder_name_depth = folder_name_depth
        self.transform_depth = transform_depth
        self.transform_img = transform_img
        self.extension_img = extension_img
        self.extension_depth = extension_depth
        self.random_view = random_view
        self.with_camera = with_camera
        self.camera_mode = camera_mode
        self.sdf_path = sdf_path
        self.imageio_kwargs = dict()

    def load(self, model_path, idx):
        folder_img = os.path.join(model_path, self.folder_name_img)
        files_img = glob.glob(os.path.join(folder_img, '*.%s' % self.extension_img))
        files_img.sort()
        folder_depth = os.path.join(model_path, self.folder_name_depth)
        files_depth = glob.glob(os.path.join(folder_depth, '*.%s' % self.extension_depth))
        files_depth.sort()
        if self.random_view:
            idx_img = random.randint(0, len(files_img)-1)
        else:
            idx_img = 0

        filename_img = files_img[idx_img]
        filename_depth = files_depth[idx_img]

        image = imageio.imread(filename_img, **self.imageio_kwargs)
        image = np.asarray(image)
        if image.shape[2] == 4:
            image = image[:,:,:3]

        depth = imageio.imread(filename_depth, **self.imageio_kwargs)

        depth = np.asarray(depth)

        if len(depth.shape) == 2:
            depth = depth.reshape((depth.shape[0], depth.shape[1], 1)).astype(np.float32)
            # chair dataset, should be parameterized
            depth = depth / 1000
        
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255
        else:
            image = image.astype(np.float32)

        if self.transform_img is not None:
            image = self.transform_img(image)

        if self.transform_depth is not None:
            depth = self.transform_depth(depth)

        image = image.transpose(2, 0, 1)
        #TODO adapt depth transpose
        depth = depth.transpose(2, 0, 1)

        data = {
            'img': image,
            'depth': depth
            }

        if self.with_camera:
            if self.camera_mode == 'shapenet':
                Rt, K = read_camera(self.sdf_path, model_path, os.path.basename(filename_img).split(".")[0])
                data['world_mat'] = Rt
                data['camera_mat'] = K
            elif self.camera_mode == 'matterport':
                Rt, K = read_camera_matterport(self.sdf_path, model_path, os.path.basename(filename_img).split(".")[0])
                data['world_mat'] = Rt
                data['camera_mat'] = K
            else:
                camera_file = os.path.join(folder_depth, 'cameras.npz')
                camera_dict = np.load(camera_file)
                Rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
                K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
                data['world_mat'] = Rt
                data['camera_mat'] = K

        return data

    def check_complete(self, files):
        complete = (self.folder_name_img in files)
        # TODO: check camera
        return complete


def read_camera(sdf_path, model_path, filename):
    def read_matrix_file(mat_path):
        with open(mat_path, "r") as fptr:
            lines = fptr.read().splitlines()
            lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
            mat = np.asarray(lines, dtype=np.float32)
            return mat
    path_to_sdf = os.path.join(sdf_path, os.path.basename(model_path)+".npz")
    path_to_pose = os.path.join(model_path, "pose", filename + ".txt")
    path_to_intr = os.path.join(model_path, "intrinsic", filename + ".txt")
    world2grid = np.load(path_to_sdf)['world2grid'].reshape(4, 4).astype(np.float32)
    pose = read_matrix_file(path_to_pose)
    intrinsics = adjust_intrinsic(read_matrix_file(path_to_intr), (240, 320), (128, 128))
    # cam_W = torch.FloatTensor(np.eye(4, dtype=np.float32)[:3, :4]).unsqueeze(0)
    cam_W = np.matmul(world2grid, pose)[:3, :4]
    cam_K = intrinsics[:3, :4]
    return cam_W, cam_K


def read_camera_matterport(sdf_path, model_path, filename):
    def read_matrix_file(mat_path):
        with open(mat_path, "r") as fptr:
            lines = fptr.read().splitlines()
            lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
            pose = np.asarray(lines[:4], dtype=np.float32)
            intrinsic = np.asarray(lines[4:], dtype=np.float32)
            return pose, intrinsic

    def read_sdf(sdf_path):
        fin = open(sdf_path, 'rb')
        fin.read(28)
        world2grid = struct.unpack('f' * 4 * 4, fin.read(4 * 4 * 4))
        fin.close()
        return np.array(world2grid).reshape((4, 4)).astype(np.float32)
    basename = os.path.basename(model_path)
    path_to_sdf = os.path.join(sdf_path, basename.split("_")[0]+"_"+basename.split("_")[1]+"__cmp__"+basename.split("_")[2]+".sdf")
    path_to_cam = os.path.join(model_path, "camera", filename + ".txt")
    world2grid = read_sdf(path_to_sdf)
    pose, intrinsics = read_matrix_file(path_to_cam)
    intrinsics = adjust_intrinsic(intrinsics, (256, 320), (128, 128))
    # cam_W = torch.FloatTensor(np.eye(4, dtype=np.float32)[:3, :4]).unsqueeze(0)
    cam_W = np.matmul(world2grid, pose)[:3, :4]
    cam_K = intrinsics[:3, :4]
    return cam_W, cam_K


def adjust_intrinsic(intrinsic, old_shape, new_shape):
    intrinsic[0, 0] *= (new_shape[1] / old_shape[1])
    intrinsic[1, 1] *= (new_shape[0] / old_shape[0])
    intrinsic[0, 2] *= ((new_shape[1] - 1) / (old_shape[1] - 1))
    intrinsic[1, 2] *= ((new_shape[0] - 1) / (old_shape[0] - 1))
    return intrinsic