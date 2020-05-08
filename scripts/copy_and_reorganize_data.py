import sys
import os
from pathlib import Path
from shutil import copyfile, rmtree
from tqdm import tqdm


def copy_and_reorganize_data_shapenet(path_src, path_tgt):
    src = Path(path_src)
    dest = Path(path_tgt)
    for s in tqdm(list(src.iterdir())):
        if not (dest / (s.name + ".npz")).exists():
           if (dest / s.name).exists():
              rmtree(dest / s.name)
           print(s.name)
           continue
        (dest / s.name).mkdir(exist_ok=True)
        depth_dir = dest / s.name / "depth"
        image_dir = dest / s.name / "image"
        intrinsic_dir = dest / s.name / "intrinsic"
        pose_dir = dest / s.name / "pose"
        input_image_dir = dest / s.name / "input_image"
        visualize_dir = dest / s.name / "visualize"
        visualize_depth_dir = dest / s.name / "visualize" / "depth"
        visualize_image_dir = dest / s.name / "visualize" / "image"
        points_path = dest / s.name / "pointcloud.npz"
        depth_dir.mkdir(exist_ok=True)
        intrinsic_dir.mkdir(exist_ok=True)
        pose_dir.mkdir(exist_ok=True)
        image_dir.mkdir(exist_ok=True)
        input_image_dir.mkdir(exist_ok=True)
        visualize_dir.mkdir(exist_ok=True)
        visualize_depth_dir.mkdir(exist_ok=True)
        visualize_image_dir.mkdir(exist_ok=True)
        for i in range(39):
            copyfile(str(s / (s.name + f"_intr_{i}.txt")), str(intrinsic_dir / f"{i:03d}.txt"))
            copyfile(str(s / (s.name + f"_pose_{i}.txt")), str(pose_dir / f"{i:03d}.txt"))
            if 0 <= i <= 9:
                copyfile(str(s / (s.name + f"_depth_{i}.png")), str(depth_dir / f"{i:03d}.png"))
                copyfile(str(s / (s.name + f"_color_{i}.png")), str(image_dir / f"{i:03d}.png"))
            if 10 <= i <= 33:
                copyfile(str(s / (s.name + f"_color_{i}.png")), str(input_image_dir / f"{i:03d}.png"))
            if 34 <= i <= 38:
                copyfile(str(s / (s.name + f"_depth_{i}.png")), str(visualize_depth_dir / f"{i:03d}.png"))
                copyfile(str(s / (s.name + f"_color_{i}.png")), str(visualize_image_dir / f"{i:03d}.png"))
        copyfile(str(dest / (s.name + ".npz")), str(points_path))


def copy_and_reorganize_matterport(frames_directory, frame_associations, dest):
    dest = Path(dest)
    frame_associations = Path(frame_associations)
    for x in tqdm([y for y in frame_associations.iterdir() if "__cmp__" in y.name]):
        room = x.name.split("_")[0]
        basename = x.name.split(".")[0].split("__cmp__")[0] + "_" + x.name.split(".")[0].split("__cmp__")[1]
        frames = x.read_text().split("\n")
        if len(frames) <= 2:
            continue
        (dest / basename).mkdir(exist_ok=True)
        depth_dir = dest / basename / "depth"
        image_dir = dest / basename / "image"
        camera_dir = dest / basename / "camera"
        input_image_dir = dest / basename / "input_image"
        visualize_dir = dest / basename / "visualize"
        visualize_depth_dir = dest / basename / "visualize" / "depth"
        visualize_image_dir = dest / basename / "visualize" / "image"
        points_path = dest / basename / "pointcloud.npz"
        depth_dir.mkdir(exist_ok=True)
        camera_dir.mkdir(exist_ok=True)
        image_dir.mkdir(exist_ok=True)
        input_image_dir.mkdir(exist_ok=True)
        visualize_dir.mkdir(exist_ok=True)
        visualize_depth_dir.mkdir(exist_ok=True)
        visualize_image_dir.mkdir(exist_ok=True)
        for f in frames:
            copyfile(frames_directory / room / "color" / f"{f}.jpg", image_dir / f"{f}.jpg")
            copyfile(frames_directory / room / "color" / f"{f}.jpg", input_image_dir / f"{f}.jpg")
            copyfile(frames_directory / room / "depth" / f"{f}.png", depth_dir / f"{f}.png")
            copyfile(frames_directory / room / "camera" / f"{f}.txt", camera_dir / f"{f}.txt")
        for f in frames[:3]:
            copyfile(frames_directory / room / "color" / f"{f}.jpg", visualize_image_dir / f"{f}.jpg")
            copyfile(frames_directory / room / "depth" / f"{f}.png", visualize_depth_dir / f"{f}.png")
        copyfile(str(dest / (basename + ".npz")), str(points_path))


if __name__ == '__main__':
    # path_src = sys.argv[1]
    # path_tgt = sys.argv[2]
    # copy_and_reorganize_data(path_src, path_tgt)
    path_frames = sys.argv[1]
    path_associations = sys.argv[2]
    path_tgt = sys.argv[3]
    copy_and_reorganize_matterport(path_frames, path_associations, path_tgt)


