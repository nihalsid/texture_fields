import sys
import os
from pathlib import Path
from shutil import copyfile


def copy_and_reorganize_data(path_src, path_tgt):
    src = Path(path_src)
    dest = Path(path_tgt)
    for s in src.iterdir():
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


if __name__ == '__main__':
    root_path = Path(sys.argv[1])
    all_items = [x.name for x in root_path.iterdir()]
    all_items = sorted(all_items, key=lambda x: int(x.split(".")[0].split("_")[-1]))
    ext = all_items[0].split(".")[-1]
    for x in all_items:
        os.rename(root_path / x, root_path / f"{int(x.split('.')[0].split('_')[-1]):03d}.{ext}")
