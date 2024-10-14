import os
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from pycolmap import SceneManager

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

from .dnerf_loader import load_blender_data


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths

class DNeRFDataset:
    """D-NeRF dataset class."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
    ):
        images, poses, times, _, _, hwf, i_split = load_blender_data(data_dir, False, 1)
        train_num = len(i_split[0])
        val_num = len(i_split[1])
        test_num = len(i_split[2])

        [trainset, valset, testset] = np.split(images[...,:3], [train_num, train_num+val_num]) # delete alpha channel
        [trainpose, valpose, testpose] = np.split(poses, [train_num, train_num+val_num])
        [traintime, valtime, testtime] = np.split(times, [train_num, train_num+val_num])

        if split == "train":
            self.images = trainset
            self.poses = trainpose
            self.times = traintime
        elif split == "val":
            self.images = valset
            self.poses = valpose
            self.times = valtime
        elif split == "test":
            self.images = testset
            self.poses = testpose
            self.times = testtime

        self.hwf = hwf
        self.i_split = i_split

        # flip yz axis (Blender coord -> COLMAP coord)
        # breakpoint()
        rotmat = self.poses[:,:3,:3]
        rotmat[:,:,1] = -rotmat[:,:,1]
        rotmat[:,:,2] = -rotmat[:,:,2]
        self.poses[:,:3,:3] = rotmat

        # size of the scene measured by cameras
        camera_locations = self.poses[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

    def __len__(self):
        return len(self.images)

    def get_intrinsics(self, height, width, focal):
        #TODO
        cx = (width-1) / 2
        cy = (height-1) / 2

        intrinsics = np.array(
            [
                [focal, 0, cx],
                [0, focal, cy],
                [0, 0, 1]
            ]
        )
        return intrinsics

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image = self.images[index]
        # camera_id = self.parser.camera_ids[index]
        # K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        [h, w, f] = self.hwf
        K = self.get_intrinsics(h,w,f)
        camtoworlds = self.poses[index]

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float() * 255,
            "image_id": index,
        }

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./dnerf_dataset/lego")
    args = parser.parse_args()

    # Parse DNeRF data.
    dataset = DNeRFDataset(data_dir=args.data_dir, split="train")
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/train.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):
        # breakpoint()
        image = data["image"].numpy()
        writer.append_data(image)
    writer.close()
