import os
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import json
import tqdm
from pycolmap import SceneManager

# from .normalize import (
from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths

def make_transform_matrix(rotation_matrix, position_vector):
    R = rotation_matrix
    P = position_vector

    # Start with a 4x4 identity matrix
    transform_matrix = np.eye(4)

    # Populate the upper 3x3 block with the rotation matrix
    transform_matrix[:3, :3] = R

    # Populate the last column (except the last row) with the position vector
    transform_matrix[:3, 3] = P
    
    return transform_matrix

def load_iphone_data(data_dir):
    splits = ["train", "val"]
    metas = {}
    for s in splits:
        with open(os.path.join(data_dir, "splits", "{}.json".format(s)), "r") as fp:
            metas[s] = json.load(fp)

    i_split = []
    imgs = []
    poses = []
    hwfs = []
    principal_points = []
    times = []

    for s in splits:
        meta = metas[s]

        print("reading iphone dataset...")
        for frame, t in zip(tqdm.tqdm(meta["frame_names"]), meta["time_ids"]):
            # scale: hardcoding now
            # read imgs
            imgname = os.path.join(data_dir, "rgb", "2x", frame+".png")
            imgs.append(imageio.imread(imgname))

            # read poses, hwfs
            posename = os.path.join(data_dir, "camera", frame+".json")
            with open(posename, "r") as p:
                posefile = json.load(p)
                h,w = posefile["image_size"]
                f = posefile["focal_length"]
                principal_point = posefile["principal_point"]

                # convert w2c -> c2w
                orientation = np.array(posefile["orientation"])
                position = np.array(posefile["position"])
                camtoworld = np.linalg.inv(posefile["orientation"])
                # position = -orientation @ position
                transform = make_transform_matrix(camtoworld, posefile["position"])
                # transform = make_transform_matrix(orientation, position)
                hwfs.append([h,w,f])
                principal_points.append(principal_point)
                poses.append(transform)

            times.append(t)

        i_split.append(len(meta["frame_names"]))

    imgs = np.array(imgs)

    return imgs, poses, times, hwfs, i_split, principal_points

class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        # test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        # self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective"
            ), f"Only support perspective camera model, got {type_}"

            params_dict[camera_id] = params

            # image size
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)

        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load images.
        if factor > 1:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        # breakpoint()
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                K, params, (width, height), 0
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                K, params, None, K_undist, (width, height), cv2.CV_32FC1
            )
            self.Ks_dict[camera_id] = K_undist
            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.roi_undist_dict[camera_id] = roi_undist

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class IphoneDataset:
    """Iphone dataset class."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        use_colmap: bool = True,
    ):
        self.use_colmap = use_colmap
        if use_colmap:
            self.parser = Parser(
                data_dir=os.path.join(data_dir, "colmap"),
                factor=4,
                normalize=True,
            )
            camera_ids = self.parser.camera_ids
            poses = self.parser.camtoworlds
            images = []
            params = []
            Ks = []
            i = 0
            for camera_id in tqdm.tqdm(camera_ids):
                image = imageio.imread(self.parser.image_paths[i])
                params.append(self.parser.params_dict[camera_id])
                Ks.append(self.parser.Ks_dict[camera_id])
                if len(self.parser.params_dict[camera_id]) > 0:
                    # If images are distorted, undistort them.
                    mapx, mapy = (
                        self.parser.mapx_dict[camera_id],
                        self.parser.mapy_dict[camera_id],
                    )
                    images = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
                    x, y, w, h = self.parser.roi_undist_dict[camera_id]
                    image = image[y : y + h, x : x + w]
                
                images.append(image)
                i += 1

            images = np.array(images)
            _, _, times, _, i_split, _ = load_iphone_data(data_dir)
            self.Ks = Ks
        else:
            images, poses, times, hwfs, i_split, principal_points = load_iphone_data(data_dir)
            self.hwfs = hwfs
            self.principals = principal_points

        train_num = i_split[0]
        val_num = i_split[1]

        [trainset, valset, _] = np.split(images[...,:3], [train_num,train_num+val_num])
        [trainpose, valpose, _] = np.split(poses, [train_num,train_num+val_num])
        [traintime, valtime, _] = np.split(times, [train_num,train_num+val_num])

        if split == "train":
            self.images = trainset
            self.poses = trainpose
            self.times = traintime
        elif split == "val":
            self.images = valset
            self.poses = valpose
            self.times = valtime

        
        self.i_split = i_split
        
        # size of the scene measured by cameras
        camera_locations = self.poses[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        # breakpoint()
        self.scene_scale = np.max(dists)

    def __len__(self):
        return len(self.images)

    def get_intrinsics(self, height, width, focal, cx, cy):
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
        if self.use_colmap:
            K = self.Ks[index]
        else:
            [h, w, f] = self.hwfs[index]
            [cx, cy] = self.principals[index]
            K = self.get_intrinsics(h,w,f,cx,cy)
        camtoworlds = self.poses[index]

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": index,
        }

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./iphone_dataset/paper-windmill")
    args = parser.parse_args()

    # Parse iphone data.
    dataset = IphoneDataset(data_dir=args.data_dir, split="train")
    print(f"Dataset: {len(dataset)} images.")
    # print(load_iphone_data(args.data_dir))


    # dataset = DNeRFDataset(data_dir=args.data_dir, split="train")
    # print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/iphone_colmap.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):
        # breakpoint()
        image = data["image"].numpy()
        writer.append_data(image)
    writer.close()
