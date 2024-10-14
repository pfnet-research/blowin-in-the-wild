import random

import numpy as np
import torch
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import torch.nn.functional as F



class CirclePointMover:
    def __init__(self, r, speed=0.1):
        self.r = r
        self.speed = speed
        self.t = 0
        self.position = (r*np.cos(self.t), r*np.sin(self.t))

    def update(self):
        t = self.t
        v1 = self.r * np.cos(t) + 0.5
        v2 = self.r * np.sin(t) + 0.5
        self.position = (v1,v2)

        t += self.speed
        self.t = t

    def get_position(self):
        return self.position

class LissajousPointMover:
    def __init__(self, r, speed=0.05):
        self.r = r
        self.speed = speed
        self.t = 0
        self.position = (0,0)
    
    def update(self):
        t = self.t
        r = self.r * (0.25 * np.cos(t) + 0.5)
        # r = self.r * random.uniform(0.2, 1.2)
        v1 = r * np.cos(7*t) + 0.5
        v2 = r * np.sin(13*t) + 0.5
        self.position = (v1,v2)

        t += 1 * self.speed
        self.t = t

    def get_position(self):
        return self.position

class MomentumPointMover:
    def __init__(self, r, center=(0.5, 0.5), momentum=0.4, randomness=0.2):
        self.r = r
        self.center = np.array(center, dtype=float)
        self.position = np.random.rand(2) * r * 0.5 + center
        self.momentum = momentum
        self.randomness = randomness
        self.velocity = np.zeros(2)

    def update(self):
        center_vec = self.center - self.position
        self.velocity += center_vec * self.momentum
        self.velocity += np.random.randn(2) * self.randomness
        self.position += self.velocity

    def get_position(self):
        return self.position

class MomentumPointMover:
    def __init__(self, r, center=(0.5, 0.5), momentum=0.4, randomness=0.2):
        self.r = r
        self.center = np.array(center, dtype=float)
        self.position = np.random.rand(2) * r * 0.5 + center
        self.momentum = momentum
        self.randomness = randomness
        self.velocity = np.zeros(2)

    def update(self):
        center_vec = self.center - self.position
        center_dist = np.clip(np.linalg.norm(center_vec), 1e-8, None)
        center_dir = center_vec / center_dist
        # self.velocity -= self.velocity * self.momentum
        self.velocity += center_vec * self.momentum
        self.velocity += np.random.randn(2) * self.randomness

        self.position += self.velocity
        # if np.linalg.norm(self.center - self.position) > self.r:
        #     self.position = self.center + (self.position - self.center) / np.linalg.norm(self.position - self.center) * self.r

    def get_position(self):
        return self.position

class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def scipy_knn(pts, num_knn):
    tree = cKDTree(pts)
    distances, indices = tree.query(pts, k=num_knn + 1)
    # Exclude the first neighbor, which is the point itself
    return distances[:, 1:], indices[:, 1:]


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def o3d_knn(pts, num_knn):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)

def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()