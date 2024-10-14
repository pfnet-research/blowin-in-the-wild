import torch
from torch import Tensor
import numpy as np
from sklearn import config_context
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union
from gsplat.rendering import rasterization


def bilinear_interpolation(grid: Tensor, pts: Tensor):
    x = pts[0]
    y = pts[1]

    x1 = torch.floor(x).long()
    x2 = x1 + 1
    y1 = torch.floor(y).long()
    y2 = y1 + 1

    Q11 = grid[x1, y1]
    Q12 = grid[x1, y2]
    Q21 = grid[x2, y1]
    Q22 = grid[x2, y2]

    interp_val = (Q11*(x2-x)*(y2-y) + Q21*(x-x1)*(y2-y) + Q12*(x2-x)*(y-y1) + Q22*(x-x1)*(y-y1))

    return interp_val

class MotionBaker:
    """Class to bake transforms."""
    def __init__(self, splats, pca, cfg, mlp, use_baking):
        self.baked_transforms = []
        self.splats = splats
        self.pca = pca
        self.cfg = cfg
        self.transform_mlp = mlp
        self.use_baking = use_baking
        self.gaussian_num = splats["means"].shape[0]

        self.baking_flag = False

    def bake_motion_mlp(self, shot_embeddings_2d):
        print("baking mlp approximation...")
        self.shot_embeddings_2d = shot_embeddings_2d
        means = self.splats["means"].clone()
        quats = self.splats["quats"].clone()
        gaussian_embeddings = self.splats["gaussian_embeddings"]

        if self.splats["shot_embeddings"].shape[1] == 2:
                embedding_expanded = self.shot_embeddings_2d.repeat(len(means), 1)
        elif self.pca is not None:
            with config_context(array_api_dispatch=True):
                shot_embeddings_nd = self.pca.inverse_transform(self.shot_embeddings_2d)
                embedding_expanded = shot_embeddings_nd.repeat(len(means), 1)
        if self.cfg.use_rotationMLP:
            position_offset, quarternion_offset = self.transform_mlp(gaussian_embeddings, embedding_expanded, means, quats)
            means = means + position_offset
            if self.cfg.use_quats_offset_with_normalization:
                quats = torch.nn.functional.normalize(quats, dim=1) + quarternion_offset
            else:
                quats = torch.nn.functional.normalize(quats + quarternion_offset, dim=-1)
            self.baked_transforms.append(torch.cat([means, quats], dim=1))
        else:
            position_offset = self.transform_mlp(gaussian_embeddings, embedding_expanded, means)
            quarternion_offset = torch.zeros(1).to(position_offset.device)

            means = means + position_offset
            self.baked_transforms.append(means)

    def approximate_mlp(self, x):
        print("Approximating mlp matrix...")
        self.baked_transforms = torch.stack(self.baked_transforms, dim=0).permute(1, 0, 2) # [N, frames, 7]
        x_with_ones = torch.cat([x, torch.ones(self.baked_transforms.shape[1],1).cuda()], dim=1) # [frames, 3]
        x_expanded = x_with_ones.unsqueeze(0).expand(self.gaussian_num, -1, -1)  # [N, frames, 3]
        self.mlp_matrix = torch.linalg.lstsq(x_expanded, self.baked_transforms).solution

        self.baking_flag = True

    @torch.no_grad()
    def rasterize_baked_splats_lstsq(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        shot_embeddings_2d: Tuple[float, float],
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Transform gaussians using approximated MLP matrix."""
        # torch.cuda.synchronize()
        # start = time.time()
        
        semb_with_ones = torch.cat((shot_embeddings_2d, torch.tensor([1]).cuda()), dim=0)
        approximate_val = semb_with_ones @ self.mlp_matrix
        
        means, quats = torch.split(approximate_val, [3, 4], dim=1)
        scales = torch.exp(self.splats["scales"])  
        opacities = torch.sigmoid(self.splats["opacities"])  
        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"

        # torch.cuda.synchronize()
        # elapsed_time = time.time() - start
        # print("MLP approximation: ", elapsed_time, "sec.")

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            **kwargs,
        )

        return render_colors, render_alphas, info

    @torch.no_grad()
    def bake_motion_bilinear(self):        
        print("Baking bilinear interpolation...")
        cfg = self.cfg

        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        gaussian_embeddings = self.splats["gaussian_embeddings"]

        grid_res = cfg.grid_resolution
        emb_scale = cfg.emb_2d_scale
        grid = torch.zeros(grid_res, grid_res, scales.shape[0], 7) # 7: position + quaternion

        for x in range(grid_res):
            for y in range(grid_res):
                means = self.splats["means"].clone()  # [N, 3]
                quats = self.splats["quats"].clone()  # [N, 4]

                # (0, grid_res-1) -> (min(semb)*emb_scale, max(semb)*emb_scale)
                v1 = x / (grid_res-1)
                v2 = y / (grid_res-1)
                v_min, _ = self.splats["shot_embeddings"].min(dim=0)
                v_max, _ = self.splats["shot_embeddings"].max(dim=0)
                v1_scale = (v_max[0] - v_min[0]) 
                v2_scale = (v_max[1] - v_min[1]) 
                v1 = v1 * v1_scale * emb_scale
                v2 = v2 * v2_scale * emb_scale
                v1 = v1 + v_min[0] * emb_scale
                v2 = v2 + v_min[1] * emb_scale

                shot_embeddings_2d = torch.stack([v1,v2], dim=0)

                if self.splats["shot_embeddings"].shape[1] == 2:
                    embedding_expanded = shot_embeddings_2d.repeat(len(means), 1)
                elif self.pca is not None:
                    with config_context(array_api_dispatch=True):
                        shot_embeddings_nd = self.pca.inverse_transform(shot_embeddings_2d)
                        embedding_expanded = shot_embeddings_nd.repeat(len(means), 1)

                if self.cfg.use_rotationMLP:
                    position_offset, quarternion_offset = self.transform_mlp(gaussian_embeddings, embedding_expanded, means, quats)
                    means = means + position_offset
                    if self.cfg.use_quats_offset_with_normalization:
                        quats = torch.nn.functional.normalize(quats, dim=1) + quarternion_offset
                    else:
                        quats = torch.nn.functional.normalize(quats + quarternion_offset, dim=-1)
                else:
                    position_offset = self.transform_mlp(gaussian_embeddings, embedding_expanded, means)
                    quarternion_offset = torch.zeros(1).to(position_offset.device)

                    means = means + position_offset
                
                transform = torch.cat([means, quats], dim=1) # vec 7
                grid[x,y] = transform

        self.grid = grid.cuda()
        self.baking_flag = True

    @torch.no_grad()
    def rasterize_baked_splats_bilinear(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        shot_embeddings_2d: Tuple[float, float],
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Transform from baked motion grid."""
        # torch.cuda.synchronize()
        # start = time.time()

        # interpolate position/rotation offsets
        grid_res = self.cfg.grid_resolution
        emb_scale = self.cfg.emb_2d_scale

        # (min(semb)*emb_scale, max(semb)*emb_scale) -> (0, grid_res-1)
        v1 = shot_embeddings_2d[0]
        v2 = shot_embeddings_2d[1]
        v_min, _ = self.splats["shot_embeddings"].min(dim=0)
        v_max, _ = self.splats["shot_embeddings"].max(dim=0)
        v1_scale = (v_max[0] - v_min[0]) 
        v2_scale = (v_max[1] - v_min[1]) 
        v1 = (v1 - emb_scale*v_min[0])/(emb_scale*v1_scale) * (grid_res - 1)
        v2 = (v2 - emb_scale*v_min[1])/(emb_scale*v2_scale) * (grid_res - 1)
        
        shot_embeddings_2d = torch.clamp(torch.stack([v1,v2], dim=0),min=0, max=self.cfg.grid_resolution-1.001)

        interp_val = bilinear_interpolation(self.grid, shot_embeddings_2d)
        
        means, quats = torch.split(interp_val, [3, 4], dim=1)
        scales = torch.exp(self.splats["scales"])  
        opacities = torch.sigmoid(self.splats["opacities"])  
        image_ids = kwargs.pop("image_ids", None)
        
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"

        # torch.cuda.synchronize()
        # elapsed_time = time.time() - start
        # print("MLP interpolation: ", elapsed_time, "sec.")

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            **kwargs,
        )

        return render_colors, render_alphas, info

    