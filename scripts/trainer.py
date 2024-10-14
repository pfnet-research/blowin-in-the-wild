import json
import math
import os
import time
import colorsys
from typing import Dict, List, Optional, Tuple, Union

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from scipy.spatial.transform import Rotation as R
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from functools import reduce
from operator import mul
from sklearn.decomposition import PCA
from sklearn import config_context
import matplotlib.pyplot as plt

from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization

from strategy import DefaultStrategy, MCMCStrategy
from datasets.colmap import Dataset, Parser
from datasets.traj import generate_interpolated_path
from motion_baker import MotionBaker
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed, \
scipy_knn, weighted_l2_loss_v2, MomentumPointMover, LissajousPointMover, CirclePointMover
from mlps import PosMLP, PosRotMLP, ColorMLP, ColorOpacityMLP
from config import Config


def create_splats_with_optimizers(
    parser: Optional[Parser],
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    use_embedding: Optional[bool] = True,
    cfg : Optional[Config] = None,
    img_num: Optional[int] = None,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """Initialize gaussian parameters"""
    
    if init_type == "sfm":
        print("sfm init")
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        print("random init")
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    elif init_type == "randomsfm":
        print("random & sfm init")
        sfm_points = torch.from_numpy(parser.points).float()
        random_points = init_extent * scene_scale * (torch.rand((sfm_points.shape[0], 3)) * 2 - 1)
        sfm_rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        random_rgbs = torch.rand((sfm_points.shape[0], 3))
        points = torch.cat([sfm_points, random_points], dim=0)
        rgbs = torch.cat([sfm_rgbs, random_rgbs], dim=0)
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.zeros((N, 4))  # [N, 4]  # inria and per-gaussian
    quats[:, 0] = 1
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), cfg.means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), cfg.scales_lr),
        ("quats", torch.nn.Parameter(quats), cfg.quats_lr),
        ("opacities", torch.nn.Parameter(opacities), cfg.opacities_lr),
    ]

    if use_embedding:
        if cfg.use_posenc:
            # init gembs with positional encoding
            gaussian_embeddings = _get_fourier_features(points, num_features=cfg.transform_n_fourier_freqs)
        else:
            gaussian_embeddings = torch.zeros(points.shape[0], cfg.transform_n_fourier_freqs*6)
        params.append(("gaussian_embeddings", gaussian_embeddings, cfg.gaussian_embeddings_lr))
        shot_embeddings = torch.normal(0, cfg.semb_init_scale,  (img_num, cfg.shot_embeddings_dim), dtype=torch.float32, requires_grad=True)
        params.append(("shot_embeddings", shot_embeddings, cfg.shot_embeddings_lr))

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), cfg.sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), cfg.sh0_lr * cfg.shN_lr_scale))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), cfg.sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), cfg.sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    weight_decays = {
        "gaussian_embeddings": cfg.gemb_weight_decay,
        "shot_embeddings": cfg.semb_weight_decay,
    }
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.AdamW)(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            weight_decay=weight_decays.get(name, 0.0),
        )
        for name, _, lr in params
    }
    return splats, optimizers

def _get_fourier_features(xyz: Tensor, num_features=3):
    xyz = xyz.to(dtype=torch.float32)
    xyz = xyz - xyz.mean(dim=0, keepdim=True)
    xyz = xyz / torch.quantile(xyz.abs(), 0.97, dim=0) * 0.5 + 0.5
    freqs = torch.repeat_interleave(
        2**torch.linspace(0, num_features-1, num_features, dtype=xyz.dtype, device=xyz.device), 2)
    offsets = torch.tensor([0, 0.5 * math.pi] * num_features, dtype=xyz.dtype, device=xyz.device)
    feat = xyz[..., None] * freqs[None, None] * 2 * math.pi + offsets[None, None]
    feat = torch.sin(feat).view(-1, reduce(mul, feat.shape[1:]))
    return feat

class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        self.pca = None
        self.motion_baker = None
        self.emb_2d_scale = cfg.emb_2d_scale
        self.looping = False
        self.mover = None

        # Where to dump results.
        self.save_dir = cfg.result_dir
        i = 1
        while os.path.exists(self.save_dir):
            self.save_dir = cfg.result_dir + f"_{str(i)}"
            i += 1
        os.makedirs(self.save_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{self.save_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{self.save_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{self.save_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{self.save_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        if cfg.dataset_type == "colmap":
            self.parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=True,
                test_every=cfg.test_every,
            )
            self.trainset = Dataset(
                self.parser,
                split="all" if cfg.use_all_train else "train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
            )
            self.valset = Dataset(self.parser, split="val")
            self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
            self.val_idxs = self.valset.indices
            print("Scene scale:", self.scene_scale)
        elif cfg.dataset_type == "dnerf":
            from datasets.dnerf import DNeRFDataset
            self.trainset = DNeRFDataset(
                data_dir=cfg.data_dir,
                split="train",
            )
            self.valset = DNeRFDataset(
                data_dir=cfg.data_dir,
                split="val",
            )
            self.scene_scale = self.trainset.scene_scale * 1.1 * cfg.global_scale
            print("Scene scale:", self.scene_scale)

            # compute shot_embeddings idx of valset
            self.train_times = self.trainset.times
            self.val_times = self.valset.times
            val_idxs = []
            if cfg.interpolate_val:
                # interpolate idx for accurate fitting
                for vt in self.val_times:
                    val_idx = np.searchsorted(self.train_times, vt)
                    if val_idx >= len(self.train_times):
                        val_idx = len(self.train_times) - 1
                    t1 = self.train_times[val_idx-1]
                    t2 = self.train_times[val_idx]
                    a = vt - t1
                    b = t2 - vt
                    val_interp_idx = (a*val_idx + b*(val_idx-1)) / (a+b)
                    val_idxs.append(val_interp_idx)
                self.val_idxs = val_idxs
            else:
                for vt in self.val_times:
                    val_idx = np.searchsorted(self.train_times, vt)
                    if val_idx >= len(self.train_times):
                        val_idx = len(self.train_times) - 1
                    val_idxs.append(val_idx)
                self.val_idxs = val_idxs
        elif cfg.dataset_type == "iphone":
            from datasets.iphone import IphoneDataset
            self.trainset = IphoneDataset(
                data_dir=cfg.data_dir,
                split="train",
                use_colmap=True,
            )
            self.valset = IphoneDataset(
                data_dir=cfg.data_dir,
                split="val",
                use_colmap=True,
            )
            self.parser = self.trainset.parser
            self.scene_scale = self.trainset.scene_scale * 1.1 * cfg.global_scale
            print("Scene scale:", self.scene_scale)

            # compute shot_embeddings idx of valset
            self.train_times = self.trainset.times
            self.val_times = self.valset.times
            val_idxs = []
            if cfg.interpolate_val:
                # interpolate idx for accurate fitting
                for vt in self.val_times:
                    val_idx = np.searchsorted(self.train_times, vt)
                    if val_idx >= len(self.train_times):
                        val_idx = len(self.train_times) - 1
                    t1 = self.train_times[val_idx-1]
                    t2 = self.train_times[val_idx]
                    a = vt - t1
                    b = t2 - vt
                    val_interp_idx = (a*val_idx + b*(val_idx-1)) / (a+b)
                    val_idxs.append(val_interp_idx)
                self.val_idxs = val_idxs
            else:
                for vt in self.val_times:
                    val_idx = np.searchsorted(self.train_times, vt)
                    if val_idx >= len(self.train_times):
                        val_idx = len(self.train_times) - 1
                    val_idxs.append(val_idx)
                self.val_idxs = val_idxs
        elif cfg.dataset_type == "hypernerf":
            from datasets.hypernerf import HyperNeRFDataset
            self.trainset = HyperNeRFDataset(
                data_dir=cfg.data_dir,
                split="train",
            )
            self.valset = HyperNeRFDataset(
                data_dir=cfg.data_dir,
                split="val",
            )
            self.parser = self.trainset.parser
            self.scene_scale = self.trainset.scene_scale * 1.1 * cfg.global_scale
            print("Scene scale:", self.scene_scale)

            # compute shot_embeddings idx of valset
            self.train_times = self.trainset.times
            self.val_times = self.valset.times
            val_idxs = []
            if cfg.interpolate_val:
                # interpolate idx for accurate fitting
                for vt in self.val_times:
                    val_idx = np.searchsorted(self.train_times, vt)
                    if val_idx >= len(self.train_times):
                        val_idx = len(self.train_times) - 1
                    t1 = self.train_times[val_idx-1]
                    t2 = self.train_times[val_idx]
                    a = vt - t1
                    b = t2 - vt
                    val_interp_idx = (a*val_idx + b*(val_idx-1)) / (a+b)
                    val_idxs.append(val_interp_idx)
                self.val_idxs = val_idxs
            else:
                for vt in self.val_times:
                    val_idx = np.searchsorted(self.train_times, vt)
                    if val_idx >= len(self.train_times):
                        val_idx = len(self.train_times) - 1
                    val_idxs.append(val_idx)
                self.val_idxs = val_idxs

        # Set random init for dnerf dataset
        if cfg.dataset_type == "dnerf":
            self.init_type = "random"
            self.parser = None
        elif cfg.dataset_type in ["hypernerf", "iphone"]:
            self.init_type = cfg.init_type
            # self.parser = None
        else:
            self.init_type = cfg.init_type

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=self.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            use_embedding=cfg.use_embeddings,
            img_num=len(self.trainset),
            cfg=cfg,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        # MLP Strategy
        self.transform_optimizers = []
        self.color_optimizers = []
        self.color_opacity_optimizers = []
        if cfg.use_embeddings:
            if cfg.use_rotationMLP:
                self.transform_mlp = PosRotMLP(cfg).to(self.device)
            else:
                self.transform_mlp = PosMLP(cfg).to(self.device)
            self.transform_optimizers = [
                torch.optim.AdamW(
                    self.transform_mlp.parameters(),
                    lr=cfg.transform_mlp_lr,
                    weight_decay=cfg.transform_mlp_weight_decay,
                )
            ]
            if cfg.use_opacityMLP:
                self.color_opacity_mlp = ColorOpacityMLP(cfg).to(self.device)
                self.color_opacity_optimizers = [
                    torch.optim.AdamW(
                        self.color_opacity_mlp.parameters(),
                        lr=cfg.opacity_mlp_lr,
                        weight_decay=cfg.opacity_mlp_weight_decay
                    )
                ]
            elif cfg.use_colorMLP:
                self.color_mlp = ColorMLP(cfg).to(self.device)
                self.color_optimizers = [
                    torch.optim.AdamW(
                        self.color_mlp.parameters(),
                        lr=cfg.color_mlp_lr,
                        weight_decay=cfg.color_mlp_weight_decay
                    )
                ]
            
        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)

            # a trick for setting a training pose as the default pose in viser
            @self.server.on_client_connect
            def _(client: viser.ClientHandle) -> None:
                camtoworld = self.trainset[0]["camtoworld"]  # first pose
                position = camtoworld[:3, 3]
                rotation_matrix = camtoworld[:3, :3]
                quaternion = R.from_matrix(rotation_matrix).as_quat()  # xyzw
                quaternion_wxyz = np.roll(quaternion, 1)  # xyzw -> wxyz
                client.camera.position = position
                client.camera.wxyz = quaternion_wxyz

            self.shot_embeddings_viewer = 0
            self.shot_embeddings_2d = None
            
            # define custom UI
            with self.server.gui.add_folder("Shot Embeddings"):
                gui_timestep = self.server.gui.add_slider(
                    "Timestep",
                    min=0,
                    max=len(self.trainset),
                    step=1,
                    initial_value=0,
                )
                self.gui_2dembeddings = self.server.gui.add_rgb(
                    "2D Embeddings",
                    (255,255,255)
                )
                gui_baking = self.server.gui.add_checkbox(
                    "Bake Motion (eval only)",
                    initial_value=cfg.bake_motion,
                )
                gui_looping = self.server.gui.add_checkbox(
                    "Loop Motion",
                    initial_value=False,
                )
                gui_looptype = self.server.gui.add_dropdown(
                    "Loop Type",
                    ("circle", "momentum", "lissajous"),
                    initial_value=cfg.looping_type
                )
                self.gui_framerate = self.server.gui.add_slider(
                    "FPS", min=1, max=60, step=1, initial_value=30
                )
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )
            # change shot_embeddings when slider move
            @gui_timestep.on_update
            def _(_) -> None:
                self.shot_embeddings_viewer = gui_timestep.value
                self.shot_embeddings_2d = None
                self.viewer.rerender(_)
            # change shot_embeddings when color picker move
            @gui_baking.on_update
            def _(_) -> None:
                self.motion_baker.use_baking = gui_baking.value
                self.viewer.rerender(_)
            @gui_looping.on_update
            def _(_) -> None:
                self.looping = gui_looping.value
                self.viewer.rerender(_)
            @gui_looptype.on_update
            def _(_) -> None:
                self.get_mover(gui_looptype.value)
                self.viewer.rerender(_)
            @self.gui_2dembeddings.on_update
            def _(_) -> None:
                [r,g,b] = self.gui_2dembeddings.value
                _, v1, v2 = colorsys.rgb_to_hsv(r,g,b)
                v2 /= 255
                emb_scale = self.emb_2d_scale

                # (0,1) -> (min(semb)*2,max(semb)*2)
                v_min, _ = self.splats["shot_embeddings"].min(dim=0)
                v_max, _ = self.splats["shot_embeddings"].max(dim=0)
                v1_scale = (v_max[0] - v_min[0]) 
                v2_scale = (v_max[1] - v_min[1]) 
                v1 = v1 * v1_scale * emb_scale
                v2 = v2 * v2_scale * emb_scale
                v1 = v1 + v_min[0] * emb_scale
                v2 = v2 + v_min[1] * emb_scale

                self.shot_embeddings_2d = torch.stack([v1,v2], dim=0)
                self.viewer.rerender(_)

    def get_mover(self, loop_type: str):
        if loop_type == "circle":
            self.mover = CirclePointMover(r=0.4)
        elif loop_type == "momentum":
            self.mover = MomentumPointMover(r=0.4)
        elif loop_type == "lissajous":
            self.mover = LissajousPointMover(r=0.4)

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        shot_embeddings_2d: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        # torch.cuda.synchronize()
        # start = time.time()

        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        sh0 = self.splats["sh0"]
        opacities = self.splats["opacities"]
        
        sh0_offset = None
        position_offset = None
        quarternion_offset = None
        is_training = kwargs.pop("is_training", False)

        
        if self.cfg.use_embeddings:
            gaussian_embeddings = self.splats["gaussian_embeddings"]
            # get shot_embeddings from image_ids
            # TODO: multiple image ids
            try:
                if kwargs["image_ids"] is None:
                    print("image_ids is None. use image_ids = 0.")
                    shot_embeddings = self.splats["shot_embeddings"][0]
                elif kwargs["image_ids"] >= len(self.splats["shot_embeddings"]) - 1:
                    shot_embeddings = self.splats["shot_embeddings"][-1]
                else:
                    # interpolate shot_embeddings of val img
                    i = math.floor(kwargs["image_ids"])
                    a = kwargs["image_ids"] - i
                    b = i + 1 - kwargs["image_ids"]
                    shot_embeddings = b * self.splats["shot_embeddings"][i] + a * self.splats["shot_embeddings"][i+1]
            except TypeError as e:
                print("use image_ids = 0. exception: ", e)
                shot_embeddings = self.splats["shot_embeddings"][0]

            # if shot_embeddings is not 2D, do PCA
            if shot_embeddings_2d is not None:
                if self.splats["shot_embeddings"].shape[1] == 2:
                    embedding_expanded = shot_embeddings_2d.repeat(len(means), 1)
                elif self.pca is not None:
                    with config_context(array_api_dispatch=True):
                        shot_embeddings_nd = self.pca.inverse_transform(shot_embeddings_2d)
                        embedding_expanded = shot_embeddings_nd.repeat(len(means), 1)
                else:
                    embedding_expanded = shot_embeddings.repeat(len(means), 1)
            else:
                embedding_expanded = shot_embeddings.repeat(len(means), 1)

            if self.cfg.semb_noise > 0.0 and is_training:
                embedding_expanded = embedding_expanded * (1 + torch.randn_like(embedding_expanded) * self.cfg.semb_noise)
            if self.cfg.gemb_noise > 0.0 and is_training:
                gaussian_embeddings = gaussian_embeddings * (1 + torch.randn_like(gaussian_embeddings) * self.cfg.gemb_noise)

            # get transform offsets from MLP
            if self.cfg.use_rotationMLP:
                position_offset, quarternion_offset = self.transform_mlp(gaussian_embeddings, embedding_expanded, means, quats)
                means = means + position_offset
                if self.cfg.use_quats_offset_with_normalization:
                    quats = torch.nn.functional.normalize(quats, dim=1) + quarternion_offset
                else:
                    quats = torch.nn.functional.normalize(quats + quarternion_offset, dim=-1)
            else:
                position_offset = self.transform_mlp(gaussian_embeddings, embedding_expanded, means)
                means = means + position_offset

            # get color/opacity offsets from MLP
            if self.cfg.use_opacityMLP:
                sh0 = self.splats["sh0"].squeeze()
                opacity = self.splats["opacities"].unsqueeze(1)
                sh0_offset, opacity_offset = self.color_opacity_mlp(gaussian_embeddings, embedding_expanded, sh0, opacity)
                sh0 = sh0 + sh0_offset
                opacity = opacity + opacity_offset
                sh0 = sh0.unsqueeze(1)
                opacities = opacity.squeeze()
            elif self.cfg.use_colorMLP:
                sh0 = self.splats["sh0"].squeeze()
                sh0_offset = self.color_mlp(gaussian_embeddings, embedding_expanded, sh0)
                sh0 = sh0 + sh0_offset
                sh0 = sh0.unsqueeze(1)

        image_ids = kwargs.pop("image_ids", None)
        opacities = torch.sigmoid(opacities)  # [N,]
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
            colors = torch.cat([sh0, self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"

        # torch.cuda.synchronize()
        # elapsed_time = time.time() - start
        # print("naive MLP: ", elapsed_time, "sec.")

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
            distributed=self.world_size > 1,
            **kwargs,
        )

        # torch.cuda.synchronize()
        # elapsed_time = time.time() - start
        # print("naive 2D embeddings: ", elapsed_time, "sec.")

        if position_offset is not None:
            info["position_offset"] = position_offset
        if quarternion_offset is not None:
            info["quarternion_offset"] = quarternion_offset
        if sh0_offset is not None:
            info["sh0_offset"] = sh0_offset
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{self.save_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if self.transform_optimizers:
            # transform MLP and shot embeddings have a learning rate schedule
            # see A.2 https://arxiv.org/pdf/2404.03613.pdf
            if cfg.exp_lr_mlp:
                schedulers.append(
                    torch.optim.lr_scheduler.ExponentialLR(
                        self.transform_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                    )
                )
            if cfg.exp_lr_semb:
                schedulers.append(
                    torch.optim.lr_scheduler.ExponentialLR(
                        self.optimizers["shot_embeddings"], gamma=0.01 ** (1.0 / max_steps)
                    )
                )
            if cfg.exp_lr_gemb:
                schedulers.append(
                    torch.optim.lr_scheduler.ExponentialLR(
                        self.optimizers["gaussian_embeddings"], gamma=0.01 ** (1.0 / max_steps)
                    )
                )

        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        prev_num_gs = 0
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)
    
            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            assert image_ids is not None

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                is_training=True,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0:
                loss = (
                    loss
                    + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )
            if cfg.gemb_smoothness_reg > 0.0 and cfg.use_embeddings == True:
                # gaussian embeddings smoothness regularization using knn (https://github.com/JonathonLuiten/Dynamic3DGaussians)
                if prev_num_gs != len(self.splats["means"]):
                    neighbor_sq_dist, neighbor_indices = scipy_knn(self.splats["means"].detach().cpu().numpy(), 20)
                    neighbor_weight = np.exp(-cfg.gemb_smoothness_lambda * neighbor_sq_dist)
                    neighbor_indices = torch.tensor(neighbor_indices).cuda().long().contiguous()
                    neighbor_weight = torch.tensor(neighbor_weight).cuda().float().contiguous()
                    prev_num_gs = len(self.splats["means"])
                
                emb = self.splats["gaussian_embeddings"][:,None,:].repeat(1,20,1)
                emb_knn = self.splats["gaussian_embeddings"][neighbor_indices]
                loss += cfg.gemb_smoothness_reg * weighted_l2_loss_v2(emb, emb_knn, neighbor_weight)

            if cfg.stability_pos_reg > 0.0 and "position_offset" in info:
                loss += (abs(info["position_offset"]) * cfg.stability_pos_reg).mean()  # .mean()
            if cfg.stability_rot_reg > 0.0 and "quarternion_offset" in info:
                loss += (abs(info["quarternion_offset"]) * cfg.stability_rot_reg).mean()  # .mean()
            if cfg.stability_sh0_reg > 0.0 and "sh0_offset" in info:
                loss += (abs(info["sh0_offset"]) * cfg.stability_sh0_reg).mean()  # .mean()

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "elapsed_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.use_embeddings:
                    data["mlp"] = self.transform_mlp.state_dict()

                if cfg.use_opacityMLP:
                    data["opacity_mlp"] = self.color_opacity_mlp.state_dict()
                elif cfg.use_colorMLP:
                    data["color_mlp"] = self.color_mlp.state_dict()

                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )

            if isinstance(self.cfg.strategy, DefaultStrategy):
                if len(self.splats["means"]) < 1000000:  # skip if many gaussians
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        packed=cfg.packed,
                    )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )
            if cfg.use_embeddings:
                if step < cfg.use_embeddings_start_iter:
                    self.transform_optimizers[0].zero_grad(set_to_none=True)
                    self.optimizers["shot_embeddings"].zero_grad(set_to_none=True)
                    self.optimizers["gaussian_embeddings"].zero_grad(set_to_none=True)
                    if self.color_opacity_optimizers:
                        self.color_opacity_optimizers[0].zero_grad(set_to_none=True)
                    if self.color_optimizers:
                        self.color_optimizers[0].zero_grad(set_to_none=True)

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.transform_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.color_opacity_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.color_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                if cfg.dataset_type == "colmap":
                    # self.render_traj(step)
                    pass
                elif cfg.dataset_type == "dnerf":
                    self.render_traj_dnerf(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # for training
        if self.pca is None:
            with config_context(array_api_dispatch=True):
                pca = PCA(n_components=2, svd_solver="randomized", power_iteration_normalizer="QR")
                pca.fit(self.splats["shot_embeddings"])
                self.pca = pca

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        elapsed_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}

        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        cbar = None

        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            image_ids = data["image_id"].to(device)

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
            )  # [1, H, W, 3]
            colors = torch.clamp(colors, 0.0, 1.0)
            torch.cuda.synchronize()
            elapsed_time += time.time() - tic

            if world_rank == 0:
                # write images
                canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_{i:04d}.png",
                    (canvas * 255).astype(np.uint8),
                )
                # write errormap
                errormap = np.abs(pixels.cpu().numpy() - colors.cpu().numpy()).squeeze()
                im1 = ax.imshow(errormap, cmap='jet')
                if cbar is None:
                    cbar = plt.colorbar(im1)
                else:
                    cbar.update_normal(im1)
                plt.savefig(f"{self.render_dir}/{stage}_{i:04d}_errormap.png")

                # write stats
                pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors, pixels))
                metrics["ssim"].append(self.ssim(colors, pixels))
                metrics["lpips"].append(self.lpips(colors, pixels))

        if world_rank == 0:
            elapsed_time /= len(valloader)

            psnr = torch.stack(metrics["psnr"]).mean()
            ssim = torch.stack(metrics["ssim"]).mean()
            lpips = torch.stack(metrics["lpips"]).mean()
            print(
                f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
                f"Time: {elapsed_time:.3f}s/image "
                f"Number of GS: {len(self.splats['means'])}"
            )
            # save stats as json
            stats = {
                "psnr": psnr.item(),
                "ssim": ssim.item(),
                "lpips": lpips.item(),
                "elapsed_time": elapsed_time,
                "num_GS": len(self.splats["means"]),
            }
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

            if cfg.use_embeddings:
                # self.render_various_shot_embeddings()
                self.render_looping_shot_embeddings()
                pass

            return stats
        
    @torch.no_grad()
    def ablation(self):
        """For ablation study."""
        cfg = self.cfg
        device = self.device

        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=1, shuffle=False, num_workers=1
        )

        # test on first 10 shots in training data
        sembeddings_length = max(self.splats["shot_embeddings"].shape[0], 10)

        # test the effect of 2d compression
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        for j, data in enumerate(trainloader):
            try:
                if j >= sembeddings_length:
                    break
                camtoworlds = data["camtoworld"].to(device)
                Ks = data["K"].to(device)
                pixels = data["image"].to(device) / 255.0
                height, width = pixels.shape[1:3]
                image_ids = data["image_id"].to(device)

                i = math.floor(image_ids)
                a = image_ids - i
                b = 1 - a
                shot_embeddings = b * self.splats["shot_embeddings"][i] + a * self.splats["shot_embeddings"][i+1]
                
                with config_context(array_api_dispatch=True):
                    shot_embeddings_2d = self.pca.transform(shot_embeddings.unsqueeze(0))

                colors, _, _ = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=None,
                    shot_embeddings_2d=shot_embeddings_2d
                )  # [1, H, W, 3]
                pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors = torch.clamp(colors, 0.0, 1.0)

                metrics["psnr"].append(self.psnr(colors, pixels))
                metrics["ssim"].append(self.ssim(colors, pixels))
                metrics["lpips"].append(self.lpips(colors, pixels))
                
            except Exception as e:
                print(e)
        
        naive2d_psnr = torch.stack(metrics["psnr"]).mean()
        naive2d_ssim = torch.stack(metrics["ssim"]).mean()
        naive2d_lpips = torch.stack(metrics["lpips"]).mean()

        # test the effect of MLP interpolation 
        if cfg.bake_type == "bilinear":
            metrics = {"psnr": [], "ssim": [], "lpips": []}
            for j, data in enumerate(trainloader):
                try:
                    if j >= sembeddings_length:
                        break
                    camtoworlds = data["camtoworld"].to(device)
                    Ks = data["K"].to(device)
                    pixels = data["image"].to(device) / 255.0
                    height, width = pixels.shape[1:3]
                    image_ids = data["image_id"].to(device)

                    i = math.floor(image_ids)
                    a = image_ids - i
                    b = 1 - a
                    shot_embeddings = b * self.splats["shot_embeddings"][i] + a * self.splats["shot_embeddings"][i+1]
                    with config_context(array_api_dispatch=True):
                        shot_embeddings_2d = self.pca.transform(shot_embeddings.unsqueeze(0))

                    colors, _, _ = self.motion_baker.rasterize_baked_splats_bilinear(
                        camtoworlds=camtoworlds,
                        Ks=Ks,
                        width=width,
                        height=height,
                        sh_degree=cfg.sh_degree,
                        near_plane=cfg.near_plane,
                        far_plane=cfg.far_plane,
                        image_ids=None,
                        shot_embeddings_2d=shot_embeddings_2d.squeeze()
                    )  # [1, H, W, 3]
                    pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    colors = torch.clamp(colors, 0.0, 1.0)

                    metrics["psnr"].append(self.psnr(colors, pixels))
                    metrics["ssim"].append(self.ssim(colors, pixels))
                    metrics["lpips"].append(self.lpips(colors, pixels))
                except Exception as e:
                    print(e)
            
            interpolation_psnr = torch.stack(metrics["psnr"]).mean()
            interpolation_ssim = torch.stack(metrics["ssim"]).mean()
            interpolation_lpips = torch.stack(metrics["lpips"]).mean()

        # test the effect of MLP approximation
        if cfg.bake_type == "mlp_approximation":
            metrics = {"psnr": [], "ssim": [], "lpips": []}
            for j, data in enumerate(trainloader):
                try:
                    if j >= sembeddings_length:
                        break
                    camtoworlds = data["camtoworld"].to(device)
                    Ks = data["K"].to(device)
                    pixels = data["image"].to(device) / 255.0
                    height, width = pixels.shape[1:3]
                    image_ids = data["image_id"].to(device)

                    i = math.floor(image_ids)
                    a = image_ids - i
                    b = 1 - a
                    shot_embeddings = b * self.splats["shot_embeddings"][i] + a * self.splats["shot_embeddings"][i+1]
                    with config_context(array_api_dispatch=True):
                        shot_embeddings_2d = self.pca.transform(shot_embeddings.unsqueeze(0))

                    colors, _, _ = self.motion_baker.rasterize_baked_splats_lstsq(
                        camtoworlds=camtoworlds,
                        Ks=Ks,
                        width=width,
                        height=height,
                        sh_degree=cfg.sh_degree,
                        near_plane=cfg.near_plane,
                        far_plane=cfg.far_plane,
                        image_ids=None,
                        shot_embeddings_2d=shot_embeddings_2d.squeeze()
                    )  # [1, H, W, 3]
                    pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    colors = torch.clamp(colors, 0.0, 1.0)

                    metrics["psnr"].append(self.psnr(colors, pixels))
                    metrics["ssim"].append(self.ssim(colors, pixels))
                    metrics["lpips"].append(self.lpips(colors, pixels))
                except Exception as e:
                    print(e)
            
            approximation_psnr = torch.stack(metrics["psnr"]).mean()
            approximation_ssim = torch.stack(metrics["ssim"]).mean()
            approximation_lpips = torch.stack(metrics["lpips"]).mean()

        # save stats as json
        if cfg.bake_type == "bilinear":
            stats = {
                "naive2d_psnr": naive2d_psnr.item(),
                "naive2d_ssim": naive2d_ssim.item(),
                "naive2d_lpips": naive2d_lpips.item(),
                "interpolation_psnr": interpolation_psnr.item(),
                "interpolation_ssim": interpolation_ssim.item(),
                "interpolation_lpips": interpolation_lpips.item(),
            }
        elif cfg.bake_type == "mlp_approximation":
            stats = {
                "naive2d_psnr": naive2d_psnr.item(),
                "naive2d_ssim": naive2d_ssim.item(),
                "naive2d_lpips": naive2d_lpips.item(),
                "approximation_psnr": approximation_psnr.item(),
                "approximation_ssim": approximation_ssim.item(),
                "approximation_lpips": approximation_lpips.item(),
            }
        with open(f"{self.stats_dir}/ablation.json", "w") as f:
            json.dump(stats, f)

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds = self.parser.camtoworlds[5:-5]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{self.save_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def render_traj_dnerf(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=1,
        )
        valloader_iter = iter(valloader)
        max_steps = 100
        init_step = 0

        # Validation loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        canvas_all = []
        
        for step in pbar:
            try:
                data = next(valloader_iter)
            except StopIteration:
                valloader_iter = iter(valloader)
                data = next(valloader_iter)
    
            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            height, width = pixels.shape[1:3]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                image_ids=image_ids
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{self.save_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")
        
    @torch.no_grad()
    def render_various_shot_embeddings(self):
        """Render the scene with various shot embeddings."""
        print("Running sembeddings rendering...")
        cfg = self.cfg
        device = self.device

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=False,  # for reproducibility
            num_workers=1,
            persistent_workers=True,
            pin_memory=True,
        )

        trainloader_iter = iter(trainloader)
        data = next(trainloader_iter)
        sembeddings_length = self.splats["shot_embeddings"].shape[0]

        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        pixels = data["image"].to(device) / 255.0
        height, width = pixels.shape[1:3]

        canvas_all = []
        for i in tqdm.trange(sembeddings_length, desc="Rendering sembeddings"):

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                image_ids=i
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{self.save_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/various_embeddings.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/various_embeddings.mp4")

    @torch.no_grad()
    def render_looping_shot_embeddings(self):
        """Render the scene with looping shot embeddings."""
        print("Running sembeddings rendering...")
        cfg = self.cfg
        device = self.device
        self.frames = cfg.rendering_frames

        # get camera parameters from trainset
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)
        data = next(trainloader_iter)
        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        pixels = data["image"].to(device) / 255.0
        height, width = pixels.shape[1:3]

        canvas_all = []
        xs = []
        for i in tqdm.trange(self.frames, desc="Rendering sembeddings"):
            # shot_embeddings manipulation in 2D space
            if self.mover is None:
                self.get_mover(cfg.looping_type)
            self.mover.update()
            v1, v2 = self.mover.get_position()

            emb_scale = self.emb_2d_scale
            v_min, _ = self.splats["shot_embeddings"].min(dim=0)
            v_max, _ = self.splats["shot_embeddings"].max(dim=0)
            v1_scale = (v_max[0] - v_min[0])
            v2_scale = (v_max[1] - v_min[1])

            v1 = v1 * v1_scale * emb_scale
            v2 = v2 * v2_scale * emb_scale

            v1 = v1 + v_min[0] * emb_scale
            v2 = v2 + v_min[1] * emb_scale
            self.shot_embeddings_2d = torch.stack([v1,v2], dim=0)
            xs.append(self.shot_embeddings_2d)

            # bake looping motion (if bake_type is "mlp_approximation")
            if cfg.bake_type == "mlp_approximation":
                if self.motion_baker is not None:
                    self.motion_baker.bake_motion_mlp(self.shot_embeddings_2d)

            # render frames
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                image_ids=i,
                shot_embeddings_2d=self.shot_embeddings_2d,
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)
        
        if cfg.bake_type == "mlp_approximation":
            if self.motion_baker is not None:
                xs = torch.stack(xs, dim=0).cuda()
                self.motion_baker.approximate_mlp(xs)

        # save to video
        video_dir = f"{self.save_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/looping_embeddings.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/looping_embeddings.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{self.save_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        if self.motion_baker is None:
            render_colors, _, _ = self.rasterize_splats(
                camtoworlds=c2w[None],
                Ks=K[None],
                width=W,
                height=H,
                sh_degree=self.cfg.sh_degree,  # active all SH degrees
                radius_clip=0.1,  # skip GSs that have small image radius (in pixels)
                image_ids=self.shot_embeddings_viewer,
                shot_embeddings_2d=self.shot_embeddings_2d,
            )  # [1, H, W, 3]
            return render_colors[0].cpu().numpy()
        
        if self.motion_baker.use_baking and self.motion_baker.baking_flag:
            if cfg.bake_type == "bilinear":
                render_colors, _, _ = self.motion_baker.rasterize_baked_splats_bilinear(
                    camtoworlds=c2w[None],
                    Ks=K[None],
                    width=W,
                    height=H,
                    sh_degree=self.cfg.sh_degree,  # active all SH degrees
                    radius_clip=0.1,  # skip GSs that have small image radius (in pixels)
                    image_ids=self.shot_embeddings_viewer,
                    shot_embeddings_2d=self.shot_embeddings_2d if self.shot_embeddings_2d is not None else [0,0],
                )  # [1, H, W, 3]
            elif cfg.bake_type == "mlp_approximation":
                render_colors, _, _ = self.motion_baker.rasterize_baked_splats_lstsq(
                    camtoworlds=c2w[None],
                    Ks=K[None],
                    width=W,
                    height=H,
                    sh_degree=self.cfg.sh_degree,  # active all SH degrees
                    radius_clip=0.1,  # skip GSs that have small image radius (in pixels)
                    image_ids=self.shot_embeddings_viewer,
                    shot_embeddings_2d=self.shot_embeddings_2d if self.shot_embeddings_2d is not None else [0,0],
                )  # [1, H, W, 3]
            else:
                raise ValueError("Please specify a correct bake_type: bilinear or mlp_approximation")
            return render_colors[0].cpu().numpy()
        else:
            render_colors, _, _ = self.rasterize_splats(
                camtoworlds=c2w[None],
                Ks=K[None],
                width=W,
                height=H,
                sh_degree=self.cfg.sh_degree,  # active all SH degrees
                radius_clip=0.1,  # skip GSs that have small image radius (in pixels)
                image_ids=self.shot_embeddings_viewer,
                shot_embeddings_2d=self.shot_embeddings_2d,
            )  # [1, H, W, 3]
            return render_colors[0].cpu().numpy()

def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device, weights_only=False)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        print("loading mlp weight...")
        runner.transform_mlp.load_state_dict(ckpt["mlp"])
        if cfg.use_opacityMLP:
            runner.color_opacity_mlp.load_state_dict(ckpt["opacity_mlp"])
        elif cfg.use_colorMLP:
            runner.color_mlp.load_state_dict(ckpt["color_mlp"])
        
        if cfg.compression is not None:
            runner.run_compression(step=ckpt["step"])
        if cfg.use_embeddings:
            with config_context(array_api_dispatch=True):
                pca = PCA(n_components=2, svd_solver="randomized", power_iteration_normalizer="QR")
                pca.fit(runner.splats["shot_embeddings"])
                runner.pca = pca
        runner.motion_baker = MotionBaker(
            splats=runner.splats, 
            pca=runner.pca, 
            cfg=runner.cfg, 
            mlp=runner.transform_mlp, 
            use_baking=cfg.bake_motion)
        if cfg.bake_type=="bilinear":
            runner.motion_baker.bake_motion_bilinear()
        runner.eval(step=ckpt["step"])
        # runner.ablation()
    else:
        runner.train()
        metrics = runner.eval(step=cfg.max_steps)
        print(metrics)

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        # play looping motion while checkboxed
        frame = 0
        frame_num = 100
        while True:
            if runner.looping:
                if runner.mover is None:
                    runner.get_mover(cfg.looping_type)
                runner.mover.update()
                s, v = runner.mover.get_position()
                r,g,b = colorsys.hsv_to_rgb(0.5, s, v)
                r = int(r*255)
                g = int(g*255)
                b = int(b*255)
                runner.gui_2dembeddings.value = (r,g,b)
                frame = (frame + 1) % frame_num

            time.sleep(1.0 / runner.gui_framerate.value)


if __name__ == "__main__":
    """
    Usage:
    ```
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default
    """
    yaml_config = {}
    # config_yaml_path = "configs/tmpopt_lego_v5.yaml"
    config_yaml_path = "configs/tmp_rose_v5.yaml"

    if config_yaml_path:
        with open(config_yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f)
            # TODO: strategy config
            for k in ["init_opa",
                      "init_scale",
                      "opacity_reg",
                      "scale_reg"]:
                if k in yaml_config:
                    del yaml_config[k]
            print(yaml_config)

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
                **yaml_config,
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
                **yaml_config,
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    cli(main, cfg, verbose=True)
