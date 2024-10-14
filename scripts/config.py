from dataclasses import dataclass, field
from typing import List, Optional, Union
from typing_extensions import Literal, assert_never
from strategy import DefaultStrategy, MCMCStrategy

@dataclass
class Config:
    # ---- Settings ----
    config_yaml_path: str | None = None
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Dataset: "colmap" or "dnerf" or "iphone" or "hypernerf"
    dataset_type: str = "dnerf"
    # Path to the dnerf dataset
    data_dir: str = "./dnerf_dataset/lego"
    # Directory to save results
    result_dir: str = "./result/dnerf"
    # Port for the viewer server
    port: int = 8080
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [2_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [1_000, 30_000])
    # Initialization strategy (If dataset_type is "dnerf", it's automatically set to "random")
    init_type: str = "sfm"
    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False
    # Downsample factor for the dataset
    data_factor: int = 1
    # Every N images there is a test image
    test_every: int = 8
    use_all_train: bool = False
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0
    # Number of training steps
    max_steps: int = 30_000
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # Use random background for training to discourage transparency
    random_bkgd: bool = False
    # Enable camera optimization.
    pose_opt: bool = False
    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    lpips_net: Literal["vgg", "alex"] = "alex"

    # ---- Hyperparameters ----
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10
    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # ---- Hyperparameters for embeddings ----
    # Use gaussian & shot embeddings
    use_embeddings: bool = True
    use_embeddings_start_iter: int = -1

    # Use rotation MLP
    transform_hidden_dim: int = 128
    transform_n_hidden_layers: int = 2
    use_rotationMLP: bool = True
    feat_in_mlp: Literal["inject", "inject-detach", "none"] = "inject"
    use_quats_offset_with_normalization: bool = False

    # Use color MLP
    use_colorMLP: bool = False
    color_mlp_lr: float = 1e-3
    color_mlp_weight_decay: float = 0.0
    color_mlp_coeff: float = 1.0

    # Use opacity MLP
    use_opacityMLP: bool = False
    opacity_mlp_lr: float = 1e-4
    opacity_mlp_weight_decay: float = 0.0
    opacity_mlp_coeff: float = 1e-5

    # Use interpolation for val shot embeddings
    interpolate_val: bool = False

    # Embeddings hyperparameters
    gaussian_embeddings_lr: float = 1e-3
    shot_embeddings_lr: float = 1e-3
    shot_embeddings_dim: int = 32
    semb_weight_decay: float = 0.0
    gemb_weight_decay: float = 0.0
    transform_mlp_weight_decay: float = 0.0

    semb_noise: float = 0.0
    gemb_noise: float = 0.0

    semb_init_scale: float = 0.0

    exp_lr_mlp: bool = True
    exp_lr_semb: bool = True
    exp_lr_gemb: bool = False
    use_posenc: bool = True
    emb_2d_scale: float = 2

    # MLP hyperparameters
    transform_n_fourier_freqs: int = 4
    transform_mlp_lr: float = 1e-3
    transform_mlp_pos_coeff: float = 1.0
    transform_mlp_rot_coeff: float = 1.0

    # Smoothness regularization for gaussian embeddings
    gemb_smoothness_reg: float = 1
    gemb_smoothness_lambda: float = 2000.0
    stability_pos_reg: float = 0.0
    stability_rot_reg: float = 0.0
    stability_sh0_reg: float = 0.0

    # Gaussian Hyperparameters
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shN_lr_scale: float = 0.05  # 1/20

    # Define move of shot_embeddings. "circle", "momentum", "lissajous"
    looping_type: str = "lissajous"
    # Define looping video length
    rendering_frames: int = 100
    # Bake motion to skip MLP computation
    bake_motion: bool = True
    # Define type of motion bake. "bilinear", "mlp_approximation"
    bake_type: str = "bilinear"
    # Define grid resolution of bilinear interpolation
    grid_resolution: int = 2


    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)

        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)
