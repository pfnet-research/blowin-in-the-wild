import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed, o3d_knn, weighted_l2_loss_v2
from functools import reduce
from operator import mul

from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
# from gsplat.strategy import DefaultStrategy, MCMCStrategy
from strategy import DefaultStrategy, MCMCStrategy
from simple_trainer import Config, Runner
import optuna
import copy
from functools import partial


def objective(trial, cfg: Config):
    cfg = copy.deepcopy(cfg)
    cfg.eval_steps = []
    cfg.save_steps = []

    def sample(name, *args, **kwargs):
        first = args[0]
        if isinstance(first, float):
            suggest = trial.suggest_float
        elif isinstance(first, int):
            suggest = trial.suggest_int
        elif isinstance(first, list):
            suggest = trial.suggest_categorical
        else:
            print(name, first, "unknown type", type(first))
            raise NotImplementedError()
        short_name = name.split(".")[-1]
        setattr(cfg, name, suggest(short_name, *args, **kwargs))

    # params
    sample("transform_hidden_dim", [64, 96, 128])
    sample("feat_in_mlp", ["inject", "inject-detach", "none"])
    sample("gaussian_embeddings_lr", 1e-5, 3e-3, log=True)
    sample("shot_embeddings_lr", 5e-5, 3e-3, log=True)
    sample("shot_embeddings_dim", 8, 128, log=True)
    sample("transform_n_fourier_freqs", 2, 6, log=True)

    sample("transform_mlp_lr", 1e-3, 5e-2, log=True)
    sample("transform_mlp_pos_coeff", 1e-2, 10.0, log=True)
    sample("transform_mlp_rot_coeff", 1e-2, 10.0, log=True)

    sample("gemb_smoothness_reg", 1e-5, 0.5, log=True)
    sample("gemb_smoothness_lambda", 200.0, 20000.0, log=True)
    sample("semb_weight_decay", 1e-5, 0.1, log=True)
    sample("gemb_weight_decay", 1e-5, 1.0, log=True)
    sample("transform_mlp_weight_decay", [0.0, 1e-6, 1e-5, 1e-3, 1e-2, 1e-1])

    sample("stability_pos_reg", [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    sample("stability_rot_reg", [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])

    sample("semb_noise", [0.0, 1e-4, 1e-3, 1e-2, 1e-1])
    sample("gemb_noise", [0.0, 1e-4, 1e-3, 1e-2, 1e-1])

    sample("semb_init_scale", [0.0, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0])

    # sample("sh_degree", [1, 2, 3])

    torch.cuda.empty_cache()
    # cfg.exp_name = f"{trial.number:03d}_" + cfg.exp_name
    print("started. sampled trial.params are", trial.params)
    runner = Runner(0, 0, 1, cfg)
    runner.train()
    metrics = runner.eval(step=cfg.max_steps)

    score = 1.0 - metrics["lpips"]
    print(f"[score={score}] finished. sampled trial.params are", trial.params)
    return score


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    cfg.disable_viewer = True

    storage = (
        "sqlite:///sqlite_optuna.db"
    )
    sampler = optuna.samplers.TPESampler(n_startup_trials=20)
    study_name = str(cfg.result_dir).split("/")[-1]
    print(f"{study_name=}")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",  # psnr
        sampler=sampler,
    )
    study.optimize(partial(objective, cfg=cfg), n_trials=50)


if __name__ == "__main__":
    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
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
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    cli(main, cfg, verbose=True)
