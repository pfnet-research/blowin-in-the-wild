from typing import Sequence
import torch

from config import Config

class MLP(torch.nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dims: Sequence[int], output_scales: Sequence[float] | None = None,
                 n_hidden_layers: int = 2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dims = output_dims
        self.output_scales = output_scales
        self.n_hidden_layers = n_hidden_layers
        layers = [
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_dim, sum(output_dims), bias=False))
        super().__init__(*layers)
        self[-1].weight.data[:].zero_()  # output is initialized to zero

    def forward(self, x):
        out = super().forward(x)
        out_tuple = torch.split(out, self.output_dims, dim=-1)
        if self.output_scales:
            out_tuple = [out * scale for out, scale in zip(out_tuple, self.output_scales)]
        if len(out_tuple) == 1:
            return out_tuple[0]
        return out_tuple

class PosMLP(torch.nn.Module):
    def __init__(self, config: Config):
        self.config = config
        feat_in = 3 # xyz
        if "inject" not in self.config.feat_in_mlp:
            feat_in = 0
        self.mlp = MLP(
            input_dim=config.transform_embedding_dim + feat_in + 6 * self.config.transform_n_fourier_freqs,
            hidden_dim=config.transform_hidden_dim,
            output_dims=[3],
            output_scales=[self.config.transform_mlp_pos_coeff],  # TODO: different scales
            n_hidden_layers=config.transform_n_hidden_layers,
        )
        print("using transform Pos MLP...")

    def forward(self, gembedding, sembedding, mean, viewdir=None):
        if "inject" not in self.config.feat_in_mlp:
            inp = torch.cat((gembedding, sembedding), dim=-1)
        else:
            if self.config.feat_in_mlp == "inject-detach":
                mean = mean.detach()
            inp = torch.cat((mean, gembedding, sembedding), dim=-1)
        out = self.mlp(inp)
        return out

class PosRotMLP(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        feat_in = 7 # position + quarternion
        if "inject" not in self.config.feat_in_mlp:
            feat_in = 0
        self.mlp = MLP(
            input_dim=config.shot_embeddings_dim + feat_in + 6 * self.config.transform_n_fourier_freqs,
            hidden_dim=config.transform_hidden_dim,
            output_dims=[3, 4],
            output_scales=[self.config.transform_mlp_pos_coeff, self.config.transform_mlp_rot_coeff],  # TODO: different scales
            n_hidden_layers=config.transform_n_hidden_layers,
        )
        print("using transform PosRot MLP...")

    def forward(self, gembedding, sembedding, mean, quats, viewdir=None):
        if "inject" not in self.config.feat_in_mlp:
            inp = torch.cat((gembedding, sembedding), dim=-1)
        else:
            if self.config.feat_in_mlp == "inject-detach":
                mean = mean.detach()
                quats = quats.detach()
            inp = torch.cat((mean, quats, gembedding, sembedding), dim=-1)
        outs = self.mlp(inp)
        return outs

class ColorMLP(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        feat_in = 3 # rgb
        if "inject" not in self.config.feat_in_mlp:
            feat_in = 0
        self.mlp = MLP(
            input_dim=config.shot_embeddings_dim + feat_in + 6 * self.config.transform_n_fourier_freqs,
            hidden_dim=config.transform_hidden_dim,
            output_dims=[3],
            output_scales=[self.config.color_mlp_coeff],
            n_hidden_layers=config.transform_n_hidden_layers,
        )
        print("using color MLP...")

    def forward(self, gembedding, sembedding, sh0, viewdir=None):
        if "inject" not in self.config.feat_in_mlp:
            inp = torch.cat((gembedding, sembedding), dim=-1)
        else:
            if self.config.feat_in_mlp == "inject-detach":
                sh0 = sh0.detach()
            inp = torch.cat((sh0, gembedding, sembedding), dim=-1)
        outs = self.mlp(inp)
        return outs

class ColorOpacityMLP(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        feat_in = 4 # rgba
        if "inject" not in self.config.feat_in_mlp:
            feat_in = 0
        self.mlp = MLP(
            input_dim=config.shot_embeddings_dim + feat_in + 6 * self.config.transform_n_fourier_freqs,
            hidden_dim=config.transform_hidden_dim,
            output_dims=[3,1],
            output_scales=[self.config.color_mlp_coeff, self.config.opacity_mlp_coeff],
            n_hidden_layers=config.transform_n_hidden_layers,
        )
        print("using color/opacity MLP...")

    def forward(self, gembedding, sembedding, sh0, opacity, viewdir=None):
        if "inject" not in self.config.feat_in_mlp:
            inp = torch.cat((gembedding, sembedding), dim=-1)
        else:
            if self.config.feat_in_mlp == "inject-detach":
                sh0 = sh0.detach()
                opacity = opacity.detach()
            inp = torch.cat((sh0, opacity, gembedding, sembedding), dim=-1)
        outs = self.mlp(inp)
        return outs