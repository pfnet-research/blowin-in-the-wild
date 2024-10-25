# Blowin' in the Wild: Dynamic Looping Gaussians from Still Images
Blowin' in the Wild is a method for 4D Gaussian splatting with in-the-wild still images. The photos are casually taken at irregular intervals, NOT as a video, which makes 4D novel view synthesis more challenging.

We can reconstruct in-the-wild scenes, "blowin' in the wind," by optimizing per-shot embeddings, per-Gaussian embeddings, and a transformation MLP, in addition to 3DGS parameters.
Upon reconstruction, we can see 4D novel view synthesis from the real-time viewer with manual or automatic looping dynamics by manipulating a per-shot embedding.

I'd like you to please read [the blog](https://tech.preferred.jp/ja/blog/blowin-in-the-wild-4dgs-from-still-images/) for more understanding.

![marigold_nvs](https://github.com/user-attachments/assets/712e09f2-64ad-47c5-b63f-e2f578917714)

https://github.com/user-attachments/assets/85a8ef10-2a43-450a-a3ca-c60637e70cca

Let us emphasize again: *no video source was used* for this novel view synthesis.

## Installation
This code is partially based on [gsplat](https://github.com/nerfstudio-project/gsplat), an open-source library for Gaussian splatting.

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

```bash
pip install -r requirements.txt
```

## Data Preparation
**For synthetic scenes**:  
The dataset provided in [D-NeRF](https://github.com/albertpumarola/D-NeRF) is supported. It is available on [their dropbox](https://www.dropbox.com/scl/fi/cdcmkufncwcikk1dzbgb4/data.zip?rlkey=n5m21i84v2b2xk6h7qgiu8nkg&e=1&dl=0).

**For real scenes**:  
Any colmap data can be used. Use [this script](https://github.com/graphdeco-inria/gaussian-splatting/blob/493535a5766a0f1a9b850c28f6c7d4185e8431a1/convert.py).  
We also provide [two sample datasets](https://github.com/pfnet-research/blowin-in-the-wild/releases/download/v1.0.0/blowin_dataset.zip).

## Training
For training dnerf scenes, run
```
CUDA_VISIBLE_DEVICES=0 python scripts/trainer.py default --data_dir <path-to-dnerf-data> --result_dir <path-to-result-dir> --ckpt None --init_extent 1.0 --interpolate-val --dataset_type dnerf
```

For traning colmap scenes, run
```
CUDA_VISIBLE_DEVICES=0 python scripts/trainer.py default --data_dir <path-to-dnerf-data> --result_dir <path-to-result-dir> --ckpt None --init_extent 1.0 --dataset_type colmap --strategy.refine_stop_iter 4000 --data_factor 4
```

## Rendering
For launch viewer from trained scenes, please run with ckpt:
```
# for dnerf data
CUDA_VISIBLE_DEVICES=0 python scripts/trainer.py default --data_dir <path-to-dnerf-data> --result_dir <path-to-result-dir> --ckpt <path-to-ckpt> --init_extent 1.0 --interpolate-val --dataset_type dnerf

# for real data
CUDA_VISIBLE_DEVICES=0 python scripts/trainer.py default --data_dir <path-to-dnerf-data> --result_dir <path-to-result-dir> --ckpt <path-to-ckpt> --init_extent 1.0 --dataset_type colmap --strategy.refine_stop_iter 4000 --data_factor 4
```


## Related Work

[E-D3DGS (Bae et al. 2024)](https://arxiv.org/abs/2404.03613) proposes a similar framework for 4D reconstruction, but based on well-captured *multi-view videos*.
We found that, even when observed images are taken at irregular and long intervals (i.e., when we can no longer assume temporal smoothness and continuity), our method can still reconstruct each image well using per-Gaussian and per-shot embeddings.
Furthermore, we demonstrated that the embedding space allows for smoothly controllable dynamic novel view synthesis.  
[WildGaussians (Kulhanek et al., 2024)](https://arxiv.org/abs/2407.08447) extends 3DGS to in-the-wild settings where the appearance may vary.
Analogically, our method extends 3DGS to in-the-wild settings where objects may move.


## Citation

If you use this repository or refer to the ideas in this work, please cite it as follows:

```
@misc{kohyama2024blowin,
    author = {Kai Kohyama and Toru Matsuoka and Sosuke Kobayashi and Hiroharu Kato},
    title = {Blowin' in the Wild: Dynamic Looping Gaussians from Still Images},
    howpublished = {\url{https://github.com/pfnet-research/blowin-in-the-wild}},
    year = {2024}
}
```

This work is derived from a 2024 internship project at Preferred Networks, Inc.
