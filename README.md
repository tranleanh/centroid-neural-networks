# CentNN & Variants

CentNN:
[![Paper](https://img.shields.io/badge/Paper-IEEE-blue)](https://ieeexplore.ieee.org/document/839021)
[![Blog](https://img.shields.io/badge/Blog-Towards_AI-blue)](https://towardsai.net/p/l/centroid-neural-network-an-efficient-and-stable-clustering-algorithm) | 
FastCentNN:
[![Preprint](https://img.shields.io/badge/Preprint-arXiv-red)](https://arxiv.org/)

The implementations of different variants of the Centroid Neural Network (CentNN) clustering algorithm.

## Introduction

CentNN is an unsupervised competitive learning algorithm that initializes with two centroids positioned at the global mean of the dataset and iteratively splits and adjusts the centroids until the desired number of clusters is reached.

<!-- <p align="center">
<img src="docs/centnn_algorithm.png" width="600">
</p>

### Process Illustration -->

<p align="center">
<img src="docs/cnn_result_gif_delay.gif" width="600">
</p>

## Variants

### ⭐ CentNN (Original)
```bash
python test_CentNN.py
```

#### Citation

```bibtex
@article{park2000centroid,
  title={Centroid neural network for unsupervised competitive learning},
  author={Park, Dong-Chul},
  journal={IEEE Transactions on Neural Networks},
  volume={11},
  number={2},
  pages={520--528},
  year={2000},
  publisher={IEEE}
}
```

### ⚡ FastCentNN (Accelerated)
```bash
python test_CentNN_vs_FastCentNN.py
```

<p align="center">
<img src="docs/movement_comparison.png" width="600">
</p>

#### Citation

```bibtex
@article{tran2026fastcentnn,
  title={FastCentNN: Accelerating Centroid Neural Network with Entropy Proxy},
  author={Tran, Le-Anh},
  journal={arXiv preprint arXiv:2607.13613},
  year={2026}
}
```

Have fun!

LA Tran
