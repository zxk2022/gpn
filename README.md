# Graph-based Point Cloud Networks (GPN)

A comprehensive framework for 3D point cloud classification using graph neural networks. This project implements multiple graph construction methods and various graph neural network architectures for 3D object classification tasks.

## 📋 Table of Contents

- [Graph-based Point Cloud Networks (GPN)](#graph-based-point-cloud-networks-gpn)
  - [📋 Table of Contents](#-table-of-contents)
  - [🔍 Overview](#-overview)
  - [✨ Features](#-features)
  - [🤖 Supported Models](#-supported-models)
  - [📊 Datasets](#-datasets)
  - [🛠 Installation](#-installation)
    - [Prerequisites](#prerequisites)
    - [Environment Setup](#environment-setup)
  - [🚀 Quick Start](#-quick-start)
    - [Basic Training](#basic-training)
    - [Model Performance Evaluation](#model-performance-evaluation)
    - [Batch Experiments](#batch-experiments)
  - [📈 Experimental Results](#-experimental-results)
    - [Visual Results](#visual-results)
    - [Key Findings](#key-findings)
  - [⚙️ Configuration](#️-configuration)
    - [YAML Configuration Format](#yaml-configuration-format)
    - [Graph Construction Methods](#graph-construction-methods)
  - [📁 Project Structure](#-project-structure)
  - [📄 License](#-license)
  - [🙏 Acknowledgments](#-acknowledgments)

## 🔍 Overview

This project explores the effectiveness of different graph construction methods and graph neural network architectures for 3D point cloud classification. We implement and compare:

- **Graph Construction Methods**: k-Nearest Neighbors (k-NN), Gabriel Graph, Relative Neighborhood Graph (RNG)
- **GNN Architectures**: Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), Graph Isomorphism Networks (GIN), EdgeConv, and Message Passing Networks (MR)
- **Datasets**: ModelNet10, ModelNet40, ScanObjectNN, ShapeNet-Skeleton

## ✨ Features

- **Multiple Graph Construction Methods**: Supports k-NN, Gabriel, and RNG graph construction
- **Diverse GNN Architectures**: Implements 5 different graph neural network models
- **Comprehensive Evaluation**: Evaluation on 4 different 3D point cloud datasets
- **Flexible Configuration**: YAML-based configuration system for easy experimentation
- **Performance Analysis**: Built-in metrics calculation including throughput, FLOPs, and parameters
- **Reproducible Experiments**: Supports multiple runs with statistical analysis

## 🤖 Supported Models

| Model | Description | Paper |
|-------|-------------|-------|
| **GCN** | Graph Convolutional Network | [Kipf & Welling, 2017](https://arxiv.org/abs/1609.02907) |
| **GAT** | Graph Attention Network | [Veličković et al., 2018](https://arxiv.org/abs/1710.10903) |
| **GIN** | Graph Isomorphism Network | [Xu et al., 2019](https://arxiv.org/abs/1810.00826) |
| **EdgeConv** | Edge Convolution Network | [Wang et al., 2019](https://arxiv.org/abs/1801.07829) |
| **MRNet** | Message Passing Network | Custom implementation |

## 📊 Datasets

- **ModelNet10**: 10-class 3D CAD model dataset
- **ModelNet40**: 40-class 3D CAD model dataset  
- **ScanObjectNN**: Real-world 3D scanned objects (with/without background)
- **ShapeNet-Skeleton**: Skeleton-based shape dataset

## 🛠 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric
- CUDA (optional, for GPU acceleration)

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/zxk2022/gpn.git
cd gpn
```

2. Create conda environment:
```bash
conda env create -f environment.yml
conda activate gpn
```

3. Install additional dependencies:
```bash
pip install torch-geometric
pip install tqdm pyyaml scikit-learn tensorboard
```

## 🚀 Quick Start

### Basic Training

```bash
# Train GCN with Gabriel graph on ModelNet40
python main.py --cfgs cfgs_exp/ModelNet40/gabriel/gcn_gabriel.yaml

# Train GAT with k-NN graph on ModelNet10
python main.py --cfgs cfgs_exp/ModelNet10/knn/gat_knn.yaml
```

### Model Performance Evaluation

```bash
# Evaluate model performance metrics
python main.py --cfgs cfgs_exp/ModelNet40/gabriel/gcn_gabriel.yaml --metric True --model_path path/to/model.pth
```

### Batch Experiments

```bash
# Run all experiments for a specific dataset
for config in cfgs_exp/ModelNet40/*/*.yaml; do
    python main.py --cfgs $config
done
```

## 📈 Experimental Results

### Visual Results

<div align="center">
<img src="table_exp/qualitative_results.png" width="800" alt="Qualitative Results">
<p><em>Qualitative classification results showing successful and failure cases</em></p>
</div>

<div align="center">
<img src="table_exp/performance_quantitative_test.png" width="800" alt="Performance Analysis">
<p><em>Quantitative performance analysis across different models and graph types</em></p>
</div>

<div align="center">
<img src="table_exp/graph_isomorphism_test.png" width="800" alt="Graph Isomorphism Test">
<p><em>Graph isomorphism analysis and structural properties</em></p>
</div>

### Key Findings

1. **EdgeConv consistently achieves the best performance** across all datasets and graph construction methods
2. **Gabriel graphs generally outperform k-NN graphs** for most model architectures
3. **Graph construction method impact varies by model**: GIN shows larger performance gaps between graph types
4. **ScanObjectNN without background is significantly easier** than the version with background clutter
5. **Model complexity vs. performance trade-off**: EdgeConv offers the best accuracy but with higher computational cost

## ⚙️ Configuration

### YAML Configuration Format

```yaml
# Model and training configuration
model: GCNNet                    # Model architecture
type: 0                         # Training type (0: full, 1: partial, 2: minimal)
data_dir: data/ModelNet40       # Dataset directory
output: output/Gabriel_ModelNet40 # Output directory
epochs: 300                     # Training epochs
batch: 256                      # Batch size
lr: 0.001                      # Learning rate
min_lr: 0.0000001              # Minimum learning rate
dropout: 0.5                   # Dropout rate
keys: pos                      # Input features ('pos' for coordinates, '1' for ones)
graph_type: gabriel            # Graph construction method
undirected: true               # Convert to undirected graph
configs:                       # Model-specific architecture configuration
  - [['gcn', 32], ['relu'], ['gcn', 64], ['relu'], ['gcn', 128]]
```

### Graph Construction Methods

- **k-NN**: Connects each point to its k nearest neighbors
- **Gabriel**: Gabriel graph based on empty circumcircle property
- **RNG**: Relative Neighborhood Graph with lune-based connectivity

## 📁 Project Structure

```
gpn/
├── cfgs_exp/                   # Experiment configurations
│   ├── ModelNet10/            # ModelNet10 configs
│   ├── ModelNet40/            # ModelNet40 configs
│   ├── ScanObjectNN_BG/       # ScanObjectNN with background
│   ├── ScanObjectNN_NOBG/     # ScanObjectNN without background
│   └── SHAPENET_SKEL/         # ShapeNet skeleton configs
├── datasets/                   # Dataset implementations
│   ├── modelnet10.py
│   ├── modelnet40.py
│   ├── scanobject.py
│   └── shapenet_skel.py
├── models/                     # Model implementations
│   ├── GCN.py                 # Graph Convolutional Network
│   ├── GAT.py                 # Graph Attention Network
│   ├── GIN.py                 # Graph Isomorphism Network
│   ├── Edge.py                # EdgeConv Network
│   └── MR.py                  # Message Passing Network
├── py_utils/                   # Utility functions
│   ├── train_script.py        # Training utilities
│   ├── metrics.py             # Performance metrics
│   └── knn_graph.py           # Graph construction
├── table_exp/                  # Experimental results
│   ├── qualitative_results.png
│   ├── performance_quantitative_test.png
│   └── graph_isomorphism_test.png
├── main.py                     # Main training script
└── README.md                   # This file
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for graph neural network implementations
- [ModelNet](https://modelnet.cs.princeton.edu/) and [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/) for datasets
- The graph neural network community for inspiration and baseline implementations
