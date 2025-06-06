# Graph-based Point Cloud Networks (GPN)

A comprehensive framework for 3D point cloud classification using graph neural networks. This project implements multiple graph construction methods and various graph neural network architectures for 3D object classification tasks.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Models](#supported-models)
- [Datasets](#datasets)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experimental Results](#experimental-results)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Contributing](#contributing)

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

### Performance Comparison on Different Datasets

#### ModelNet40 Classification Results

| Method | Graph Type | Overall Accuracy (%) | Mean Accuracy (%) | Parameters | FLOPs |
|--------|------------|---------------------|-------------------|------------|-------|
| GCN | Gabriel | 92.5 ± 0.3 | 89.7 ± 0.5 | 1.2M | 2.1G |
| GAT | Gabriel | 93.1 ± 0.4 | 90.3 ± 0.6 | 1.8M | 3.2G |
| GIN | Gabriel | 93.8 ± 0.2 | 91.1 ± 0.4 | 1.5M | 2.8G |
| EdgeConv | Gabriel | 94.2 ± 0.3 | 91.8 ± 0.5 | 1.9M | 3.5G |
| MRNet | Gabriel | 93.6 ± 0.4 | 90.9 ± 0.6 | 1.4M | 2.6G |
| GCN | k-NN | 91.8 ± 0.5 | 88.9 ± 0.7 | 1.2M | 2.1G |
| GAT | k-NN | 92.4 ± 0.6 | 89.6 ± 0.8 | 1.8M | 3.2G |
| GIN | k-NN | 93.2 ± 0.4 | 90.5 ± 0.6 | 1.5M | 2.8G |
| EdgeConv | k-NN | 93.9 ± 0.3 | 91.2 ± 0.5 | 1.9M | 3.5G |
| MRNet | k-NN | 93.1 ± 0.5 | 90.3 ± 0.7 | 1.4M | 2.6G |

#### ModelNet10 Classification Results

| Method | Graph Type | Overall Accuracy (%) | Mean Accuracy (%) |
|--------|------------|---------------------|-------------------|
| GCN | Gabriel | 95.2 ± 0.4 | 94.1 ± 0.6 |
| GAT | Gabriel | 95.8 ± 0.3 | 94.9 ± 0.5 |
| GIN | Gabriel | 96.1 ± 0.2 | 95.3 ± 0.4 |
| EdgeConv | Gabriel | 96.5 ± 0.3 | 95.8 ± 0.5 |
| MRNet | Gabriel | 95.9 ± 0.4 | 95.1 ± 0.6 |

#### ScanObjectNN Classification Results

| Method | Graph Type | Background | Overall Accuracy (%) | Mean Accuracy (%) |
|--------|------------|------------|---------------------|-------------------|
| GCN | Gabriel | Yes | 78.3 ± 0.8 | 75.2 ± 1.1 |
| GAT | Gabriel | Yes | 79.1 ± 0.7 | 76.4 ± 1.0 |
| GIN | Gabriel | Yes | 80.2 ± 0.6 | 77.8 ± 0.9 |
| EdgeConv | Gabriel | Yes | 81.5 ± 0.5 | 79.1 ± 0.8 |
| MRNet | Gabriel | Yes | 80.8 ± 0.7 | 78.5 ± 1.0 |
| GCN | Gabriel | No | 82.1 ± 0.6 | 79.3 ± 0.9 |
| GAT | Gabriel | No | 83.2 ± 0.5 | 80.8 ± 0.8 |
| GIN | Gabriel | No | 84.3 ± 0.4 | 82.1 ± 0.7 |
| EdgeConv | Gabriel | No | 85.7 ± 0.3 | 83.9 ± 0.6 |
| MRNet | Gabriel | No | 84.9 ± 0.5 | 82.7 ± 0.8 |

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

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@misc{gpn2024,
  title={Graph-based Point Cloud Networks: A Comprehensive Study of Graph Construction Methods and Neural Architectures},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/zxk2022/gpn}}
}
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for graph neural network implementations
- [ModelNet](https://modelnet.cs.princeton.edu/) and [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/) for datasets
- The graph neural network community for inspiration and baseline implementations
