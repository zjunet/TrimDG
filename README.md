# TrimDG

Official implementation of **"Trimming the Fat: Redundancy-Aware Acceleration Framework for DGNNs"**

## Overview

TrimDG is a redundancy-aware acceleration framework designed to speed up Dynamic Graph Neural Networks (DGNNs) by intelligently reducing computational overhead while maintaining model performance. This framework implements various state-of-the-art DGNN models and provides efficient training and evaluation pipelines for temporal graph learning tasks.

## Features

- **Multiple DGNN Models**: Support for 9 state-of-the-art dynamic graph neural network models
- **Acceleration Framework**: Redundancy-aware sampling and caching mechanisms to accelerate training
- **Link Prediction**: Comprehensive link prediction task implementation with both transductive and inductive settings
- **Flexible Configuration**: Extensive hyperparameter configuration options
- **Multiple Datasets**: Support for 17+ temporal graph datasets
- **Efficient Neighbor Sampling**: Advanced sampling strategies including temporal PageRank-based sampling

## Supported Models

- **JODIE**: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks
- **DyRep**: Learning Representations over Dynamic Graphs
- **TGAT**: Temporal Graph Attention Networks
- **TGN**: Temporal Graph Networks
- **CAWN**: Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks
- **EdgeBank**: Simple baseline using edge memory
- **TCL**: Temporal Contrastive Learning
- **GraphMixer**: Graph Mixer Networks
- **DyGFormer**: Dynamic Graph Transformer

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Pandas
- tqdm

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/TrimDG.git
cd TrimDG

# Install dependencies (create requirements.txt if needed)
pip install torch numpy pandas tqdm
```

## Dataset Preparation

### Supported Datasets

The framework supports the following datasets:
- Wikipedia, Reddit, MOOC, LastFM, Myket
- Enron, SocialEvo, UCI
- Flights, CanParl, USLegis, UNtrade, UNvote, Contacts
- BitAlpha, BitOtc

### Preprocessing

To preprocess your dataset:

```bash
python preprocess_data/preprocess_data.py --dataset_name <dataset_name>
```

The preprocessed data will be saved in the `processed_data/` directory.

## Usage

### Basic Training

Train a model using the default configuration:

```bash
python trim_link.py --model_name TGAT --dataset_name wikipedia --gpu 0
```

### Advanced Configuration

Train with custom hyperparameters:

```bash
python trim_link.py \
    --model_name DyGFormer \
    --dataset_name reddit \
    --num_neighbors 20 \
    --num_layers 2 \
    --batch_size 200 \
    --num_epochs 50 \
    --gpu 0 \
    --sample_neighbor_strategy recent \
    --cache 1 \
    --presampling_total_rate 0.6
```

### Using Shell Scripts

Run experiments using provided shell scripts:

```bash
# Run with TrimDG acceleration
bash scripts/run_trim.sh

# Run original models without acceleration
bash scripts/run_origin.sh

# Run with input-level optimization
bash scripts/run_trim_input.sh
```

## Key Parameters

- `--model_name`: Choose from JODIE, DyRep, TGAT, TGN, CAWN, TCL, GraphMixer, DyGFormer
- `--dataset_name`: Dataset to use for training
- `--num_neighbors`: Number of neighbors to sample for each node (default: 20)
- `--num_layers`: Number of model layers (default: 2)
- `--batch_size`: Training batch size (default: 200)
- `--gpu`: GPU device ID
- `--cache`: Enable caching mechanism (0 or 1)
- `--sample_neighbor_strategy`: Sampling strategy (uniform, recent, time_interval_aware)
- `--presampling_total_rate`: Rate for presampling acceleration (default: 0.6)
- `--batch_sampling`: Enable batch sampling strategy (0 or 1)
- `--num_runs`: Number of runs with different random seeds (default: 1)

## Project Structure

```
TrimDG/
├── models/              # DGNN model implementations
│   ├── TGAT.py
│   ├── DyGFormer.py
│   ├── GraphMixer.py
│   ├── TCL.py
│   ├── CAWN.py
│   ├── MemoryModel.py
│   ├── EdgeBank.py
│   └── modules.py
├── utils/               # Utility functions
│   ├── DataLoader.py
│   ├── utils.py
│   ├── metrics.py
│   ├── EarlyStopping.py
│   └── load_configs.py
├── preprocess_data/     # Data preprocessing scripts
├── processed_data/      # Preprocessed datasets
├── scripts/             # Shell scripts for running experiments
├── trim_link.py         # Main training script for link prediction
└── README.md
```

## Results

The framework provides comprehensive evaluation metrics including:
- Average Precision (AP)
- Area Under ROC Curve (AUC)
- Both transductive and inductive evaluation settings
- Support for new node evaluation

## Citation

Coming soon!


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation builds upon several open-source DGNN implementations and temporal graph learning frameworks.

## Contact

For questions or issues, please open an issue on GitHub or contact [renh2@zju.edu.cn].
