# GNN Cora Graph Classification

This project implements a simple Graph Convolutional Network (GCN) using PyTorch Geometric on the Cora citation network dataset.

## Project Files

- dataset.py: loads and inspects the Cora dataset
- model.py: defines the GCN model
- train_eval.py: trains and evaluates the model

## Dataset

The project uses the Planetoid Cora dataset, which is automatically loaded from the local directory:

    data/Planetoid/Cora

If the dataset is already downloaded, PyTorch Geometric will reuse the local files.

## Requirements

- Python 3.9 or later
- PyTorch
- PyTorch Geometric
- torch-scatter, torch-sparse, torch-cluster, torch-spline-conv depending on your installation method

## Installation

Create and activate a virtual environment, then install the dependencies.

    pip install torch
    pip install torch-geometric

If your platform requires additional PyTorch Geometric packages, follow the official installation instructions for your PyTorch and CUDA version.

## How It Works

1. The dataset is loaded with normalization applied to node features.
2. The GCN model takes node features and graph connectivity as input.
3. The model is trained using cross-entropy loss on the training nodes.
4. Accuracy is measured on the test nodes after training.

## Usage

Run the training script:

    python train_eval.py

You can also inspect the dataset:

    python dataset.py

## Model Architecture

The model is a two-layer GCN:

- First GCN layer maps input features to a hidden representation
- ReLU activation is applied
- Dropout is used during training
- Second GCN layer maps hidden representations to class scores

## Output

During training, the script prints:

- epoch number
- training loss

After training, it prints the final test accuracy.

## Notes

- The current project is focused on node classification on the Cora graph.
- The model uses train, validation, and test masks provided by the dataset.
- If you want a cleaner structure for a GitHub repository, it is usually better to keep dataset loading out of model.py and place it in the training script instead.

## Example Repository Structure

    GNN/
    ├── dataset.py
    ├── model.py
    ├── train_eval.py
    └── data/
        └── Planetoid/
            └── Cora/

If you want, I can also write a better polished README version with badges, project description, and installation steps formatted for GitHub.
