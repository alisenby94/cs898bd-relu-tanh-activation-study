# CS898BD AlexNet Activation Study

## Disclosure

This README was initially generated using GitHub Copilot to provide a basic project structure and overview during Repository initialization. This is a temporary placeholder that will be replaced with a comprehensive, manually-written README upon project completion.

## Assignment 2 - Deep Learning Spring 2025

This repository contains the implementation and experiments for CS898BD Assignment 2, focusing on dataset preparation and activation function analysis.

## Assignment Overview

### Question 1: Dataset Preparation
- Download and process Tiny ImageNet dataset
- Create custom dataset with 100 classes, 500 images per class
- Split into training (30,000), testing (10,000), and validation (10,000) sets
- Implement data preprocessing as described in AlexNet paper

### Question 2: Activation Function Experiments
- Implement 4-layer CNN for CIFAR-10 dataset
- Compare ReLU vs Tanh activation functions
- Train until ≤25% training error
- Analyze training time per epoch
- Generate comparative performance graphs

## Repository Structure
```
├── notebooks/                              # Jupyter notebooks for analysis and experiments
│   ├── 01_dataset_preparation.ipynb        # Question 1: Tiny ImageNet processing
│   └── 02_activation_experiments.ipynb     # Question 2: ReLU vs Tanh comparison
├── src/                                    # Source code modules
│   ├── __init__.py                         # Package initialization
│   ├── utils.py                            # Helper functions and utilities
│   ├── data_processing.py                  # Data loading and preprocessing functions
│   └── models.py                           # Neural network model definitions
├── data/                                   # Dataset storage
│   ├── raw/                                # Original datasets
│   ├── processed/                          # Preprocessed datasets
│   └── tiny_imagenet/                      # Custom 100-class subset
├── models/                                 # Saved model checkpoints
├── results/                                # Generated plots and analysis outputs
└── report/                                 # Final report and documentation
```

## Requirements
- Python 3.x
- Jupyter Notebook/JupyterLab
- PyTorch or TensorFlow
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- Other dependencies (to be updated)

## Usage
Before running the notebooks, make sure your environment is set up.

Quickstart from the project root directory:
- **Virtual Environment Setup**: Run `python3 -m venv .venv`
- **Source Environment**: Run `source .venv/bin/activate`
- **Requirements Installation**: Run `pip3 install -r requirements.txt` to install the project requirements.

The project is organized as interactive Jupyter notebooks:
1. **Dataset Preparation**: Run `01_dataset_preparation.ipynb` to download and process Tiny ImageNet
2. **Activation Experiments**: Run `02_activation_experiments.ipynb` for CIFAR-10 CNN comparisons
3. Each notebook contains detailed explanations, code, and analysis in a step-by-step format

## Results
*Results and analysis will be updated upon completion*

---
**Due Date:** October 15, 2025