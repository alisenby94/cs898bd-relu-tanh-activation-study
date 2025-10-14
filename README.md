# CS898BD AlexNet Activation Study

## Disclosure

This README was initially generated using GitHub Copilot to provide a basic project structure and overview during Repository initialization. It has since been modified in order to provide a more accurate summary and layout of the project. Some AI generated elements are still present in the final README.

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
├── src/                                    # Source code modules
│   ├── 01_dataset_preparation.py           # Question 1: Tiny ImageNet processing
│   └── 02_activation_experiments.py        # Question 2: ReLU vs Tanh comparison
├── data/                                   # Dataset storage
│   ├── processed/                          # Preprocessed datasets
│   └── cifar-10*                           # Raw CIFAR-10 dataset data
├── results/                                # Generated plots and analysis outputs
└── report/                                 # Final report
```

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Datasets

## Usage
Ensure your environment is set up prior to execution.

Quickstart from the project root directory:
- **Virtual Environment Setup**: Run `python3 -m venv .venv`
- **Source Environment**: Run `source .venv/bin/activate`
- **Requirements Installation**: Run `pip3 install -r requirements.txt` to install the project requirements.

The project is organized as two python source files in the src directory:
1. **Dataset Preparation**: Run `01_dataset_preparation.py` to download and process Tiny ImageNet
2. **Activation Experiments**: Run `02_activation_experiments.py` for CIFAR-10 CNN comparisons
3. Each program is commented thoroughly with explanations of the steps.

## Results
The ReLU activation function was significantly easier to train than Tanh. It became clear after a few attempts that the Tanh vanishing gradient was extremely hard to overcome in order to train the model to an error of less than 25%. Batch normalization after each convolutional level helped to reduce this effect, finally the maxpooling applied to the final convolutional output before the dense layers appeared to help it prioritize the important features -- allowing it to converge faster.

See `report/` for a more comprehensive summary of the research.