#!/usr/bin/env python3
"""
CS898BD Assignment 2 - Question 1: Dataset Preparation

This script processes the Tiny ImageNet dataset according to assignment requirements:
- Consider only 100 classes from the dataset
- From each class, take 500 images  
- Split into training (30,000), testing (10,000), and validation (10,000) sets
- Prepare data for model training as described in the AlexNet paper

Author: Andrew Lisenby
"""

import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import os



def setup_environment():
    """Seed the environment for reproducibility"""
    print("Setting up environment...")
    
    # Setting seed for randomness
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    print("Environment setup complete!")


def load_tiny_imagenet():
    """Load Tiny ImageNet dataset from Hugging Face"""
    print("Loading Tiny ImageNet dataset...")
    print("Note: This could take a while.")
    
    # Load dataset from Hugging Face
    dataset = load_dataset("zh-plus/tiny-imagenet")
    
    # Print dataset information safely
    print("Dataset loaded successfully!")
    
    return dataset


def visualize_sample(dataset):
    """Show a sample image from the dataset"""
    print("\nVisualizing sample data...")
    
    train_data = dataset['train']

    # Show first image for verification
    sample = train_data[0]
    plt.figure(figsize=(6, 6))
    plt.imshow(sample['image'])
    plt.title(f"Sample image, Label: {sample['label']}")
    plt.axis('off')
    plt.show()
    
    return sample


def create_custom_dataset(dataset, num_classes=100, samples_per_class=500):
    """
    Create custom dataset with specified number of classes and samples per class
    """
    print(f"\nCreating custom dataset with {num_classes} classes, {samples_per_class} samples each...")
    
    # Combine train and validation sets for redistribution
    all_data = []
    all_data.extend(dataset['train'])
    all_data.extend(dataset['valid'])
    
    # Get all available labels
    all_labels = list(set([sample['label'] for sample in all_data]))
    print(f"Total classes available: {len(all_labels)}")
    
    # Select random classes (reproducible due to seed)
    selected_classes = random.sample(all_labels, num_classes)
    print(f"Selected {len(selected_classes)} random classes")
    
    # Filter dataset for selected classes and sample specified number per class
    filtered_data = []
    classes_with_insufficient_data = 0
    
    # Iterate over selected classes and extract samples from each
    for label in selected_classes:
        class_samples = [sample for sample in all_data if sample['label'] == label]
        
        if len(class_samples) >= samples_per_class:
            # Randomly sample the required number
            class_samples = random.sample(class_samples, samples_per_class)
        else:
            # Use all available samples if less than required
            classes_with_insufficient_data += 1
            print(f"Warning: Class {label} has only {len(class_samples)} samples (< {samples_per_class})")
        
        # Add to filtered dataset
        filtered_data.extend(class_samples)
    
    print(f"Total samples after filtering: {len(filtered_data)}")
    return filtered_data, selected_classes


def split_dataset(data, train_size=30000, test_size=10000, valid_size=10000):
    """
    Split dataset into train/test/validation sets
    """
    print(f"\nSplitting dataset into:")
    print(f"  Training: {train_size:,} samples")
    print(f"  Testing: {test_size:,} samples") 
    print(f"  Validation: {valid_size:,} samples")
    
    # Verify we have enough data
    total_required = train_size + test_size + valid_size
    if len(data) < total_required:
        raise ValueError(f"Insufficient data: have {len(data)}, need {total_required}")
    
    # Shuffle data for random distribution
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    # Split the data
    train_split = data_copy[:train_size]
    test_split = data_copy[train_size:train_size + test_size]
    valid_split = data_copy[train_size + test_size:train_size + test_size + valid_size]

    print(f"\nDistribution of dataset splits:")
    print(f"  Train: {len(train_split):,}")
    print(f"  Test: {len(test_split):,}")
    print(f"  Valid: {len(valid_split):,}")
    
    return train_split, test_split, valid_split


def save_dataset(train_split, test_split, valid_split, selected_classes, output_dir='data/processed'):
    """
    Save the dataset splits for later
    """
    print(f"\nSaving dataset to {output_dir}...")
    
    # Create output directory if it doesn't exist
    abs_output_dir = os.path.abspath(output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    print(f"Created directory: {abs_output_dir}")
    
    # Save each split using pickle
    files_saved = []
    
    # Pickle files for each split
    train_path = os.path.join(abs_output_dir, 'train_split.pkl')
    with open(train_path, 'wb') as f:
        pickle.dump(train_split, f)
        files_saved.append('train_split.pkl')
    
    test_path = os.path.join(abs_output_dir, 'test_split.pkl')
    with open(test_path, 'wb') as f:
        pickle.dump(test_split, f)
        files_saved.append('test_split.pkl')
    
    valid_path = os.path.join(abs_output_dir, 'valid_split.pkl')
    with open(valid_path, 'wb') as f:
        pickle.dump(valid_split, f)
        files_saved.append('valid_split.pkl')
    
    # Save selected classes for reference
    classes_path = os.path.join(abs_output_dir, 'selected_classes.pkl')
    with open(classes_path, 'wb') as f:
        pickle.dump(selected_classes, f)
        files_saved.append('selected_classes.pkl')
    
    print("Dataset saved successfully!")
    print(f"Files saved: {files_saved}")
    print(f"Location: {abs_output_dir}")

def main():
    """Main workflow for dataset preparation"""
    print("="*80)
    print("CS898BD Assignment 2 - Question 1: Dataset Preparation")
    print("Tiny ImageNet Dataset Processing")
    print("="*80)
    
    # Setup environment
    setup_environment()
    
    # Load dataset
    dataset = load_tiny_imagenet()
    
    # Show sample data
    sample = visualize_sample(dataset)
    
    # Create custom dataset (100 classes, 500 samples each)
    filtered_data, selected_classes = create_custom_dataset(dataset)
    
    # Split into train/test/validation
    train_split, test_split, valid_split = split_dataset(filtered_data)
    
    # Save dataset
    save_dataset(train_split, test_split, valid_split, selected_classes)

if __name__ == "__main__":
    main()