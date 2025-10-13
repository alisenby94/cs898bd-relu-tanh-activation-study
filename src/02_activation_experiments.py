#!/usr/bin/env python3
"""
CS898BD Assignment 2 - Question 2: Activation Function Experiments

This script implements a 4-layer CNN to compare ReLU vs Tanh activation functions
on a custom Tiny ImageNet dataset. The model trains until reaching ≤25% training error
and generates comparative analysis plots.

Author: CS898BD Student
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import os
from PIL import Image
from tqdm import tqdm


def setup_environment():
    """Setup PyTorch environment and device selection"""
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Select CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    return device


class TinyImageNetDataset(Dataset):
    """Custom Dataset class for Tiny ImageNet data"""
    
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
        # Create label mapping (labels might not be 0-99)
        unique_labels = sorted(list(set([item['label'] for item in data])))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']  # PIL Image
        label = self.label_map[item['label']]  # Map to 0-99
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def load_dataset():
    """Load the preprocessed dataset from Question 1"""
    print("Loading preprocessed dataset from Question 1...")
    
    try:
        with open('data/processed/train_split.pkl', 'rb') as f:
            train_data = pickle.load(f)
        
        with open('data/processed/test_split.pkl', 'rb') as f:
            test_data = pickle.load(f)
            
        with open('data/processed/valid_split.pkl', 'rb') as f:
            valid_data = pickle.load(f)
            
        with open('data/processed/selected_classes.pkl', 'rb') as f:
            selected_classes = pickle.load(f)
            
        print(f"Dataset loaded successfully!")
        print(f"  Train samples: {len(train_data)}")
        print(f"  Test samples: {len(test_data)}")
        print(f"  Validation samples: {len(valid_data)}")
        print(f"  Classes: {len(selected_classes)}")
        
        return train_data, test_data, valid_data, selected_classes
        
    except FileNotFoundError as e:
        print(f"Error: Could not find dataset files. Please run 01_dataset_preparation.ipynb first.")
        raise e


def create_data_loaders(train_data, test_data, valid_data, batch_size=64):
    """Create PyTorch DataLoaders with proper transforms"""
    
    # Define transforms - convert grayscale to RGB to handle channel mismatch
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Lambda(lambda x: x.convert('RGB')),  # Ensure 3 channels
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Lambda(lambda x: x.convert('RGB')),  # Ensure 3 channels
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = TinyImageNetDataset(train_data, transform=train_transform)
    test_dataset = TinyImageNetDataset(test_data, transform=test_transform)
    valid_dataset = TinyImageNetDataset(valid_data, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Data loaders created!")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Validation batches: {len(valid_loader)}")
    
    return train_loader, test_loader, valid_loader


class SimpleCNN(nn.Module):
    """4-Layer CNN with improved spatial preservation for Tiny ImageNet"""
    
    def __init__(self, num_classes=100, activation='relu'):
        super(SimpleCNN, self).__init__()
        
        self.activation_type = activation
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # REDESIGNED: Only 2 MaxPool operations - preserve much more spatial detail!
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)    # 64x64 -> 64x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                   # 64x64 -> 32x32
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 32x32 -> 32x32
        # NO pooling here - preserve spatial detail!
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) # 32x32 -> 32x32
        # NO pooling here - preserve spatial detail!
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1) # 32x32 -> 32x32
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)                   # 32x32 -> 16x16 (FINAL pool)
        
        self.dropout = nn.Dropout(0.5)
        
        # MUCH BETTER: 16x16 spatial size retains meaningful feature maps!
        self.fc1 = nn.Linear(512 * 16 * 16, 2048)  # 16X more spatial information than 8x8!
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # REDESIGNED: Minimal spatial reduction - only 2 pools total
        x = self.pool1(self.activation(self.conv1(x)))  # 64x64 -> 32x32
        x = self.activation(self.conv2(x))              # 32x32 -> 32x32 (NO pool - preserve!)
        x = self.activation(self.conv3(x))              # 32x32 -> 32x32 (NO pool - preserve!)
        x = self.pool4(self.activation(self.conv4(x)))  # 32x32 -> 16x16 (final pool only)
        
        # Flatten for FC layers - much larger feature maps!
        x = x.view(x.size(0), -1)
        
        # FC layers with dropout
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.fc3(x)
        
        return x


def create_models(device, num_classes=100):
    """Create ReLU and Tanh models"""
    print("Creating CNN models...")
    
    model_relu = SimpleCNN(num_classes=num_classes, activation='relu').to(device)
    model_tanh = SimpleCNN(num_classes=num_classes, activation='tanh').to(device)
    
    print("Models created!")    
    
    return model_relu, model_tanh


def train_model(model, train_loader, valid_loader, model_name, device, max_epochs=100, target_error=0.25):
    """
    Train model until training error <= 25% or max epochs reached
    Returns: training_losses, training_errors, epoch_times, validation_errors
    """
    criterion = nn.CrossEntropyLoss()
    alpha = 0.003  # Learning rate
    optimizer = optim.Adam(model.parameters(), lr=alpha)
    
    training_losses = []
    training_errors = []
    validation_errors = []
    epoch_times = []
    
    print(f"\n" + "="*70)
    print(f"TRAINING {model_name.upper()} MODEL")
    print(f"Target: Stop when training error ≤ {target_error*100}%")
    print(f"Optimizer: Adam (lr={alpha})")
    print("="*70)
    
    for epoch in range(max_epochs):
        # Start timing this epoch
        epoch_start_time = time.time()
        
        # Training phase with progress bar
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Create progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:3d}/{max_epochs} - Training', 
                         leave=False, ncols=100, colour='blue')
        
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct_train += pred.eq(target.view_as(pred)).sum().item()
            total_train += target.size(0)
            
            # Update progress bar with current metrics
            current_acc = correct_train / total_train * 100
            current_loss = total_loss / (batch_idx + 1)
            train_pbar.set_postfix({
                'Loss': f'{current_loss:.3f}',
                'Acc': f'{current_acc:.1f}%'
            })
        
        # Calculate training metrics
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_error = 1 - train_accuracy
        
        # Validation phase with progress bar
        model.eval()
        correct_val = 0
        total_val = 0
        
        # Create progress bar for validation batches
        val_pbar = tqdm(valid_loader, desc=f'Epoch {epoch+1:3d}/{max_epochs} - Validation', 
                       leave=False, ncols=100, colour='green')
        
        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct_val += pred.eq(target.view_as(pred)).sum().item()
                total_val += target.size(0)
                
                # Update validation progress bar
                current_val_acc = correct_val / total_val * 100
                val_pbar.set_postfix({'Acc': f'{current_val_acc:.1f}%'})
        
        val_accuracy = correct_val / total_val
        val_error = 1 - val_accuracy
        
        # End timing this epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # Store metrics
        training_losses.append(avg_train_loss)
        training_errors.append(train_error)
        validation_errors.append(val_error)
        epoch_times.append(epoch_duration)
        
        # Print detailed progress
        print(f"Epoch {epoch+1:3d}/{max_epochs} | "
              f"Train: {train_error*100:5.2f}% | "
              f"Valid: {val_error*100:5.2f}% | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Time: {epoch_duration:5.2f}s")
        
        # Show progress towards target every 5 epochs
        if epoch == 0 or (epoch + 1) % 5 == 0:
            progress = ((target_error - train_error) / target_error * 100) if target_error > 0 else 0
            print(f"   Progress: {progress:+5.1f}% to target ({target_error*100:.1f}%)")
        
        # Check stopping condition
        if train_error <= target_error:
            print(f"\nTARGET REACHED! Training error {train_error:.4f} ({train_error*100:.2f}%) ≤ {target_error} ({target_error*100}%)")
            print(f"Final validation error: {val_error:.4f} ({val_error*100:.2f}%)")
            break
    
    # Training summary
    final_train_error = training_errors[-1]
    final_val_error = validation_errors[-1]
    total_time = sum(epoch_times)
    avg_epoch_time = total_time / len(epoch_times)
    
    print(f"\n{model_name.upper()} TRAINING SUMMARY:")
    print(f"   Completed after {len(training_losses)} epochs")
    print(f"   Final training error: {final_train_error:.4f} ({final_train_error*100:.2f}%)")
    print(f"   Final validation error: {final_val_error:.4f} ({final_val_error*100:.2f}%)")
    print(f"   Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    return training_losses, training_errors, epoch_times, validation_errors


def create_comparison_plots(relu_results, tanh_results, save_path='results/activation_comparison.png'):
    """Create comparative analysis plots as required by assignment"""
    
    relu_losses, relu_errors, relu_times, relu_val_errors = relu_results
    tanh_losses, tanh_errors, tanh_times, tanh_val_errors = tanh_results
    
    print("\nCreating comparative analysis plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. REQUIRED: Epoch Times Comparison (x=epoch, y=time in seconds)
    ax1.plot(range(1, len(relu_times) + 1), relu_times, 'b-o', label='ReLU', linewidth=2, markersize=4)
    ax1.plot(range(1, len(tanh_times) + 1), tanh_times, 'r-s', label='Tanh', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch Number')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Training Time per Epoch: ReLU vs Tanh')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Training Error Comparison
    ax2.plot(range(1, len(relu_errors) + 1), [e*100 for e in relu_errors], 'b-o', label='ReLU', linewidth=2, markersize=4)
    ax2.plot(range(1, len(tanh_errors) + 1), [e*100 for e in tanh_errors], 'r-s', label='Tanh', linewidth=2, markersize=4)
    ax2.axhline(y=25, color='k', linestyle='--', alpha=0.7, label='Target (25%)')
    ax2.set_xlabel('Epoch Number')
    ax2.set_ylabel('Training Error (%)')
    ax2.set_title('Training Error: ReLU vs Tanh')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Training Loss Comparison
    ax3.plot(range(1, len(relu_losses) + 1), relu_losses, 'b-o', label='ReLU', linewidth=2, markersize=4)
    ax3.plot(range(1, len(tanh_losses) + 1), tanh_losses, 'r-s', label='Tanh', linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch Number')
    ax3.set_ylabel('Training Loss')
    ax3.set_title('Training Loss: ReLU vs Tanh')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Validation Error Comparison
    ax4.plot(range(1, len(relu_val_errors) + 1), [e*100 for e in relu_val_errors], 'b-o', label='ReLU', linewidth=2, markersize=4)
    ax4.plot(range(1, len(tanh_val_errors) + 1), [e*100 for e in tanh_val_errors], 'r-s', label='Tanh', linewidth=2, markersize=4)
    ax4.set_xlabel('Epoch Number')
    ax4.set_ylabel('Validation Error (%)')
    ax4.set_title('Validation Error: ReLU vs Tanh')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plots saved to: {save_path}")


def print_summary_statistics(relu_results, tanh_results):
    """Print final summary statistics"""
    
    relu_losses, relu_errors, relu_times, relu_val_errors = relu_results
    tanh_losses, tanh_errors, tanh_times, tanh_val_errors = tanh_results
    
    print("\n" + "="*80)
    print("FINAL TRAINING SUMMARY")
    print("="*80)
    
    print(f"ReLU Results:")
    print(f"   Epochs: {len(relu_times)}")
    print(f"   Avg time/epoch: {np.mean(relu_times):.2f}s")
    print(f"   Final train error: {relu_errors[-1]*100:.2f}%")
    print(f"   Final valid error: {relu_val_errors[-1]*100:.2f}%")
    print(f"   Total training time: {sum(relu_times)/60:.1f} minutes")
    
    print(f"\nTanh Results:")
    print(f"   Epochs: {len(tanh_times)}")
    print(f"   Avg time/epoch: {np.mean(tanh_times):.2f}s")
    print(f"   Final train error: {tanh_errors[-1]*100:.2f}%")
    print(f"   Final valid error: {tanh_val_errors[-1]*100:.2f}%")
    print(f"   Total training time: {sum(tanh_times)/60:.1f} minutes")
    
    # Comparison
    relu_faster = len(relu_times) < len(tanh_times)
    faster_model = "ReLU" if relu_faster else "Tanh"
    epoch_diff = abs(len(relu_times) - len(tanh_times))
    
    print(f"\nCOMPARISON:")
    print(f"   {faster_model} converged faster by {epoch_diff} epochs")
    print(f"   ReLU vs Tanh final training error: {relu_errors[-1]*100:.2f}% vs {tanh_errors[-1]*100:.2f}%")
    print(f"   ReLU vs Tanh final validation error: {relu_val_errors[-1]*100:.2f}% vs {tanh_val_errors[-1]*100:.2f}%")


def main():
    """Main execution function"""
    print("Starting CS898BD Assignment 2 - Activation Function Experiments")
    print("="*80)
    
    # Setup environment
    device = setup_environment()
    
    # Load dataset
    train_data, test_data, valid_data, selected_classes = load_dataset()
    
    # Create data loaders
    train_loader, test_loader, valid_loader = create_data_loaders(
        train_data, test_data, valid_data, batch_size=64
    )
    
    # Create models
    model_relu, model_tanh = create_models(device, num_classes=len(selected_classes))
    
    # Train ReLU model
    print("\n" + "="*80)
    print("TRAINING RELU MODEL")
    print("="*80)
    relu_results = train_model(
        model_relu, train_loader, valid_loader, "ReLU", device
    )
    
    # Train Tanh model
    print("\n" + "="*80)
    print("TRAINING TANH MODEL")
    print("="*80)
    tanh_results = train_model(
        model_tanh, train_loader, valid_loader, "Tanh", device
    )
    
    # Create comparative plots
    create_comparison_plots(relu_results, tanh_results)
    
    # Print summary statistics
    print_summary_statistics(relu_results, tanh_results)
    
    print(f"\nDONE!")
    print(f"  Results saved to: results/activation_comparison.png")

if __name__ == "__main__":
    main()