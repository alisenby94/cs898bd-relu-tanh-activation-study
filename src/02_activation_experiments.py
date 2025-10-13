#!/usr/bin/env python3
"""
CS898BD Assignment 2 - Question 2: Activation Function Experiments

This script implements an experimental 4-layer CNN to compare ReLU vs Tanh activation functions
on the CIFAR-10 dataset. The model trains until reaching ≤25% training error and generates
comparative analysis plots.

Author: Andrew Lisenby
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os
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


def load_cifar10_dataset():
    """Load CIFAR-10 dataset with standard normalization values"""
    print("Loading CIFAR-10 dataset...")
    
    # Define transforms for CIFAR-10 (32x32 images) using standard normalization
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation slowed down convergence with tanh
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                           std=[0.2023, 0.1994, 0.2010])  # Standard CIFAR-10 normalization
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                           std=[0.2023, 0.1994, 0.2010])  # Standard CIFAR-10 normalization
    ])
    
    # Download and create datasets with standard normalization
    train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                   download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                  download=True, transform=test_transform)
    
    # CIFAR-10 classes
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"CIFAR-10 dataset loaded successfully!")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Classes: {len(cifar10_classes)}")
    print(f"  Image size: 32x32x3")
    print(f"  Using standard CIFAR-10 normalization")
    
    return train_dataset, test_dataset, cifar10_classes


def create_data_loaders(train_dataset, test_dataset, batch_size=64, validation_split=0.1):
    """Create PyTorch DataLoaders with train/validation split"""
    
    # Split training data into train and validation
    train_size = len(train_dataset)
    val_size = int(validation_split * train_size)
    train_size = train_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create data loaders (may need to adjust num_workers based on system)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Data loaders created!")
    print(f"  Batch size: {batch_size}")
    print(f"  Train samples: {len(train_dataset)} (batches: {len(train_loader)})")
    print(f"  Validation samples: {len(val_dataset)} (batches: {len(valid_loader)})")
    print(f"  Test samples: {len(test_dataset)} (batches: {len(test_loader)})")
    
    return train_loader, test_loader, valid_loader


class SimpleCNN(nn.Module):
    """Modified 4-Layer CNN: conv1+bn1, conv2+bn2, conv3+bn3, conv4+bn4+pool4, fc1, fc2, fc3"""
    
    def __init__(self, num_classes=10, activation='relu'):
        super(SimpleCNN, self).__init__()
        
        self.activation_type = activation
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # LARGER KERNEL ARCHITECTURE WITH BATCH NORMALIZATION: More spatial context + BN for stable training
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)    # 32x32 -> 32x32 (large receptive field!)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)   # 32x32 -> 32x32 (medium kernel + no downsampling)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 32x32 -> 32x32 (medium spatial context)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # 32x32 -> 32x32 (final downsampling)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16

        self.dropout = nn.Dropout(0.5)
        
        # LARGER: 32x32 final feature maps = 131,072 features (more spatial data for tanh!)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)     # 10 classes for CIFAR-10
        
    def forward(self, x):
        # Each conv layer followed by BN and activation. BN was necessary for tanh due to vanishing gradients.
        # Final pooling to reduce feature map size for FC layers.
        x = self.activation(self.bn1(self.conv1(x)))    # 32x32 -> 32x32 (large kernel + BN)  
        x = self.activation(self.bn2(self.conv2(x)))    # 32x32 -> 32x32 (medium kernel + BN)
        x = self.activation(self.bn3(self.conv3(x)))    # 32x32 -> 32x32 (small kernel + BN)
        x = self.activation(self.bn4(self.conv4(x)))    # 32x32 -> 32x32 (small kernel + BN)
        x = self.pool4(x)                              # 32x32 -> 16x16 (final pooling for manageable feature count)

        # Flatten for FC layers
        x = x.view(x.size(0), -1)
        
        # FC layers with dropout
        # This really slowed down training without doing much to improve tanh.
        # x = self.dropout(self.activation(self.fc1(x)))
        # x = self.dropout(self.activation(self.fc2(x)))

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        
        # No activation on final layer
        x = self.fc3(x)
        
        return x


def create_models(device, num_classes=10):
    """Create ReLU and Tanh models for CIFAR-10"""
    print("Creating CNN models for CIFAR-10...")
    
    # Initialize each model with appropriate test activation function
    # Move to relevant device
    model_relu = SimpleCNN(num_classes=num_classes, activation='relu').to(device)
    model_tanh = SimpleCNN(num_classes=num_classes, activation='tanh').to(device)
    
    print("Models created!")    
    
    return model_relu, model_tanh


def train_model(model, train_loader, valid_loader, model_name, device, max_epochs=100, target_error=0.25):
    """
    Train model until training error <= 25% or max epochs reached
    Returns: training_losses, training_errors, epoch_times, validation_errors
    """
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Learning rate
    alpha = 0.001

    # Optimizer. Adam seems to work well for both activations.
    optimizer = optim.Adam(model.parameters(), lr=alpha)
    
    # Lists for tracking metrics
    training_losses = []
    training_errors = []
    validation_errors = []
    epoch_times = []
    
    print(f"\n" + "="*70)
    print(f"TRAINING {model_name.upper()} MODEL")
    print(f"Target: Training Error <= {target_error*100}%")
    print(f"Optimizer: Adam (lr={alpha})")
    print("="*70)

    # Training loop
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
        
        # End timing this epoch (ONLY training time, not validation!)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # Validation phase with progress bar (AFTER timing)
        model.eval()
        correct_val = 0
        total_val = 0
        
        # Create progress bar for validation batches
        val_pbar = tqdm(valid_loader, desc=f'Epoch {epoch+1:3d}/{max_epochs} - Validation', 
                       leave=False, ncols=100, colour='green')

        # Disable gradient calculation for validation
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
    """Create comparative analysis plots to analyze ReLU vs Tanh performance"""
    
    relu_losses, relu_errors, relu_times, relu_val_errors = relu_results
    tanh_losses, tanh_errors, tanh_times, tanh_val_errors = tanh_results
    
    print("\nCreating comparative analysis plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Epoch Times Comparison (x=epoch, y=time in seconds)
    ax1.plot(range(1, len(relu_times) + 1), relu_times, 'b-o', label='ReLU', linewidth=2, markersize=4)
    ax1.plot(range(1, len(tanh_times) + 1), tanh_times, 'r-s', label='Tanh', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch Number')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Training Time per Epoch: ReLU vs Tanh')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Training Error Comparison
    ax2.plot(range(1, len(relu_errors) + 1), [e*100 for e in relu_errors], 'b-o', label='ReLU', linewidth=2, markersize=4)
    ax2.plot(range(1, len(tanh_errors) + 1), [e*100 for e in tanh_errors], 'r-s', label='Tanh', linewidth=2, markersize=4)
    ax2.axhline(y=25, color='k', linestyle='--', alpha=0.7, label='Target (25%)')
    ax2.set_xlabel('Epoch Number')
    ax2.set_ylabel('Training Error (%)')
    ax2.set_title('Training Error: ReLU vs Tanh')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Training Loss Comparison (not required)
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


def evaluate_test_set(model_relu, model_tanh, test_loader, device):
    """Evaluate both models on the test set for final performance comparison"""
    print("\n" + "="*80)
    print("FINAL TEST SET EVALUATION")
    print("="*80)
    
    def evaluate_model(model, model_name):
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        print(f"\nEvaluating {model_name} model on test set...")
        
        # Create progress bar for test evaluation
        test_pbar = tqdm(test_loader, desc=f'{model_name} Test Evaluation', 
                        leave=False, ncols=100, colour='yellow')
        
        with torch.no_grad():
            for data, target in test_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Update progress bar
                current_acc = correct / total * 100
                test_pbar.set_postfix({'Acc': f'{current_acc:.1f}%'})
        
        test_accuracy = correct / total
        test_error = 1 - test_accuracy
        avg_test_loss = test_loss / len(test_loader)
        
        print(f"{model_name} Test Results:")
        print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"   Test Error: {test_error*100:.2f}%")
        print(f"   Test Loss: {avg_test_loss:.4f}")
        print(f"   Correct/Total: {correct}/{total}")
        
        return test_accuracy, test_error, avg_test_loss
    
    # Evaluate both models
    relu_acc, relu_err, relu_loss = evaluate_model(model_relu, "ReLU")
    tanh_acc, tanh_err, tanh_loss = evaluate_model(model_tanh, "Tanh")
    
    # Final comparison
    print(f"\n" + "="*60)
    print("FINAL TEST SET COMPARISON")
    print("="*60)
    print(f"ReLU Test Performance:")
    print(f"   Accuracy: {relu_acc*100:.2f}%  |  Error: {relu_err*100:.2f}%  |  Loss: {relu_loss:.4f}")
    print(f"Tanh Test Performance:")
    print(f"   Accuracy: {tanh_acc*100:.2f}%  |  Error: {tanh_err*100:.2f}%  |  Loss: {tanh_loss:.4f}")
    
    # Determine winner
    if relu_acc > tanh_acc:
        winner = "ReLU"
        acc_diff = (relu_acc - tanh_acc) * 100
    else:
        winner = "Tanh"
        acc_diff = (tanh_acc - relu_acc) * 100
    
    print(f"\nFINAL WINNER: {winner} by {acc_diff:.2f} percentage points!")
    
    return (relu_acc, relu_err, relu_loss), (tanh_acc, tanh_err, tanh_loss)


def main():
    """Main execution function"""
    print("Starting CS898BD Assignment 2 - Activation Function Experiments with CIFAR-10")
    print("="*80)
    
    # Setup environment
    device = setup_environment()
    
    # Load CIFAR-10 dataset
    train_dataset, test_dataset, cifar10_classes = load_cifar10_dataset()
    
    # Create data loaders
    train_loader, test_loader, valid_loader = create_data_loaders(
        train_dataset, test_dataset, batch_size=64
    )
    
    # Create models
    model_relu, model_tanh = create_models(device, num_classes=len(cifar10_classes))
    
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
    
    # Evaluate both models on test set for final performance (just because)
    _ = evaluate_test_set(model_relu, model_tanh, test_loader, device)
    
    print(f"\nDONE!")
    print(f"  Training results saved to: results/activation_comparison.png")

if __name__ == "__main__":
    main()