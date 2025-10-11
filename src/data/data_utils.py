"""
Utility functions for data preprocessing and analysis.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json

def compute_dataset_stats(data_loader, desc="Computing stats"):
    """
    Compute mean and std of dataset for normalization.
    
    Args:
        data_loader: PyTorch DataLoader
        desc: Description for progress bar
    
    Returns:
        mean, std of the dataset
    """
    mean = 0.0
    std = 0.0
    total_samples = 0
    
    for data, _ in tqdm(data_loader, desc=desc):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return mean.item(), std.item()

def plot_class_distribution(stats_file, save_path='results/class_distribution.png'):
    """
    Plot emotion class distribution across splits.
    
    Args:
        stats_file: Path to dataset_stats.json
        save_path: Where to save the plot
    """
    # load statistics
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    # prepare data for plotting
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    splits = ['train', 'val', 'test']
    
    # create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, split in enumerate(splits):
        counts = [stats[split].get(emotion, 0) for emotion in emotions]
        total = stats[split]['total']
        
        # create bar plot
        bars = axes[idx].bar(emotions, counts)
        axes[idx].set_title(f'{split.capitalize()} Set (n={total})')
        axes[idx].set_xlabel('Emotion')
        axes[idx].set_ylabel('Number of Samples')
        axes[idx].tick_params(axis='x', rotation=45)
        
        # add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                         f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Class distribution plot saved to {save_path}")

def create_sample_grid(data_loader, save_path='results/sample_grid.png', samples_per_class=4):
    """
    Create a grid showing sample images from each emotion class.
    
    Args:
        data_loader: PyTorch DataLoader
        save_path: Where to save the grid
        samples_per_class: Number of samples to show per emotion
    """
    emotion_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    samples_dict = {i: [] for i in range(7)}
    
    # collect samples for each class
    for images, labels in data_loader:
        for img, label in zip(images, labels):
            label_idx = label.item()
            if len(samples_dict[label_idx]) < samples_per_class:
                samples_dict[label_idx].append(img)
        
        # check if we have enough samples
        if all(len(samples) >= samples_per_class for samples in samples_dict.values()):
            break
    
    # create grid
    fig, axes = plt.subplots(7, samples_per_class, figsize=(samples_per_class * 2, 14))
    
    for row, (label_idx, samples) in enumerate(samples_dict.items()):
        for col, img in enumerate(samples[:samples_per_class]):
            # denormalize image for display
            img_display = img.squeeze().numpy()
            img_display = (img_display * 0.5) + 0.5  # Convert from [-1,1] to [0,1]
            
            axes[row, col].imshow(img_display, cmap='gray')
            axes[row, col].axis('off')
            
            if col == 0:
                axes[row, col].set_ylabel(emotion_names[label_idx], rotation=0, 
                                        labelpad=40, fontsize=12)
    
    plt.suptitle('Sample Images by Emotion Class', fontsize=16)
    plt.tight_layout()
    
    # save grid
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Sample grid saved to {save_path}")

def check_data_quality(data_loader, num_batches=10):
    """
    Check data quality: look for corrupted images, check value ranges, etc.
    
    Args:
        data_loader: PyTorch DataLoader
        num_batches: Number of batches to check
    """
    print("Checking data quality...")
    
    min_vals = []
    max_vals = []
    mean_vals = []
    std_vals = []
    
    for i, (images, labels) in enumerate(data_loader):
        if i >= num_batches:
            break
        
        # check for NaN or Inf values
        if torch.isnan(images).any():
            print(f"WARNING: Batch {i} contains NaN values!")
        if torch.isinf(images).any():
            print(f"WARNING: Batch {i} contains Inf values!")
        
        # collect statistics
        min_vals.append(images.min().item())
        max_vals.append(images.max().item())
        mean_vals.append(images.mean().item())
        std_vals.append(images.std().item())
    
    # print summary
    print(f"\nData quality summary (checked {num_batches} batches):")
    print(f"  Min values: {np.min(min_vals):.3f} to {np.max(min_vals):.3f}")
    print(f"  Max values: {np.min(max_vals):.3f} to {np.max(max_vals):.3f}")
    print(f"  Mean values: {np.mean(mean_vals):.3f} ± {np.std(mean_vals):.3f}")
    print(f"  Std values: {np.mean(std_vals):.3f} ± {np.std(std_vals):.3f}")
    
    # check if normalized properly
    if np.mean(mean_vals) < -0.1 or np.mean(mean_vals) > 0.1:
        print("  Note: Data might not be properly zero-centered")
    if np.mean(std_vals) < 0.4 or np.mean(std_vals) > 0.6:
        print("  Note: Data might not be properly normalized")