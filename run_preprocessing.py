"""
Script to run the complete preprocessing pipeline for FER2013 dataset.
Run this from the project root directory.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.preprocess import (
    organize_data_splits,
    create_data_loaders,
    visualize_augmentations
)
from src.data.data_utils import (
    compute_dataset_stats,
    plot_class_distribution,
    create_sample_grid,
    check_data_quality
)

def main():
    """Run the complete preprocessing pipeline."""

    # configuration
    raw_data_dir = "data/raw"
    processed_data_dir = "data/processed"

    print("=" * 60)
    print("FER2013 Preprocessing Pipeline")
    print("=" * 60)

    # check if raw data exists
    if not os.path.exists(raw_data_dir):
        print(f"ERROR: Raw data directory '{raw_data_dir}' not found!")
        print("Please ensure your FER2013 data is in the data/raw/ directory")
        return
    
    # organize data splits
    print("\n1. Organizing data into train/val/test splits...")
    stats = organize_data_splits(raw_data_dir, processed_data_dir, val_split=0.15)
    
    # create data loaders
    print("\n2. Creating data loaders...")
    loaders = create_data_loaders(processed_data_dir, batch_size=32)
    
    # compute dataset statistics
    print("\n3. Computing dataset statistics...")
    
    # use a loader without normalization to compute true stats
    from src.data.preprocess import FER2013Dataset, transforms
    temp_dataset = FER2013Dataset(
        processed_data_dir, 
        split='train',
        transform=transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor()
        ])
    )
    temp_loader = torch.utils.data.DataLoader(
        temp_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    
    mean, std = compute_dataset_stats(temp_loader)
    print(f"   Dataset mean: {mean:.3f}, std: {std:.3f}")
    
    # check data quality
    print("\n4. Checking data quality...")
    check_data_quality(loaders['train'], num_batches=10)
    
    # create visualizations
    print("\n5. Creating visualizations...")
    
    # plot class distribution
    plot_class_distribution(
        os.path.join(processed_data_dir, 'dataset_stats.json'),
        'results/class_distribution.png'
    )
    
    # create sample grid
    create_sample_grid(
        loaders['train'],
        'results/sample_grid.png',
        samples_per_class=5
    )
    
    # visualize augmentations
    visualize_augmentations(processed_data_dir)
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    
    # print final summary
    print("\nDataset Summary:")
    for split in ['train', 'val', 'test']:
        total = stats[split]['total']
        print(f"  {split}: {total} images")


if __name__ == "__main__":
    import torch
    main()