"""
Data preprocessing pipeline for FER2013 emotion detection dataset.
Handles normalization, data augmentation setup, and train/val/test splits.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
from pathlib import Path
from tqdm import tqdm
import shutil

class FER2013Dataset(Dataset):
    """Custom Dataset for FER2013 emotion detection"""

    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Directory with all the images organized by emotion
            split (str): 'train', 'val', or 'test'
            transform: Optional transform to be applied on images
        """
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.emotion_labels = {
            'angry': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'neutral': 4,
            'sad': 5,
            'surprise': 6
        }

        # load all image paths and labels
        self.samples = []
        for emotion_name, label in self.emotion_labels.items():
            emotion_dir = self.root_dir / emotion_name
            if emotion_dir.exists():
                for img_path in emotion_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), label))

        print(f"Loaded {len(self.samples)} samples for {split} set")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # load image
        image = Image.open(img_path).convert('L')   # ensure grayscale

        if self.transform:
            image = self.transform(image)

        return image, label
    
def get_transforms(split='train', augment=True):
    """
    Get appropriate transforms for different splits.
    
    Args:
        split: 'train', 'val', or 'test'
        augment: Whether to apply data augmentation
    """
    # base transforms - always applied
    base_transforms = [
        transforms.Resize((48, 48)), # ensure 48x48 size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # normalize to [-1, 1]
    ]

    if split == 'train' and augment:
        # training transforms with augmentation
        train_transforms = [
            transforms.Resize((48, 48)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]
        return transforms.Compose(train_transforms)
    
    # validation and test transforms (no augmentation)
    return transforms.Compose(base_transforms)

def organize_data_splits(source_dir, target_dir, val_split=0.15):
    """
    Organize data into train/val/test splits.
    FER2013 already has train/test folders, we'll create a validation split from train.
    
    Args:
        source_dir: Directory containing original data
        target_dir: Directory to save organized data
        val_split: Fraction of training data to use for validation
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # create target directories
    for split in ['train', 'val', 'test']:
        for emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
            (target_path / split / emotion).mkdir(parents=True, exist_ok=True)

    # copy test data as is
    print("Copying test data...")
    test_source = source_path / 'test'
    if test_source.exists():
        for emotion_dir in test_source.iterdir():
            if emotion_dir.is_dir():
                emotion_name = emotion_dir.name
                for img_file in tqdm(list(emotion_dir.glob('*.jpg')),
                                     desc=f"Test {emotion_name}"):
                    shutil.copy2(img_file, target_path / 'test' / emotion_name / img_file.name)

    # split training data into train and validation
    print("\nSplitting training data into train/val...")
    train_source = source_path / 'train'
    if train_source.exists():
        for emotion_dir in train_source.iterdir():
            if emotion_dir.is_dir():
                emotion_name = emotion_dir.name
                img_files = list(emotion_dir.glob('*.jpg'))
                np.random.shuffle(img_files)

                # calculate split point
                n_val = int(len(img_files) * val_split)
                val_files = img_files[:n_val]
                train_files = img_files[n_val:]

                # copy validation files
                for img_file in tqdm(val_files, desc=f"Val {emotion_name}"):
                    shutil.copy2(img_file, target_path / 'val' / emotion_name / img_file.name)

                # copy training files
                for img_file in tqdm(train_files, desc=f"Train {emotion_name}"):
                    shutil.copy2(img_file, target_path / 'train' / emotion_name / img_file.name)

    # save dataset statistics
    stats = {}
    for split in ['train', 'val', 'test']:
        stats[split] = {}
        split_path = target_path / split
        total = 0
        for emotion_dir in split_path.iterdir():
            if emotion_dir.is_dir():
                count = len(list(emotion_dir.glob('*.jpg')))
                stats[split][emotion_dir.name] = count
                total += count
        stats[split]['total'] = total

    # save statistics
    with open(target_path / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print("\nDataset statistics:")
    for split, emotions in stats.items():
        print(f"\n{split.upper()}:")
        for emotion, count in emotions.items():
            print(f"  {emotion}: {count}")
    
    return stats

def create_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        data_dir: Directory containing organized data
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
    
    Returns:
        Dictionary with train, val, and test data loaders
    """
    # create datasets
    train_dataset = FER2013Dataset(
        data_dir,
        split='train',
        transform=get_transforms('train', augment=True)
    )

    val_dataset = FER2013Dataset(
        data_dir,
        split='val',
        transform=get_transforms('val', augment=False)
    )

    test_dataset = FER2013Dataset(
        data_dir,
        split='test',
        transform=get_transforms('test', augment=False)
    )

    # create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

def visualize_augmentations(data_dir, save_dir='results/augmentation_samples'):
    """
    Visualize augmentation effects on sample images.
    """
    import matplotlib.pyplot as plt
    
    # create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # load a sample image
    dataset = FER2013Dataset(data_dir, split='train', transform=None)
    img_path, label = dataset.samples[0]
    original_img = Image.open(img_path).convert('L')
    
    # different augmentation settings
    augmentations = {
        'Original': transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor()
        ]),
        'With Rotation': transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor()
        ]),
        'With Flip': transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor()
        ]),
        'With Brightness': transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ColorJitter(brightness=0.3),
            transforms.ToTensor()
        ]),
        'All Augmentations': get_transforms('train', augment=True)
    }
    
    # create visualization
    fig, axes = plt.subplots(1, len(augmentations), figsize=(15, 3))
    
    for idx, (name, transform) in enumerate(augmentations.items()):
        # apply transform
        aug_img = transform(original_img)
        
        # convert tensor to numpy for visualization
        if isinstance(aug_img, torch.Tensor):
            aug_img = aug_img.numpy().squeeze()
        
        axes[idx].imshow(aug_img, cmap='gray')
        axes[idx].set_title(name)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/augmentation_examples.png', dpi=150)
    plt.close()
    
    print(f"Augmentation examples saved to {save_dir}/augmentation_examples.png")

if __name__ == "__main__":
    # configuration
    raw_data_dir = "data/raw"
    processed_data_dir = "data/processed"

    # organize data into train/val/test splits
    print("Organizing data splits...")
    stats = organize_data_splits(raw_data_dir, processed_data_dir, val_split=0.15)

    # create data loaders
    print("\nCreating data loaders...")
    loaders = create_data_loaders(processed_data_dir, batch_size=32)

    # test data loading
    print("\nTesting data loaders...")
    for split_name, loader in loaders.item():
        images, labels = next(iter(loader))
        print(f"{split_name} - Batch shape: {images.shape}, Labels shape: {labels.shape}")
        print(f"  Min pixel value: {images.min():.3f}, Max pixel value: {images.max():.3f}")

    # visualize augmentations
    print("\nVisualizing augmentations...")
    visualize_augmentations(processed_data_dir)

    print("\nPreprocessing complete!")