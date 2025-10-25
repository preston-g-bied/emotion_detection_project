"""
Test script for class-specific attention visualization.
Run from project root: python test_class_specific.py
"""

import torch
import sys
import os
from pathlib import Path

# add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.attention_cnn import AttentionCNN
from src.data.preprocess import create_data_loaders
from src.visualization.class_specific_attention import ClassSpecificVisualizer

def main():
    print("=" * 70)
    print("CLASS-SPECIFIC ATTENTION VISUALIZATION TEST")
    print("=" * 70)
    
    # setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"\nUsing device: {device}")
    
    # load trained model
    model_path = 'models/checkpoints/best_attention_model.pth'
    if not Path(model_path).exists():
        print(f"\nError: Model not found at {model_path}")
        print("Please train the attention model first using train_attention.py")
        return
    
    print(f"\nLoading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model = AttentionCNN(num_classes=7)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded (Val Acc: {checkpoint['val_acc']:.2f}%)")

    # load test data
    print("\nLoading test data...")
    loaders = create_data_loaders('data/processed', batch_size=8, num_workers=2)
    test_loader = loaders['test']
    
    # get a batch
    images, labels = next(iter(test_loader))
    print(f"Loaded batch: {images.shape}")
    
    # create class-specific visualizer
    print("\nCreating class-specific attention visualizer...")
    visualizer = ClassSpecificVisualizer(model, device)

    # Test 1: All classes for one image
    print("\n" + "-" * 70)
    print("TEST 1: All Emotion Classes for Single Image")
    print("-" * 70)
    print("This shows what the model focuses on when predicting each emotion")
    visualizer.visualize_all_classes(
        images[0:1],
        save_path='results/step2_class_specific_all.png'
    )
    
    # Test 2: Class comparison across multiple images
    print("\n" + "-" * 70)
    print("TEST 2: Class-Specific Attention Comparison")
    print("-" * 70)
    print("Comparing true vs predicted class attention")
    visualizer.visualize_class_comparison(
        images,
        labels,
        num_samples=4,
        save_path='results/step2_class_specific_comparison.png'
    )
    
    # Test 3: Average attention patterns per class
    print("\n" + "-" * 70)
    print("TEST 3: Average Attention Patterns per Emotion")
    print("-" * 70)
    print("Analyzing which facial regions each emotion typically focuses on...")
    visualizer.analyze_class_attention_patterns(
        test_loader,
        num_batches=5,
        save_path='results/step2_class_specific_patterns.png'
    )
    
    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE: Class-Specific Attention")
    print("=" * 70)
    print("\nGenerated visualizations:")
    print("  • results/step2_class_specific_all.png")
    print("     → Shows all 7 emotion attention maps for one image")
    print("  • results/step2_class_specific_comparison.png")
    print("     → Compares true vs predicted class attention")
    print("  • results/step2_class_specific_patterns.png")
    print("     → Average attention patterns for each emotion")
    print("\nKey Insights to Look For:")
    print("  • Happy: Should focus on mouth/smile region")
    print("  • Angry: Should focus on eyebrows/forehead")
    print("  • Surprise: Should focus on eyes and mouth")
    print("  • Sad: Should focus on mouth corners and eyes")


if __name__ == "__main__":
    main()