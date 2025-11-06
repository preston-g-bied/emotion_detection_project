"""
Test script for multi-layer attention visualization.
Run from project root: python test_multi_layer.py
"""

import torch
import sys
import os
from pathlib import Path

# add src to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.attention_cnn import AttentionCNN
from src.data.preprocess import create_data_loaders
from src.visualization.multi_layer_attention import MultiLayerAttentionExtractor

def main():
    print("=" * 70)
    print("MULTI-LAYER ATTENTION VISUALIZATION TEST")
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

    # create multi-layer extractor
    print("\nCreating multi-layer attention extractor...")
    extractor = MultiLayerAttentionExtractor(model, device)

    # test 1: single image visualization
    print("\n" + "-" * 70)
    print("TEST 1: Single Image Multi-Layer Visualization")
    print("-" * 70)
    extractor.visualize_multi_layer(
        images[0:1].to(device),
        save_path='results/step1_multi_layer_single.png'
    )
    
    # test 2: batch visualization
    print("\n" + "-" * 70)
    print("TEST 2: Batch Multi-Layer Visualization")
    print("-" * 70)
    extractor.visualize_batch_multi_layer(
        images,
        labels,
        num_samples=4,
        save_path='results/step1_multi_layer_batch.png'
    )
    
    # Cleanup
    extractor.cleanup()
    
    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE: Multi-Layer Attention")
    print("=" * 70)
    print("\nGenerated visualizations:")
    print("  • results/step1_multi_layer_single.png")
    print("  • results/step1_multi_layer_batch.png")

if __name__ == "__main__":
    main()