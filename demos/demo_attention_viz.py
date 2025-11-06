"""
Demo script for attention visualization.
Loads trained model and visualizes attention on test samples.
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
from src.visualization.attention_visualizer import AttentionVisualizer

def main():
    """Main demo function."""
    print("="*60)
    print("Attention Visualization Demo")
    print("="*60)

    # setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}\n")

    # load trained model
    model_path = 'models/checkpoints/best_attention_model.pth'
    print(f"Loading model from: {model_path}")

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_attention.py")
        return
    
    checkpoint = torch.load(model_path, map_location=device)

    model = AttentionCNN(num_classes=7)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded! (Val acc: {checkpoint['val_acc']:.2f}%)\n")

    # create visualizer
    visualizer = AttentionVisualizer(model, device)

    # load test data
    print("Loading test data...")
    loaders = create_data_loaders('data/processed', batch_size=32, num_workers=0)
    test_loader = loaders['test']

    # get a batch of test images
    images, labels = next(iter(test_loader))
    print(f'Loaded batch: {images.shape}\n')

    # create output directory
    output_dir = Path('results/attention_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # visualize individual samples
    print('Generating visualization for individual samples...')
    for i in range(min(5, len(images))):
        save_path = output_dir / f'attention_sample_{i+1}.png'
        visualizer.visualize_single(
            images[i],
            save_path=save_path,
            show_plot=False
        )
    
    print(f'\n{min(5, len(images))} individual visualizations saved!')

    # visualize batch
    print("\nGenerating batch visualization...")
    visualizer.visualize_batch(
        images,
        labels=labels,
        save_dir=output_dir,
        num_samples=8,
        show_plot=False
    )

    print("\n" + "="*60)
    print("Visualization Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()