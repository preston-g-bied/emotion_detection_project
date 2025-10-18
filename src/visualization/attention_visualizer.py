"""
Attention Visualization System for Emotion Detection.
Generates and overlays attention heatmaps on facial images.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

class AttentionVisualizer:
    """
    Visualizes attention maps from the Attention CNN model.
    Supports multiple visualization modes and overlay options.
    """

    def __init__(self, model, device='cpu'):
        """
        Initialize the attention visualizer.
        
        Args:
            model: Trained AttentionCNN model
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 
                               'Neutral', 'Sad', 'Surprise']
        
    def extract_attention_map(self, image):
        """
        Extract attention map from a single image.
        
        Args:
            image: Input tensor [1, 1, 48, 48] or [1, 48, 48]
            
        Returns:
            attention_map: Attention weights [H, W]
            prediction: Predicted emotion class
            confidence: Prediction confidence
        """
        # ensure correct shape
        if image.dim() == 3:
            image = image.unsqueeze(0)  # add batch dimension
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0) # add batch and channel

        image = image.to(self.device)

        with torch.no_grad():
            # get prediction and attention map
            logits, attention = self.model(image, return_attention=True)

            # get prediction
            probs = F.softmax(logits, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)

            # process attention map
            # attention shape: [B, 1, H, W]
            attention_map = attention.squeeze().cpu().numpy()

        return attention_map, pred_class.item(), confidence.item()
    
    def apply_colormap(self, attention_map, colormap=cv2.COLORMAP_JET):
        """
        Apply color mapping to attention map.
        
        Args:
            attention_map: 2D numpy array [H, W]
            colormap: OpenCV colormap constant
            
        Returns:
            colored_map: RGB image [H, W, 3]
        """
        # normalize to 0-255
        attention_normalized = attention_map - attention_map.min()
        if attention_normalized.max() > 0:
            attention_normalized = attention_normalized / attention_normalized.max()
        attention_normalized = (attention_normalized * 255).astype(np.uint8)

        # apply colormap
        colored_map = cv2.applyColorMap(attention_normalized, colormap)
        colored_map = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)

        return colored_map
    
    def overlay_attention(self, original_image, attention_map, alpha=0.4,
                          colormap=cv2.COLORMAP_JET):
        """
        Overlay attention heatmap on original image.
        
        Args:
            original_image: Original image [H, W] or [H, W, 1]
            attention_map: Attention weights [H_att, W_att]
            alpha: Transparency of heatmap overlay (0=transparent, 1=opaque)
            colormap: OpenCV colormap for attention
            
        Returns:
            overlay: Combined image [H, W, 3]
        """
        # prepare original image
        if original_image.ndim == 2:
            original_image = original_image[..., np.newaxis]
        if original_image.shape[2] == 1:
            original_image = np.repeat(original_image, 3, axis=2)

        # normalize original image to 0-255
        img_normalized = original_image - original_image.min()
        if img_normalized.max() > 0:
            img_normalized = img_normalized / img_normalized.max()
        img_normalized = (img_normalized * 255).astype(np.uint8)

        # resize attention map to match image size
        h, w = img_normalized.shape[:2]
        attention_resized = cv2.resize(attention_map, (w, h))

        # apply colormap to attention
        heatmap = self.apply_colormap(attention_resized, colormap)

        # blend images
        overlay = cv2.addWeighted(img_normalized, 1-alpha, heatmap, alpha, 0)

        return overlay
    
    def visualize_single(self, image, save_path=None, show_plot=True):
        """
        Visualize attention for a single image.
        
        Args:
            image: Input tensor or numpy array
            save_path: Path to save visualization
            show_plot: Whether to display the plot
            
        Returns:
            fig: Matplotlib figure
        """
        # convert numpy to tensor if needed
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).float()
        else:
            image_tensor = image

        # extract attention and prediction
        attention_map, pred_class, confidence = self.extract_attention_map(image_tensor)

        # prepare original image for display
        if image_tensor.dim() > 2:
            img_display = image_tensor.squeeze().cpu().numpy()
        else:
            img_display = image_tensor.cpu().numpy()

        # create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # original image
        axes[0].imshow(img_display, cmap='gray')
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # attention heatmap
        im = axes[1].imshow(attention_map, cmap='jet')
        axes[1].set_title('Attention Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # overlay
        overlay = self.overlay_attention(img_display, attention_map)
        axes[2].imshow(overlay)
        axes[2].set_title('Attention Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        # add prediction info
        emotion = self.emotion_labels[pred_class]
        fig.suptitle(f'Predicted: {emotion} (Confidence: {confidence:.2%})',
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig
    
    def visualize_batch(self, images, labels=None, save_dir=None,
                        num_samples=8, show_plot=True):
        """
        Visualize attention for a batch of images.
        
        Args:
            images: Batch of images [B, 1, 48, 48]
            labels: True labels (optional)
            save_dir: Directory to save visualizations
            num_samples: Number of samples to visualize
            show_plot: Whether to display the plot
        """
        num_samples = min(num_samples, len(images))

        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes[np.newaxis, :]

        for idx in range(num_samples):
            # get image
            img = images[idx]

            # extract attention
            attention_map, pred_class, confidence = self.extract_attention_map(img)

            # prepare display image
            img_display = img.squeeze().cpu().numpy()

            # plot original
            axes[idx, 0].imshow(img_display, cmap='gray')
            if idx == 0:
                axes[idx, 0].set_title('Original', fontsize=12, fontweight='bold')
            axes[idx, 0].axis('off')

            # plot attention heatmap
            im = axes[idx, 1].imshow(attention_map, cmap='jet')
            if idx == 0:
                axes[idx, 1].set_title('Attention Heatmap', fontsize=12, fontweight='bold')
            axes[idx, 1].axis('off')

            # plot overlay
            overlay = self.overlay_attention(img_display, attention_map)
            axes[idx, 2].imshow(overlay)
            if idx == 0:
                axes[idx, 2].set_title('Overlay', fontsize=12, fontweight='bold')
            axes[idx, 2].axis('off')
            
            # add label
            emotion = self.emotion_labels[pred_class]
            label_text = f'{emotion} ({confidence:.1%})'
            if labels is not None:
                true_emotion = self.emotion_labels[labels[idx]]
                label_text = f'True: {true_emotion} | Pred: {label_text}'
                
            axes[idx, 0].set_ylabel(label_text, fontsize=10, fontweight='bold')
        
        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / 'batch_attention_visualization.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Batch visualization saved to: {save_path}")
            
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return fig
    
def test_visualizer():
    """Test the attention visualizer with a trained model."""
    import sys
    import os

    # add src to path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.attention_cnn import AttentionCNN

    print("Testing Attention Visualizer")
    print("=" * 60)

    # setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # create dummy model
    model = AttentionCNN(num_classes=7)
    model.eval()

    # create visualizeer
    visualizer = AttentionVisualizer(model, device)

    # test with dummy image
    dummy_image = torch.randn(1, 1, 48, 48)

    print("\nTesting single image visualization...")
    visualizer.visualize_single(dummy_image, show_plot=True)

    print("\nTesting batch visualization...")
    dummy_batch = torch.randn(4, 1, 48, 48)
    visualizer.visualize_batch(dummy_batch, num_samples=4, show_plot=True)

    print("\nAttention Visualizer working correctly!")

if __name__ == "__main__":
    test_visualizer()