"""
Multi-layer Attention Extractor
Captures attention maps from different depths of the network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class MultiLayerAttentionExtractor:
    """
    Extract attention-like maps from multiple layers of the CNN.
    This helps visualize what features the network focuses on at different depths.
    """

    def __init__(self, model, device='cpu'):
        """
        Args:
            model: Trained AttentionCNN model
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # storage for intermediate activations
        self.activations = {}
        self.hooks = []

        # register hooks for each conv layer
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations."""

        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # hook into each conv layer
        self.hooks.append(
            self.model.conv1.register_forward_hook(get_activation('conv1'))
        )
        self.hooks.append(
            self.model.conv2.register_forward_hook(get_activation('conv2'))
        )
        self.hooks.append(
            self.model.conv3.register_forward_hook(get_activation('conv3'))
        )
        self.hooks.append(
            self.model.conv4.register_forward_hook(get_activation('conv4'))
        )

    def extract_layer_attention(self, image):
        """
        Extract attention-like maps from all conv layers.
        
        Args:
            image: Input image tensor [1, 1, 48, 48]
            
        Returns:
            dict: Layer names mapped to attention maps
        """
        self.activations = {}

        with torch.no_grad():
            # forward pass (also gets spatial attention from attention module)
            outputs, final_attention = self.model(image, return_attention=True)

        attention_maps = {}

        # process each layer's attention
        for layer_name, activation in self.activations.items():
            # use variance across channels as attention indicator
            attention = torch.var(activation, dim=1, keepdim=True)

            # normalize to [0, 1]
            attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

            # resize to match input size for visualization
            attention_resized = F.interpolate(
                attention,
                size=(48, 48),
                mode='bilinear',
                align_corners=False
            )

            attention_maps[layer_name] = attention_resized.squeeze().cpu().numpy()

        # add the final spatial attention map
        final_attention_resized = F.interpolate(
            final_attention,
            size=(48, 48),
            mode='bilinear',
            align_corners=False
        )
        attention_maps['spatial_attention'] = final_attention_resized.squeeze().cpu().numpy()

        return attention_maps, outputs

    def visualize_multi_layer(self, image, save_path=None):
        """
        Visualize attention at all layers for a single image.
        
        Args:
            image: Input image tensor [1, 1, 48, 48] (normalized)
            save_path: Where to save the visualization
        """
        # extract attention from all layers
        attention_maps, outputs = self.extract_layer_attention(image)

        # get prediction
        probs = torch.softmax(outputs, dim=1)
        pred_class = outputs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        # denormalize image for visualization
        img_np = image.squeeze().cpu().numpy()
        img_np = (img_np * 0.5 + 0.5)   # denormalize from [-1, 1] to [0, 1]
        img_np = np.clip(img_np, 0, 1)

        # create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f'Multi-Layer Attention Visualization\n'
            f'Prediction: {emotion_labels[pred_class]} ({confidence:.2%})',
            fontsize=16, fontweight='bold'
        )

        # original image
        axes[0, 0].imshow(img_np, cmap='gray')
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # plot each layer's attention
        layers = ['conv1', 'conv2', 'conv3', 'conv4', 'spatial_attention']
        layer_titles = ['Layer 1\n(Early Features)', 'Layer 2\n(Edges)', 
                       'Layer 3\n(Textures)', 'Layer 4\n(High-level)', 
                       'Final Spatial\nAttention']
        
        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        
        for idx, (layer, title) in enumerate(zip(layers, layer_titles)):
            row, col = positions[idx]
            attention = attention_maps[layer]
            
            # overlay attention on original
            axes[row, col].imshow(img_np, cmap='gray', alpha=0.6)
            im = axes[row, col].imshow(attention, cmap='jet', alpha=0.6, 
                                       vmin=0, vmax=1)
            axes[row, col].set_title(title, fontsize=11, fontweight='bold')
            axes[row, col].axis('off')
            
            # add colorbar
            cbar = plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            cbar.set_label('Attention', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Multi-layer visualization saved to {save_path}")
        
        plt.close()
        
        return attention_maps
    
    def visualize_batch_multi_layer(self, images, labels, num_samples=4, save_path=None):
        """
        Visualize multi-layer attention for a batch of images.
        
        Args:
            images: Batch of images [B, 1, 48, 48]
            labels: True labels [B]
            num_samples: Number of samples to visualize
            save_path: Where to save
        """
        num_samples = min(num_samples, images.shape[0])
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3 * num_samples))
        fig.suptitle('Multi-Layer Attention Analysis', fontsize=16, fontweight='bold')
        
        for idx in range(num_samples):
            image = images[idx:idx+1].to(self.device)
            true_label = labels[idx].item()
            
            # extract attention
            attention_maps, outputs = self.extract_layer_attention(image)
            pred_label = outputs.argmax(dim=1).item()
            
            # denormalize image
            img_np = image.squeeze().cpu().numpy()
            img_np = (img_np * 0.5 + 0.5)
            img_np = np.clip(img_np, 0, 1)
            
            # original
            axes[idx, 0].imshow(img_np, cmap='gray')
            axes[idx, 0].set_title(
                f'True: {emotion_labels[true_label]}\nPred: {emotion_labels[pred_label]}',
                fontsize=10
            )
            axes[idx, 0].axis('off')
            
            # each layer
            layers = ['conv1', 'conv2', 'conv3', 'conv4', 'spatial_attention']
            for col_idx, layer in enumerate(layers, start=1):
                attention = attention_maps[layer]
                axes[idx, col_idx].imshow(img_np, cmap='gray', alpha=0.5)
                axes[idx, col_idx].imshow(attention, cmap='jet', alpha=0.5, vmin=0, vmax=1)
                if idx == 0:
                    axes[idx, col_idx].set_title(layer.replace('_', ' ').title(), fontsize=10)
                axes[idx, col_idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Batch multi-layer visualization saved to {save_path}")
        
        plt.close()
    
    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()