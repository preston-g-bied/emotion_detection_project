"""
Class-Specific Attention Visualization
Shows how attention differs for each emotion class using Grad-CAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    Generates class-specific attention maps showing what the model looks at
    for each emotion prediction.
    """

    def __init__(self, model, target_layer, device='cpu'):
        """
        Args:
            model: Trained AttentionCNN model
            target_layer: Layer to compute gradients from (e.g., model.conv4)
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.target_layer = target_layer

        # storage for gradients and activations
        self.gradients = None
        self.activations = None

        # register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, image, target_class):
        """
        Generate class activation map for a specific class.
        
        Args:
            image: Input image [1, 1, 48, 48]
            target_class: Target emotion class (0-6)
            
        Returns:
            cam: Class activation map [48, 48]
        """
        # forward pass
        self.model.zero_grad()
        output = self.model(image, return_attention=False)

        # backward pass for target class
        target = output[0, target_class]
        target.backward()

        # generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]

        # global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))

        # weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # apply ReLU
        cam = F.relu(cam)

        # normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        # resize to input size
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(48, 48),
            mode='bilinear',
            align_corners=False
        )

        return cam.squeeze().cpu().numpy()
    
class ClassSpecificVisualizer:
    """
    Visualize attention maps for all emotion classes.
    Shows what the model looks at when predicting each emotion.
    """

    def __init__(self, model, device='cpu'):
        """
        Args:
            model: Trained AttentionCNN model
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 
                               'Neutral', 'Sad', 'Surprise']
        
        # create GradCAM using the last conv layer
        self.grad_cam = GradCAM(model, model.conv4, device)

    def visualize_all_classes(self, image, save_path=None):
        """
        Visualize attention for all 7 emotion classes on one image.
        
        Args:
            image: Input image [1, 1, 48, 48] (normalized)
            save_path: Where to save the visualization
        """
        # Denormalize image
        img_np = image.squeeze().cpu().numpy()
        img_np = (img_np * 0.5 + 0.5)
        img_np = np.clip(img_np, 0, 1)
        
        # Get actual prediction
        with torch.no_grad():
            output = self.model(image.to(self.device), return_attention=False)
            probs = torch.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            pred_conf = probs[0, pred_class].item()
        
        # Generate CAM for each class
        cams = {}
        for class_idx in range(7):
            cam = self.grad_cam.generate_cam(image.to(self.device), class_idx)
            cams[class_idx] = cam
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(
            f'Class-Specific Attention Maps\n'
            f'Model Prediction: {self.emotion_labels[pred_class]} ({pred_conf:.2%})',
            fontsize=16, fontweight='bold'
        )
        
        # Original image
        axes[0, 0].imshow(img_np, cmap='gray')
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Plot CAM for each emotion
        positions = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
        
        for idx, (emotion, pos) in enumerate(zip(self.emotion_labels, positions)):
            row, col = pos
            cam = cams[idx]
            
            # Overlay CAM on original
            axes[row, col].imshow(img_np, cmap='gray', alpha=0.5)
            im = axes[row, col].imshow(cam, cmap='jet', alpha=0.6, vmin=0, vmax=1)
            
            # Highlight predicted class
            if idx == pred_class:
                title = f'{emotion}\n★ PREDICTED ★'
                axes[row, col].set_title(title, fontsize=11, 
                                        fontweight='bold', color='green')
            else:
                axes[row, col].set_title(emotion, fontsize=11)
            
            axes[row, col].axis('off')
            
            # Add colorbar to first emotion
            if idx == 0:
                cbar = plt.colorbar(im, ax=axes[row, col], 
                                   fraction=0.046, pad=0.04)
                cbar.set_label('Attention', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Class-specific visualization saved to {save_path}")
        
        plt.close()
        
        return cams
    
    def visualize_class_comparison(self, images, labels, num_samples=4, save_path=None):
        """
        Compare class-specific attention across multiple images.
        Shows only the predicted class vs. one alternative class for clarity.
        
        Args:
            images: Batch of images [B, 1, 48, 48]
            labels: True labels [B]
            num_samples: Number of samples to visualize
            save_path: Where to save
        """
        num_samples = min(num_samples, images.shape[0])
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        if num_samples == 1:
            axes = axes[np.newaxis, :]
        
        fig.suptitle('Class-Specific Attention Comparison', 
                    fontsize=16, fontweight='bold')
        
        for idx in range(num_samples):
            image = images[idx:idx+1].to(self.device)
            true_label = labels[idx].item()
            
            # Get prediction
            with torch.no_grad():
                output = self.model(image, return_attention=False)
                pred_label = output.argmax(dim=1).item()
            
            # Denormalize
            img_np = image.squeeze().cpu().numpy()
            img_np = (img_np * 0.5 + 0.5)
            img_np = np.clip(img_np, 0, 1)
            
            # Generate CAMs for predicted and true class
            pred_cam = self.grad_cam.generate_cam(image, pred_label)
            true_cam = self.grad_cam.generate_cam(image, true_label)
            
            # Original
            axes[idx, 0].imshow(img_np, cmap='gray')
            axes[idx, 0].set_title(f'Original\nTrue: {self.emotion_labels[true_label]}',
                                  fontsize=10)
            axes[idx, 0].axis('off')
            
            # True class attention
            axes[idx, 1].imshow(img_np, cmap='gray', alpha=0.5)
            axes[idx, 1].imshow(true_cam, cmap='jet', alpha=0.6, vmin=0, vmax=1)
            axes[idx, 1].set_title(f'Focus for "{self.emotion_labels[true_label]}"',
                                  fontsize=10)
            axes[idx, 1].axis('off')
            
            # Predicted class attention
            axes[idx, 2].imshow(img_np, cmap='gray', alpha=0.5)
            im = axes[idx, 2].imshow(pred_cam, cmap='jet', alpha=0.6, vmin=0, vmax=1)
            axes[idx, 2].set_title(f'Focus for "{self.emotion_labels[pred_label]}"',
                                  fontsize=10, fontweight='bold')
            axes[idx, 2].axis('off')
            
            # Difference (show where they differ)
            diff = np.abs(pred_cam - true_cam)
            axes[idx, 3].imshow(img_np, cmap='gray', alpha=0.5)
            axes[idx, 3].imshow(diff, cmap='coolwarm', alpha=0.6, vmin=0, vmax=1)
            axes[idx, 3].set_title('Attention Difference', fontsize=10)
            axes[idx, 3].axis('off')
            
            # Add colorbar for first row
            if idx == 0:
                cbar = plt.colorbar(im, ax=axes[idx, 2], 
                                   fraction=0.046, pad=0.04)
                cbar.set_label('Attention', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Class comparison visualization saved to {save_path}")
        
        plt.close()

    def analyze_class_attention_patterns(self, dataloader, num_batches=5, save_path=None):
        """
        Analyze average attention patterns for each emotion class.
        Shows which facial regions each emotion typically focuses on.
        
        Args:
            dataloader: Test dataloader
            num_batches: Number of batches to analyze
            save_path: Where to save
        """
        # Accumulate CAMs for each class
        class_cams = {i: [] for i in range(7)}
        
        print("Analyzing class-specific attention patterns...")
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            for img, label in zip(images, labels):
                img_tensor = img.unsqueeze(0).to(self.device)
                label_idx = label.item()
                
                # Generate CAM for this image's true class
                cam = self.grad_cam.generate_cam(img_tensor, label_idx)
                class_cams[label_idx].append(cam)
        
        # Average CAMs for each class
        avg_cams = {}
        for class_idx in range(7):
            if len(class_cams[class_idx]) > 0:
                avg_cams[class_idx] = np.mean(class_cams[class_idx], axis=0)
            else:
                avg_cams[class_idx] = np.zeros((48, 48))
        
        # Visualize average patterns
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Average Attention Patterns per Emotion Class',
                    fontsize=16, fontweight='bold')
        
        positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
        
        for idx, pos in enumerate(positions):
            row, col = pos
            cam = avg_cams[idx]
            
            im = axes[row, col].imshow(cam, cmap='jet', vmin=0, vmax=1)
            axes[row, col].set_title(f'{self.emotion_labels[idx]}\n'
                                    f'(n={len(class_cams[idx])} samples)',
                                    fontsize=11, fontweight='bold')
            axes[row, col].axis('off')
            
            cbar = plt.colorbar(im, ax=axes[row, col], 
                               fraction=0.046, pad=0.04)
            cbar.set_label('Avg Attention', fontsize=8)
        
        # Remove last empty subplot
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Average patterns saved to {save_path}")
        
        plt.close()
        
        return avg_cams