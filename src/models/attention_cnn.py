"""
CNN with Spatial Attention for emotion recognition.
Same architecture as baseline but with attention mechanism integrated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_attention import SpatialAttention

class AttentionCNN(nn.Module):
    """
    CNN with spatial attention for emotion recognition.
    Architecture matches baseline CNN but adds attention after conv layers.
    """

    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(AttentionCNN, self).__init__()

        # convolutional layers (same as baseline)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # attention module
        self.attention = SpatialAttention(in_channels=512, reduction_ratio=16)

        # dropout
        self.dropout = nn.Dropout(dropout_rate)

        # fully connected layers (same as baseline)
        self.fc1 = nn.Linear(512 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x, return_attention=False):
        """
        Forward pass with optional attention map return.
        
        Args:
            x: Input images [B, 1, 48, 48]
            return_attention: If True, return attention maps for visualization
            
        Returns:
            logits: Class predictions [B, num_classes]
            attention_map: (optional) Spatial attention weights [B, 1, H, W]
        """
        # conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)    # 48x48 -> 24x24

        # conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)    # 24x24 -> 12x12

        # conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)    # 12x12 -> 6x6

        # conv bloock 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)    # 6x6 -> 3x3

        # apply attention
        x, attention_map = self.attention(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        if return_attention:
            return x, attention_map
        return x
    
    def get_num_params(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
def test_model():
    """Test the attention CNN model."""
    print("Testing Attention CNN Model")
    print("=" * 60)
    
    model = AttentionCNN(num_classes=7)
    
    # test with dummy input
    dummy_input = torch.randn(2, 1, 48, 48)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # test forward pass without attention return
    output = model(dummy_input)
    print(f"Output shape (logits): {output.shape}")
    
    # test forward pass with attention return
    output, attention_map = model(dummy_input, return_attention=True)
    print(f"Output shape (with attention): {output.shape}")
    print(f"Attention map shape: {attention_map.shape}")
    
    # parameter count
    total_params = model.get_num_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    # compare with baseline
    from baseline_cnn import BaselineCNN
    baseline = BaselineCNN(num_classes=7)
    baseline_params = baseline.get_num_params()
    
    print(f"Baseline parameters: {baseline_params:,}")
    print(f"Additional parameters: {total_params - baseline_params:,} ({100*(total_params-baseline_params)/baseline_params:.1f}% increase)")
    
    print("\nAttention CNN working correctly!")
    
    return model

if __name__ == "__main__":
    test_model()