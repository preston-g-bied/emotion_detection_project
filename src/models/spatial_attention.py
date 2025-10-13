"""
Spatial Attention Module for Emotion Recognition.
Helps the model focus on important facial regions (eyes, mouth, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism that learns to focus on important regions.
    
    Uses a simple approach:
    1. Generate attention map from feature maps
    2. Apply attention weights to features
    3. Combine original and attended features
    """

    def __init__(self, in_channels, reduction_ratio=16):
        """
        Args:
            in_channels: Number of input channels
            reduction_ratio: Channel reduction for computational efficiency
        """
        super(SpatialAttention, self).__init__()

        # channel attention branch
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # shared MLP for channel attention
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )

        # spatial attention branch
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1)
        )

        self.sigmoid = nn.Sigmoid()

    def channel_attention(self, x):
        """Generate channel attention weights."""
        b, c, _, _ = x.size()

        # average and max pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        # combine and generate weights
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1)
    
    def spatial_attention(self, x):
        """Generate spatial attention map."""
        # compute average and max across channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # concatenate and generate spatial attention map
        concat = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.spatial_conv(concat)

        return self.sigmoid(attention_map)
    
    def forward(self, x):
        """
        Apply spatial attention to input features.
        
        Args:
            x: Input feature maps [B, C, H, W]
            
        Returns:
            attended_features: Features with attention applied
            attention_map: Spatial attention weights for visualization
        """
        # channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att

        # spatial attention
        spacial_att = self.spatial_attention(x)
        attended_features = x * spacial_att

        return attended_features, spacial_att
    
def test_attention():
    """Test the spatial attention module."""
    print("Testing Spatial Attention Module")
    print("=" * 50)

    # create module
    attention = SpatialAttention(in_channels=512, reduction_ratio=16)

    # test with dummy input (batch_size=4, channels=512, height=3, width=3)
    dummy_input = torch.randn(4, 512, 3, 3)

    print(f"Input shape: {dummy_input.shape}")
    
    # forward pass
    attended_features, attention_map = attention(dummy_input)

    print(f"Output shape: {attended_features.shape}")
    print(f"Attention map shape: {attention_map.shape}")
    print(f"Attention map range: [{attention_map.min():.3f}, {attention_map.max():.3f}]")

    # count parameters
    num_params = sum(p.numel() for p in attention.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {num_params:,}")
    
    print("\nSpatial attention module working correctly!")

if __name__ == "__main__":
    test_attention()