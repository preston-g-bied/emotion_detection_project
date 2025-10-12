
"""
Simple CNN baseline model for FER2013 emotion classification.
Architecture: 4 convolutional layers + 2 fully connected layers
No attention mechanisms - pure baseline for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    """
    Simple CNN baseline for emotion recognition.
    Input: 1x48x48 grayscale images
    Output: 7 emotion classes
    """

    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(BaselineCNN, self).__init__()

        # convolutional layers
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

        # dropout
        self.dropout = nn.Dropout(dropout_rate)

        # fully connected layers
        # after 4 pooling operations: 48 -> 24 -> 12 -> 6 -> 4
        # feature map size: 3x3x512 = 4608
        self.fc1 = nn.Linear(512 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
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

        # flatten
        x = x.view(x.size(0), -1)

        # fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
    def get_num_params(self):
        """Return the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
def test_model():
    """Test function to verify model architecture."""
    model = BaselineCNN(num_classes=7)

    # test with dummy input
    dummy_input = torch.randn(1, 1, 48, 48)
    output = model(dummy_input)

    print("Model Architecture Test:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {model.get_num_params():,}")
    print("\nModel structure:")
    print(model)

    return model

if __name__ == "__main__":
    test_model()