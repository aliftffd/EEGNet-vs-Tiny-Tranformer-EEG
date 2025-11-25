"""
EEGNet: Compact Convolutional Neural Network for EEG-based BCIs
Based on: Lawhern et al. (2018) "EEGNet: A Compact Convolutional Network for EEG-based BCIs"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet architecture for Motor Imagery classification
    
    Architecture:
    1. Temporal Convolution: learns frequency filters
    2. Depthwise Convolution: learns spatial filters (channel combinations)
    3. Separable Convolution: learns temporal patterns
    4. Classification layer
    """
    def __init__(self,
                 num_classes=2,
                 channels=3,  # C3, Cz, C4
                 samples=500,  # 2 seconds at 250 Hz
                 dropout_rate=0.5,
                 kernel_length=64,
                 F1=8,
                 D=2,
                 F2=16):
        """
        Args:
            num_classes: number of output classes
            channels: number of EEG channels
            samples: number of time samples per trial
            dropout_rate: dropout probability
            kernel_length: length of temporal convolution
            F1: number of temporal filters
            D: depth multiplier (spatial filters per temporal filter)
            F2: number of pointwise filters
        """
        super(EEGNet, self).__init__()
        
        self.num_classes = num_classes
        self.channels = channels
        self.samples = samples
        self.dropout_rate = dropout_rate
        
        # Layer 1: Temporal Convolution
        # Input: (batch, 1, channels, samples)
        # Output: (batch, F1, channels, samples)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        # Layer 2: Depthwise Convolution (Spatial Filter)
        # Input: (batch, F1, channels, samples)
        # Output: (batch, F1*D, 1, samples)
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(channels, 1),
            groups=F1,
            bias=False
        )
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU()
        self.pooling1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Layer 3: Separable Convolution
        # Input: (batch, F1*D, 1, samples/4)
        # Output: (batch, F2, 1, samples/4)
        self.separable_conv = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 16),
            padding=(0, 8),
            bias=False
        )
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Calculate size after convolutions and pooling
        self.feature_size = self._get_feature_size()
        
        # Classification layer
        self.fc = nn.Linear(self.feature_size, num_classes)
        
    def _get_feature_size(self):
        """
        Calculate the flattened feature size after convolutions
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.channels, self.samples)
            x = self.conv1(dummy_input)
            x = self.depthwise_conv(x)
            x = self.pooling1(x)
            x = self.separable_conv(x)
            x = self.pooling2(x)
            feature_size = x.numel()
        return feature_size
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch, 1, channels, samples)
        Returns:
            logits: (batch, num_classes)
        """
        # Block 1: Temporal Convolution
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        # Block 2: Depthwise Spatial Convolution
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.pooling1(x)
        x = self.dropout1(x)
        
        # Block 3: Separable Convolution
        x = self.separable_conv(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.pooling2(x)
        x = self.dropout2(x)
        
        # Flatten and classify
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def extract_features(self, x):
        """
        Extract features before classification layer (useful for transfer learning)
        """
        # Block 1: Temporal Convolution
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        # Block 2: Depthwise Spatial Convolution
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.pooling1(x)
        x = self.dropout1(x)
        
        # Block 3: Separable Convolution
        x = self.separable_conv(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.pooling2(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        return x


class EEGNetTransfer(nn.Module):
    """
    EEGNet with Transfer Learning support
    Allows freezing/unfreezing layers for fine-tuning
    """
    def __init__(self, pretrained_model, num_classes=2):
        super(EEGNetTransfer, self).__init__()
        
        self.feature_extractor = pretrained_model
        
        # Replace the classification layer
        feature_size = pretrained_model.feature_size
        self.fc = nn.Linear(feature_size, num_classes)
        
        # Freeze feature extractor initially
        self.freeze_features()
        
    def freeze_features(self):
        """Freeze all layers except the final classification layer"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # Ensure FC layer is trainable
        for param in self.fc.parameters():
            param.requires_grad = True
            
    def unfreeze_features(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
            
    def unfreeze_last_n_layers(self, n=1):
        """
        Unfreeze last n convolutional blocks
        n=1: unfreeze separable conv
        n=2: unfreeze separable + depthwise conv
        n=3: unfreeze all
        """
        self.freeze_features()
        
        if n >= 1:
            # Unfreeze block 3 (separable conv)
            for param in self.feature_extractor.separable_conv.parameters():
                param.requires_grad = True
            for param in self.feature_extractor.batchnorm3.parameters():
                param.requires_grad = True
                
        if n >= 2:
            # Unfreeze block 2 (depthwise conv)
            for param in self.feature_extractor.depthwise_conv.parameters():
                param.requires_grad = True
            for param in self.feature_extractor.batchnorm2.parameters():
                param.requires_grad = True
                
        if n >= 3:
            # Unfreeze block 1 (temporal conv)
            for param in self.feature_extractor.conv1.parameters():
                param.requires_grad = True
            for param in self.feature_extractor.batchnorm1.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        features = self.feature_extractor.extract_features(x)
        logits = self.fc(features)
        return logits
