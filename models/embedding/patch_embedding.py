import torch
import torch.nn as nn


class SequenceEmbedding(nn.Module):
    """
    Projects a 1D time-series sequence to an embedding dimension.
    For time-series data with shape [B, C, L] where C is the number of channels 
    and L is the sequence length.
    
    This is a pure Transformer embedding without patch-based vision operations.
    """
    def __init__(self, in_channels, embedding_dim=256, method='conv1d', 
                 segment_size=None):
        """
        Args:
            in_channels: Number of input channels (e.g., EEG electrodes, audio channels).
            embedding_dim: Dimension to project to (e.g., 256, 512).
            method: 'conv1d' or 'segment'.
            segment_size: Size of segments (required if method='segment').
        """
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.method = method
        self.segment_size = segment_size
        
        if method == 'conv1d':
            # Full sequence: 1D Convolution with kernel_size=1
            self.projection = nn.Conv1d(
                in_channels, 
                embedding_dim, 
                kernel_size=1
            )
        elif method == 'segment':
            # Segment-based: 1D Convolution with stride
            if segment_size is None:
                raise ValueError("segment_size is required for 'segment' method")
            self.projection = nn.Conv1d(
                in_channels,
                embedding_dim,
                kernel_size=segment_size,
                stride=segment_size
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'conv1d' or 'segment'")

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, C, L] (batch, channels, length).
        
        Returns:
            Embedded sequence of shape [B, L, D] (for conv1d)
            or [B, num_segments, D] (for segment), where D is the embedding dimension.
        """
        # Input: [B, C, L] -> Output: [B, D, L or num_segments]
        x = self.projection(x)
        # Transpose to [B, L or num_segments, D] for Transformer
        x = x.transpose(1, 2)
        return x

