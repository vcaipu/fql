import functools
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.networks import MLP


class ResnetStack(nn.Module):
    """ResNet stack module."""

    def __init__(self, num_features: int, num_blocks: int, max_pooling: bool = True):
        super().__init__()
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.max_pooling = max_pooling

        # Initial convolutional layer
        self.initial_conv = nn.Conv2d(
            in_channels=3,  # Assuming RGB input
            out_channels=self.num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        nn.init.xavier_uniform_(self.initial_conv.weight)
        
        # ResNet blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.num_features,
                    out_channels=self.num_features,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.Conv2d(
                    in_channels=self.num_features,
                    out_channels=self.num_features,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ])
            nn.init.xavier_uniform_(block[0].weight)
            nn.init.xavier_uniform_(block[1].weight)
            self.blocks.append(block)

    def forward(self, x):
        # Initial convolution
        conv_out = self.initial_conv(x)
        
        # Optional max pooling
        if self.max_pooling:
            conv_out = F.max_pool2d(
                conv_out,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        
        # ResNet blocks
        for block in self.blocks:
            block_input = conv_out
            conv_out = F.relu(conv_out)
            conv_out = block[0](conv_out)
            conv_out = F.relu(conv_out)
            conv_out = block[1](conv_out)
            conv_out = conv_out + block_input
            
        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    def __init__(
        self,
        width: int = 1,
        stack_sizes: tuple = (16, 32, 32),
        num_blocks: int = 2,
        dropout_rate: float = None,
        mlp_hidden_dims: Sequence[int] = (512,),
        layer_norm: bool = False,
    ):
        super().__init__()
        self.width = width
        self.stack_sizes = stack_sizes
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.mlp_hidden_dims = mlp_hidden_dims
        self.layer_norm = layer_norm
        
        # Create ResNet stacks
        self.stack_blocks = nn.ModuleList([
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ])
        
        # Optional dropout
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(p=self.dropout_rate)
            
        # Placeholder for MLP (will be initialized in forward)
        self.mlp = None
        self._built = False

    def forward(self, x, train=True, cond_var=None):
        # Normalize input
        x = x.float() / 255.0
        
        # Process through ResNet stacks
        conv_out = x
        for idx, stack in enumerate(self.stack_blocks):
            conv_out = stack(conv_out)
            if self.dropout_rate is not None and train:
                conv_out = self.dropout(conv_out)
                
        # Final activation
        conv_out = F.relu(conv_out)
        if self.layer_norm:
            # Apply LayerNorm to the channel dimension
            # Note: PyTorch's LayerNorm requires moving channels to last dimension
            shape = conv_out.shape
            conv_out = conv_out.permute(0, 2, 3, 1).reshape(-1, shape[1])
            conv_out = nn.LayerNorm(shape[1])(conv_out)
            conv_out = conv_out.reshape(shape[0], shape[2], shape[3], shape[1]).permute(0, 3, 1, 2)
            
        # Flatten spatial dimensions
        out = conv_out.reshape(x.shape[0], -1)
        
        # Process through MLP
        if not self._built:
            self.mlp = MLP(self.mlp_hidden_dims, input_dim=out.shape[-1], 
                           activate_final=True, layer_norm=self.layer_norm)
            self._built = True
            
        out = self.mlp(out)
        
        return out


# Dictionary of encoder modules
encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
}
