import torch
import torch.nn as nn
from torchonn.layers.base_layer import ONNBaseLayer
from torch.types import Device, _size
from torch.nn.modules.utils import _pair

from typing import Tuple

class MaxPool(ONNBaseLayer):
    '''
    -Frequency domain 2D max-pooling layer.
    Selects the values with the maximum magnitude from complex-valued tensors within pooling windows
    Inputs:
        Mandatory:
            x (Tensor) -
                Input tensor of shape (batch_size, in_channels, height, width)
            kernel_size (int or tuple) -
                Size of the pooling window
        Optional:
            stride (int or tuple) -
                Stride of the pooling window. Default value is kernel_size
            padding (int or tuple) -
                Implicit zero padding to be added on both sides
            dtype (torch.dtype) -
                data type of the layer parameters, default is torch.float64
            device (str) -
                Device to run the layer on. Default is 'cpu'
    Outputs:
        out (Tensor) -
            output tensor of shape (batch_size, out_channels, height_{out}, width_{out})
    '''
    __constants__ = [
        "kernel_size",
        "stride",
        "padding"
    ]

    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dtype: torch.dtype

    def __init__(self,
                 kernel_size: _size,
                 stride: _size = None,
                 padding: _size = 0,
                 dtype: torch.dtype = torch.float64,
                 device: Device = torch.device("cpu")):
        super(MaxPool, self).__init__(device=device)
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = _pair(padding)
        self.dtype = dtype
        self.pool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)
    
    def forward(self, x):
        # call pytorch.nn.MaxPool2d on the magnitude of the input
        filter=nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, return_indices=True)
        output, indices = filter(torch.abs(x))
        x = torch.reshape(torch.gather(x.flatten(-2,-1), -1, indices.flatten(-2,-1)), output.shape)
        return x

class AdaptiveAvgPool(ONNBaseLayer):
    '''
    -Frequency domain 2D adaptive average-pooling layer.
    Applies average-pooling over complex-valued tensors to obtain a specified output size
    Inputs:
        Mandatory:
            x (Tensor) -
                Input tensor of shape (batch_size, in_channels, height, width)
            output_size (int or tuple) -
                Target output size (height, width)
        Optional:
            dtype (torch.dtype) -
                data type of the layer parameters, default is torch.float64
            device (str) -
                Device to run the layer on. Default is 'cpu'
    Outputs:
        out (Tensor) -
            output tensor of shape (batch_size, out_channels, height_{out}, width_{out})
    '''
    __constants__ = [
        "output_size"
    ]

    output_size: Tuple[int, ...]
    dtype: torch.dtype

    def __init__(self,
                 output_size: _size,
                 dtype: torch.dtype = torch.float64,
                 device: Device = torch.device("cpu")):
        super(AdaptiveAvgPool, self).__init__(device=device)
        self.output_size = torch.tensor(_pair(output_size))
        self.dtype = dtype
    
    def forward(self, x):
        input_size = torch.tensor(x.size()[-2:])
        stride = input_size // self.output_size
        kernel_size = input_size - (self.output_size - 1) * stride
        # Calculate cummulative sum tensor
        cumsum = torch.cumsum(torch.cumsum(x, dim=-2), dim=-1)
        # output tensor
        output = torch.zeros(x.size(0), x.size(1), self.output_size[0], self.output_size[1], dtype=self.dtype, device=self.device)
        # Average pooling
        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                h_start = i * stride[0]
                h_end = h_start + kernel_size[0] - 1
                w_start = j * stride[1]
                w_end = w_start + kernel_size[1] - 1
                # Calculate sum over pooling window using cummulative sum tensor
                top_left = cumsum[:, :, h_start, w_start]
                top_right = cumsum[:, :, h_start, w_end]
                bottom_left = cumsum[:, :, h_end, w_start]
                bottom_right = cumsum[:, :, h_end, w_end]
                sum_region = bottom_right - top_right - bottom_left + top_left
                output[:, :, i, j] = sum_region / (kernel_size[0] * kernel_size[1])
        x = output
        return x