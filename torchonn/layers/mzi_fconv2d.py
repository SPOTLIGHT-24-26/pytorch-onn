import torch
from torch import Tensor
from torch.nn import Parameter, init
from torch.types import Device, _size
from torch.nn.modules.utils import _pair
from torchonn.layers.base_layer import ONNBaseLayer
from torchonn.op.mzi_op import (
    PhaseQuantizer,
    phase_to_voltage,
    voltage_to_phase,
)
from torch.nn.functional import pad

import numpy as np
from pyutils.compute import gen_gaussian_noise, merge_chunks
from pyutils.general import logger, print_stat
from pyutils.quantize import input_quantize_fn

from typing import Any, Dict, Tuple, Optional

class FourierConv2d(ONNBaseLayer):
    '''
    Fourier convolutional layer or layer of 2D array of MZIs in attenuation mode
    i.e., MZI with only one input and ouput and the bottom I/O ports attenuated.
    Note: This layer also performs spectral pooling
    Inputs: 
        in_channels (int) - 
            Number of channels in input
        out_channels (int) -
            Number of channels in output
        kernel_size (int, Tuple(int, int))- 
            Size of the kernels
        pool_size (int, Tuple(int, int)) - 
            desired size of the output image
    '''
    __constants__ = [
        "in_channels",
        "out_channels",
        "kernel_size",
        "pool_size",
        #"miniblock"
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    _in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    pool_size: Tuple[int, ...]
    weight: Tensor
    bias: Optional[Tensor]
    #miniblock: int
    dtype: torch.dtype

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size,
            pool_size: _size,
            bias: bool = False,
            #miniblock: int = 4,
            dtype: torch.dtype = torch.float64,
            mode: str  = "weight",
            photodetect: bool = True,
            device: Device = torch.device("cpu")
    ):
        super(FourierConv2d, self).__init__(device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.pool_size = _pair(pool_size)
        self.dtype = dtype
        #self.miniblock = miniblock
        #self.in_channels_flat = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        #self.grid_dim_x = int(np.ceil(self.in_channels_flat / miniblock))
        #self.grid_dim_y = int(np.ceil(self.out_channels / miniblock))
        self.mode = mode
        assert mode in {"weight", "phase"}, logger.error(
            f"Mode not supported. Expected one from (weight, phase) but got {mode}."
        )
        self.v_max = 10.8
        self.v_pi = 4.36
        self.gamma = np.pi / self.v_pi**2
        self.w_bit = 32
        self.in_bit = 32
        self.photodetect = photodetect

        ### build trainable parameters
        self.build_parameters(mode)

        ### quantization tool
        self.input_quantizer = input_quantize_fn(self.in_bit, alg="dorefa", device=self.device)
        self.phase_S_quantizer = PhaseQuantizer(
            self.w_bit,
            self.v_pi,
            self.v_max,
            gamma_noise_std=0,
            crosstalk_factor=0,
            crosstalk_filter_size=5,    # from clements decomposition
            random_state=0,
            mode="diagonal",
            device=self.device
        )

        ### default set to slow forward
        self.disable_fast_forward()
        ### default set no phase variation
        self.set_phase_variation(0)
        ### default set no gamma noise
        self.set_gamma_noise(0)
        ### default set no crosstalk
        self.set_crosstalk_factor(0)

        if bias:
            self.bias = Parameter(torch.zeros(out_channels, dtype=self.dtype).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def complex_norm_polar(self, x, upper):
        '''
        Normalization for complex numbers. Scales the magnitude of input to fit them
        on a unit circle in the complex plane
        '''
        rad = torch.abs(x)
        max_rad = torch.max(rad)
        norm_rad = rad*upper/max_rad
        x_angle = x.angle()
        return torch.view_as_complex(torch.stack([norm_rad*torch.cos(x_angle), norm_rad*torch.sin(x_angle)], dim=-1))

    def build_parameters(self, mode: str = "weight") -> None:
        # weight mode [out_channels, in_channels, pool_size[0], pool_size[1]]
        weight = torch.zeros((self.out_channels, self.in_channels, self.pool_size[0], 
                              self.pool_size[1]), dtype=self.dtype).to(self.device)

        #weight = torch.empty((self.out_channels, self.in_channels, self.kernel_size[0], 
        #                      self.kernel_size[1]), dtype=self.dtype).to(self.device)

        # Sigma mode (analogous to USV mode in MZIs in regular configuration)
        #S = torch.empty((self.grid_dim_y, self.grid_dim_x, 1, self.miniblock), dtype=self.dtype).to(self.device)

        # phase mode
        phase_S = torch.empty((self.out_channels, self.in_channels, self.pool_size[0], 
                              self.pool_size[1]), dtype=self.dtype).to(self.device)
        #phase_S = torch.empty((self.out_channels, self.in_channels, self.kernel_size[0], 
        #                      self.kernel_size[1]), dtype=self.dtype).to(self.device)
        # TIA gain
        S_scale = torch.empty((self.out_channels, self.in_channels, 1, 1), dtype=self.dtype).to(self.device).float()

        """ weight = torch.empty((self.grid_dim_y, self.grid_dim_x, self.miniblock, self.miniblock), dtype=self.dtype).to(self.device)
        # Sigma mode (analogous to USV mode in MZIs in regular configuration)
        #S = torch.empty((self.grid_dim_y, self.grid_dim_x, 1, self.miniblock), dtype=self.dtype).to(self.device)
        # phase mode
        phase_S = torch.empty((self.grid_dim_y, self.grid_dim_x, self.miniblock, self.miniblock), dtype=self.dtype).to(self.device)
        # TIA gain
        S_scale = torch.empty((self.grid_dim_y, self.grid_dim_x, self.miniblock, 1), dtype=self.dtype).to(self.device).float()
        """

        if mode == "weight":
            self.weight = Parameter(weight)
        #elif mode == "sigma":
        #    self.S = Parameter(S)
        elif mode == "phase":
            self.phase_S = Parameter(phase_S)
            self.S_scale = Parameter(S_scale)
        else:
            raise NotImplementedError
        
        for p_name, p in {
            "weight": weight,
            #"S": S,
            "phase_S": phase_S,
            "S_scale": S_scale
        }.items():
            if not hasattr(self, p_name):
                self.register_buffer(p_name, p)

    def reset_parameters(self) -> None:
        if self.mode == "weight":
            init.kaiming_normal_(self.weight.data)
        # elif self.mode == "sigma":
        #     S = init.kaiming_normal_(
        #         torch.empty(
        #             self.grid_dim_y,
        #             self.grid_dim_x,
        #             1,
        #             self.miniblock,
        #             dtype=self.dtype,
        #             device=self.device
        #         )
        #     )
        #     self.S.data.copy_(S)
        elif self.mode == "phase":
            S = init.kaiming_normal_(
                torch.empty(
                    self.out_channels,
                    self.in_channels,
                    self.kernel_size[0],
                    self.kernel_size[1],
                    dtype=self.dtype,
                    device=self.device
                )
                # torch.empty(
                #     self.grid_dim_y,
                #     self.grid_dim_x,
                #     self.miniblock,
                #     self.miniblock,
                #     dtype=self.dtype,
                #     device=self.device
                # )
            )
            self.S_scale.data.copy_(S.abs().max(dim=-1, keepdim=True)[0])
            self.phase_S.data.copy_(S.div(self.S_scale.data).acos())
        else:
            raise NotImplementedError
        
        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    """ def build_weight_from_sigma(self, S: Tensor) -> Tensor:
        weight = S
        self.weight.data.copy_(weight)
        return weight """
    
    def build_weight_from_phase(self, phase_S: Tensor) -> Tensor:
        '''
        While the MZI implementations take an update list parameter, here we have only "phase_S"
        Hence the update_list argument is not necessary
        '''
        ### not differentiable
        ### reconstruct is time-consuming, a fast method is to only reconstruct based on updated phases
        weight = phase_S.cos().mul_(self.S_scale)
        self.weight.data.copy_(weight)
        
        return weight
    
    """ def build_phase_from_sigma(self, S: Tensor) -> Tuple[Tensor, Tensor]:
        self.S_scale.data.copy_(S.data.abs().max(dim=-1, keepdim=True)[0])
        self.phase_S.data.copy_(S.data.div(self.S_scale.data).acos())

        return self.phase_S, self.S_scale """
    
    """ def build_sigma_from_phase(self, phase_S: Tensor, S_scale: Tensor) -> Tensor:
        '''
        While the MZI implementations take an update list parameter, here we have only "phase_S"
        Hence the update_list argument is not necessary
        '''
        self.S.data.copy_(phase_S.data.cos().mul_(S_scale))
        return self.S """
    
    """ def build_sigma_from_weight(self, weight:Tensor) -> Tensor:
        self.S.data.copy_(weight)
        return self.S """
    
    def build_phase_from_weight(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        self.S_scale.data.copy_(weight.data.abs().max(dim=-1, keepdim=True)[0])
        self.phase_S.data.copy_(weight.data.div(self.S_scale.data).acos())
        return self.phase_S, self.S_scale
    
    def sync_parameters(self, src: str = "weight") -> None:
        '''
        description: synchronize all parameters from the source parameters
        '''
        if src == "weight":
            self.build_phase_from_weight(self.weight)
        # elif src == "sigma":
        #     self.build_phase_from_sigma(self.S)
        #     self.build_weight_from_sigma(self.S)
        elif src == "phase":
            if self.w_bit < 16:
                phase_S = self.phase_S_quantizer(self.phase_S.data)
            else:
                phase_S = self.phase_S
            # phase_S is assumed to be protected, so we do not add noise to it
            self.build_weight_from_phase(phase_S)
        else:
            raise NotImplementedError
        
    def build_weight(self) -> Tensor:
        if self.mode == "weight":
            weight = self.weight
        # elif self.mode == "sigma":
        #     S = self.S
        #     weight = self.build_weight_from_sigma(S)
        elif self.mode == "phase":
            # not differentiable
            if self.w_bit < 16 or self.gamma_noise_std > 1e-5 or self.crosstalk_factor > 1e-5:
                phase_S = self.phase_S_quantizer(self.phase_S.data)
            else:
                phase_S = self.phase_S

            # phase_S is assumed to be protected, so we do not add noise to it
            weight = self.build_weight_from_phase(phase_S)
        else:
            raise NotImplementedError
        
        return weight
    
    def set_gamma_noise(self, noise_std: float, random_state: Optional[int] = None) -> None:
        self.gamma_noise_std = noise_std
        self.phase_S_quantizer.set_gamma_noise(noise_std, self.phase_S.size(), random_state)

    def set_crosstalk_factor(self, crosstalk_factor: float) -> None:
        self.crosstalk_factor = crosstalk_factor
        self.phase_S_quantizer.set_crosstalk_factor(crosstalk_factor)

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        self.phase_S_quantizer.set_bitwidth(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        self.input_quantizer.set_bitwidth(in_bit)

    def load_parameters(self, param_dict: Dict[str, Any]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        super().load_parameters(param_dict=param_dict)
        if self.mode == "phase":
            self.build_weight(update_list=param_dict)

    def forward(self, x: Tensor) -> Tensor:
        if self.in_bit < 16:
            x = self.input_quantizer(x)
        if not self.fast_forward_flag or self.weight is None:
            weight = self.build_weight()
        else:
            weight = self.weight
        """weight = merge_chunks(weight)[: self.out_channels, : self.in_channels_flat].view(
            -1, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        )"""

        # Reshape x and weights to (bacth_size, output_channels, input_channels, x, y) to avoid loops and take
        #       advantage of pytorch matrix multiplication optimizations
        # Sum the output over input_channels to get final shape (bacth_size, output_channels, x, y)
        batchSize = x.shape[0]
        x = x.repeat(1, self.out_channels, 1, 1)
        x = torch.reshape(x, (batchSize, self.out_channels, self.in_channels,x.size(dim=-2),x.size(dim=-1)))
        weight = weight.repeat(batchSize,1,1,1)
        weight = torch.reshape(weight, (batchSize, self.out_channels, self.in_channels, weight.size(dim=-2), weight.size(dim=-1)))
        # Select only subset of inputs within pooling range for Fourier convolution
        poolStartIdx = (int(np.ceil((x.size(dim=-2)-self.pool_size[0])/2)), int(np.ceil((x.size(dim=-1)-self.pool_size[1])/2)))
        poolEndIdx = (poolStartIdx[0]+self.pool_size[0], poolStartIdx[1]+self.pool_size[1])
        # Fourier convolutions
        x_out = x[:,:,:,poolStartIdx[0]:poolEndIdx[0],poolStartIdx[1]:poolEndIdx[1]] * weight
        x_out = torch.sum(x_out, dim=2, dtype=self.dtype)

        if self.photodetect:
            x_out = x_out.square()

        if self.bias is not None:
            x_out = x_out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return x_out