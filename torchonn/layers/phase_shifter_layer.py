"""
Description:
Author: V. Anirudh Puligandla (vpuligan@irb.hr)
Date: 2025-01-28
LastEditors: V. Anirudh Puligandla (vpuligan@irb.hr)
LastEditTime: 2025-01-28
"""

import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter, init
from torch.types import Device
from torchonn.layers.base_layer import ONNBaseLayer
from torchonn.op.mzi_op import (
    PhaseQuantizer,
    phase_to_voltage,
    voltage_to_phase,
)

import numpy as np
from pyutils.compute import gen_gaussian_noise, merge_chunks
from pyutils.general import logger, print_stat
from pyutils.quantize import input_quantize_fn

from typing import Any, Dict, Tuple, Optional

class PhaseShifter(ONNBaseLayer):
    '''
    Phase shifter layer or layer of vertically cascaded MZIs in attenuation mode
    i.e., MZI with only one input and ouput and the bottom I/O ports cascaded
    '''
    __constants__ = ["in_features"]
    in_features: int
    miniblock: int
    weight: Tensor
    dtype: torch.dtype

    def __init__(
            self,
            in_features: int,
            bias: bool = False,
            miniblock: int = 4,
            dtype: torch.dtype = torch.float64,
            mode: str  = "weight",
            device: Device = torch.device("cpu")
    ):
        super(PhaseShifter, self).__init__(device=device)
        self.in_features = in_features
        self.out_features = 1
        self.dtype = dtype
        self.miniblock = miniblock
        self.grid_dim_x = int(np.ceil(self.in_features / miniblock))
        self.grid_dim_y = int(np.ceil(self.out_features / miniblock))
        self.mode = mode
        assert mode in {"weight", "sigma", "phase"}, logger.error(
            f"Mode not supported. Expected one from (weight, sigma, phase) but got {mode}."
        )
        self.v_max = 10.8
        self.v_pi = 4.36
        self.gamma = np.pi / self.v_pi**2
        self.w_bit = 32
        self.in_bit = 32

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
            self.bias = Parameter(torch.Tensor(in_features).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def build_parameters(self, mode: str = "weight") -> None:
        # weight mode
        weight = torch.empty((self.grid_dim_y, self.grid_dim_x, 1, self.miniblock), dtype=self.dtype).to(self.device)
        # Sigma mode (analogous to USV mode in MZIs in regular configuration)
        S = torch.empty((self.grid_dim_y, self.grid_dim_x, 1, self.miniblock), dtype=self.dtype).to(self.device)
        # phase mode
        phase_S = torch.empty((self.grid_dim_y, self.grid_dim_x, 1, self.miniblock), dtype=self.dtype).to(self.device)
        # TIA gain
        S_scale = torch.empty((self.grid_dim_y, self.grid_dim_x, 1, 1), dtype=self.dtype).to(self.device).float()

        if mode == "weight":
            self.weight = Parameter(weight)
        elif mode == "sigma":
            self.S = Parameter(S)
        elif mode == "phase":
            self.phase_S = Parameter(phase_S)
            self.S_scale = Parameter(S_scale)
        else:
            raise NotImplementedError
        
        for p_name, p in {
            "weight": weight,
            "S": S,
            "phase_S": phase_S,
            "S_scale": S_scale
        }.items():
            if not hasattr(self, p_name):
                self.register_buffer(p_name, p)

    def reset_parameters(self) -> None:
        if self.mode == "weight":
            init.kaiming_normal_(self.weight.data)
        elif self.mode == "sigma":
            S = init.kaiming_normal_(
                torch.empty(
                    self.grid_dim_y,
                    self.grid_dim_x,
                    1,
                    self.miniblock,
                    dtype=self.dtype,
                    device=self.device
                )
            )
            self.S.data.copy_(S)
        elif self.mode == "phase":
            S = init.kaiming_normal_(
                torch.empty(
                    self.grid_dim_y,
                    self.grid_dim_x,
                    1,
                    self.miniblock,
                    dtype=self.dtype,
                    device=self.device
                )
            )
            self.S_scale.data.copy_(S.abs().max(dim=-1, keepdim=True)[0])
            self.phase_S.data.copy_(S.div(self.S_scale.data).acos())
        else:
            raise NotImplementedError
        
        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    def build_weight_from_sigma(self, S: Tensor) -> Tensor:
        weight = S
        self.weight.data.copy_(weight)
        return weight
    
    def build_weight_from_phase(self, phase_S: Tensor) -> Tensor:
        '''
        While the MZI implementations take an update list parameter, here we have only "phase_S"
        Hence the update_list argument is not necessary
        '''
        ### not differentiable
        ### reconstruct is time-consuming, a fast method is to only reconstruct based on updated phases
        self.S.data.copy_(phase_S.cos().mul_(self.S_scale))
        
        return self.build_weight_from_sigma(self.S)
    
    def build_phase_from_sigma(self, S: Tensor) -> Tuple[Tensor, Tensor]:
        self.S_scale.data.copy_(S.data.abs().max(dim=-1, keepdim=True)[0])
        self.phase_S.data.copy_(S.data.div(self.S_scale.data).acos())

        return self.phase_S, self.S_scale
    
    def build_sigma_from_phase(self, phase_S: Tensor, S_scale: Tensor) -> Tensor:
        '''
        While the MZI implementations take an update list parameter, here we have only "phase_S"
        Hence the update_list argument is not necessary
        '''
        self.S.data.copy_(phase_S.data.cos().mul_(S_scale))
        return self.S
    
    def build_sigma_from_weight(self, weight:Tensor) -> Tensor:
        self.S.data.copy_(weight)
        return self.S
    
    def build_phase_from_weight(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        return self.build_phase_from_sigma(*self.build_sigma_from_weight(weight))
    
    def sync_parameters(self, src: str = "weight") -> None:
        '''
        description: synchronize all parameters from the source parameters
        '''
        if src == "weight":
            self.build_phase_from_weight(self.weight)
        elif src == "sigma":
            self.build_phase_from_sigma(self.S)
            self.build_weight_from_sigma(self.S)
        elif src == "phase":
            if self.w_bit < 16:
                phase_S = self.phase_S_quantizer(self.phase_S.data)
            else:
                phase_S = self.phase_S
            # phase_S is assumed to be protected, so we do not add noise to it
        else:
            raise NotImplementedError
        
    def build_weight(self) -> Tensor:
        if self.mode == "weight":
            weight = self.weight
        elif self.mode == "sigma":
            S = self.S
            weight = self.build_weight_from_sigma(S)
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
        weight = merge_chunks(weight)[: self.out_features, : self.in_features]
        x = weight*x
        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)

        return x