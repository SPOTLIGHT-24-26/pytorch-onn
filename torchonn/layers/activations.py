"""
Description:
Author: V. Anirudh Puligandla (vpuligan@irb.hr)
Date: 2025-01-28
LastEditors: V. Anirudh Puligandla (vpuligan@irb.hr)
LastEditTime: 2025-01-28
"""

import numpy as np
import torch
from torch.types import Device
from torchonn.layers.base_layer import ONNBaseLayer
from torch.nn import Parameter, init
from torch import Tensor
from torch.autograd import Function
from torch.nn.functional import relu

__all__ = [
    "ElectroOptic",
    "CRelu"
    ]

class EOActivation(Function):
    '''
    Electro-optic activations as described in - 
        {Williamson, Ian AD, et al. "Reprogrammable electro-optic nonlinear 
        activation functions for optical neural networks." IEEE Journal of 
        Selected Topics in Quantum Electronics 26.1 (2019): 1-12.}
    '''
    @staticmethod
    def forward(Z: Tensor,
                alpha: Tensor,
                g: Tensor,
                phi_b: Tensor) -> Tensor:
        '''
        Forward-pass of EO activation function
        Z: tensor, Input tensor
        alpha: tensor, parameter 'alpha'
        g: tensor, parameter 'g'
        phi_b: tensor, parameter 'phi_b'
        '''
        return 1j * torch.sqrt(1 - alpha) * torch.exp(
            -1j*0.5*g*torch.conj(Z)*Z - 1j*0.5*phi_b) * torch.cos(
                0.5*g*torch.conj(Z)*Z + 0.5*phi_b) * Z
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        '''
        ctx: Context object
        inputs: Inputs are the inputs to forward()
        output: Output tensor of forward()
        '''
        # Save parameters and output of forward for backward pass
        input, alpha, g, phi_b = inputs
        ctx.save_for_backward(input, alpha, g, phi_b)
    
    @staticmethod
    def backward(ctx, grad_Z: Tensor) -> Tensor:
        '''
        ctx: context object
        grad_Z: backpropagated gradient signal from (l+1)th layer
        '''
        # get the parameters and output field computed during forward-pass
        Z, alpha, g, phi_b = ctx.saved_tensors
        zR, zI = Z.real, Z.imag
        # df_dRe - Gradient w.r.t. real part of the input
        df_dRe = torch.sqrt(1 - alpha) * torch.exp((-0.5*1j)*g*(zR - 1j*zI)*(zR + 1j*zI) - (
            0.5*1j)*phi_b) * (zR*g*(zI - 1j*zR) * torch.sin(0.5*(zR**2)*g + 0.5*(
                zI**2)*g + 0.5*phi_b) + ((zR**2)*g + 1j*zR*zI*g + 1j) * torch.cos(
                    0.5*(zR**2)*g + 0.5*(zI**2)*g + 0.5*phi_b))
        #df_dIm - Gradient w.r.t. imaginary part of the input
        df_dIm = torch.sqrt(1 - alpha) * torch.exp((-0.5*1j)*g*(zR - 1j*zI)*(zR + 1j*zI) - (
            0.5*1j) * phi_b) * (zI*g*(zI - 1j*zR) * torch.sin(0.5*(zR**2)*g + 0.5*(
                zI**2)*g + 0.5*phi_b) + (zR*zI*g + 1j*(zI**2)*g - 1) * torch.cos(
                    0.5*(zR**2)*g + 0.5*(zI**2)*g + 0.5*phi_b))
        # final gradients
        grad_Z_out = torch.conj(grad_Z) * 0.5 * (df_dRe + 1j * df_dIm) + \
                        grad_Z * 0.5 * torch.conj(df_dRe - 1j * df_dIm)
        # Return the gradient and 'None' for parameters in forward()
        return grad_Z_out, None, None, None

class ElectroOptic(ONNBaseLayer):
    '''
    Class implementing EOActivation as layer
    '''
    __constants__ = ["in_features"]
    in_features: int

    def __init__(
            self,
            in_features: int,
            bias: bool = False,
            alpha: float = 0.2,
            responsivity: float = 0.8,
            area: float = 1.0,
            V_pi: float = 10.0,
            V_bias: float = 10.0,
            R: float = 1e3,
            impedance = 120 * np.pi,
            g: float = None,
            phi_b: float = None,
            device: Device = torch.device("cpu")
            ):
        super(ElectroOptic, self).__init__(device=device)
        self.in_features = in_features

        self.alpha = torch.tensor(alpha, requires_grad=False, device=device)
        if g is not None and phi_b is not None:
            self.g = torch.tensor(g, requires_grad=False, device=device)
            self.phi_b = torch.tensor(phi_b, requires_grad=False, device=device)
        else:
            # convert "feedforward phase gain" and "phase bias" parameters
            self.g = torch.tensor((np.pi*alpha*R*responsivity*area*1e-12/2/V_pi/impedance), 
                                  requires_grad=False, device=device)
            self.phi_b = torch.tensor((np.pi*V_bias/V_pi), requires_grad=False, device=device)

        if bias:
            self.bias = Parameter(torch.Tensor(in_features).to(self.device))
            init.uniform_(self.bias, 0, 0)
        else:
            self.register_parameter("bias", None)

    def forward(self, Z: Tensor) -> Tensor:
        '''
        Z: Input tensor from (l-1)th layer
        Z_out: Output tensor after forward propagation (activation)
        '''
        Z = EOActivation.apply(Z, self.alpha, self.g, self.phi_b)
        # There is no need to implement custom backward() as this forward (below)
        # computation is well-handled by pytorch autograd
        """ Z = 1j * torch.sqrt(1 - self.alpha) * torch.exp(
            -1j*0.5*self.g*torch.conj(Z)*Z - 1j*0.5*self.phi_b) * torch.cos(
                0.5*self.g*torch.conj(Z)*Z + 0.5*self.phi_b) * Z """
        if self.bias is not None:
            Z = Z + self.bias.unsqueeze(0)

        return Z


class CRelu(ONNBaseLayer):
    '''
    Complex relu activation layer
    '''
    __constants__ = ["in_features"]
    in_features: int

    def __init__(
            self,
            in_features: int,
            bias: bool = False,
            device: Device = torch.device("cpu")
    ):
        super(CRelu, self).__init__(device=device)
        self.in_features = in_features

        if bias:
            self.bias = Parameter(torch.Tensor(in_features).to(self.device))
            init.uniform_(self.bias, 0, 0)
        else:
            self.register_parameter("bias", None)

    def forward(self, Z: Tensor) -> Tensor:
        '''
        Z: Input tensor
        '''
        Z = relu(Z.real).type(torch.complex128) + 1j * relu(Z.imag).type(torch.complex128)
        if self.bias is not None:
            Z = Z + self.bias.unsqueeze(0)

        return Z
