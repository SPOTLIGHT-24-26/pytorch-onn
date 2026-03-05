<h1><p align="center">Torch-ONN-Complex</p></h1>
This repository is an extension of the <a href="https://github.com/JeremieMelo/pytorch-onn">torch-onn</a> package to support complex-valued operations.

# Citation
This code refers to the following "most innovative paper" award winning publication:
```
@InProceedings{Puligandla_2025_ICCV,
    author    = {Puligandla, Venkata Anirudh and Ceperic, Vladimir and Knezevic, Tihomir},
    title     = {Scalable Optical Convolutional Neural Networks For Edge Applications},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2025},
    pages     = {1724-1733}
}
```
Please cite the paper if you find this useful.

#### Added Features
- MZIBlockConv2D and MZIBlockLinear layers support complex numbers
- MZIfConv2D: supports convolution in the Fourier domain (i.e., scalar multiplication of the input and weights in the Fourier domain)
  - Group convolution is supported
- Activations: complex-valued activation functions:
  - Complex Relu
  - <a href="https://github.com/fancompute/electro-optic-activation">electro-optic</a> activations
  - Complex valued Swish and Mish activation functions
  - Separable Mish which is applied independently on the real and imaginary parts
  - Polar Relu and Tanh activation functions that act only on the magnitude of the input leaving the phase unchanged
- Complex valued max-pooling and adaptive average pooling

<br/>
<br/>

## Installation

### From Source

#### Dependencies
- Python >= 3.6
- PyTorch >= 1.13.0
- Tensorflow-gpu >= 2.5.0
- [pyutils](https://github.com/JeremieMelo/pyutility) >= 0.0.2
- Others are listed in requirements.txt
- GPU model training requires NVIDIA GPUs and compatible CUDA

#### Get the PyTorch-ONN Source
```bash
git clone https://github.com/SPOTLIGHT-24-26/pytorch-onn.git
```

#### Install PyTorch-ONN
```bash
cd pytorch-onn
python3 setup.py install --user clean
```
or
```bash
./setup.sh
```

## Usage
Please see [pytorch-onn](https://github.com/JeremieMelo/pytorch-onn) for usage instructions and examples
