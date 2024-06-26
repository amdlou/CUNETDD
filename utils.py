""" Utility functions and classes for the complex-valued neural network.
    This module contains cross-correlation functions
    using Fast Fourier Transform (FFT),
    Inverse Fast Fourier Transform (IFFT),
    convolutional layers for complex-valued inputs,
    upsampling layer for complex-valued inputs,
    convolutional blocks with n_depth layers,
    batch normalization,dropout, and activation,
    class (ConvComplex2D) implements the complex convolution
    between two complex vectors (W = A + iB) and input vector (X = x + iy).
    The weights initialized in this class are complex and can be
    convolved with complex inputs such as
    fourier representation of image inputs...
    Adapted from implementation of
    complex convolutions in Deep Complex Networks by Trabelsi et. al. (2018)
"""
from typing import Optional, Tuple, Union
import torch
import torch.fft
from torch import nn
import torch.nn.functional as F


def cross_correlate_fft(
    cbed: torch.Tensor, probe: torch.Tensor
) -> torch.Tensor:
    """
    Perform cross-correlation of CBED (Convergent Beam Electron Diffraction)
    patterns using Fast Fourier Transform (FFT).

    Args:
        cb (torch.Tensor): CBED patterns tensor with shape
        (batch, channel, height, width).
        pr (torch.Tensor): Probe patterns tensor with shape
        (batch, channel, height, width).

    Returns:
        torch.Tensor: Combined cross-correlation tensor with shape
        (batch, 2 * channel, height, width).
    """
    # Assuming cb and pr are torch tensors in
    # the shape of (batch, channel, height, width)

    # Shift the probe and CBED patterns to the origin
    pr_shifted = torch.fft.ifftshift(probe, dim=(-2, -1))
    cb_shifted = torch.fft.ifftshift(cbed, dim=(-2, -1))

    # Perform FFT
    cbed_fft = torch.fft.fft2(cb_shifted.to(torch.complex64))
    probe_fft = torch.fft.fft2(pr_shifted.to(torch.complex64))

    # Multiply CBED FFT with the conjugate of probe FFT
    ccff = cbed_fft * torch.conj(probe_fft)

    # Normalize each cross-correlation
    ccff_norm = torch.norm(ccff, dim=(-2, -1), keepdim=True)
    epsilon = 1e-8
    ccff_norm += epsilon
    ccff_normalized = ccff / ccff_norm

    # Split into real and imaginary parts
    # and concatenate along the channel dimension
    ccff_real = ccff_normalized.real
    ccff_imag = ccff_normalized.imag
    ccff_combined = torch.cat([ccff_real, ccff_imag], dim=1)

    return ccff_combined


def cross_correlate_ifft(x: torch.Tensor) -> torch.Tensor:
    """
    Perform cross-correlation using IFFT.

    Args:
        x (torch.Tensor): Input tensor of shape
        (batch, 2 * channel, height, width),
        where the first half channels are real
        and the second half are imaginary parts.

    Returns:
        torch.Tensor: Output tensor after performing
        cross-correlation using IFFT.
    """
    x = x.to(torch.float32)  # Convert to float32 for torch.fft
    input_channel = x.size(1) // 2
    input_complex = torch.complex(x[:, :input_channel, :, :],
                                  x[:, input_channel:, :, :])

    # Perform IFFT and shift
    output_complex = torch.fft.fftshift(torch.fft.ifft2(input_complex),
                                        dim=(-2, -1))

    # Take the real part
    output = output_complex.real

    return output


class Conv2D(nn.Module):
    """
    Custom Conv2D module that performs multiple convolutional operations
    with optional batch normalization,
    activation, and dropout.

    Args:
        in_channels (int): Number of input channels.
        n_filters (int): Number of filters/output channels.
        n_depth (int, optional): Number of convolutional layers
        to stack (default: 2).
        kernel_size (int or tuple, optional): Size of the convolutional kernel.
        activation (torch.nn.Module, optional): Activation function to apply
        after each convolutional layer (default: torch.nn.ReLU).
        dp_rate (float, optional): Dropout rate (default: 0.1).
        batchnorm (bool, optional): Whether to apply batch normalization
        after each convolutional layer (default: True).
    Returns:
        torch.Tensor: Output tensor after passing through the Conv2D module.
    """

    def __init__(self, in_channels: int, n_filters: int, n_depth: int = 2,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 activation: Optional[type[nn.Module]] = nn.ReLU,
                 dp_rate: float = 0.1, batchnorm: bool = True) -> None:
        super().__init__()
        self.n_depth = n_depth
        self.activation = activation
        self.dp_rate = dp_rate
        self.batchnorm = batchnorm

        self.layers = nn.ModuleList()
        for _ in range(n_depth):
            self.layers.append(nn.Conv2d(in_channels, n_filters, kernel_size,
                                         padding='same', bias=True))
            if batchnorm:
                self.layers.append(nn.BatchNorm2d(n_filters))
            if activation is not None:
                self.layers.append(activation())
            self.layers.append(nn.Dropout(dp_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through all the layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class ConvComplex2D(nn.Module):
   """
    Convolutional layer for complex-valued inputs.

    This module expects the input tensor to contain an equal split of real and imaginary parts
    in the channel dimension, effectively doubling the `in_channels` specified. For example,
    if the actual input tensor has 20 channels, 10 of these should be real and 10 imaginary,
    and `in_channels` should be set to 10.

    Args:
        in_channels (int): Number of input channels for each of the real and imaginary parts.
                           The total channel count of the tensor should be 2 * in_channels.
        out_channels (int): Number of output channels for each of the real and imaginary parts.
                            The output tensor will have 2 * out_channels channels.
        kernel_size (int or tuple): Size of the convolutional kernel.
        stride (int or tuple, optional): Stride of the convolution.
        padding (int or tuple or 'same', optional): Padding added to all sides of the input.
        dilation (int or tuple, optional): Spacing between kernel elements.
        groups (int, optional): Number of blocked connections from input channels to output channels.
        bias (bool, optional): If True, adds a learnable bias to the output.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int], str] = 'same',
                 dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1,
                 bias: bool = True) -> None:
        super().__init__()

        # Initialize convolution layers for real and imaginary parts
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the complex-valued convolutional layer.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape
            [batch_size, 2*in_channels, height, width].
            The first half of the channels are
            real and the second half are imaginary.

        Returns:
            torch.Tensor: Output tensor of shape
            [batch_size, 2*out_channels, height, width].
            The real and imaginary parts are concatenated
            in the channel dimension.
        """
        # Split input tensor into real and imaginary parts
        # based on the channel dimension
        real_input, imag_input = torch.chunk(input_tensor, 2, dim=1)

        # Perform convolution on real and imaginary parts separately
        real_output = self.real_conv(real_input) - self.imag_conv(imag_input)
        imag_output = self.imag_conv(real_input) + self.real_conv(imag_input)

        # Concatenate the real and imaginary parts in the channel dimension
        # Now, both real and imaginary outputs have out_channels channels
        # doubling the output channels
        return torch.cat([real_output, imag_output], dim=1)


class ComplexUpsample2d(nn.Module):
    """
    2D Complex Upsampling module.

    Args:
        scale_factor (int): Upsampling scale factor.
        mode (str): Upsampling mode. Default is 'bilinear'.
        align_corners (bool): Whether to align corners during upsampling.

    Attributes:
        scale_factor (int): Upsampling scale factor.
        mode (str): Upsampling mode.
        align_corners (bool): Whether to align corners during upsampling.

    """

    def __init__(self, scale_factor: int = 2, mode: str = 'bilinear',
                 align_corners: bool = False) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        modes = ['linear', 'bilinear', 'bicubic', 'trilinear']
        assert mode in modes, (
            f"Invalid mode {mode}. Choose from 'linear', 'bilinear', "
            "'bicubic', 'trilinear'."
        )
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ComplexUpsample2d module.

        Args:
            x (torch.Tensor): Input tensor of shape
            (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Upsampled tensor of shape
            (batch_size, channels, new_height, new_width).
        """
        real_input, imag_input = torch.chunk(x, 2, dim=1)
        real_output = F.interpolate(real_input, scale_factor=self.scale_factor,
                                    mode=self.mode,
                                    align_corners=self.align_corners)
        imag_output = F.interpolate(imag_input, scale_factor=self.scale_factor,
                                    mode=self.mode,
                                    align_corners=self.align_corners)
        output = torch.cat([real_output, imag_output], dim=1)
        return output


class ConvSpec2D(nn.Module):
    """
    Convolutional layer with configurable specifications.

    Args:
        in_channels (int): Number of input channels.
        n_filters (int): Number of filters.
        n_depth (int, optional): Number of convolutional layers.
        kernel_size (int, optional): Size of the convolutional kernel.
        activation (torch.nn.Module, optional): Activation function to apply
        after each convolutional layer. Defaults to torch.nn.ReLU
        dp_rate (float, optional): Dropout rate. Defaults to 0.1.
        batchnorm (bool, optional): Whether to apply batch normalization
          after each convolutional layer.Defaults to True
        bias (bool, optional): Whether to include bias
        in the convolutional layers.
        Defaults to True.
    """

    def __init__(self, in_channels: int, n_filters: int, n_depth: int = 1,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 activation: Optional[type[nn.Module]] = nn.ReLU,
                 dp_rate: float = 0.1, batchnorm: bool = True,
                 bias: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(n_depth):
            conv_layer = ConvComplex2D(in_channels, n_filters, kernel_size,
                                       padding='same', bias=bias)
            self.layers.append(conv_layer)

            # Following layers are now correctly placed
            # after convolution as in the TensorFlow model
            if batchnorm:
                # Assuming n_filters doubles after ConvComplex2D
                self.layers.append(nn.BatchNorm2d(n_filters * 2))

            if activation is not None:
                self.layers.append(activation())

            if dp_rate > 0.0:
                self.layers.append(nn.Dropout(dp_rate))

            in_channels = n_filters  # Adjust for the next layer, if any

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x
