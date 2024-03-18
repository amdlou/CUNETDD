"""FCU_net model implementation. """
from typing import Optional, Type
import torch
from torch import nn
import numpy as np
from utils import cross_correlate_fft, cross_correlate_ifft
from utils import ConvSpec2D, Conv2D, ComplexUpsample2d


class ComplexUNet(nn.Module):
    """
    ComplexUNet model implementation.

    Args:
        input_channel (int): Number of input channels.
        image_size (int): Size of the input image.
        filter_size (int): Size of the filters in the model.
        n_depth (int): Number of convolutional layers in each block.
        dp_rate (float, optional): Dropout rate. Defaults to 0.1.
        activation (torch.nn.Module, optional): Activation function.
        batchnorm (bool, optional): Whether to use batch normalization.
        bias (bool, optional): Whether to include
        bias in the convolutional layers.
    """

    def __init__(self,
                 input_channel: int,
                 image_size: int,
                 filter_size: int,
                 n_depth: int,
                 dp_rate: float = 0.1,
                 activation: Optional[Type[nn.Module]] = nn.ReLU,
                 batchnorm: bool = True,
                 bias: bool = True) -> None:

        super(ComplexUNet, self).__init__()
        self.cross_correlate = cross_correlate_fft
        self.inverse_fft = cross_correlate_ifft
        self.initial_conv = ConvSpec2D(input_channel, filter_size, 3,
                                       n_depth, activation, dp_rate,
                                       bias, batchnorm)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        current_channels = filter_size
        max_channels = 256
        current_image_size = image_size
         
        # Encoder - Convolution followed by Pooling
        for _ in range(int(np.log2(image_size)) - 1):
            next_channels = min(current_channels * 2, max_channels)
            self.encoder.append(ConvSpec2D(current_channels, next_channels, 3,
                                           n_depth, activation, dp_rate, bias,
                                           batchnorm))
            self.encoder.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = next_channels
            current_image_size //= 2
        
        # Decoder - Convolution followed by Upsampling
        for idx in range(len(self.encoder) // 2):
            if current_image_size < filter_size:
                upsample_channels = max_channels
            else:
                upsample_channels = current_channels // 2
            if idx in range(1):  # The first decoder has the same number of input channels as the last encoder
                self.decoder.append(ConvSpec2D(current_channels,
                                               upsample_channels, 3,
                                               n_depth, activation, dp_rate,
                                               bias, batchnorm))
                self.decoder.append(ComplexUpsample2d(scale_factor=2,
                                                      mode='bilinear'))
            else:  # The rest of the decoders have double the number of input channels
                self.decoder.append(ConvSpec2D(current_channels*2,
                                               upsample_channels, 3,
                                               n_depth, activation, dp_rate,
                                               bias, batchnorm))
                self.decoder.append(ComplexUpsample2d(scale_factor=2,
                                                      mode='bilinear'))
            current_channels = upsample_channels
            current_image_size *= 2

        self.additional_conv = ConvSpec2D(current_channels, filter_size, 3,
                                          n_depth, activation, dp_rate,
                                          bias, batchnorm)
        self.conv2d = Conv2D(filter_size, filter_size, n_depth, 3, activation,
                             dp_rate, batchnorm)
        self.final_conv = nn.Conv2d(filter_size, 1, kernel_size=3,
                                    padding=1, bias=bias)
        self.actv = nn.ReLU()
    
    def forward(self,
                inputsa: torch.Tensor,
                inputsb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CUNETD model.

        Args:
            inputsa (torch.Tensor): Input tensor A.
            inputsb (torch.Tensor): Input tensor B.

        Returns:
            torch.Tensor: Output tensor after passing through the CUNETD model.
        """
        x = self.cross_correlate(inputsa, inputsb)
        x = self.initial_conv(x)
        skips = []

        # Encoder path
        for i in range(0, len(self.encoder), 2):  # Step by 2 to handle conv+pool pairs
            x = self.encoder[i](x)  # Convolution
            x = self.encoder[i + 1](x)  # Pooling
            skips.append(x)

        skip_connection = skips.pop()  # Remove the last skip connection

        # Decoder path
        for i in range(0, len(self.decoder) - 2, 2):
            # Exclude the last upsample for now
            x = self.decoder[i](x)  # Convolution
            x = self.decoder[i + 1](x)  # Upsampling
            skip_connection = skips.pop()
            x = torch.cat((x, skip_connection), dim=1)
        # Last decoder block
        x = self.decoder[-2](x)  # Last Convolution
        x = self.decoder[-1](x)  # Last Upsampling
        x = self.additional_conv(x)
        x = self.inverse_fft(x)
        x = self.conv2d(x)
        x = self.final_conv(x)
        x = self.actv(x)
        return x
