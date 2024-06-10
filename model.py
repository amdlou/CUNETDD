
"""FCU_net model implementation.
   Disentangling multiple scattering with deep learning:
   application to strain mapping from electron diffraction patterns
   npj Computational Materials (2022)8:254 ;
   https://doi.org/10.1038/s41524-022-00939-9
   ComplexUNet a efficient complex-valued U-Net model
   for electron diffraction pattern analysis.
   The model is based on the U-Net architecture and is designed to
   process complex-valued input data.
"""
from typing import Optional, Type
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import cross_correlate_fft, cross_correlate_ifft
from utils import ConvSpec2D, Conv2D, ComplexUpsample2d

class AttentionGate(nn.Module):
    """
    AttentionGate module that performs attention mechanism on input tensors.

    Args:
        in_channels (int): Number of input channels.
        gating_channels (int): Number of gating channels.
        inter_channels (int, optional): Number of intermediate channels. Defaults to None.

    Attributes:
        in_channels (int): Number of input channels.
        gating_channels (int): Number of gating channels.
        inter_channels (int): Number of intermediate channels.
        W_g (nn.Sequential): Sequential module for gating channel convolution.
        W_x (nn.Sequential): Sequential module for input channel convolution.
        psi (nn.Sequential): Sequential module for sigmoid activation.
        relu (nn.ReLU): ReLU activation function.

    """

    def __init__(self, in_channels, gating_channels, inter_channels=None, pos_embedding=None):
        super().__init__()

        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.pos_embedding = pos_embedding

        self.inter_channels = in_channels // 2 if inter_channels is None else inter_channels

        self.W_g = nn.Sequential(
            ConvSpec2D(in_channels=self.gating_channels, n_filters=self.in_channels),
        )

        self.W_x = nn.Sequential(
            ConvSpec2D(in_channels=self.gating_channels, n_filters=self.in_channels),
        )

        self.psi = nn.Sequential(
            ConvSpec2D(in_channels=self.in_channels, n_filters=1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Forward pass of the AttentionGate module.

        Args:
            g (torch.Tensor): Gating tensor.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after attention mechanism.

        """
        shape_x = x.size()
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        upsample_psi = F.interpolate(psi, size=(shape_x[2], shape_x[3]),
                                     mode='bilinear', align_corners=True)
        real_psi, imag_psi = torch.chunk(upsample_psi, 2, dim=1)
        real_x, imag_x = torch.chunk(x, 2, dim=1)
        real_product = real_psi * real_x
        imag_product = imag_psi * imag_x
        x=torch.cat((real_product, imag_product), dim=1)

        return x

class RelativePositionalEmbedding(nn.Module):
    """
    A module that generates relative positional embeddings for input sequences.

    Args:
        d_model (int): The dimensionality of the embedding vectors.
        max_len (int, optional): The maximum length of the input sequences. Defaults to 5000.

    Attributes:
        d_model (int): The dimensionality of the embedding vectors.
        max_len (int): The maximum length of the input sequences.
        embedding (nn.Embedding): The embedding layer used to generate the positional embeddings.

    Methods:
        forward(x): Performs a forward pass of the module.
        generate_relative_positions_matrix(length, max_length): Generates the relative positions matrix.

    """

    def __init__(self, input_channel, d_model):
        super().__init__()
        self.d_model = d_model
        self.max_len = d_model*d_model
        self.input_channel = input_channel
        self.embedding = nn.Embedding(2*self.max_len-1, d_model)

    def forward(self, x):
        """
        Performs a forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The input tensor with the positional embeddings added.

        """
        position_matrix = self.generate_relative_positions_matrix(self.input_channel, self.max_len).to(x.device)
        embeddings = self.embedding(position_matrix)
        return x.add_(embeddings)

    def generate_relative_positions_matrix(self, length, max_length):
        """
        Generates the relative positions matrix.

        Args:
            length (int): The length of the input sequence.
            max_length (int): The maximum length of the input sequences.

        Returns:
            torch.Tensor: The relative positions matrix of shape (length, length).

        """
        range_vec = torch.arange(length)
        range_mat = range_vec[None, :].expand(length, length)
        distance_mat = range_mat - torch.t(range_mat)
        distance_mat_clipped = torch.clamp(distance_mat, -max_length, max_length)
        final_mat = distance_mat_clipped + max_length - 1
        return final_mat

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

        super().__init__()
        self.cross_correlate = cross_correlate_fft
        self.inverse_fft = cross_correlate_ifft
        self.initial_conv = ConvSpec2D(input_channel, filter_size, 3,
                                       n_depth, activation, dp_rate,
                                       bias, batchnorm)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pos_embedding = RelativePositionalEmbedding(input_channel, d_model=256)
        self.attention_blocks = nn.ModuleList()
        current_channels = filter_size
        max_channels = 256
        current_image_size = image_size

        # Encoder - Convolution followed by Pooling
        for _ in range(int(np.log2(image_size)) - 1):
            self.attention_blocks = nn.ModuleList()
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
            if idx in range(1):
                # The first decoder has the same number of
                # input channels as the last encoder
                self.decoder.append(ConvSpec2D(current_channels,
                                               upsample_channels, 3,
                                               n_depth, activation, dp_rate,
                                               bias, batchnorm))
                self.decoder.append(ComplexUpsample2d(scale_factor=2,
                                                      mode='bilinear'))
            else:
                # The rest of the decoders have double
                # the number of input channels
                self.decoder.append(ConvSpec2D(current_channels*2,
                                               upsample_channels, 3,
                                               n_depth, activation, dp_rate,
                                               bias, batchnorm))
                self.decoder.append(ComplexUpsample2d(scale_factor=2,
                                                      mode='bilinear'))
            self.attention_blocks.append(AttentionGate(current_channels, upsample_channels,pos_embedding=self.pos_embedding))
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
            inputsa (torch.Tensor): Input tensor A CBED patterns tensor.
            inputsb (torch.Tensor): Input tensor B Probe patterns tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the CUNETD model.
        """
        inputsa = self.pos_embedding(inputsa)
        inputsb = self.pos_embedding(inputsb)
        x = self.cross_correlate(inputsa, inputsb)
        x = self.initial_conv(x)
        skips = []

        # Encoder path
        # Step by 2 to handle conv+pool pairs
        for i in range(0, len(self.encoder), 2):
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
            x = self.attention_blocks[i // 2](x, skip_connection)
     
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
    