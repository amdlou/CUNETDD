from re import T
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


def cross_correlate_fft(cb, pr):
    # Assuming cb and pr are torch tensors in the shape of (batch, channel, height, width)

    # Shift the probe and CBED patterns to the origin
    pr_shifted = torch.fft.ifftshift(pr, dim=(-2, -1))
    cb_shifted = torch.fft.ifftshift(cb, dim=(-2, -1))

    # Perform FFT
    cbed_fft = torch.fft.fft2(cb_shifted.to(torch.complex64))
    probe_fft = torch.fft.fft2(pr_shifted.to(torch.complex64))

    # Multiply CBED FFT with the conjugate of probe FFT
    ccff = cbed_fft * torch.conj(probe_fft)

    # Normalize each cross-correlation
    ccff_norm = torch.linalg.vector_norm(ccff, dim=(-2, -1), keepdim=True)
    ccff_normalized = ccff / ccff_norm

    # Split into real and imaginary parts and concatenate along the channel dimension
    ccff_real = ccff_normalized.real
    ccff_imag = ccff_normalized.imag
    ccff_combined = torch.cat([ccff_real, ccff_imag], dim=1)  # Concatenating on channel dimension

    return ccff_combined

def cross_correlate_ifft(x):
    # Assuming x is a torch tensor in the shape of (batch, 2 * channel, height, width)
    # where first half channels are real and second half are imaginary parts

    input_channel = x.size(1) // 2
    input_complex = torch.complex(x[:, :input_channel, :, :], x[:, input_channel:, :, :])

    # Perform IFFT and shift
    output_complex = torch.fft.fftshift(torch.fft.ifft2(input_complex), dim=(-2, -1))

    # Take the real part
    output = output_complex.real

    return output


class MonteCarloDropout(nn.Module):
    def __init__(self, p=0.1):
        super(MonteCarloDropout, self).__init__()
        self.p = p

    def forward(self, inputs):
        # Apply dropout both in training and evaluation modes
        return F.dropout(inputs, self.p, training=True)


class Conv2D(nn.Module):
    def __init__(self, in_channels, n_filters, n_depth=2, kernel_size=3, activation=nn.ReLU, dp_rate=0.1, batchnorm=True,bias = True):
        super(Conv2D, self).__init__()
        self.n_depth = n_depth
        self.activation = activation
        self.dp_rate = dp_rate
        self.batchnorm = batchnorm

        self.layers = nn.ModuleList()
        for _ in range(n_depth):
            self.layers.append(nn.Conv2d(in_channels, n_filters, kernel_size, padding='same',bias=True ))
            if batchnorm:
                self.layers.append(nn.BatchNorm2d(n_filters))              
            if activation is not None:
                self.layers.append(activation())
            self.layers.append(MonteCarloDropout(dp_rate))
            in_channels = n_filters  # Output channels become input for the next layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConvComplex2D_py(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=True):
        super(ConvComplex2D_py, self).__init__()

        # No need to adjust in_channels and out_channels for real and imaginary parts
        # Both the real and imaginary convolutions will now output out_channels channels each

        # Initialize convolution layers for real and imaginary parts
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        # Assume input is of shape [batch_size, 2*in_channels, height, width]
        # Where the first half of the channels are real and the second half are imaginary
        real_input, imag_input = torch.chunk(input, 2, dim=1)

        # Perform convolution on real and imaginary parts separately
        real_output = self.real_conv(real_input) - self.imag_conv(imag_input)
        imag_output = self.imag_conv(real_input) + self.real_conv(imag_input)

        # Concatenate the real and imaginary parts in the channel dimension
        # Now, both real and imaginary outputs have out_channels channels, doubling the output channels
        return torch.cat([real_output, imag_output], dim=1)

class ComplexUpsample2d(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False):
        super(ComplexUpsample2d, self).__init__()
        # Define the upsampling parameters
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners if mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None
 
    def forward(self, x):
        # Split input tensor into real and imaginary parts based on the channel dimension
        real_input, imag_input = torch.chunk(x, 2, dim=1)
 
        # Apply upsampling separately to real and imaginary components
        real_output = F.interpolate(real_input, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        imag_output = F.interpolate(imag_input, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
 
        # Concatenate the outputs to maintain the structure
        output = torch.cat([real_output, imag_output], dim=1)
        return output




class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super(ComplexConvTranspose2d, self).__init__()
        # Define transposed convolution for real and imaginary separately
        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, output_padding, groups, bias, dilation)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, output_padding, groups, bias, dilation)

    def forward(self, x):
        # Split input tensor into real and imaginary parts based on the channel dimension
        real_input, imag_input = torch.chunk(x, 2, dim=1)

        # Apply transposed convolution separately and perform complex arithmetic
        real_output = self.tconv_re(real_input) - self.tconv_im(imag_input)
        imag_output = self.tconv_im(real_input) + self.tconv_re(imag_input)

        # Concatenate the outputs to maintain the structure
        output = torch.cat([real_output, imag_output], dim=1)
        return output
    

class ConvSpec2D(nn.Module):
    def __init__(self, in_channels, n_filters, n_depth=1, kernel_size=3, activation=nn.ReLU, dp_rate=0.1, batchnorm=True, bias=True):
        super(ConvSpec2D, self).__init__()
        self.layers = nn.ModuleList()

        #if batchnorm:
            # Apply BN to the input channels before convolution
            #self.layers.append(nn.BatchNorm2d(in_channels))

        for _ in range(n_depth):
            conv_layer = ConvComplex2D_py(in_channels, n_filters, kernel_size, padding='same',bias=bias )
            self.layers.append(conv_layer)

            # Following layers are now correctly placed after convolution as in the TensorFlow model
            if batchnorm:
                self.layers.append(nn.BatchNorm2d(n_filters*2))  # Assuming n_filters doubles after ConvComplex2D_py

            if activation is not None:
                self.layers.append(activation())

            if dp_rate > 0.0:
                self.layers.append(MonteCarloDropout(dp_rate))

            in_channels = n_filters # Adjust for the next layer, if any
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
