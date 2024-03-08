import torch
import torch.nn as nn
import numpy as np
from utils import cross_correlate_fft, cross_correlate_ifft
from utils import ConvSpec2D,Conv2D,ComplexUpsample2d

def normalize_data(data):
    # Assuming data is a PyTorch tensor of shape [batch_size, 1, 256, 256]
    # Find the maximum value for each image in the batch
    max_vals = data.view(data.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)
    
    # Avoid division by zero for images with all pixels equal to zero
    max_vals[max_vals == 0] = 1
    
    # Normalize each image individually
    normalized_data = data / max_vals
    return normalized_data
class ComplexUNet(nn.Module):
    def __init__(self, input_channel, image_size, filter_size, n_depth, dp_rate=0.1, activation=nn.ReLU, batchnorm=True, bias=True):
        super(ComplexUNet, self).__init__()
        self.cross_correlate = cross_correlate_fft  
        self.inverse_fft = cross_correlate_ifft  
        self.initial_conv = ConvSpec2D(input_channel, filter_size, 3, n_depth, activation, dp_rate,bias, batchnorm)
        self.conv_e = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        self.additional_conv = ConvSpec2D(filter_size, filter_size, 3, n_depth, activation, dp_rate, bias, batchnorm)
        self.conv2d =Conv2D(filter_size, filter_size, n_depth, 3, activation, dp_rate, bias,batchnorm)  
        self.final_conv = nn.Conv2d(filter_size, 1, kernel_size=3, padding=1, bias=bias) 
        self.actv = nn.ReLU()
        self.filter_size = filter_size
        current_channels = self.filter_size
        max_channels = 256
        current_image_size = image_size
        
        for _ in range(int(np.log2(image_size)) - 1):
            next_channels = min(current_channels * 2, max_channels)
            
            self.conv_e.append(ConvSpec2D(current_channels, next_channels, 3, n_depth, activation, dp_rate, bias, batchnorm))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = next_channels
            current_image_size //= 2  # Each pooling layer halves the image size
        
        for idx in range(len(self.conv_e)):  
            if  current_image_size < self.filter_size :
                upsample_channels = max_channels
            else:
                upsample_channels = current_channels // 2
            
            if idx == 0:  # The first decoder has the same number of input channels as the last encoder
                self.conv_d.append(ConvSpec2D(current_channels, upsample_channels, 3, n_depth, activation, dp_rate, bias, batchnorm))
            else:  # The rest of the decoders have double the number of input channels
                self.conv_d.append(ConvSpec2D(current_channels*2, upsample_channels, 3, n_depth, activation, dp_rate, bias, batchnorm))
            self.upsamples.append(ComplexUpsample2d(scale_factor=2, mode='bilinear'))
            current_channels = upsample_channels
            current_image_size *= 2 # Each upsampling layer doubles the image size
        
        

    def forward(self, inputsA, inputsB):
        #print(f"Input A: {inputsA.size()}", f"Input B: {inputsB.size()}")  
        x = self.cross_correlate(inputsA, inputsB)    
        #print(f"After cross-correlation: {x.size()}")
        #print("====")
        x = self.initial_conv(x)  
        #print(f"After initial conv: {x.size()}")
        #print("====")
        skips = []
        # Encoder path
        for conv_e, pool in zip(self.conv_e, self.pools):
            x = conv_e(x)
            #print(f"After encoder: {x.size()}")
            x = pool(x)
            #print(f"After pooling: {x.size()}")
            #print('====')
            skips.append(x)
        skip_connection = skips.pop()
        #print('_'*30)
        #print(f"Skip connection: {skip_connection.size()}")
        for idx, conv_d in enumerate(self.conv_d[:-1]):  # Exclude the last decoder for now
            x = conv_d(x)
            #print(f"After decoder: {x.size()}")
            x = self.upsamples[idx](x)
            #print(f"After upsampling: {x.size()}")
            skip_connection = skips.pop()
            x = torch.cat((x, skip_connection), dim=1)
            #print(f"After cat: {x.size()}")

        # Process the last decoder without an upsample and without a skip connection
        x = self.conv_d[-1](x)
        #print(f"After last decoder: {x.size()}")
        x = self.upsamples[-1](x)
        #print(f"After last upsampling: {x.size()}")

        

            
                

        #print(("___"*30))
        #print(f"After decoder: {x.size()}")
        #print('====')
        x = self.additional_conv(x)
        #print(f"After additional conv: {x.size()}")
        #print('====') 
        x = self.inverse_fft(x)
        #print(f"After inverse fft: {x.size()}")
        #print('====')
        x = self.conv2d(x)
        #print(f"After conv2d: {x.size()}")
        #print('====')
        x = self.final_conv(x)
        #print(f"After final conv: {x.size()}")
        #print('====')   
        x=self.actv(x)
        #x = normalize_data(x)
        #print(f"Output: {x.size()}")                  
        return x


#input_channel = 1  
#image_size = 256
#dummy_inputA = torch.randn(1, input_channel, image_size, image_size)
#dummy_inputB = torch.randn(1, input_channel, image_size, image_size)
#model = ComplexUNet(input_channel=input_channel, image_size=image_size, filter_size=32, n_depth=4, dp_rate=0.1, activation=nn.ReLU, batchnorm=True, bias=True)
#output = model(dummy_inputA, dummy_inputB)
