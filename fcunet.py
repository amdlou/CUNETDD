import torch
import torch.nn as nn
import numpy as np
from utils import cross_correlate_fft, cross_correlate_ifft
from utils import ConvSpec2D, ComplexConvTranspose2d, Conv2D,ComplexUpsample2d

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
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.additional_conv = ConvSpec2D(filter_size, filter_size, 3, n_depth, activation, dp_rate, bias, batchnorm)
        self.conv2d =Conv2D(filter_size, filter_size, n_depth, 3, activation, dp_rate, batchnorm, bias)  
        self.final_conv = nn.Conv2d(filter_size, 1, kernel_size=3, padding=1, bias=bias) 
        current_channels = filter_size
        for _ in range(int(np.log2(image_size)) - 1):
            next_channels = current_channels * 2
            self.encoders.append(ConvSpec2D(current_channels, next_channels, 3, n_depth, activation, dp_rate, bias, batchnorm))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = next_channels 
        for idx in range(len(self.encoders)):
            upsample_channels = current_channels // 2
            self.upsamples.append(ComplexUpsample2d(scale_factor=2, mode='bilinear'))
            
            if idx == 0:
                self.decoders.append(ConvSpec2D(current_channels, upsample_channels, 3, n_depth, activation, dp_rate, bias, batchnorm))
            else:
                self.decoders.append(ConvSpec2D(current_channels * 2, upsample_channels, 3, n_depth, activation, dp_rate, bias, batchnorm))
            
            current_channels = upsample_channels       
        #for _ in (range(len(self.encoders))):
            #upsample_channels = current_channels // 2  
            #self.upsamples.append(ComplexConvTranspose2d(current_channels, upsample_channels, kernel_size=2, stride=2,bias=bias))
            #self.decoders.append(ConvSpec2D(current_channels, upsample_channels, 3, n_depth, activation, dp_rate, bias, batchnorm))
            #current_channels = upsample_channels

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
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            #print(f"After encoder: {x.size()}")
            x = pool(x)
            #print(f"After pooling: {x.size()}")
            #print('====')
            skips.append(x)
        skip_connection = skips.pop()
        #print('_'*30)
        # Decoder path
        for i,(upsample, decoder) in enumerate(zip((self.upsamples), (self.decoders))):            
            
            x = decoder(x)
            #print( ' After decoder:',x.shape)
            
            
            x = upsample(x)
            #print( 'After upsampling:',x.shape,)
            if i >= 0:  # Since we want to add skip connections starting from the second layer, we check if i is greater than or equal to 0
                if len(skips) > 0:  # Ensure there is a skip connection available to add
                    skip_connection = skips.pop()  # Get the last skip connection
                    x = torch.cat((x, skip_connection), dim=1)  # Concatenate skip connection
                    #print('After cat:', x.shape)

            #if i < len(self.upsamples) - 1:  
                #skip_connection = skips.pop()
                #x = torch.cat((x, skip_connection), dim=1)
                #print('====')
                

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
        x=nn.ReLU()(x)
        #x = normalize_data(x)
        #print(f"Output: {x.size()}")                  
        return x


#input_channel = 1  
#image_size = 256
#dummy_inputA = torch.randn(1, input_channel, image_size, image_size)
#dummy_inputB = torch.randn(1, input_channel, image_size, image_size)
#model = ComplexUNet(input_channel=input_channel, image_size=image_size, filter_size=4, n_depth=2, dp_rate=0.1, activation='relu')
#output = model(dummy_inputA, dummy_inputB)
