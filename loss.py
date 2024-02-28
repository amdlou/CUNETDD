import torch
from pytorch_msssim import ssim

def CustomSSIMLoss(y_true, y_pred, data_range=255.0):
    # Calculate SSIM
    # The data_range parameter is set to 255 to account for images in the 0-255 range
    ssim_val = ssim(y_true, y_pred, data_range=data_range)
    
    # Calculate SSIM loss
    ssim_loss = 1 - ssim_val
    
    return ssim_loss