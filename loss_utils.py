""" This module contains custom loss functions.
    The custom_ssim_loss function calculates
    the SSIM and MSE between the target and output images.
    SSIM: Structural Similarity Index Measure
    #define ssim formula
    ssim = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2) /
    (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    mu_x = mean of x, mu_y = mean of y,
    sigma_x = variance of x, sigma_y = variance of y,
    MSE: Mean Squared Error
"""
from typing import Tuple
from pytorch_msssim import ssim
import torch
import torch.nn.functional as F

def custom_ssim_loss(
    targets: torch.Tensor,
    outputs: torch.Tensor,
    data_range: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    assert isinstance(targets, torch.Tensor), "Expected 'targets' to be a PyTorch Tensor"
    assert isinstance(outputs, torch.Tensor), "Expected 'outputs' to be a PyTorch Tensor"
    assert targets.shape == outputs.shape, "Targets and outputs must have the same dimensions"

    # Convert tensors to float32
    targets = targets.float()
    outputs = outputs.float()

    # Define thresholds and weights
    
    thresh1 = 100.0
    thresh2 = 500.0
    weight1 = 0.6
    weight2 = 0.3
    weight3 = 0.1
    
    mask1 = targets < thresh1
    mask1.requires_grad = False

    mask2 = (targets >= thresh1) & (targets < thresh2)
    mask2.requires_grad = False

    mask3 = targets >= thresh2
    mask3.requires_grad = False

    masked_targets1 = torch.where(mask1, targets, torch.zeros_like(targets))
    masked_outputs1 = torch.where(mask1, outputs, torch.zeros_like(outputs))

    masked_targets2 = torch.where(mask2, targets, torch.zeros_like(targets))
    masked_outputs2 = torch.where(mask2, outputs, torch.zeros_like(outputs))

    masked_targets3 = torch.where(mask3, targets, torch.zeros_like(targets))
    masked_outputs3 = torch.where(mask3, outputs, torch.zeros_like(outputs))

    def calculate_losses(masked_targets, masked_outputs):
        if masked_targets.numel() > 0:
            ssim_loss = 1 - ssim(masked_targets, masked_outputs, data_range=data_range)
            #kl_div_loss = F.kl_div(F.log_softmax(masked_outputs, dim=1), F.softmax(masked_targets, dim=1))
            mse_loss = F.mse_loss(masked_targets, masked_outputs)
        else:
            ssim_loss = torch.tensor(0.0)
            #kl_div_loss = torch.tensor(0.0)
            mse_loss = torch.tensor(0.0)
        return ssim_loss, mse_loss

    ssim_loss1, mse_loss1 = calculate_losses(masked_targets1, masked_outputs1)
    ssim_loss2, mse_loss2 = calculate_losses(masked_targets2, masked_outputs2)
    ssim_loss3, mse_loss3 = calculate_losses(masked_targets3, masked_outputs3)

    total_loss1 = ssim_loss1 + mse_loss1
    total_loss2 = ssim_loss2 + mse_loss2
    total_loss3 = ssim_loss3 + mse_loss3

    total_loss = weight1 * total_loss1 + weight2 * total_loss2 + weight3 * total_loss3

    return total_loss, ssim_loss1 + ssim_loss2 + ssim_loss3, mse_loss1 + mse_loss2 + mse_loss3