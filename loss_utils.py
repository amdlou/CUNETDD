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
    data_range: float = 255.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    custom_ssim_loss: Calculate the SSIM and MSE between
    the target and output images.
    
    Parameters:
    - targets: The target tensor.
    - outputs: The output tensor.
    - data_range: The range of the data, default is 255.0 for images in the 0-255 range.

    Returns:
    - A tuple containing the total loss, SSIM loss, and MSE loss.
    """
    # Assert that both targets and outputs are tensors
    assert isinstance(targets, torch.Tensor), "Expected 'targets' to be a PyTorch Tensor"
    assert isinstance(outputs, torch.Tensor), "Expected 'outputs' to be a PyTorch Tensor"

    # Assert that targets and outputs have the same dimensions
    assert targets.shape == outputs.shape, "Targets and outputs must have the same dimensions"

    # Calculate SSIM
    ssim_val = ssim(targets, outputs, data_range=data_range)
    ssim_val = torch.clamp(ssim_val, min=1e-7)
    # Calculate losses
    loss_1 = 1 - ssim_val  # SSIM loss component
    loss_2 = F.mse_loss(targets, outputs)  # MSE loss component
    loss_2 = loss_2 + 1e-7
    total_loss = loss_1 + loss_2  # Combined loss

    return total_loss, loss_1, loss_2
