""" This module contains custom loss functions.c
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
    - y_true: The true image.
    - y_pred: The predicted image.
    - targets: The target tensor.
    - outputs: The output tensor.
    - data_range: The range of the data. Default is 255.0 for images in the 0-255 range.

    Returns:
    - A tuple containing the SSIM loss, total loss, loss_1, and loss_2.
    """
    # Calculate SSIM
    ssim_val = ssim(targets, outputs, data_range=data_range)
    # Calculate SSIM loss
    ssim_loss = 1 - ssim_val

    # Calculate losses
    loss_1 = ssim_loss
    loss_2 = F.mse_loss(targets, outputs)
    total_loss = loss_1 + loss_2

    return total_loss, loss_1, loss_2
