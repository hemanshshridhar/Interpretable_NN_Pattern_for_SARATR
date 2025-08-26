import numpy as np
import scipy.signal
import torch
import torch.nn.functional as f
from scipy import signal


def complex_correlation(S_1,S_2, k=1):

    ####### In the model
    # filter = torch.ones((1, inputdata1.size(1), 2*k+1, 2*k+1)).cuda()

    #Out of the model
   
    S_1,S_2 = S_1.unsqueeze(0).unsqueeze(0), S_2.unsqueeze(0).unsqueeze(0)

    corr = S_1*torch.conj(S_2)
    filter = torch.ones((1, 1, 2*k+1, 2*k+1), dtype=torch.real(corr).dtype).cuda()

    numerator_real = f.conv2d(torch.real(corr), filter, padding='same')
    numerator_img = f.conv2d(torch.imag(corr), filter, padding='same')
    corr_map = numerator_real + 1j * numerator_img
    S_11_2 = torch.abs(S_1**2)
    S_12_2 = torch.abs(S_2**2)
    conv_s1 = f.conv2d(S_11_2, filter, padding='same')
    conv_s2 = f.conv2d(S_12_2, filter, padding='same')
    denominator = torch.sqrt(conv_s1 * conv_s2) + 1e-8
    corr_map = corr_map.squeeze()
    corr_map = corr_map.squeeze()  # Remove dimensions of size 1

    mu = torch.mean(corr_map)
    sigma = torch.sqrt(torch.mean(torch.abs(corr_map - mu) ** 2)) + 1e-8
    corr_map = (corr_map - mu) / sigma
    corr_map = torch.abs(corr_map)
    corr_map = corr_map.to(dtype=torch.float32)
    # corr_map = corr_map.squeeze()
    return corr_map


import torch

def rotate(SHH, SHV, SVH, SVV, theta_deg):
    """
    GPU-accelerated version of the rotate function using PyTorch.
    Inputs must be torch tensors of shape (H, W), dtype=torch.cfloat.
    """
    theta = torch.deg2rad(torch.tensor(theta_deg, device=SHH.device))
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Define rotation matrix R and its transpose
    R = torch.tensor([[cos_theta, sin_theta],
                      [-sin_theta, cos_theta]], dtype=torch.cdouble, device=SHH.device)

    R = R.to('cuda')
    R_T = R.T
    
    # Stack input scattering matrices
    # S = (H, W, 2, 2)
    S = torch.stack([
        torch.stack([SHH, SHV], dim=-1),
        torch.stack([SVH, SVV], dim=-1)
    ], dim=-2)  # shape: (H, W, 2, 2)

    # Matrix multiply: R @ S @ R.T
    # First compute R @ S: (H, W, 2, 2)
    RS = torch.matmul(R, S)
    S_rot = torch.matmul(RS, R_T)  # (H, W, 2, 2)

    # Extract rotated channels
    SHH_rot = S_rot[..., 0, 0]
    SHV_rot = S_rot[..., 0, 1]
    SVV_rot = S_rot[..., 1, 1]

    # Construct outputs (in original function’s order)
    A1 = SHH_rot - SVV_rot             # HH - VV
    A2 = SHH_rot + SVV_rot             # HH + VV
    A3 = SHH_rot                       # just HH
    A4 = SVV_rot                       # just VV
    A5 = SHV_rot                       # rotated HV

    return A5, A4, A3, A2, A1