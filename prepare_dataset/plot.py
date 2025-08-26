import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as f
from scipy import signal

def local_corr(A, B, window_size=3):
    num_real = uniform_filter((A * np.conj(B)).real, window_size)
    num_imag = uniform_filter((A * np.conj(B)).imag, window_size)
    numerator = num_real + 1j * num_imag
    denom_A = np.sqrt(uniform_filter(np.abs(A)**2, window_size))
    denom_B = np.sqrt(uniform_filter(np.abs(B)**2, window_size))
    denom = denom_A * denom_B + 1e-8
    return np.abs(numerator / denom)

def rotate_and_compute(SHH, SHV, SVH, SVV, theta_deg):
    theta = np.deg2rad(theta_deg)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])

    H, W = SHH.shape
    S_rot = np.zeros((H, W, 2, 2), dtype=complex)

    for i in range(H):
        for j in range(W):
            S = np.array([
                [SHH[i,j], SHV[i,j]],
                [SVH[i,j], SVV[i,j]]
            ])
            S_rot[i,j] = R @ S @ R.T

    SHH_rot = S_rot[:,:,0,0]
    SHV_rot = S_rot[:,:,0,1]
    SVV_rot = S_rot[:,:,1,1]

    A1 = SHH_rot - SVV_rot
    A2 = SHH_rot - SHV_rot
    A3 = A1 - SHV_rot
    A4 = (SHH_rot + SVV_rot) - A1

    gamma1 = np.mean(local_corr(SHH_rot, A1))
    gamma2 = np.mean(local_corr(SHH_rot, A2))
    gamma3 = np.mean(local_corr(A1, SHV_rot))
    gamma4 = np.mean(local_corr(SHH_rot + SVV_rot, A1))

    return gamma1, gamma2, gamma3, gamma4

# def circular_plot(thetas, values, title):
#     thetas_rad = np.deg2rad(thetas)
#     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#     ax.plot(thetas_rad, values, linewidth=2)
#     ax.set_title(title, va='bottom')
#     ax.set_theta_direction(-1)
#     ax.set_theta_zero_location('N')
#     plt.tight_layout()
#     plt.show()

# === Load full-pol data ===
SHH = np.load("/content/drive/MyDrive/Polsar/GOTCHA-CP_Disc2/DATA/pass8/VEHICLES/Fordtaurus/pass8/HH_NPY/data_3dsar_pass8_az004_HH.npy")
SHV = np.load("/content/drive/MyDrive/Polsar/GOTCHA-CP_Disc2/DATA/pass8/VEHICLES/Fordtaurus/pass8/HV_NPY/data_3dsar_pass8_az004_HV.npy")
SVH = np.load("/content/drive/MyDrive/Polsar/GOTCHA-CP_Disc2/DATA/pass8/VEHICLES/Fordtaurus/pass8/VH_NPY/data_3dsar_pass8_az003_VH.npy")
SVV = np.load("/content/drive/MyDrive/Polsar/GOTCHA-CP_Disc2/DATA/pass8/VEHICLES/Fordtaurus/pass8/VV_NPY/data_3dsar_pass8_az004_VV.npy")

# === Theta sweep ===
thetas = list(range(0, 360, 5))
gamma1_all, gamma2_all, gamma3_all, gamma4_all = [], [], [], []

for theta in thetas:
    g1, g2, g3, g4 = rotate_and_compute(SHH, SHV, SVH, SVV, theta)
    gamma1_all.append(g1)
    gamma2_all.append(g2)
    gamma3_all.append(g3)
    gamma4_all.append(g4)



def circular_plot(thetas, gamma1, gamma2, gamma3, gamma4):
    thetas_rad = np.deg2rad(thetas)
    titles = [
        r"$|\gamma_{\mathrm{HH, VV}}(\theta)|$",
        r"$|\gamma_{\mathrm{HH, HV}}(\theta)|$",
        r"$|\gamma_{(\mathrm{HH-VV}), HV}(\theta)|$",
        r"$|\gamma_{(\mathrm{HH+VV}), (HH-VV)}(\theta)|$"
    ]
    gammas = [gamma1, gamma2, gamma3, gamma4]

    fig, axs = plt.subplots(1, 4, subplot_kw={'projection': 'polar'}, figsize=(16, 4))

    for ax, gamma, title in zip(axs, gammas, titles):
        ax.plot(thetas_rad, gamma, linewidth=2)
        ax.set_title(title, va='bottom', fontsize=10)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')

    plt.tight_layout()
    plt.show()



def complex_correlation(S_1,S_2, k=1):

    ####### In the model
    # filter = torch.ones((1, inputdata1.size(1), 2*k+1, 2*k+1)).cuda()

    #Out of the model
    S_1,S_2 = torch.tensor(S_1), torch.tensor(S_2)
    S_1,S_2 = S_1.unsqueeze(0).unsqueeze(0), S_2.unsqueeze(0).unsqueeze(0)

    corr = S_1*torch.conj(S_2)
    filter = torch.ones((1, 1, 2*k+1, 2*k+1), dtype=torch.real(corr).dtype)

    numerator_real = f.conv2d(torch.real(corr), filter, padding='same')
    numerator_img = f.conv2d(torch.imag(corr), filter, padding='same')
    corr_map = numerator_real + 1j * numerator_img
    S_11_2 = torch.abs(S_1**2)
    S_12_2 = torch.abs(S_2**2)
    conv_s1 = f.conv2d(S_11_2, filter, padding='same')
    conv_s2 = f.conv2d(S_12_2, filter, padding='same')
    denominator = torch.sqrt(conv_s1 * conv_s2) + 1e-8
    corr_map = corr_map.squeeze()
    corr_map = corr_map.numpy()

    corr_map = corr_map/np.abs(corr_map).max()
    # c = (f.conv2d(torch.real(inputdata1*torch.conj(inputdata2)), filter, padding="same") + 1j * f.conv2d(torch.imag(inputdata1*torch.conj(inputdata2)), filter, padding="same"))/ \
    #     (torch.sqrt(f.conv2d(torch.abs(inputdata1)**2, filter, padding="same")*
    #                 f.conv2d(torch.abs(inputdata2)**2, filter, padding="same")))
    # corr_map = corr_map.squeeze()
    return corr_map
# === Plot circular graphs ===
circular_plot(thetas, gamma1_all, "|γ_{HH_VV}(θ)|")
circular_plot(thetas, gamma2_all, "|γ_{HH_HV}(θ)|")
circular_plot(thetas, gamma3_all, "|γ_{(HH_VV)_HV}(θ)|")
circular_plot(thetas, gamma4_all, "|γ_{(HH+VV)_HH_VV)}(θ)|")