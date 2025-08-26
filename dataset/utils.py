import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Optional
import os


def visualize_sar_image(sar_data: np.ndarray, 
                       title: str = "SAR Image",
                       channel_names: List[str] = ["HH", "HV", "VH", "VV"],
                       save_path: Optional[str] = None):
    """
    Visualize SAR image data.
    
    Args:
        sar_data: SAR data of shape (4, H, W) for 4 polarization channels
        title: Title for the plot
        channel_names: Names of the polarization channels
        save_path: Path to save the plot (if None, plot is displayed)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i in range(4):
        channel_data = sar_data[i]
        
        # Convert to dB scale for better visualization
        if np.any(channel_data > 0):
            channel_data_db = 10 * np.log10(np.abs(channel_data) + 1e-10)
        else:
            channel_data_db = np.abs(channel_data)
        
        im = axes[i].imshow(channel_data_db, cmap='gray')
        axes[i].set_title(f'{channel_names[i]} Channel')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def compute_polarization_ratios(sar_data: np.ndarray) -> dict:
    """
    Compute various polarization ratios from SAR data.
    
    Args:
        sar_data: SAR data of shape (4, H, W) for channels [HH, HV, VH, VV]
        
    Returns:
        Dictionary containing various polarization ratios
    """
    HH = sar_data[0]
    HV = sar_data[1]
    VH = sar_data[2]
    VV = sar_data[3]
    
    ratios = {}
    
    # Basic ratios
    ratios['HH/VV'] = np.abs(HH) / (np.abs(VV) + 1e-10)
    ratios['HV/VV'] = np.abs(HV) / (np.abs(VV) + 1e-10)
    ratios['HH/HV'] = np.abs(HH) / (np.abs(HV) + 1e-10)
    
    # Cross-polarization ratio
    ratios['Cross_pol_ratio'] = np.abs(HV) / (np.abs(HH) + 1e-10)
    
    # Co-polarization ratio
    ratios['Co_pol_ratio'] = np.abs(HH) / (np.abs(VV) + 1e-10)
    
    # Total power
    ratios['Total_power'] = np.abs(HH)**2 + np.abs(HV)**2 + np.abs(VH)**2 + np.abs(VV)**2
    
    return ratios


def normalize_sar_data(sar_data: np.ndarray, 
                      method: str = 'minmax',
                      per_channel: bool = True) -> np.ndarray:
    """
    Normalize SAR data.
    
    Args:
        sar_data: SAR data of shape (4, H, W)
        method: Normalization method ('minmax', 'zscore', 'log')
        per_channel: Whether to normalize each channel separately
        
    Returns:
        Normalized SAR data
    """
    normalized_data = np.copy(sar_data)
    
    if per_channel:
        for i in range(sar_data.shape[0]):
            channel_data = sar_data[i]
            
            if method == 'minmax':
                min_val = np.min(channel_data)
                max_val = np.max(channel_data)
                normalized_data[i] = (channel_data - min_val) / (max_val - min_val + 1e-10)
            
            elif method == 'zscore':
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                normalized_data[i] = (channel_data - mean_val) / (std_val + 1e-10)
            
            elif method == 'log':
                normalized_data[i] = np.log(np.abs(channel_data) + 1e-10)
    
    else:
        if method == 'minmax':
            min_val = np.min(sar_data)
            max_val = np.max(sar_data)
            normalized_data = (sar_data - min_val) / (max_val - min_val + 1e-10)
        
        elif method == 'zscore':
            mean_val = np.mean(sar_data)
            std_val = np.std(sar_data)
            normalized_data = (sar_data - mean_val) / (std_val + 1e-10)
        
        elif method == 'log':
            normalized_data = np.log(np.abs(sar_data) + 1e-10)
    
    return normalized_data


def compute_statistics(sar_data: np.ndarray) -> dict:
    """
    Compute statistical measures for SAR data.
    
    Args:
        sar_data: SAR data of shape (4, H, W)
        
    Returns:
        Dictionary containing statistical measures
    """
    stats = {}
    
    for i, channel_name in enumerate(['HH', 'HV', 'VH', 'VV']):
        channel_data = sar_data[i]
        
        stats[f'{channel_name}_mean'] = np.mean(channel_data)
        stats[f'{channel_name}_std'] = np.std(channel_data)
        stats[f'{channel_name}_min'] = np.min(channel_data)
        stats[f'{channel_name}_max'] = np.max(channel_data)
        stats[f'{channel_name}_median'] = np.median(channel_data)
        
        # Compute histogram
        hist, bins = np.histogram(channel_data, bins=50)
        stats[f'{channel_name}_histogram'] = {'hist': hist, 'bins': bins}
    
    return stats


def augment_sar_data(sar_data: np.ndarray, 
                    rotation_angle: float = 0,
                    flip_horizontal: bool = False,
                    flip_vertical: bool = False,
                    noise_level: float = 0.0) -> np.ndarray:
    """
    Apply data augmentation to SAR data.
    
    Args:
        sar_data: SAR data of shape (4, H, W)
        rotation_angle: Rotation angle in degrees
        flip_horizontal: Whether to flip horizontally
        flip_vertical: Whether to flip vertically
        noise_level: Standard deviation of Gaussian noise to add
        
    Returns:
        Augmented SAR data
    """
    augmented_data = np.copy(sar_data)
    
    # Apply rotation
    if rotation_angle != 0:
        from scipy.ndimage import rotate
        for i in range(sar_data.shape[0]):
            augmented_data[i] = rotate(sar_data[i], rotation_angle, reshape=False)
    
    # Apply flips
    if flip_horizontal:
        augmented_data = np.flip(augmented_data, axis=2)
    
    if flip_vertical:
        augmented_data = np.flip(augmented_data, axis=1)
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, sar_data.shape)
        augmented_data = augmented_data + noise
    
    return augmented_data


def save_sar_sample(sar_data: np.ndarray, 
                   label: int,
                   save_dir: str,
                   sample_id: str = "sample"):
    """
    Save a SAR sample with visualization.
    
    Args:
        sar_data: SAR data of shape (4, H, W)
        label: Class label
        save_dir: Directory to save the sample
        sample_id: Unique identifier for the sample
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save raw data
    data_path = os.path.join(save_dir, f"{sample_id}_label_{label}.npy")
    np.save(data_path, sar_data)
    
    # Save visualization
    viz_path = os.path.join(save_dir, f"{sample_id}_label_{label}.png")
    visualize_sar_image(sar_data, f"SAR Sample - Label {label}", save_path=viz_path)
    
    # Save statistics
    stats = compute_statistics(sar_data)
    stats_path = os.path.join(save_dir, f"{sample_id}_label_{label}_stats.txt")
    
    with open(stats_path, 'w') as f:
        f.write(f"SAR Sample Statistics - Label {label}\n")
        f.write("=" * 50 + "\n")
        for key, value in stats.items():
            if not key.endswith('_histogram'):
                f.write(f"{key}: {value}\n")
    
    print(f"Sample saved to {save_dir}")


def load_sar_sample(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load a SAR sample from file.
    
    Args:
        file_path: Path to the .npy file
        
    Returns:
        Tuple of (sar_data, label)
    """
    sar_data = np.load(file_path)
    
    # Extract label from filename
    filename = os.path.basename(file_path)
    if '_label_' in filename:
        label_str = filename.split('_label_')[1].split('.')[0]
        label = int(label_str)
    else:
        label = 0  # Default label
    
    return sar_data, label
