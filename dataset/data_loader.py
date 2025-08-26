import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
from typing import List, Tuple, Optional, Dict, Any
import glob


class PolarimetricSARDataset(Dataset):
    """
    Dataset class for loading polarimetric SAR data.
    
    This dataset loads SAR images from different polarization channels (HH, HV, VH, VV)
    and organizes them by target classes.
    """
    
    def __init__(self, 
                 root_dir: str,
                 channel_dirs: List[str] = ["HH_NPY", "HV_NPY", "VH_NPY", "VV_NPY"],
                 transform: Optional[transforms.Compose] = None,
                 train: bool = True,
                 train_split: float = 0.8,
                 target_classes: Optional[List[str]] = None,
                 max_samples_per_class: Optional[int] = None,
                 random_seed: int = 42):
        """
        Initialize the PolarimetricSARDataset.
        
        Args:
            root_dir: Root directory containing the SAR data
            channel_dirs: List of directory names for different polarization channels
            transform: Optional transforms to apply to the data
            train: Whether this is for training (True) or validation (False)
            train_split: Fraction of data to use for training
            target_classes: List of target classes to include (if None, all classes are included)
            max_samples_per_class: Maximum number of samples per class (if None, all samples are used)
            random_seed: Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.channel_dirs = channel_dirs
        self.transform = transform
        self.train = train
        self.train_split = train_split
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Get all target classes
        if target_classes is None:
            self.target_classes = self._get_target_classes()
        else:
            self.target_classes = target_classes
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.target_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load data paths and labels
        self.data_paths, self.labels = self._load_data_paths(max_samples_per_class)
        
        print(f"Loaded {len(self.data_paths)} samples for {'training' if train else 'validation'}")
        print(f"Classes: {self.target_classes}")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _get_target_classes(self) -> List[str]:
        """Get all available target classes from the root directory."""
        classes = []
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                classes.append(item)
        return sorted(classes)
    
    def _load_data_paths(self, max_samples_per_class: Optional[int] = None) -> Tuple[List[str], List[int]]:
        """Load data paths and labels for all samples."""
        data_paths = []
        labels = []
        
        for class_name in self.target_classes:
            class_path = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Get all pass directories
            pass_dirs = [d for d in os.listdir(class_path) if d.startswith('pass') and os.path.isdir(os.path.join(class_path, d))]
            
            class_samples = []
            for pass_dir in pass_dirs:
                pass_path = os.path.join(class_path, pass_dir)
                
                # Check if all channel directories exist
                channel_paths = []
                for channel_dir in self.channel_dirs:
                    channel_path = os.path.join(pass_path, channel_dir)
                    if os.path.exists(channel_path):
                        channel_paths.append(channel_path)
                    else:
                        print(f"Warning: Channel directory {channel_path} not found")
                
                if len(channel_paths) == len(self.channel_dirs):
                    # Get all .npy files from the first channel directory
                    first_channel_path = channel_paths[0]
                    npy_files = glob.glob(os.path.join(first_channel_path, "*.npy"))
                    
                    for npy_file in npy_files:
                        # Check if corresponding files exist in all channels
                        sample_paths = []
                        base_name = os.path.basename(npy_file)
                        
                        for channel_path in channel_paths:
                            channel_file = os.path.join(channel_path, base_name)
                            if os.path.exists(channel_file):
                                sample_paths.append(channel_file)
                            else:
                                break
                        
                        if len(sample_paths) == len(self.channel_dirs):
                            class_samples.append(sample_paths)
            
            # Limit samples per class if specified
            if max_samples_per_class is not None and len(class_samples) > max_samples_per_class:
                class_samples = np.random.choice(class_samples, max_samples_per_class, replace=False).tolist()
            
            # Split into train/val
            np.random.shuffle(class_samples)
            split_idx = int(len(class_samples) * self.train_split)
            
            if self.train:
                selected_samples = class_samples[:split_idx]
            else:
                selected_samples = class_samples[split_idx:]
            
            data_paths.extend(selected_samples)
            labels.extend([class_idx] * len(selected_samples))
        
        return data_paths, labels
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of samples across classes."""
        distribution = {}
        for class_name in self.target_classes:
            class_idx = self.class_to_idx[class_name]
            count = self.labels.count(class_idx)
            distribution[class_name] = count
        return distribution
    
    def __len__(self) -> int:
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (data, label) where data is a tensor of shape (4, H, W) for 4 polarization channels
        """
        sample_paths = self.data_paths[idx]
        label = self.labels[idx]
        
        # Load data from all channels
        channel_data = []
        for channel_path in sample_paths:
            data = np.load(channel_path)
            channel_data.append(data)
        
        # Stack channels along the first dimension
        data = np.stack(channel_data, axis=0)  # Shape: (4, H, W)
        
        # Convert to tensor
        data = torch.FloatTensor(data)
        
        # Apply transforms if specified
        if self.transform is not None:
            data = self.transform(data)
        
        return data, label


class SARDatasetFromFeatures(Dataset):
    """
    Dataset class for loading pre-extracted SAR features.
    
    This dataset loads features that were extracted using a trained model.
    """
    
    def __init__(self, 
                 features_file: str,
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize the SARDatasetFromFeatures.
        
        Args:
            features_file: Path to the pickle file containing features
            transform: Optional transforms to apply to the features
        """
        self.transform = transform
        
        # Load features from pickle file
        with open(features_file, 'rb') as f:
            data = pickle.load(f)
            self.features = torch.FloatTensor(data['features'])
            self.labels = torch.LongTensor(data['labels'])
        
        print(f"Loaded {len(self.features)} samples from {features_file}")
        print(f"Features shape: {self.features.shape}")
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (features, label)
        """
        features = self.features[idx]
        label = self.labels[idx]
        
        if self.transform is not None:
            features = self.transform(features)
        
        return features, label


def get_sar_transforms(input_size: Tuple[int, int] = (224, 224), 
                      normalize: bool = True,
                      augment: bool = False) -> transforms.Compose:
    """
    Get transforms for SAR data.
    
    Args:
        input_size: Target size for the images
        normalize: Whether to normalize the data
        augment: Whether to apply data augmentation (for training)
        
    Returns:
        Compose transform
    """
    transform_list = []
    
    if augment:
        # Add data augmentation for training
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        ])
    else:
        # Resize for validation/testing
        transform_list.append(transforms.Resize(input_size))
    
    # Convert to tensor if not already
    transform_list.append(transforms.ToTensor())
    
    if normalize:
        # Normalize SAR data (you may need to adjust these values based on your data)
        transform_list.append(transforms.Normalize(
            mean=[0.5, 0.5, 0.5, 0.5],  # For 4 channels
            std=[0.5, 0.5, 0.5, 0.5]
        ))
    
    return transforms.Compose(transform_list)


def create_sar_dataloaders(root_dir: str,
                          batch_size: int = 32,
                          num_workers: int = 4,
                          train_split: float = 0.8,
                          input_size: Tuple[int, int] = (224, 224),
                          max_samples_per_class: Optional[int] = None,
                          random_seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders for SAR data.
    
    Args:
        root_dir: Root directory containing the SAR data
        batch_size: Batch size for the dataloaders
        num_workers: Number of workers for data loading
        train_split: Fraction of data to use for training
        input_size: Target size for the images
        max_samples_per_class: Maximum number of samples per class
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create transforms
    train_transform = get_sar_transforms(input_size, normalize=True, augment=True)
    val_transform = get_sar_transforms(input_size, normalize=True, augment=False)
    
    # Create datasets
    train_dataset = PolarimetricSARDataset(
        root_dir=root_dir,
        transform=train_transform,
        train=True,
        train_split=train_split,
        max_samples_per_class=max_samples_per_class,
        random_seed=random_seed
    )
    
    val_dataset = PolarimetricSARDataset(
        root_dir=root_dir,
        transform=val_transform,
        train=False,
        train_split=train_split,
        max_samples_per_class=max_samples_per_class,
        random_seed=random_seed
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_feature_dataloaders(train_features_file: str,
                              val_features_file: str,
                              batch_size: int = 32,
                              num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders for pre-extracted features.
    
    Args:
        train_features_file: Path to training features pickle file
        val_features_file: Path to validation features pickle file
        batch_size: Batch size for the dataloaders
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = SARDatasetFromFeatures(train_features_file)
    val_dataset = SARDatasetFromFeatures(val_features_file)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
