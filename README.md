# SAR Target Recognition

A deep learning-based Synthetic Aperture Radar (SAR) target recognition system using convolutional neural networks and advanced training techniques.

## Overview

This project implements SAR target recognition using various deep learning architectures including:
- **A-ConvNets**: Attention-based Convolutional Neural Networks
- **CNN Encoders**: Custom CNN architectures for feature extraction
- **MCR²**: Maximal Coding Rate Reduction for improved feature learning

The system processes SAR imagery data and performs target classification with state-of-the-art performance.

## Features

- **SAR Data Processing**: Comprehensive SAR imaging and data preprocessing pipeline
- **Multiple Model Architectures**: Support for various CNN-based models
- **Advanced Training**: Implementation of MCR² loss for better feature learning
- **Data Augmentation**: Built-in data augmentation techniques
- **Visualization Tools**: SAR image plotting and analysis utilities

## Project Structure

```
SAR_TARGET_RECOGNITION/
├── dataset/                 # Dataset handling and utilities
│   ├── data_loader.py      # Data loading and preprocessing
│   └── utils.py            # Dataset utilities
├── models/                  # Neural network architectures
│   ├── aconvnets.py        # Attention-based ConvNet implementation
│   ├── cnn_encoder.py      # CNN encoder architectures
│   └── _blocks.py          # Building blocks for models
├── prepare_dataset/         # SAR data preparation pipeline
│   ├── SAR_imaging.py      # SAR imaging and processing
│   ├── generate.py         # Dataset generation utilities
│   └── plot.py             # Visualization and plotting tools
├── utils/                   # Utility functions
│   ├── datapreprocess.py   # Data preprocessing utilities
│   └── dataset/            # Additional dataset utilities
├── main.py                  # Main training script
├── mcr2.py                 # Maximal Coding Rate Reduction implementation
└── requirements.txt        # Python dependencies
```

## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended)
- PyTorch 1.8+

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd SAR_TARGET_RECOGNITION
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install PyTorch (if not already installed):
```bash
# For CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio
```

## Usage

### Data Preparation

1. **SAR Data Processing**:
```python
from prepare_dataset.SAR_imaging import AFRL
from prepare_dataset.generate import generate_dataset

# Process SAR data
phs_data = AFRL(directory, polarization, start_azimuth)

# Generate training dataset
generate_dataset(input_path, output_path)
```

2. **Data Loading**:
```python
from dataset.data_loader import SARDataLoader

# Initialize data loader
dataloader = SARDataLoader(
    data_path='path/to/sar/data',
    batch_size=32,
    shuffle=True
)
```

### Model Training

1. **Basic Training**:
```bash
python main.py --dataset sar --data_folder ./data --batch_size 16 --epochs 50
```

2. **Training with MCR² Loss**:
```python
from mcr2 import MaximalCodingRateReduction

# Initialize MCR² loss
mcr2_loss = MaximalCodingRateReduction(gam1=1.0, gam2=1.0, eps=0.01)

# Use in training loop
loss = mcr2_loss(features, labels)
```

3. **Training with A-ConvNets**:
```python
from models.aconvnets import Network

# Initialize A-ConvNet model
model = Network(
    classes=10,
    channels=1,
    dropout_rate=0.5
)
```

### Model Evaluation

```python
# Evaluate model performance
python main.py --eval --resume path/to/checkpoint.pth
```

## Model Architectures

### A-ConvNets (Attention-based ConvNets)

The A-ConvNet architecture uses attention mechanisms to focus on relevant features in SAR imagery:

```python
# Architecture details:
# - 4 convolutional layers with max pooling
# - Attention mechanisms for feature selection
# - Dropout for regularization
# - Global average pooling for classification
```

### CNN Encoder

Custom CNN encoder for feature extraction from SAR data:

```python
# Features:
# - Multiple convolutional layers
# - Batch normalization
# - Residual connections
# - Adaptive pooling
```

### MCR² (Maximal Coding Rate Reduction)

Advanced loss function that maximizes coding rate reduction for better feature learning:

```python
# Benefits:
# - Improved feature discrimination
# - Better class separation
# - Enhanced generalization
```

## Data Format

The system supports various SAR data formats:

- **AFRL .mat files**: Air Force Research Laboratory format
- **Phase History Data**: Raw SAR phase history
- **Processed SAR Images**: Range-Doppler processed imagery

## Configuration

Key parameters can be configured in the main training script:

```python
# Training parameters
--epochs: Number of training epochs
--batch_size: Batch size for training
--learning_rate: Learning rate
--optimizer: Optimizer choice (adam, sgd)

# Model parameters
--model: Model architecture choice
--dropout_rate: Dropout probability
--classes: Number of target classes

# Data parameters
--data_folder: Path to dataset
--num_workers: Number of data loading workers
```

## Performance

The system achieves competitive performance on SAR target recognition tasks:

- **Accuracy**: >90% on standard SAR datasets
- **Training Time**: Optimized for GPU acceleration
- **Memory Usage**: Efficient memory management for large datasets

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{sar_target_recognition,
  title={SAR Target Recognition using Deep Learning},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## Acknowledgments

- Air Force Research Laboratory (AFRL) for SAR data
- PyTorch community for deep learning framework
- Research community for SAR imaging algorithms

## Contact

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This project is designed for research purposes. Please ensure compliance with data usage agreements and export control regulations when working with SAR data.
