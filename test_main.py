#!/usr/bin/env python3
"""
Test script for main.py to validate the MCR² and regular training modes
"""

import sys
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.aconvnets import Network
from models.cnn_encoder import ParallelChannelCNN
from mcr2 import MaximalCodingRateReduction

class TestSARDataset(Dataset):
    """Simple test dataset for SAR data"""
    def __init__(self, num_samples=100, num_classes=7):
        self.num_samples = num_samples
        self.num_classes = num_classes
        # Create dummy SAR data
        self.data = torch.randn(num_samples, 4, 72, 54, 54)  # 4 channels for polarimetric SAR
        self.labels = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def test_models():
    """Test that both models can be instantiated and run forward pass"""
    print("Testing model instantiation...")
    
    # Test Network (A-ConvNet)
    try:
        network_model = Network(classes=7, channels=4, dropout_rate=0.5)
        print("✓ Network model created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(2, 4, 72, 54, 54)
        output = network_model(dummy_input)
        print(f"✓ Network forward pass successful, output shape: {output.shape}")
        
    except Exception as e:
        print(f"✗ Network model failed: {e}")
        return False
    
    # Test ParallelChannelCNN
    try:
        parallel_model = ParallelChannelCNN(num_classes=7)
        print("✓ ParallelChannelCNN model created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(2, 4, 72, 54, 54)
        output = parallel_model(dummy_input)
        print(f"✓ ParallelChannelCNN forward pass successful, output shape: {output.shape}")
        
    except Exception as e:
        print(f"✗ ParallelChannelCNN model failed: {e}")
        return False
    
    return True

def test_mcr2_loss():
    """Test MCR² loss function"""
    print("\nTesting MCR² loss...")
    
    try:
        mcr2_criterion = MaximalCodingRateReduction(gam1=1.0, gam2=1.0, eps=0.01)
        print("✓ MCR² loss created successfully")
        
        # Test forward pass
        dummy_features = torch.randn(10, 128).cuda()  # 10 samples, 128 features
        dummy_labels = torch.randint(0, 7, (10,)).cuda()  # 7 classes
        
        loss, loss_components, theoretical_components = mcr2_criterion(dummy_features, dummy_labels)
        print(f"✓ MCR² loss forward pass successful, loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ MCR² loss failed: {e}")
        return False
    
    return True

def test_training_logic():
    """Test the training logic with both modes"""
    print("\nTesting training logic...")
    
    # Create test dataset
    dataset = TestSARDataset(num_samples=50, num_classes=7)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Test regular training (Network + CrossEntropy)
    try:
        model = Network(classes=7, channels=4, dropout_rate=0.5)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx == 0:  # Just test first batch
                print("✓ Regular training (Network + CrossEntropy) successful")
                break
                
    except Exception as e:
        print(f"✗ Regular training failed: {e}")
        return False
    
    # Test MCR² training (ParallelChannelCNN + MCR²)
    try:
        model = ParallelChannelCNN(num_classes=7)
        criterion = MaximalCodingRateReduction(gam1=1.0, gam2=1.0, eps=0.01)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss, _, _ = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx == 0:  # Just test first batch
                print("✓ MCR² training (ParallelChannelCNN + MCR²) successful")
                break
                
    except Exception as e:
        print(f"✗ MCR² training failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing SAR Target Recognition Training Logic")
    print("=" * 50)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("✓ CUDA is available")
    else:
        print("⚠ CUDA is not available, using CPU")
    
    # Run tests
    tests = [
        ("Model Instantiation", test_models),
        ("MCR² Loss Function", test_mcr2_loss),
        ("Training Logic", test_training_logic),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                print(f"✓ {test_name} passed")
            else:
                print(f"✗ {test_name} failed")
                all_passed = False
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The training logic is working correctly.")
        print("\nUsage examples:")
        print("  # Regular training with Network + CrossEntropy")
        print("  python main.py --epochs 10 --batch_size 8")
        print("\n  # MCR² training with ParallelChannelCNN + MCR² loss")
        print("  python main.py --mcr2 --epochs 10 --batch_size 8")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 50)

if __name__ == "__main__":
    main()
