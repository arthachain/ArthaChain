#!/usr/bin/env python
"""
Test script for AI model modules.
Run this script to verify that the neural network models can be imported and used.
"""

import os
import sys
import numpy as np

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_models():
    """Test importing and using the neural network models."""
    try:
        # First test basic imports
        import torch
        print("✅ PyTorch imported successfully")
        
        from decoder import NeuralDecoder
        print("✅ NeuralDecoder imported successfully")
        
        from spike_detector import SpikeDetector
        print("✅ SpikeDetector imported successfully")
        
        from adaptive_network import AdaptiveNetwork
        print("✅ AdaptiveNetwork imported successfully")
        
        # Now test model instantiation
        decoder = NeuralDecoder(input_dim=100, output_dim=10)
        print("✅ NeuralDecoder instantiated successfully")
        
        spike_detector = SpikeDetector(num_channels=32, window_size=100)
        print("✅ SpikeDetector instantiated successfully")
        
        adaptive_net = AdaptiveNetwork(
            input_dim=100, 
            output_dim=10, 
            hidden_layers=[256, 128], 
            n_heads=4, 
            dropout=0.1
        )
        print("✅ AdaptiveNetwork instantiated successfully")
        
        # Test forward pass
        sample_input = np.random.rand(5, 100).astype(np.float32)  # 5 samples, 100 features
        
        decoder_output = decoder(sample_input)
        print(f"✅ NeuralDecoder forward pass shape: {decoder_output.shape}")
        
        # Reshape for spike detector (batch, channels, time)
        spike_input = np.random.rand(5, 32, 100).astype(np.float32)
        spike_output = spike_detector(spike_input)
        print(f"✅ SpikeDetector forward pass shape: {spike_output.shape}")
        
        adaptive_output = adaptive_net(sample_input)
        print(f"✅ AdaptiveNetwork forward pass shape: {adaptive_output.shape}")
        
        print("\n🎉 All tests passed! Models are working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_models() 