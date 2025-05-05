"""
AI Engine Models for Blockchain Neural Processing
================================================

This module contains neural network models for
blockchain brain-computer interfaces.
"""

# Try to import torch, fall back to our mock implementation if not available
try:
    import torch
except ImportError:
    try:
        from .mock_torch import torch
        print("Using mock PyTorch implementation. For full functionality, run install_deps.sh to install real PyTorch.")
    except ImportError:
        print("PyTorch not installed and mock implementation not found. Run install_deps.sh to install dependencies.")

# Only import model classes if torch is available
__all__ = []

try:
    from .spike_detector import SpikeDetector
    from .decoder import NeuralDecoder
    
    __all__.extend(['SpikeDetector', 'NeuralDecoder'])
except ImportError as e:
    print(f"Could not import neural models: {e}")
    pass 