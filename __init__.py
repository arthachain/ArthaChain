"""AI Engine Models Module

This module contains neural network models for the blockchain AI engine.
"""

try:
    import torch
    import numpy as np
    from torch import nn
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except ImportError:
    import mock_torch
    TORCH_AVAILABLE = False

# Import spike detector and decoder
try:
    from .spike_detector import SpikeDetector
    from .decoder import NeuralDecoder
    from .adaptive_network import AdaptiveNetwork
except ImportError:
    # Create stub classes if models can't be imported
    class SpikeDetector:
        def __init__(self, *args, **kwargs):
            pass
            
    class NeuralDecoder:
        def __init__(self, *args, **kwargs):
            pass
            
    class AdaptiveNetwork:
        def __init__(self, *args, **kwargs):
            pass

# Define version
__version__ = '0.1.0' 