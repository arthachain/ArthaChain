"""
Mock PyTorch implementation for environments where PyTorch cannot be installed.

This module provides basic stubs and mocks for PyTorch functionality to allow
code that depends on PyTorch to run in environments where PyTorch is not available.
"""

import numpy as np

# Mock torch module
class MockTorch:
    """Mock implementation of PyTorch for testing and development."""
    
    class nn:
        """Mock neural network module."""
        
        class Module:
            """Base class for neural network modules."""
            def __init__(self):
                self.training = True
            
            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)
            
            def forward(self, x):
                """Forward pass."""
                return x
            
            def parameters(self):
                """Return an empty iterator for parameters."""
                return iter([])
            
            def to(self, *args, **kwargs):
                """Mock device movement."""
                return self
            
            def eval(self):
                """Set module to evaluation mode."""
                self.training = False
                return self
            
            def train(self, mode=True):
                """Set module to training mode."""
                self.training = mode
                return self
        
        class Linear(Module):
            """Mock linear layer."""
            def __init__(self, in_features, out_features, bias=True):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.weight = np.zeros((out_features, in_features))
                self.bias = np.zeros(out_features) if bias else None
            
            def forward(self, x):
                """Forward pass of linear layer."""
                if isinstance(x, np.ndarray):
                    out = np.dot(x, self.weight.T)
                    if self.bias is not None:
                        out += self.bias
                    return out
                return x
        
        class Conv1d(Module):
            """Mock 1D convolutional layer."""
            def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding
            
            def forward(self, x):
                """Forward pass of convolutional layer."""
                return x
        
        class MaxPool1d(Module):
            """Mock 1D max pooling layer."""
            def __init__(self, kernel_size, stride=None, padding=0):
                super().__init__()
                self.kernel_size = kernel_size
                self.stride = stride if stride is not None else kernel_size
                self.padding = padding
            
            def forward(self, x):
                """Forward pass of max pooling layer."""
                return x
        
        class MultiheadAttention(Module):
            """Mock multi-head attention layer."""
            def __init__(self, embed_dim, num_heads, dropout=0.0):
                super().__init__()
                self.embed_dim = embed_dim
                self.num_heads = num_heads
                self.dropout = dropout
            
            def forward(self, query, key, value):
                """Forward pass of multi-head attention layer."""
                return query, None
        
        class LayerNorm(Module):
            """Mock layer normalization."""
            def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
                super().__init__()
                self.normalized_shape = normalized_shape
                self.eps = eps
                self.elementwise_affine = elementwise_affine
            
            def forward(self, x):
                """Forward pass of layer normalization."""
                return x
        
        class Dropout(Module):
            """Mock dropout layer."""
            def __init__(self, p=0.5, inplace=False):
                super().__init__()
                self.p = p
                self.inplace = inplace
            
            def forward(self, x):
                """Forward pass of dropout layer."""
                return x
        
        class GELU(Module):
            """Mock GELU activation."""
            def forward(self, x):
                """Forward pass of GELU activation."""
                return x
        
        class Sequential(Module):
            """Mock sequential container."""
            def __init__(self, *args):
                super().__init__()
                self.modules = list(args)
            
            def forward(self, x):
                """Forward pass through sequential layers."""
                for module in self.modules:
                    x = module(x)
                return x
        
        class ModuleList(list):
            """Mock module list."""
            def __init__(self, modules=None):
                super().__init__(modules or [])
        
        class ModuleDict(dict):
            """Mock module dictionary."""
            def __init__(self, modules=None):
                super().__init__(modules or {})
        
        class MSELoss:
            """Mock mean squared error loss."""
            def __init__(self, reduction='mean'):
                self.reduction = reduction
            
            def __call__(self, input, target):
                """Compute mean squared error loss."""
                return 0.0
        
        class BCEWithLogitsLoss:
            """Mock binary cross entropy with logits loss."""
            def __init__(self, reduction='mean'):
                self.reduction = reduction
            
            def __call__(self, input, target):
                """Compute binary cross entropy with logits loss."""
                return 0.0
    
    class optim:
        """Mock optimization module."""
        
        class Optimizer:
            """Base optimizer class."""
            def __init__(self, params, defaults):
                self.param_groups = [{'params': params}]
                self.defaults = defaults
            
            def zero_grad(self):
                """Zero out gradients."""
                pass
            
            def step(self):
                """Update parameters."""
                pass
        
        class Adam(Optimizer):
            """Mock Adam optimizer."""
            def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
                defaults = {
                    'lr': lr,
                    'betas': betas,
                    'eps': eps,
                    'weight_decay': weight_decay
                }
                super().__init__(params, defaults)
    
    def tensor(self, data, dtype=None):
        """Create a tensor from data."""
        return np.array(data)
    
    def load(self, path):
        """Mock loading from a path."""
        return {}
    
    def save(self, obj, path):
        """Mock saving to a path."""
        pass

# Create a global instance of MockTorch
torch = MockTorch()

# Define functional module
def sigmoid(x):
    """Mock sigmoid function."""
    return x

def relu(x):
    """Mock ReLU function."""
    return np.maximum(0, x) if isinstance(x, np.ndarray) else x

def softmax(x, dim=None):
    """Mock softmax function."""
    return x

# Add functional to torch
torch.nn.functional = type('MockFunctional', (), {
    'sigmoid': sigmoid,
    'relu': relu,
    'softmax': softmax,
    'gelu': lambda x: x,
})

# Create aliases for commonly used functions
F = torch.nn.functional 