import torch
import torch.nn as nn
import numpy as np

class AdaptiveNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, n_heads, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_layers[0])
        self.output_proj = nn.Linear(hidden_layers[-1], output_dim)
