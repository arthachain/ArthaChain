import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeuralDecoder(nn.Module):
    """Neural decoder that maps brain signals to command outputs."""
    
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128]):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, x):
        """Forward pass through the decoder network."""
        # Check if x is a NumPy array and convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
            
        # Handle single sample vs batch
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Pass through layers with ReLU activation
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        # Output layer
        output = self.output_layer(x)
        
        return output
    
    def train_step(self, x, y):
        """Train the decoder for one step."""
        # Convert numpy arrays to torch tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)
            
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.forward(x)
        
        # Calculate loss
        loss = self.criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        self.optimizer.step()
        
        return loss.item() 