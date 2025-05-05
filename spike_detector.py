import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpikeDetector(nn.Module):
    """Spike detector for neural signals."""
    
    def __init__(self, num_channels, window_size, threshold=5.0):
        super().__init__()
        
        self.num_channels = num_channels
        self.window_size = window_size
        self.threshold = threshold
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate the size after convolutions and pooling
        feature_size = window_size // 2 // 2 // 2  # After 3 pooling operations
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * feature_size, 512)
        self.fc2 = nn.Linear(512, num_channels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, x):
        """Forward pass through the spike detector."""
        # Convert numpy arrays to torch tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Reshape input if it's a 2D matrix [batch_size, features]
        if len(x.shape) == 2:
            # Assuming shape is [batch_size, num_channels * window_size]
            batch_size = x.shape[0]
            x = x.view(batch_size, self.num_channels, -1)
        elif len(x.shape) == 1:
            # Single sample, reshape to [1, num_channels, window_size]
            x = x.view(1, self.num_channels, -1)
        
        # Apply convolutions with ReLU activation
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def detect_spikes(self, signals, threshold=None):
        """Detect spikes in neural signals."""
        if threshold is None:
            threshold = self.threshold
            
        # Get the raw spike scores
        spike_scores = self.forward(signals)
        
        # Apply sigmoid to get probabilities
        spike_probs = torch.sigmoid(spike_scores)
        
        # Threshold the probabilities to get binary spike detection
        spikes = (spike_probs > threshold).float()
        
        return spikes, spike_probs
    
    def train_step(self, x, y):
        """Train the spike detector for one step."""
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