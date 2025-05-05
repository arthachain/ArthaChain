import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaptiveNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, n_heads, dropout):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_layers[0])
        
        # Hidden layers with attention and residual connections
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.ModuleDict({
                'attention': nn.MultiheadAttention(
                    hidden_layers[i], n_heads, dropout=dropout
                ),
                'norm1': nn.LayerNorm(hidden_layers[i]),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_layers[i], hidden_layers[i] * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_layers[i] * 4, hidden_layers[i+1])
                ),
                'norm2': nn.LayerNorm(hidden_layers[i+1])
            }))
            
        # Output projection
        self.output_proj = nn.Linear(hidden_layers[-1], output_dim)
        
        # Activation
        self.activation = nn.GELU()
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, x):
        """Forward pass through the adaptive network."""
        # Convert numpy arrays to torch tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
            
        # Handle single sample vs batch
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Input projection
        x = self.activation(self.input_proj(x))
        
        # Process through attention layers
        for layer in self.layers:
            # Self-attention
            attn_input = x.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
            attn_output, _ = layer['attention'](attn_input, attn_input, attn_input)
            attn_output = attn_output.permute(1, 0, 2)  # [batch_size, seq_len, hidden_dim]
            
            # Residual connection and normalization
            x = layer['norm1'](x + attn_output)
            
            # Feed-forward network
            ffn_output = layer['ffn'](x)
            
            # Residual connection and normalization
            x = layer['norm2'](ffn_output)
        
        # Output projection
        output = self.output_proj(x)
        
        return output
    
    def train_step(self, states, actions, rewards, next_states, dones):
        """Train the network for one step using reinforcement learning."""
        # Convert numpy arrays to torch tensors if needed
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32)
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, dtype=torch.long)
        if isinstance(rewards, np.ndarray):
            rewards = torch.tensor(rewards, dtype=torch.float32)
        if isinstance(next_states, np.ndarray):
            next_states = torch.tensor(next_states, dtype=torch.float32)
        if isinstance(dones, np.ndarray):
            dones = torch.tensor(dones, dtype=torch.float32)
            
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        current_q_values = self.forward(states)
        
        # Get Q values for the selected actions
        q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.forward(next_states)
            max_next_q = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * 0.99 * max_next_q
        
        # Calculate loss
        loss = self.criterion(q_values, target_q_values)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        self.optimizer.step()
        
        return loss.item() 