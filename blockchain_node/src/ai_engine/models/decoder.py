import torch

class NeuralDecoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
        )
        
        self.temporal_model = torch.nn.GRU(
            256, 256, num_layers=2, 
            bidirectional=True, dropout=0.2
        )
        
        self.attention = torch.nn.MultiheadAttention(
            512, num_heads=8, dropout=0.1
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        # Encode features
        x = self.encoder(x)
        
        # Temporal modeling
        x, _ = self.temporal_model(x)
        
        # Self-attention
        x, _ = self.attention(x, x, x)
        
        # Decode intent
        return self.decoder(x[-1]) 