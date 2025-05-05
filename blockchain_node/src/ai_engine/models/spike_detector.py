import torch

class SpikeDetector(torch.nn.Module):
    def __init__(self, num_channels, window_size):
        super().__init__()
        
        self.conv1 = torch.nn.Conv1d(num_channels, 64, kernel_size=5)
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=5)
        self.lstm = torch.nn.LSTM(128, 256, num_layers=2, bidirectional=True)
        self.fc = torch.nn.Linear(512, num_channels)
        
    def forward(self, x):
        # Convolutional feature extraction
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        
        # Sequence modeling
        x, _ = self.lstm(x.permute(2, 0, 1))
        
        # Spike prediction
        return self.fc(x[-1]) 