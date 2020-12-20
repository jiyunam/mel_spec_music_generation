import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, device='cpu'):
        super(Net, self).__init__()
        self.device = device
        self.conv1 = nn.Conv1d(1, 1, 1024, padding=512)
        # self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(1, 1, 100000)
        self.fc1 = nn.Linear(114000, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 14000)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = x.view(-1, 64 * 5 * 5)
        x = x[:,:,:14000]
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        x = x.squeeze(1)  # Flatten to [batch_size]
        return x
