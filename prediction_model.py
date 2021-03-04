import torch.nn as nn
import torch.nn.functional as F

class PredNet(nn.Module):
    def __init__(self):
        super(PredNet, self).__init__()
        self.conv1 = nn.Conv1d(80, 16, 7)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 3)
        self.fc1 = nn.Linear(65*32, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc1(x.view(-1, 65*32)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
