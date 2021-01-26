import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, batch_size):
        super(Net, self).__init__()
        self.bs = batch_size
        self.conv1 = nn.Conv2d(batch_size, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=1, padding=1)
        self.drp = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(32, batch_size)


    def forward(self, x):
        x = self.conv1(x)
        # x = F.relu(x)
        x = self.conv2(x)
        x = self.pool1(x)
        # x = self.drp(x)
        B, _, H, W = x.shape
        x = x.view(-1,32)
        x = F.relu(self.fc1(x))
        x = x.view(B,self.bs,H, W)
        return x