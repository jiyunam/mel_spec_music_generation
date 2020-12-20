import torch
import torch.nn as nn
import numpy as np

class Conv(torch.nn.Module):
    """
    A convolution with the option to be causal and use xavier initialization
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 dilation=1, bias=True, w_init_gain='linear', is_causal=False):
        super(Conv, self).__init__()
        self.is_causal = is_causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    dilation=dilation, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
    
    def forward(self, signal):
        return self.conv(signal)

class WaveNet(nn.Module):
    def __init__(self, layers=10, blocks=3, classes=256, residual_dim=128, skip_dim=512, kernel_size=1, device='cpu'):
        self.layers = layers
        self.blocks = blocks
        self.classes = classes
        self.sd = skip_dim
        self.rd = residual_dim
        self.ks = kernel_size
        self.device = device
        
#         assert self.blocks > 1
        
        self.init_filter = 2
        self.dilations = [2 ** i for i in range(self.layers)] * self.blocks
        self.rf = np.sum(self.dilations[:self.layers]) + self.init_filter #receptive field per block
        
        self.hidden_dim = 128
        # 1x1 convolution to create channels
        super(WaveNet, self).__init__()
        self.start_conv = torch.nn.Conv1d(in_channels=self.classes,
                                          out_channels=self.hidden_dim,
                                          kernel_size=self.init_filter)
        
        self.ydcnn = nn.ModuleList() # dilated cnn
        self.ydense = nn.ModuleList() #
        self.yskip = nn.ModuleList()  # skip connections

        for dilation in self.dilations:
            self.ydcnn.append(Conv(self.hidden_dim, self.hidden_dim*2, kernel_size=2, dilation=dilation, is_causal=True))
            self.yskip.append(Conv(self.hidden_dim, self.sd, w_init_gain='relu'))
            self.ydense.append(Conv(self.hidden_dim, self.hidden_dim, w_init_gain='linear'))
        
        self.end_conv1 = Conv(self.sd, self.sd, bias=False, w_init_gain='relu')
        self.end_conv2 = Conv(self.sd, self.classes, bias=False, w_init_gain='linear')
        
    def forward(self, y):
        # one hot encode 
        y = torch.zeros(1, 256, y.shape[-1]).to(self.device).scatter_(dim=1, index=y.reshape(1,1,-1), value=1.0)
        y = self.start_conv(y)
        
        finalout = y.size(2)-(self.rf-2)
        for idx, dilation in enumerate(self.dilations):
            # dilated convolution
            in_act = self.ydcnn[idx](y)
            t_act = torch.tanh(in_act[:, :self.hidden_dim, :])
            s_act = torch.sigmoid(in_act[:, self.hidden_dim:, :])
            acts = t_act * s_act
            
            # residual
            res_acts = self.ydense[idx](acts)
            
            # skip connection
#             import pdb
#             pdb.set_trace()
            if idx == 0: # if first iteration
                output = self.yskip[idx](acts[:,:,-finalout:]) 
            else: 
                output = self.yskip[idx](acts[:,:,-finalout:]) + output

            y = res_acts + y[:,:,dilation:]
            
        output = torch.nn.functional.relu(output, True)
        output = self.end_conv1(output)
        output = torch.nn.functional.relu(output, True)
        output = self.end_conv2(output)
        
        return output