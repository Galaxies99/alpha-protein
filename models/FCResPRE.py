import torch
import torch.nn as nn
import torch.nn.functional as F
from .resblocks import BasicBlock


class FCResPRE(nn.Module):
    def __init__(self, args = {}):
        super(FCResPRE, self).__init__()
        hidden_channel = args.get('hidden_channel', 64)
        in_channel = args.get('input_channel', 441)
        out_channel = args.get('output_channel', 10)
        droprate = args.get('droprate', 0.2)
        blocks = int(args.get('blocks', 22))

        self.conv1 = nn.Conv2d(in_channel, hidden_channel, kernel_size = 1, bias = False)
        self.bn1 = nn.InstanceNorm2d(hidden_channel)
        self.relu = nn.ReLU(inplace=True)

        layers = []
        for _ in range(blocks):
            layers.append(BasicBlock(hidden_channel, hidden_channel, droprate))
        self.layer = nn.Sequential(*layers)

        self.lastlayer = nn.Conv2d(hidden_channel, out_channel, kernel_size = 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer(x)
        x = self.lastlayer(x)
        return x
