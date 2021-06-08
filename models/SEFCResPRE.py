import torch
import torch.nn as nn
import torch.nn.functional as F
from .resblocks import BasicBlock
from .attention import SEAttention


class SEFCResPRE(nn.Module):
    def __init__(self, args = {}):
        super(SEFCResPRE, self).__init__()
        in_channel, out_channel = args.get('input_channel', 441), args.get('out_channel', 10)
        droprate = args.get('droprate', 0.2)
        hidden_channel = args.get('hidden_channel', 64)
        blocks = int(args.get('blocks', 22))
        reduction_rate = args.get('reduction', 4)

        self.conv1 = nn.Conv2d(in_channel, hidden_channel, kernel_size = 1, bias = False)
        self.in1 = nn.InstanceNorm2d(hidden_channel)
        self.relu = nn.ReLU(inplace=True)

        layer = []
        for _ in range(blocks):
            layer.append(BasicBlock(
                hidden_channel, hidden_channel, 
                droprate = droprate, 
                attention = SEAttention(hidden_channel, reduction = reduction_rate)
            ))
        self.layer = nn.Sequential(*layer)

        self.final = nn.Conv2d(hidden_channel, out_channel, kernel_size = 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        x = self.layer(x)
        x = self.final(x)
        return x
