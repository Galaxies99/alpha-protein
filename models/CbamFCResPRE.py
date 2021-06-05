import torch
import torch.nn as nn
import torch.nn.functional as F
from .resblocks import BasicBlock, Bottleneck
from .attention import Cbam


class CbamFCResPRE(nn.Module):
    def __init__(self, args = {}):
        super(CbamFCResPRE, self).__init__()
        in_channel, out_channel = args.get('input_channel', 441), args.get('out_channel', 10)
        droprate = args.get('droprate', 0.2)
        hidden_channel = args.get('hidden_channel', 64)
        blocks, block_type = int(args.get('blocks', 22)), args.get('block_type', 'BasicBlock')
        if block_type not in ['BasicBlock', 'Bottleneck']:
            raise AttributeError('Invalid block type, block type should be BasicBlock or Bottleneck')
        block = BasicBlock if block_type == 'BasicBlock' else Bottleneck

        self.conv1 = nn.Conv2d(in_channel, hidden_channel, kernel_size = 1, bias = False)
        self.in1 = nn.InstanceNorm2d(hidden_channel)
        self.relu = nn.ReLU(inplace=True)

        layer = []
        for _ in range(blocks):
            layer.append(block(
                hidden_channel, hidden_channel, 
                droprate = droprate, 
                attention = Cbam(hidden_channel)
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
