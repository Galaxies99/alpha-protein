import torch
from torch import nn, einsum
import torch.nn.functional as F
from .resblocks import BasicBlock
from .attention import GlobalNonLocal


class NLResPRE(nn.Module):
    def __init__(self, args = {}):
        super(NLResPRE, self).__init__()
        in_channel, out_channel = args.get('input_channel', 441), args.get('out_channel', 10)
        hidden_channel = args.get('hidden_channel', 64)
        blocks = args.get('blocks', 22)
        dropout_rate = args.get('dropout_rate', 0.2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, kernel_size = 1, bias = False),
            nn.InstanceNorm2d(hidden_channel),
            nn.ReLU(inplace = True)
        )

        layer = []
        for _ in range(int(blocks)):
            layer.append(BasicBlock(hidden_channel, hidden_channel, dropout_rate))
        self.layer = nn.Sequential(*layer)

        self.attn = GlobalNonLocal(in_channel = hidden_channel)

        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channel + hidden_channel, hidden_channel, kernel_size = 3, padding = 1, bias = False),
            nn.InstanceNorm2d(hidden_channel),
            nn.ReLU(inplace = True)
        )

        self.final = nn.Conv2d(hidden_channel, out_channel, kernel_size = 1)
    
    def forward(self, x):
        x = self.conv(x)
        y1 = self.layer(x)
        y2 = self.attn(x)
        y = torch.cat([y1, y2], dim = 1)
        y = self.fusion(y)
        y = self.final(y)
        return y
