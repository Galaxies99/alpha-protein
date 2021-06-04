import torch.nn as nn
import torch.nn.functional as F


class DeepCov(nn.Module):
    def __init__(self, args = {}):
        super(DeepCov, self).__init__()
        in_channel = args.get('input_channel', 441)
        out_channel = args.get('output_channel', 10)
        hidden_channel = args.get('hidden_channel', 64)
        blocks = args.get('blocks', 10)
        self.blocks = blocks

        self.MaxOutConv = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel * 2, kernel_size = 1),
            nn.BatchNorm2d(hidden_channel * 2)
        )
        
        self.interConvList = []
        for _ in range(blocks):
            self.interConvList.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channel, hidden_channel, kernel_size = 5, padding = 2),
                    nn.BatchNorm2d(hidden_channel),
                    nn.ReLU(inplace = True)
                )
            )
        self.interConvList = nn.ModuleList(self.interConvList)

        self.OutConv = nn.Conv2d(hidden_channel, out_channel, kernel_size = 1)

    def forward(self, x):
        x = self.MaxOutConv(x)

        # feature max_pooling
        n, c, w, h = x.size()
        x = x.view(n, c, w * h).permute(0, 2, 1)
        x = F.max_pool1d(x, 2)
        _, _, c = x.size()
        x = x.permute(0, 2, 1)
        x = x.view(n, c, w, h)

        for i in range(self.blocks):
            x = self.interConvList[i](x)
        
        x = self.OutConv(x)
        return x
