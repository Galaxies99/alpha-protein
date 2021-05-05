import torch.nn as nn
import torch.nn.functional as F


class DeepCov(nn.Module):
    def __init__(self, args = {}):
        super(DeepCov, self).__init__()
        in_channel = args.get('input_channel', 441)
        out_channel = args.get('output_channel', 10)
        super(DeepCov, self).__init__()

        self.MaxOutConv = nn.Sequential(
            nn.Conv2d(in_channel, 128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128)
        )
        
        self.interConvList = []
        for i in range(10):
            self.interConvList.append(
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
                    nn.BatchNorm2d(64),
                    nn.ReLU()
                )
            )
        self.interConvList = nn.ModuleList(self.interConvList)

        self.OutConv = nn.Conv2d(64, out_channel, kernel_size=(1, 1))

    def forward(self, x):
        x = self.MaxOutConv(x)

        # feature max_pooling
        n, c, w, h = x.size()
        x = x.view(n, c, w * h).permute(0, 2, 1)
        x = F.max_pool1d(x, 2)
        _, _, c = x.size()
        x = x.permute(0, 2, 1)
        x = x.view(n, c, w, h)

        for i in range(10):
            x = self.interConvList[i](x)
        
        x = self.OutConv(x)
        return x
