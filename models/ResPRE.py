import torch.nn as nn
import torch.nn.functional as F

class net_pre(nn.Module):
    # expansion = 1

    def __init__(self, inplanes, planes):
        super(net_pre, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.droprate = 0.2

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if(self.droprate>0):
            out = F.dropout(out, p=self.droprate, training=self.training)

        out += residual
        out = self.relu(out)

        return out

class ResPRE(nn.Module):
    # def __init__(self, block=net_pre, args = {}):
    def __init__(self, args = {}):
        self.inplanes = 64
        in_channel = args.get('input_channel', 441)
        out_channel = args.get('output_channel', 10)
        super(ResPRE, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        layers = []
        for i in range(0, 22):
            layers.append(net_pre(self.inplanes, self.inplanes))

        self.layer = nn.Sequential(*layers)

        self.lastlayer=nn.Conv2d(self.inplanes, 10, 3, padding=1, bias=False)

        # self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer(x)
        x = self.lastlayer(x)
        # x = self.sig(x)

        return x


    # TODO: Complete the ResPRE model by 2021/5/5, task assigned to Peishen Yan.
    