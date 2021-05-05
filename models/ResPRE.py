import torch.nn as nn
import torch.nn.functional as F

class net_pre(nn.Module):
    def __init__(self, inplanes, planes, droprate):
        super(net_pre, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.droprate = droprate

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
    def __init__(self, args = {}):
        inplanes = args.get('inplanes', 64)
        in_channel = args.get('input_channel', 441)
        out_channel = args.get('output_channel', 10)
        droprate = args.get('droprate', 0.2)
        super(ResPRE, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, inplanes, kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        layers = []
        for i in range(0, 22):
            layers.append(net_pre(inplanes, inplanes, droprate))

        self.layer = nn.Sequential(*layers)

        self.lastlayer = nn.Conv2d(inplanes, 10, 3, padding=1, bias=False)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer(x)
        x = self.lastlayer(x)

        return x

    