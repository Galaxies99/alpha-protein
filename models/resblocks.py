import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, droprate = 0.2, attention = None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 3, padding = 1, bias = False)
        self.in1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, padding = 1, bias = False)
        self.in2 = nn.InstanceNorm2d(planes)
        self.attention = attention
        self.droprate = droprate

    def forward(self, x):
        skip = x
        out = self.conv1(x)
        out = self.in1(out)
        if self.training and self.droprate > 0:
            out = F.dropout(out, p = self.droprate, training = self.training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)
        if self.training and self.droprate > 0:
            out = F.dropout(out, p = self.droprate, training = self.training)
        
        if self.attention is not None:
            out = self.attention(out)
        
        out += skip
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, droprate = 0.2, attention = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size = 1, stride = 1, bias = False)
        self.in1 = nn.InstanceNorm2d(planes)

        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.in2 = nn.InstanceNorm2d(planes)

        self.conv3 = nn.Conv2d(inplanes, planes, kernel_size = 1, stride = 1, bias = False)
        self.in3 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)

        self.skip_conv = nn.Conv2d(inplanes, planes, kernel_size = 1, stride = 1, bias = False)
        self.skip_in = nn.InstanceNorm2d(planes)

        self.attention = attention
        self.droprate = droprate

    def forward(self, x):
        skip = x
        out = self.conv1(x)
        out = self.in1(out) 
        if self.training and self.droprate > 0:
            out = F.dropout(out, p = self.droprate, training = self.training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)
        if self.training and self.droprate > 0:
            out = F.dropout(out, p = self.droprate, training = self.training)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.in3(out)
        if self.training and self.droprate > 0:
            out = F.dropout(out, p = self.droprate, training = self.training)
        
        if self.attention is not None:
            out = self.attention(out)

        skip = self.skip_conv(skip)
        skip = self.skip_in(skip)
        if self.training and self.droprate > 0:
            skip = F.dropout(skip, p = self.droprate, training = self.training)

        out += skip
        out = self.relu(out)
        return out


class DilatedBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size = 3, dilation = 1):
        super(DilatedBasicBlock, self).__init__()
        padding_size = kernel_size + (kernel_size - 1) * (dilation - 1) - 1
        assert padding_size % 2 == 0
        padding_size = int(padding_size / 2)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = kernel_size, stride = 1, padding = padding_size, dilation = dilation)
        self.in1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = kernel_size, stride = 1, padding = padding_size, dilation = dilation)
        self.in2 = nn.InstanceNorm2d(planes)
        if inplanes != planes:
            self.conv3 = nn.Conv2d(inplanes, planes, kernel_size = 1, stride = 1)
            self.in3 = nn.InstanceNorm2d(planes)
        else:
            self.conv3 = None
            self.in3 = None
   
    def forward(self, x):
        if self.conv3 is not None:
            skip = self.in3(self.conv3(x))
        else:
            skip = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += skip
        out = self.relu(out)
        return out