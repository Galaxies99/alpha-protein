import torch
import torch.nn as nn
import torch.nn.functional as F


def check_arch_config(config):
    if len(config) == 0:
        return False
    last = -1    
    for c in config:
        if len(c) != 5:
            return False
        if last != -1 and last != c[1]:
            return False
        last = c[2]
    return True


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


class DilatedResnet34(nn.Module):
    def __init__(self, args = {}):
        super(DilatedResnet34, self).__init__()
        in_channel, out_channel = args.get('input_channel', 441), args.get('out_channel', 10)
        arch_config = args.get('arch', [])
        if arch_config == []:
            arch_config = [
                [1, 16, 16, 3, 1], 
                [1, 16, 32, 3, 1], 
                [3, 32, 64, 3, 1],
                [4, 64, 96, 3, 1],
                [6, 96, 128, 3, 2],
                [3, 128, 160, 3, 4]
            ]
        
        if check_arch_config(arch_config) == False:
            raise AttributeError('Invalid architecture configuration.')
        
        self.in_conv = nn.Conv2d(in_channel, arch_config[0][1], kernel_size = 7, stride = 1, padding = 3)
        layers = []
        for config in arch_config:
            times, inplanes, planes, kernel_size, dilation = config
            for i in range(times):
                if i == 0:
                    layers.append(DilatedBasicBlock(inplanes, planes, kernel_size, dilation))
                else:
                    layers.append(DilatedBasicBlock(planes, planes, kernel_size, dilation))
        self.layers = nn.Sequential(*layers)
        
        last_layer_channel = arch_config[-1][2]
        self.final = nn.Sequential(
            nn.Conv2d(last_layer_channel, last_layer_channel, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.InstanceNorm2d(last_layer_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(last_layer_channel, last_layer_channel, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.InstanceNorm2d(last_layer_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(last_layer_channel, last_layer_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(last_layer_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(last_layer_channel, last_layer_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(last_layer_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(last_layer_channel, out_channel, kernel_size = 1, stride = 1)
        )
    
    def forward(self, x):
        out = self.in_conv(x)
        out = self.layers(out)
        out = self.final(out)
        return out
