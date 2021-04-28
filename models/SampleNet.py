import torch.nn as nn
import torch.nn.functional as F

# This is just a sample net, for testing purpose.
class SampleNet(nn.Module):
    def __init__(self, args = {}):
        super(SampleNet, self).__init__()
        in_channel = args.get('input_channel', 441)
        out_channel = args.get('output_channel', 10)
        super(SampleNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv_t = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv_t(y1)
        y3 = self.conv2(y2)
        y4 = self.conv3(y3)
        y5 = self.conv4(y4)
        res = self.final(y5)
        return res
