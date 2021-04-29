import torch.nn as nn
import torch.nn.functional as F


class DeepCov(nn.Module):
    def __init__(self, args={}):
        super(DeepCov, self).__init__()
        in_channel = args.get('input_channel', 441)
        out_channel = args.get('output_channel', 10)
        super(DeepCov, self).__init__()
        self.MaxOutConV = nn.Sequential(
            nn.Conv2d(in_channel, 128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128)
        )

        self.interConV1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.interConV2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.interConV3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.interConV4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.interConV5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.interConV6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.interConV7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.interConV8 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.interConV9 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.interConV10 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.OutConV = nn.Sequential(
            nn.Conv2d(64, out_channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.MaxOutConV(x)

        # feature max_pooling
        n, c, w, h = x.size()
        x = x.view(n, c, w * h).permute(0, 2, 1)
        x = F.max_pool1d(x, 2)
        _, _, c = x.size()
        x = x.permute(0, 2, 1)
        x = x.view(n, c, w, h)

        x = self.interConV1(x)
        x = self.interConV2(x)
        x = self.interConV3(x)
        x = self.interConV4(x)
        x = self.interConV5(x)
        x = self.interConV6(x)
        x = self.interConV7(x)
        x = self.interConV8(x)
        x = self.interConV9(x)
        x = self.interConV10(x)
        x = self.OutConV(x)
        return x
