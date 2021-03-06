import torch.nn as nn
from models.dynamic_pruning.gatedconv import GatedConv


class MyVgg(nn.Module):
    def __init__(self, num_class=10, gated=False, ratio=1):
        super(MyVgg, self).__init__()
        self.features = nn.Sequential(
            GatedConv(3, 64, gated=gated, ratio=ratio),
            GatedConv(64, 64, gated=gated, ratio=ratio),
            nn.MaxPool2d(kernel_size=2, stride=2),
            GatedConv(64, 128, gated=gated, ratio=ratio),
            GatedConv(128, 128, gated=gated, ratio=ratio),
            nn.MaxPool2d(kernel_size=2, stride=2),
            GatedConv(128, 256, gated=gated, ratio=ratio),
            GatedConv(256, 256, gated=gated, ratio=ratio),
            GatedConv(256, 256, gated=gated, ratio=ratio),
            nn.MaxPool2d(kernel_size=2, stride=2),
            GatedConv(256, 512, gated=gated, ratio=ratio),
            GatedConv(512, 512, gated=gated, ratio=ratio),
            GatedConv(512, 512, gated=gated, ratio=ratio),
            nn.MaxPool2d(kernel_size=2, stride=2),
            GatedConv(512, 512, gated=gated, ratio=ratio),
            GatedConv(512, 512, gated=gated, ratio=ratio),
            GatedConv(512, 512, gated=gated, ratio=ratio),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.reshape(output.size()[0], -1)
        output = self.classifier(output)
        return output
