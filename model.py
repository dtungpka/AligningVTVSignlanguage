import torch
import torch.nn as nn


class ModifiedLightweight3DCNN(nn.Module):
    def __init__(self,in_channels=30):
        super(ModifiedLightweight3DCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=1, padding=1)
        self.res2 = self._make_layer(block_count=3, in_channels=8, out_channels=8)  # Modified in_channels
        self.res3 = self._make_layer(block_count=4, in_channels=8, out_channels=64)  # Modified in_channels
        self.res4 = self._make_layer(block_count=6, in_channels=64, out_channels=128)
        self.res5 = self._make_layer(block_count=3, in_channels=128, out_channels=256)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1)

    def _make_layer(self, block_count, in_channels, out_channels):
        layers = []
        for _ in range(block_count):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels  # Update in_channels for the next layer
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x