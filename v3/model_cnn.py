#5 bloklu Custom CNN - makam siniflandirmasi icin

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


#Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool_size=(2, 2)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool_size)
        self.drop = nn.Dropout2d(C.CONV_DROPOUT)

    def forward(self, x):
        return self.drop(self.pool(F.relu(self.bn(self.conv(x)))))


#5 bloklu CNN, adaptive pooling sayesinde girdi boyutuna esnek
class MakamCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()

        #5 blok: 3 -> 32 -> 64 -> 128 -> 256 -> 256
        self.cnn = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(C.DROPOUT),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        #x: (B, 3, H, W)
        x = self.cnn(x)
        x = self.global_pool(x)
        return self.classifier(x)