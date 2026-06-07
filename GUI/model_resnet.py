#Pretrained ResNet18 + makam siniflandirici head
#VERSIYON C (FINAL): Sadece conv1+bn1+layer1+layer2 dondurulur
#layer3 ve layer4 fine-tune edilir (layer3 dondurma overfit fix versiyonundaydi, kaldirildi)

import torch
import torch.nn as nn
from torchvision import models

import config_resnet as C


class MakamResNet(nn.Module):

    def __init__(self, n_classes: int, freeze_early: bool = True):
        super().__init__()

        self.backbone = models.resnet18(weights="DEFAULT")

        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Dropout(C.RESNET_DROPOUT),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(C.RESNET_DROPOUT),
            nn.Linear(256, n_classes),
        )

        if freeze_early:
            self._freeze_early_layers()


    def _freeze_early_layers(self):
        #DONDUR: conv1, bn1, layer1, layer2 (universal edge/texture detectors)
        #EGIT:   layer3, layer4, fc (high-level features + yeni classifier head)
        for name, param in self.backbone.named_parameters():
            if any(prefix in name.split('.')[0:1] for prefix in
                   ["conv1", "bn1", "layer1", "layer2"]):
                param.requires_grad = False
            else:
                param.requires_grad = True


    def forward(self, x):
        return self.backbone(x)


    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())
