# model_cnn.py
# 5 bloklu Custom CNN — makam sınıflandırması için
# Girdi: (B, 1, F, T) — mel (184) veya spektrogram (1025)
# Çıktı: (B, n_classes)
%%writefile /content/model_cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import config as C


class ConvBlock(nn.Module):
    """Conv2D → BatchNorm → ReLU → MaxPool → Dropout"""
    def __init__(self, in_ch, out_ch, pool_size=(2, 2)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool_size)
        self.drop = nn.Dropout2d(C.CONV_DROPOUT)

    def forward(self, x):
        return self.drop(self.pool(F.relu(self.bn(self.conv(x)))))


class MakamCNN(nn.Module):
    """
    5 bloklu saf CNN — BiLSTM/Attention yok
    Blok kanalları: 1 → 32 → 64 → 128 → 256 → 256
    Her blok frekans ve zamanı 2x küçültür (toplam 32x)
    """
    def __init__(self, n_classes: int, in_freq: int = None):
        super().__init__()

        # in_freq verilmezse config'den al (mel varsayılan)
        if in_freq is None:
            in_freq = C.total_freq_bins()  # 184

        self.cnn = nn.Sequential(
            ConvBlock(3, 32),   # blok 1
            ConvBlock(32,   64),   # blok 2
            ConvBlock(64,  128),   # blok 3
            ConvBlock(128, 256),   # blok 4
            ConvBlock(256, 256),   # blok 5
        )

        # 5 MaxPool(2,2) sonrası frekans boyutu: in_freq // 32
        # Adaptive pooling ile sabit boyuta getir — T değişken olsa bile çalışır
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(C.DROPOUT),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.cnn(x)              # (B, 256, F', T')
        x = self.global_pool(x)      # (B, 256, 1, 1)
        return self.classifier(x)    # (B, n_classes)


# Parametre sayısını yazdır
if __name__ == "__main__":
    import torch

    # Mel girdi testi
    model_mel = MakamCNN(n_classes=16, in_freq=184)
    x = torch.randn(4, 1, 184, 1293)
    print(f"Mel girdi: {x.shape} → çıktı: {model_mel(x).shape}")
    n = sum(p.numel() for p in model_mel.parameters())
    print(f"Parametre sayısı (mel): {n:,}")

    # Spektrogram girdi testi
    model_spec = MakamCNN(n_classes=16, in_freq=1025)
    x = torch.randn(4, 1, 1025, 1293)
    print(f"\nSpektrogram girdi: {x.shape} → çıktı: {model_spec(x).shape}")
    n = sum(p.numel() for p in model_spec.parameters())
    print(f"Parametre sayısı (spec): {n:,}")
