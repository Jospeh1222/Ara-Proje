import torch
import torch.nn as nn
import torch.nn.functional as F
import config_cnn as C


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(C.CONV_DROPOUT)

    def forward(self, x):
        return self.drop(self.pool(F.relu(self.bn(self.conv(x)))))


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h):                      # h: (B, T, hidden_dim)
        scores  = self.v(torch.tanh(self.W(h))).squeeze(-1)   # (B, T)
        weights = F.softmax(scores, dim=1)                    # (B, T)
        context = (weights.unsqueeze(-1) * h).sum(dim=1)      # (B, hidden_dim)
        return context, weights


class MakamCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()

        self.cnn = nn.Sequential(
            ConvBlock(3,    64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 512),
        )

        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=192,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attn = AdditiveAttention(384, 128)   # 2*192 = 384

        self.classifier = nn.Sequential(
            nn.Dropout(C.DROPOUT),
            nn.Linear(384, n_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.freq_pool(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.gru(x)
        ctx, _ = self.attn(x)        # mean yerine attention
        return self.classifier(ctx)
