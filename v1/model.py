#CRNN mimarisi: 4 CNN bloku -> BiLSTM -> Attention -> Lineer siniflandirici
#Girdi: (B, 1, F, ~1293) - B=batch, F=toplam frekans bandi, ~1293 zaman frame
#Cikti: (B, 16) - 16 makam icin ham logit degerleri

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


#Bir CNN bloku: Konvolusyon + BatchNorm + ReLU + MaxPool + Dropout
#Spektrogram uzerindeki yerel doku oruntulerini yakalar
class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(C.CONV_DROPOUT)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        return self.drop(x)


#Additive (Bahdanau tarzi) attention mekanizmasi
#Her zaman adimi icin bir onem skoru hesaplar, onemli anlara daha cok agirlik verir
class AdditiveAttention(nn.Module):

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h):
        #h: (B, T, H) - BiLSTM ciktisi
        scores  = self.v(torch.tanh(self.W(h))).squeeze(-1) #(B, T)
        weights = F.softmax(scores, dim=1)                   #(B, T)
        context = (weights.unsqueeze(-1) * h).sum(dim=1)     #(B, H)
        return context, weights


#Ana model: tum parcalari birlestirir
class MakamCRNN(nn.Module):

    def __init__(self, n_classes: int):
        super().__init__()

        #4 CNN bloku: 1 -> 24 -> 48 -> 96 -> 96
        self.cnn = nn.Sequential(
            ConvBlock(1,   24),
            ConvBlock(24,  48),
            ConvBlock(48,  96),
            ConvBlock(96,  96),
        )

        #4 MaxPool sonrasi frekans 16'da bir indirgenir
        n_freq = C.total_freq_bins()
        feat_per_step = 96 * (n_freq // 16)

        #BiLSTM zaman boyutunda oruntuleri ogrenir (seyir, melodi akisi)
        self.lstm = nn.LSTM(
            input_size=feat_per_step,
            hidden_size=C.LSTM_HIDDEN,
            batch_first=True, bidirectional=True, num_layers=1,
        )

        #Attention zaman ekseninde onemli anlari secer
        self.attn = AdditiveAttention(C.LSTM_HIDDEN * 2, C.ATTN_DIM)

        #Dropout + son lineer katman
        self.dropout = nn.Dropout(C.DROPOUT)
        self.fc = nn.Linear(C.LSTM_HIDDEN * 2, n_classes)


    def forward(self, x):
        #x: (B, 1, F, T)
        x = self.cnn(x) #(B, 96, F', T')

        B, Cf, Fp, Tp = x.shape

        #BiLSTM zaman serisi bekler: (B, T', ozellik)
        x = x.permute(0, 3, 1, 2).reshape(B, Tp, Cf * Fp)

        x, _ = self.lstm(x)    #(B, T', 2*H)
        ctx, _ = self.attn(x)  #(B, 2*H)

        return self.fc(self.dropout(ctx)) #(B, n_classes)