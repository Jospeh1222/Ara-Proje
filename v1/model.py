#CRNN mimarisi: 4 CNN bloku -> BiLSTM -> Attention -> Lineer sınıflandırıcı
#Girdi: (B, 1, 184, ~1293) yani 1 kanallı, 184 frekans bandı, ~1293 zaman frame
#Çıktı: (B, 16) yani 16 makam için ham logit değerleri

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


#Bir CNN bloğu: Konvolüsyon + BatchNorm + ReLU + MaxPool + Dropout
#Spektrogram üzerindeki yerel doku örüntülerini yakalar
class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1) #3x3 konvolüsyon
        self.bn   = nn.BatchNorm2d(out_ch) #eğitimi stabilize eder
        self.pool = nn.MaxPool2d(2, 2)     #boyut yarıya iner (frekans ve zamanda)
        self.drop = nn.Dropout2d(C.CONV_DROPOUT) #aşırı öğrenmeyi engeller

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x))) #Conv + BN + aktivasyon
        x = self.pool(x)                   #boyutu küçült
        return self.drop(x)


#Additive (Bahdanau tarzı) attention mekanizması
#Her zaman adımı için bir önem skoru hesaplar, önemli anlara daha çok ağırlık verir
class AdditiveAttention(nn.Module):

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim) #gizli katmanı attention boyutuna projekte et
        self.v = nn.Linear(attn_dim, 1, bias=False) #skalar skor üret


    def forward(self, h):
        #h: (B, T, H) - BiLSTM çıktısı

        scores  = self.v(torch.tanh(self.W(h))).squeeze(-1) #(B, T) - her zaman adımı için skor
        weights = F.softmax(scores, dim=1) #(B, T) - softmax ile olasılığa çevir
        context = (weights.unsqueeze(-1) * h).sum(dim=1) #(B, H) - ağırlıklı toplam

        return context, weights


#Ana model: tüm parçaları birleştirir
class MakamCRNN(nn.Module):

    def __init__(self, n_classes: int):
        super().__init__()

        #4 CNN bloku, her biri kanal sayısını artırır ve boyutu yarıya indirir
        #24 -> 48 -> 96 -> 96
        self.cnn = nn.Sequential(
            ConvBlock(1,   24),
            ConvBlock(24,  48),
            ConvBlock(48,  96),
            ConvBlock(96,  96),
        )

        #4 MaxPool sonrası frekans 16'da bir indirgenmiş olur
        #184 // 16 = 11 frekans bandı kalır, 96 kanal var
        #BiLSTM'e girecek özellik sayısı: 96 * 11 = 1056
        n_freq = C.total_freq_bins()
        feat_per_step = 96 * (n_freq // 16)

        #BiLSTM zaman boyutunda örüntüleri öğrenir (seyir, melodi akışı)
        #Çift yönlü olduğu için çıktı boyutu 2 * LSTM_HIDDEN
        self.lstm = nn.LSTM(
            input_size=feat_per_step,
            hidden_size=C.LSTM_HIDDEN,
            batch_first=True, bidirectional=True, num_layers=1,
        )

        #Attention zaman ekseninde önemli anları seçer
        self.attn = AdditiveAttention(C.LSTM_HIDDEN * 2, C.ATTN_DIM)

        #Dropout + son lineer katman ile sınıf logitleri üretilir
        self.dropout = nn.Dropout(C.DROPOUT)
        self.fc = nn.Linear(C.LSTM_HIDDEN * 2, n_classes)


    def forward(self, x):
        #x: (B, 1, F, T) - batch, kanal, frekans, zaman

        x = self.cnn(x) #CNN'den geçir, (B, 96, F', T') şeklinde çıkar

        B, Cf, Fp, Tp = x.shape

        #BiLSTM zaman serisi bekler: (B, T', özellik)
        #permute ile boyutları yeniden sıralayıp, frekans ve kanalı tek vektörde topla
        x = x.permute(0, 3, 1, 2).reshape(B, Tp, Cf * Fp)

        x, _ = self.lstm(x) #(B, T', 2*H)

        ctx, _ = self.attn(x) #attention ile özet vektör çıkar (B, 2*H)

        return self.fc(self.dropout(ctx)) #son sınıflandırma katmanı (B, n_classes)