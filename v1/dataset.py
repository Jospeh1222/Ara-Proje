#Bir foldun train veya test segmentlerini tensörlere çevirir
#Eğitimde SpecAugment uygular, testte uygulamaz
#Dönüş: (tensor, sınıf indeksi, şarkı kimliği) üçlüsü

import json
import numpy as np
import torch
from torch.utils.data import Dataset

import config as C


class MakamFoldDataset(Dataset):

    def __init__(self, fold: int, split: str = "train", augment: bool = False):
        #fold: hangi fold (0..N_FOLDS-1)
        #split: "train" veya "test"
        #augment: SpecAugment uygulansın mı (sadece train için True)

        with open(C.FOLDS_FILE, encoding="utf-8") as f:
            data = json.load(f)

        self.items = data["folds"][fold][split] #segment listesi
        self.label_to_idx = data["label_to_idx"]
        self.num_classes = len(self.label_to_idx)
        self.augment = augment

        #Bu foldun normalizasyon istatistiklerini yükle
        with open(C.stats_path(fold)) as f:
            stats = json.load(f)

        #(F, 1) şeklinde sütun vektörleri, (F, T) tensoruyle yayın işlemi yapılabilsin diye
        def col(name):
            return np.array(stats[name], dtype=np.float32)[:, None]

        self.mel_mean    = col("mel_mean");    self.mel_std    = col("mel_std")
        self.mfcc_mean   = col("mfcc_mean");   self.mfcc_std   = col("mfcc_std")
        self.chroma_mean = col("chroma_mean"); self.chroma_std = col("chroma_std")


    def __len__(self):
        return len(self.items)


    #SpecAugment: rastgele frekans bandı ve zaman aralığı sıfırla
    #Modelin tek bir özelliğe takılı kalmasını engeller, overfitting'i azaltır
    def _spec_augment(self, x: np.ndarray) -> np.ndarray:

        F, T = x.shape

        #Frekans maskeleri: belli aralıktaki bandları sıfırla
        for _ in range(C.FREQ_MASK_N):
            f = np.random.randint(0, C.FREQ_MASK_PARAM + 1)
            if f == 0: continue
            f0 = np.random.randint(0, max(1, F - f))
            x[f0:f0 + f, :] = 0.0

        #Zaman maskeleri: belli zaman aralığını sıfırla
        for _ in range(C.TIME_MASK_N):
            t = np.random.randint(0, C.TIME_MASK_PARAM + 1)
            if t == 0: continue
            t0 = np.random.randint(0, max(1, T - t))
            x[:, t0:t0 + t] = 0.0

        return x


    def __getitem__(self, idx):

        item = self.items[idx]
        z = np.load(item["path"]) #disk'ten .npz oku

        #Üç matrisin zaman ekseni 1 frame fark edebilir, eşitle
        T = min(z["mel"].shape[1], z["mfcc"].shape[1], z["chroma"].shape[1])

        #Z-score normalizasyonu (her bin için ayrı), modelin daha hızlı öğrenmesini sağlar
        mel    = (z["mel"][:, :T]    - self.mel_mean)    / (self.mel_std    + 1e-6)
        mfcc   = (z["mfcc"][:, :T]   - self.mfcc_mean)   / (self.mfcc_std   + 1e-6)
        chroma = (z["chroma"][:, :T] - self.chroma_mean) / (self.chroma_std + 1e-6)

        #Üç özniteliği frekans ekseninde birleştir (128 + 20 + 36 = 184 bandlı tek matris)
        x = np.concatenate([mel, mfcc, chroma], axis=0).astype(np.float32)

        #Sadece eğitimde augmentasyon uygula
        if self.augment:
            x = self._spec_augment(x)

        x = x[None, :, :] #CNN için kanal ekseni ekle: (1, F, T)

        return torch.from_numpy(x), int(item["label_idx"]), item["song_id"]