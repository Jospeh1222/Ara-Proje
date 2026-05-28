#Mel PNG segmentlerini okuyup tensore cevirir

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import config as C


class MakamMelPNGDataset(Dataset):

    def __init__(self, fold: int, split: str = "train", augment: bool = False):

        with open(C.FOLDS_MEL_FILE, encoding="utf-8") as f:
            data = json.load(f)

        self.items        = data["folds"][fold][split]
        self.label_to_idx = data["label_to_idx"]
        self.num_classes  = len(self.label_to_idx)
        self.augment      = augment

        #Normalizasyon istatistikleri (TRAIN setinden hesaplanmis)
        with open(C.stats_mel_path(fold)) as f:
            stats = json.load(f)

        self.mean = np.array(stats["mean"], dtype=np.float32)
        self.std  = np.array(stats["std"],  dtype=np.float32)


    def __len__(self):
        return len(self.items)


    #SpecAugment: rastgele frekans bandi ve zaman araligi sifirla
    def _spec_augment(self, x: np.ndarray) -> np.ndarray:
        _, H, W = x.shape

        for _ in range(C.FREQ_MASK_N):
            f = np.random.randint(0, C.FREQ_MASK_PARAM + 1)
            if f == 0: continue
            f0 = np.random.randint(0, max(1, H - f))
            x[:, f0:f0 + f, :] = 0.0

        for _ in range(C.TIME_MASK_N):
            t = np.random.randint(0, C.TIME_MASK_PARAM + 1)
            if t == 0: continue
            t0 = np.random.randint(0, max(1, W - t))
            x[:, :, t0:t0 + t] = 0.0

        return x


    def __getitem__(self, idx):
        item = self.items[idx]

        #PNG oku -> (H, W, 3) -> float32 -> 0-1
        img = np.array(Image.open(item["path"]).convert("RGB"),
                       dtype=np.float32) / 255.0

        #(H, W, 3) -> (3, H, W)
        img = img.transpose(2, 0, 1)

        #Z-score normalizasyonu per kanal
        for c in range(3):
            img[c] = (img[c] - self.mean[c]) / (self.std[c] + 1e-6)

        if self.augment:
            img = self._spec_augment(img)

        return torch.from_numpy(img), int(item["label_idx"]), item["song_id"]