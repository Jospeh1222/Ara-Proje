%%writefile /content/dataset_cnn.py

# PNG spektrogram segmentlerini okuyup tensöre çevirir
# Eğitimde SpecAugment uygular, testte uygulamaz

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import config as C


class MakamCNNDataset(Dataset):
    def __init__(self, fold: int, split: str = "train", augment: bool = False):
        with open(C.FOLDS_SPEC_FILE, encoding="utf-8") as f:
            data = json.load(f)

        self.items        = data["folds"][fold][split]
        self.label_to_idx = data["label_to_idx"]
        self.num_classes  = len(self.label_to_idx)
        self.augment      = augment

        # Normalleştirme istatistikleri
        with open(C.stats_spec_path(fold)) as f:
            stats = json.load(f)

        self.mean = np.array(stats["mean"], dtype=np.float32)  # (3,)
        self.std  = np.array(stats["std"],  dtype=np.float32)  # (3,)

    def __len__(self):
        return len(self.items)

    def _spec_augment(self, x: np.ndarray) -> np.ndarray:
        # x: (3, H, W)
        _, H, W = x.shape

        # Gaussian gürültü
        if np.random.rand() < 0.5:
            x += np.random.normal(0, 0.05, x.shape).astype(np.float32)

        # Parlaklık
        if np.random.rand() < 0.5:
            x *= np.random.uniform(0.85, 1.15)

        # Frekans maskesi (2 adet, daha geniş)
        for _ in range(2):
            if np.random.rand() < 0.5:
                f = np.random.randint(0, 30)
                if f > 0:
                    f0 = np.random.randint(0, max(1, H - f))
                    x[:, f0:f0 + f, :] = 0.0

        # Zaman maskesi (2 adet, daha geniş)
        for _ in range(2):
            if np.random.rand() < 0.5:
                t = np.random.randint(0, 50)
                if t > 0:
                    t0 = np.random.randint(0, max(1, W - t))
                    x[:, :, t0:t0 + t] = 0.0

        # Cutout — rastgele kare bölge sıfırla
        if np.random.rand() < 0.3:
            ch = np.random.randint(20, 50)
            cw = np.random.randint(20, 50)
            ch0 = np.random.randint(0, max(1, H - ch))
            cw0 = np.random.randint(0, max(1, W - cw))
            x[:, ch0:ch0 + ch, cw0:cw0 + cw] = 0.0

        return x

    def __getitem__(self, idx):
        item = self.items[idx]

        # PNG oku → (H, W, 3) → float32 → 0-1
        img = np.array(Image.open(item["path"]).convert("RGB"),
                       dtype=np.float32) / 255.0

        # (H, W, 3) → (3, H, W)
        img = img.transpose(2, 0, 1)

        # Z-score normalleştirme
        for c in range(3):
            img[c] = (img[c] - self.mean[c]) / (self.std[c] + 1e-6)

        if self.augment:
            img = self._spec_augment(img)

        return torch.from_numpy(img), int(item["label_idx"]), item["song_id"]
