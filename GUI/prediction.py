"""
Makam siniflandirma - model yukleme ve tahmin modulu

V3.2:
  - exp4: 20 sinif (genel), sadece CNN, "featured" (default expanded)
  - exp1,2,3: 16/16/12 sinif, ResNet + CNN, "collapsible" (default kapali)
"""

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


#********************************************************************


def _extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt

    for candidate in ['model_state_dict', 'state_dict', 'model', 'weights']:
        if candidate in ckpt:
            value = ckpt[candidate]
            if isinstance(value, dict):
                if any(isinstance(v, torch.Tensor) for v in value.values()):
                    return value

    if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt

    raise ValueError(
        f"state_dict bulunamadi. Top-level keys: {list(ckpt.keys())[:10]}"
    )


def _normalize_segments(segment_images: list, mean, std) -> torch.Tensor:
    if len(segment_images) == 0:
        raise ValueError("Bos segment listesi")

    arr = np.stack(segment_images, axis=0).astype(np.float32) / 255.0

    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    arr = (arr - mean) / std

    arr = arr.transpose(0, 3, 1, 2)
    return torch.from_numpy(arr.astype(np.float32))


#********************************************************************


class ModelEntry:

    def __init__(self, display_name: str, short_name: str, exp_dir: Path,
                 model_type: str, exp_short: str, exp_label: str,
                 is_featured: bool = False):
        self.display_name = display_name
        self.short_name = short_name
        self.exp_dir = Path(exp_dir)
        self.model_type = model_type
        self.exp_short = exp_short      # "exp1", "exp4" vs
        self.exp_label = exp_label      # "16 Orijinal Sınıf" vs
        self.is_featured = is_featured  # exp4 icin True

        self.is_available = False
        self.is_loaded = False
        self.model = None
        self.label_to_idx = None
        self.idx_to_label = None
        self.n_classes = 0
        self.error_message = None

        self._norm_mean = None
        self._norm_std = None


    def _load_labels(self):
        folds_path = self.exp_dir / "folds_spec.json"
        if not folds_path.exists():
            raise FileNotFoundError(f"folds_spec.json yok: {folds_path}")
        with open(folds_path, encoding='utf-8') as f:
            data = json.load(f)
        self.label_to_idx = data['label_to_idx']
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.n_classes = len(self.label_to_idx)


    def get_class_list(self) -> list:
        """Sinif isim listesi (info popup icin)"""
        if self.label_to_idx is None:
            #Etiketleri yuklemek icin sadece dosyayi oku, modeli yukleme
            try:
                self._load_labels()
            except Exception:
                return []
        return sorted(self.label_to_idx.keys(), key=lambda k: self.label_to_idx[k])


    def _load_resnet(self):
        from model_resnet import MakamResNet

        ckpt_path = self.exp_dir / "resnet.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resnet.pt yok: {ckpt_path}")

        model = MakamResNet(n_classes=self.n_classes, freeze_early=False)

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = _extract_state_dict(ckpt)

        model.load_state_dict(state)
        model.eval()
        self.model = model

        self._norm_mean = IMAGENET_MEAN
        self._norm_std = IMAGENET_STD


    def _load_cnn(self):
        cnn_arch_path = Path("cnn_model.py")
        if not cnn_arch_path.exists():
            raise FileNotFoundError("cnn_model.py yok")

        config_path = Path("config_cnn.py")
        if not config_path.exists():
            raise FileNotFoundError("config_cnn.py yok")

        ckpt_path = self.exp_dir / "cnn.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"cnn.pt yok: {ckpt_path}")

        stats_path = self.exp_dir / "cnn_stats.json"
        if not stats_path.exists():
            alt = list(self.exp_dir.glob("stats_spec_fold*.json"))
            if alt:
                stats_path = alt[0]
            else:
                raise FileNotFoundError(
                    f"cnn_stats.json yok: {stats_path}"
                )

        with open(stats_path, encoding='utf-8') as f:
            stats = json.load(f)

        if 'mean' not in stats or 'std' not in stats:
            raise ValueError(
                f"stats dosyasinda 'mean' ve 'std' yok: {list(stats.keys())}"
            )

        self._norm_mean = stats['mean']
        self._norm_std = stats['std']

        from cnn_model import MakamCNN

        model = MakamCNN(n_classes=self.n_classes)

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = _extract_state_dict(ckpt)

        model.load_state_dict(state)
        model.eval()
        self.model = model


    def load(self) -> bool:
        try:
            self._load_labels()

            if self.model_type == "resnet":
                self._load_resnet()
            elif self.model_type == "cnn":
                self._load_cnn()
            else:
                raise ValueError(f"Bilinmeyen model tipi: {self.model_type}")

            self.is_available = True
            self.is_loaded = True
            return True

        except Exception as e:
            self.is_available = False
            self.is_loaded = False
            self.error_message = str(e)
            return False


    def predict(self, segment_images: list) -> dict:
        if not self.is_available or self.model is None:
            return None

        tensor = _normalize_segments(
            segment_images, self._norm_mean, self._norm_std
        )

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=-1)
            song_probs = probs.mean(dim=0)

        sorted_indices = song_probs.argsort(descending=True)

        top3 = []
        for i in range(min(3, self.n_classes)):
            idx = sorted_indices[i].item()
            top3.append({
                'label': self.idx_to_label[idx],
                'prob': song_probs[idx].item(),
            })

        return {
            'top1': top3[0],
            'top3': top3,
            'all': {self.idx_to_label[i]: song_probs[i].item()
                    for i in range(self.n_classes)},
        }


#********************************************************************


def build_model_registry(experiments_dir) -> list:
    """Model registry. exp4 'featured' (default expanded), digerleri collapsible."""
    experiments_dir = Path(experiments_dir)

    #(klasor, etiket, kisa, model_tipleri, featured)
    experiments = [
        ('exp4_genel20',    '20 Sınıf (Genel)',   'exp4', ['cnn'],             True),
        ('exp1_original16', '16 Orijinal Sınıf',  'exp1', ['resnet', 'cnn'],  False),
        ('exp2_modified16', '16 Modifiye Sınıf',  'exp2', ['resnet', 'cnn'],  False),
        ('exp3_dugah12',    '12 Sınıf (Dugâh)',   'exp3', ['resnet', 'cnn'],  False),
    ]

    registry = []
    for exp_folder, exp_label, exp_short, model_types, is_featured in experiments:
        exp_dir = experiments_dir / exp_folder

        for mt in model_types:
            if mt == 'resnet':
                display = f"ResNet"
                short = f"resnet_{exp_short}"
            else:
                display = f"Custom CNN"
                short = f"cnn_{exp_short}"

            registry.append(ModelEntry(
                display_name=display,
                short_name=short,
                exp_dir=exp_dir,
                model_type=mt,
                exp_short=exp_short,
                exp_label=exp_label,
                is_featured=is_featured,
            ))

    return registry