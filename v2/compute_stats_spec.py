%%writefile /content/compute_stats_spec.py

# PNG spektrogram segmentleri için fold bazında normalleştirme istatistikleri
# Sadece o foldun TRAIN setini kullanır

import sys
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image

import config as C


def compute_for_fold(fold: int):

    with open(C.FOLDS_SPEC_FILE, encoding="utf-8") as f:
        data = json.load(f)

    train_items = data["folds"][fold]["train"]
    n_val  = len(data["folds"][fold]["val"])
    n_test = len(data["folds"][fold]["test"])
    print(f"  train={len(train_items)} seg  val={n_val} seg  test={n_test} seg")
    print(f"  İstatistikler yalnızca train setinden hesaplanıyor...")

    # PNG: (H, W, 3) RGB — her kanal için ayrı mean/std
    sum_r = sum_g = sum_b = 0.0
    sumsq_r = sumsq_g = sumsq_b = 0.0
    n_pixels = 0

    for item in tqdm(train_items, desc=f"Fold {fold}"):
        img = np.array(Image.open(item["path"]).convert("RGB"), dtype=np.float64)
        # img: (H, W, 3)
        sum_r  += img[:, :, 0].sum()
        sum_g  += img[:, :, 1].sum()
        sum_b  += img[:, :, 2].sum()
        sumsq_r += (img[:, :, 0] ** 2).sum()
        sumsq_g += (img[:, :, 1] ** 2).sum()
        sumsq_b += (img[:, :, 2] ** 2).sum()
        n_pixels += img.shape[0] * img.shape[1]

    mean_r = sum_r / n_pixels
    mean_g = sum_g / n_pixels
    mean_b = sum_b / n_pixels

    std_r = np.sqrt(max(sumsq_r / n_pixels - mean_r ** 2, 1e-12))
    std_g = np.sqrt(max(sumsq_g / n_pixels - mean_g ** 2, 1e-12))
    std_b = np.sqrt(max(sumsq_b / n_pixels - mean_b ** 2, 1e-12))

    # 0-255 aralığını 0-1'e normalize et
    out = {
        "mean": [mean_r / 255, mean_g / 255, mean_b / 255],
        "std":  [std_r  / 255, std_g  / 255, std_b  / 255],
    }

    print(f"  mean: R={out['mean'][0]:.4f}  G={out['mean'][1]:.4f}  B={out['mean'][2]:.4f}")
    print(f"  std:  R={out['std'][0]:.4f}   G={out['std'][1]:.4f}   B={out['std'][2]:.4f}")

    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.stats_spec_path(fold), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Kaydedildi -> {C.stats_spec_path(fold)}")


def main():
    user_args = [a for a in sys.argv[1:] if a == "all" or a.isdigit()]
    run_mode = user_args[0] if user_args else "all"

    if run_mode == "all":
        for k in range(C.N_FOLDS):
            if C.stats_spec_path(k).exists():
                print(f"\n=== Fold {k} zaten mevcut, atlanıyor ===")
                continue
            print(f"\n=== Fold {k} ===")
            compute_for_fold(k)
    else:
        compute_for_fold(int(run_mode))


if __name__ == "__main__":
    main()
