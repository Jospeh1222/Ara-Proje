#Her fold icin ayri normalizasyon istatistikleri (mean, std) hesaplar
#Sadece o foldun TRAIN setini kullanir; val ve test verisi sizmasin diye

#Kullanim:
#python compute_stats.py 0     -> sadece fold 0 icin
#python compute_stats.py all   -> tum foldlar icin
#python compute_stats.py       -> argumansiz da tum foldlar

import sys
import json
import numpy as np
from tqdm import tqdm

import config as C


#Tek bir fold icin istatistikleri hesapla ve JSON'a yaz
def compute_for_fold(fold: int):

    if not C.FOLDS_FILE.exists():
        raise FileNotFoundError(f"folds.json bulunamadi: {C.FOLDS_FILE}. "
                                f"Once make_kfolds.py calistir.")

    with open(C.FOLDS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    #SADECE train kullanilir; val ve test istatistiklere katki saglamaz
    train_items = data["folds"][fold]["train"]
    n_val  = len(data["folds"][fold]["val"])
    n_test = len(data["folds"][fold]["test"])

    print(f"  train={len(train_items)} seg  val={n_val} seg  test={n_test} seg")
    print(f"  Istatistikler yalnizca TRAIN setinden hesaplaniyor")

    #Her oznitelik icin toplam ve kareler toplami (online varyans hesabi)
    sums   = {"mel": None, "mfcc": None, "chroma": None}
    sumsqs = {"mel": None, "mfcc": None, "chroma": None}
    n_frames = 0

    for item in tqdm(train_items, desc=f"Fold {fold}"):
        z = np.load(item["path"])
        T = None

        for key in sums:
            arr = z[key].astype(np.float64)

            if sums[key] is None:
                sums[key]   = np.zeros(arr.shape[0])
                sumsqs[key] = np.zeros(arr.shape[0])

            sums[key]   += arr.sum(axis=1)
            sumsqs[key] += (arr ** 2).sum(axis=1)
            T = arr.shape[1] if T is None else min(T, arr.shape[1])

        n_frames += T

    #Mean ve std hesapla
    out = {"n_frames": int(n_frames)}
    for key in sums:
        mean = sums[key] / n_frames
        std  = np.sqrt(np.maximum(sumsqs[key] / n_frames - mean ** 2, 1e-12))

        out[f"{key}_mean"] = mean.tolist()
        out[f"{key}_std"]  = std.tolist()

        print(f"  {key:>6} : mean[{mean.min():.2f}, {mean.max():.2f}]  "
              f"std[{std.min():.2f}, {std.max():.2f}]  bins={len(mean)}")

    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.stats_path(fold), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Kaydedildi -> {C.stats_path(fold)}")


def main():
    #Arguman yoksa veya "all" ise tum foldlar
    arg = sys.argv[1] if len(sys.argv) >= 2 else "all"

    if arg == "all":
        for k in range(C.N_FOLDS):
            print(f"\n=== Fold {k} ===")
            compute_for_fold(k)
    else:
        compute_for_fold(int(arg))


if __name__ == "__main__":
    main()