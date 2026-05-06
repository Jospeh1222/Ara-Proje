#Her fold için ayrı normalizasyon istatistikleri (mean, std) hesaplar
#Sadece o foldun TRAIN setini kullanır, test verisi sızmasın diye

#python compute_stats.py 0  (sadece fold 0 için)
#python compute_stats.py all   (tüm foldlar için)

import sys
import json
import numpy as np
from tqdm import tqdm

import config as C


#Tek bir fold için istatistikleri hesapla ve JSON'a yaz
def compute_for_fold(fold: int):

    with open(C.FOLDS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    train_items = data["folds"][fold]["train"] #sadece bu foldun train setini al

    #Her öznitelik için toplam ve kareler toplamı (online varyans hesabı için)
    sums   = {"mel": None, "mfcc": None, "chroma": None}
    sumsqs = {"mel": None, "mfcc": None, "chroma": None}
    n_frames = 0 #toplam zaman frame sayısı

    #Train setindeki her segment için
    for item in tqdm(train_items, desc=f"Fold {fold}"):
        z = np.load(item["path"])
        T = None

        #Her öznitelik tipi için (mel, mfcc, chroma)
        for key in sums:
            arr = z[key].astype(np.float64) #hassasiyet için float64

            #İlk segmentte bin sayısına göre dizi başlat
            if sums[key] is None:
                sums[key]   = np.zeros(arr.shape[0])
                sumsqs[key] = np.zeros(arr.shape[0])

            #Bu segmentin katkısını topla (her bin için ayrı)
            sums[key]   += arr.sum(axis=1)
            sumsqs[key] += (arr ** 2).sum(axis=1)
            T = arr.shape[1] if T is None else min(T, arr.shape[1])

        n_frames += T

    #Mean ve std hesapla, JSON için listele
    out = {"n_frames": int(n_frames)}
    for key in sums:
        mean = sums[key] / n_frames
        #Negatif varyansa karşı küçük epsilon ekle (sayısal kararlılık)
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
    if len(sys.argv) < 2:
        print("Kullanim: python compute_stats.py <fold_no|all>")
        sys.exit(1)

    arg = sys.argv[1]
    if arg == "all":
        #Tüm foldlar için sırayla hesapla
        for k in range(C.N_FOLDS):
            print(f"\n=== Fold {k} ===")
            compute_for_fold(k)
    else:
        compute_for_fold(int(arg))


if __name__ == "__main__":
    main()