#K=5 stratifiye fold uretir, BOLUNMEYI SARKI SEVIYESINDE YAPAR
#Her fold icin train / val / test olmak uzere 3'lu ayrim yapilir

from pathlib import Path
from collections import defaultdict
import json
import random

import config as C


#.npz dosyasinin ismindeki '_segNN' kismini sokerek sarki kimligini bulur
def song_id_from_seg(npz_path: Path) -> str:
    return npz_path.stem.rsplit("_seg", 1)[0]


#Windows'ta uretilen yollarin ters slashlerini Linux uyumlu yap (Colab icin)
def to_posix(p: Path) -> str:
    return p.as_posix()


#Stratified K-fold: her siniftan esit oranda sarkiyi her folda dagit
#Round-robin yontemiyle: ilk sarki fold 0'a, ikinci fold 1'e, ... ve dongu tekrar
def stratified_kfold_song_ids(songs_per_class: dict,
                              n_folds: int, seed: int) -> list:

    rng = random.Random(seed)
    folds = [defaultdict(list) for _ in range(n_folds)]

    for cls, songs in songs_per_class.items():
        shuffled = list(songs)
        rng.shuffle(shuffled) #sinif icindeki siralamayi karistir

        #Round-robin atama, foldlar arasinda dengeli dagilim saglar
        for i, song in enumerate(shuffled):
            folds[i % n_folds][cls].append(song)

    return [dict(f) for f in folds]


#Train sarkilarini sarki seviyesinde train/val olarak boler
#val_ratio: validation'a ayrilacak sarki orani (orn. 0.2 = %20)
#Bolunme yine sarki seviyesinde yapilir, sizinti olmaz
def split_train_val(train_songs_per_class: dict, val_ratio: float,
                    seed: int) -> tuple:

    rng = random.Random(seed + 999)  #farkli seed, fold bolunmesinden bagimsiz
    train_songs = {}
    val_songs   = {}

    for cls, songs in train_songs_per_class.items():
        shuffled = list(songs)
        rng.shuffle(shuffled)

        #En az 1 sarki val'e gitsin, en az 1 sarki train'de kalsin
        n_val = max(1, round(len(shuffled) * val_ratio))
        n_val = min(n_val, len(shuffled) - 1)  #train tamamen bosalmasin

        val_songs[cls]   = shuffled[:n_val]
        train_songs[cls] = shuffled[n_val:]

    return train_songs, val_songs


#Ana akis: tum segmentleri bul, sinif/sarki gruplari yap, foldlara bol, JSON yaz
def main():

    #Sinif -> sarki -> segment yollari seklinde sozluk olustur
    by_class = defaultdict(lambda: defaultdict(list))
    for cls_dir in sorted(p for p in C.CACHE_DIR.iterdir() if p.is_dir()):
        for npz in sorted(cls_dir.glob("*.npz")):
            song_id = song_id_from_seg(npz)
            by_class[cls_dir.name][song_id].append(to_posix(npz))

    if not by_class:
        raise SystemExit(f"Oznitelik bulunamadi: {C.CACHE_DIR}. Once features.py calistir.")

    classes = sorted(by_class.keys())
    label_to_idx = {c: i for i, c in enumerate(classes)}

    songs_per_class = {cls: list(by_class[cls].keys()) for cls in classes}

    #1. Adim: K-fold ile test setlerini belirle (sarki seviyesinde)
    test_assignments = stratified_kfold_song_ids(songs_per_class, C.N_FOLDS, C.SEED)

    folds = []

    print(f"\n{C.N_FOLDS}-fold stratifiye bolunme (sarki seviyesinde)")
    print(f"VAL_RATIO = {C.VAL_RATIO} (train havuzunun %{int(C.VAL_RATIO*100)}'i val'e ayrilir)")
    print(f"{'sinif':<22} {'sarki':>6} | " +
          " | ".join([f"f{k} test" for k in range(C.N_FOLDS)]))
    print("-" * (30 + 11 * C.N_FOLDS))

    for k in range(C.N_FOLDS):
        test_songs_per_cls  = {}
        train_pool_per_cls  = {}

        for cls in classes:
            test_set = set(test_assignments[k].get(cls, []))

            #Bu foldda test'e gitmeyen sarkilar train havuzuna girer
            train_pool = [s for s in by_class[cls].keys() if s not in test_set]
            test_songs_per_cls[cls] = list(test_set)
            train_pool_per_cls[cls] = train_pool

        #2. Adim: Train havuzunu train/val olarak bol (sarki seviyesinde)
        final_train_songs, val_songs = split_train_val(
            train_pool_per_cls, C.VAL_RATIO, seed=C.SEED + k
        )

        train_segs, val_segs, test_segs = [], [], []

        for cls in classes:
            train_set = set(final_train_songs.get(cls, []))
            val_set   = set(val_songs.get(cls, []))
            test_set  = set(test_songs_per_cls.get(cls, []))

            for song, segs in by_class[cls].items():
                if song in train_set:
                    target = train_segs
                elif song in val_set:
                    target = val_segs
                elif song in test_set:
                    target = test_segs
                else:
                    #Hicbir kumeye atanamadiysa train'e ekle (edge case)
                    target = train_segs

                for seg_path in segs:
                    target.append({
                        "path":      seg_path,
                        "label":     cls,
                        "label_idx": label_to_idx[cls],
                        "song_id":   song,
                    })

        folds.append({"train": train_segs, "val": val_segs, "test": test_segs})

    #Bolunme tablosunu yazdir
    for cls in classes:
        n_total = len(songs_per_class[cls])
        per_fold = [len(test_assignments[k].get(cls, [])) for k in range(C.N_FOLDS)]
        row = f"{cls:<22} {n_total:>6} | " + " | ".join(f"{c:>7}" for c in per_fold)
        print(row)

    print("-" * (30 + 11 * C.N_FOLDS))
    for k, fold in enumerate(folds):
        print(f"Fold {k}: train={len(fold['train']):>6} seg  "
              f"val={len(fold['val']):>6} seg  "
              f"test={len(fold['test']):>6} seg")

    #Sonucu JSON olarak yaz
    out = {
        "label_to_idx": label_to_idx,
        "n_folds": C.N_FOLDS,
        "val_ratio": C.VAL_RATIO,
        "folds": folds,
    }
    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.FOLDS_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nKaydedildi -> {C.FOLDS_FILE}")


if __name__ == "__main__":
    main()