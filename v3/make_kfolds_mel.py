#Mel PNG segmentleri icin K-fold uretir 

from pathlib import Path
from collections import defaultdict
import json
import random

import config as C


def song_id_from_png(png_path: Path) -> str:
    return png_path.stem.rsplit("_seg", 1)[0]


def to_posix(p: Path) -> str:
    return p.as_posix()


def stratified_kfold_song_ids(songs_per_class: dict,
                              n_folds: int, seed: int) -> list:
    rng = random.Random(seed)
    folds = [defaultdict(list) for _ in range(n_folds)]

    for cls, songs in songs_per_class.items():
        shuffled = list(songs)
        rng.shuffle(shuffled)
        for i, song in enumerate(shuffled):
            folds[i % n_folds][cls].append(song)

    return [dict(f) for f in folds]


def split_train_val(train_songs_per_class: dict, val_ratio: float,
                    seed: int) -> tuple:
    rng = random.Random(seed + 999)
    train_songs = {}
    val_songs   = {}

    for cls, songs in train_songs_per_class.items():
        shuffled = list(songs)
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_ratio))
        n_val = min(n_val, len(shuffled) - 1)
        val_songs[cls]   = shuffled[:n_val]
        train_songs[cls] = shuffled[n_val:]

    return train_songs, val_songs


def main():
    if not C.MEL_PNG_CACHE_DIR.exists():
        raise SystemExit(f"Mel PNG klasoru yok: {C.MEL_PNG_CACHE_DIR}. "
                         f"Once features_mel_png.py calistir.")

    by_class = defaultdict(lambda: defaultdict(list))
    for cls_dir in sorted(p for p in C.MEL_PNG_CACHE_DIR.iterdir() if p.is_dir()):
        for png in sorted(cls_dir.glob("*.png")):
            song_id = song_id_from_png(png)
            by_class[cls_dir.name][song_id].append(to_posix(png))

    if not by_class:
        raise SystemExit(f"PNG bulunamadi: {C.MEL_PNG_CACHE_DIR}")

    classes      = sorted(by_class.keys())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    songs_per_class = {cls: list(by_class[cls].keys()) for cls in classes}

    test_assignments = stratified_kfold_song_ids(songs_per_class, C.N_FOLDS, C.SEED)

    folds = []

    print(f"\n{C.N_FOLDS}-fold stratifiye bolunme - MEL PNG")
    print(f"VAL_RATIO = {C.VAL_RATIO}")
    print(f"{'sinif':<22} {'sarki':>6} | " +
          " | ".join([f"f{k} test" for k in range(C.N_FOLDS)]))
    print("-" * (30 + 11 * C.N_FOLDS))

    for k in range(C.N_FOLDS):
        test_songs_per_cls  = {}
        train_songs_per_cls = {}

        for cls in classes:
            test_set   = set(test_assignments[k].get(cls, []))
            train_pool = [s for s in by_class[cls].keys() if s not in test_set]
            test_songs_per_cls[cls]  = list(test_set)
            train_songs_per_cls[cls] = train_pool

        final_train_songs, val_songs = split_train_val(
            train_songs_per_cls, C.VAL_RATIO, seed=C.SEED + k
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
                    target = train_segs

                for seg_path in segs:
                    target.append({
                        "path":      seg_path,
                        "label":     cls,
                        "label_idx": label_to_idx[cls],
                        "song_id":   song,
                    })

        folds.append({"train": train_segs, "val": val_segs, "test": test_segs})

    for cls in classes:
        n_total  = len(songs_per_class[cls])
        per_fold = [len(test_assignments[k].get(cls, [])) for k in range(C.N_FOLDS)]
        print(f"{cls:<22} {n_total:>6} | " + " | ".join(f"{c:>7}" for c in per_fold))

    print("-" * (30 + 11 * C.N_FOLDS))
    for k, fold in enumerate(folds):
        print(f"Fold {k}: train={len(fold['train']):>6}  "
              f"val={len(fold['val']):>6}  "
              f"test={len(fold['test']):>6} (segment)")

    out = {
        "label_to_idx": label_to_idx,
        "n_folds":   C.N_FOLDS,
        "val_ratio": C.VAL_RATIO,
        "folds":     folds,
    }
    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.FOLDS_MEL_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nKaydedildi -> {C.FOLDS_MEL_FILE}")


if __name__ == "__main__":
    main()