#K=5 stratifiye fold üretir, BÖLÜNMEYİ ŞARKI SEVİYESİNDE YAPAR
#Aynı şarkının segmentleri asla farklı foldlara dağılmaz, sızıntı olmaz
#Her fold her sınıftan orantılı şarkı içerir

from pathlib import Path
from collections import defaultdict
import json
import random

import config as C


#.npz dosyasının ismindeki '_segNN' kısmını ile şarkı kimliğini bulur
def song_id_from_seg(npz_path: Path) -> str:
    return npz_path.stem.rsplit("_seg", 1)[0]


#Windows'ta üretilen yolların ters slashlerini Linux uyumlu yap colab için
def to_posix(p: Path) -> str:
    return p.as_posix()


#Stratified K-fold: her sınıftan eşit oranda şarkıyı her folda dağıt
#Round-robin yöntemiyle: ilk şarkı fold 0'a, ikinci fold 1'e, ... ve döngü tekrar
def stratified_kfold_song_ids(songs_per_class: dict[str, list[str]],
                              n_folds: int, seed: int) -> list[dict[str, list[str]]]:

    rng = random.Random(seed)
    folds = [defaultdict(list) for _ in range(n_folds)]

    for cls, songs in songs_per_class.items():
        shuffled = list(songs)
        rng.shuffle(shuffled) #sınıf içindeki sıralamayı karıştır

        #Round-robin atama, fold'lar arasında dengeli dağılım sağlar
        for i, song in enumerate(shuffled):
            folds[i % n_folds][cls].append(song) #i'nci şarkı (i mod n_folds)'inci folda

    return [dict(f) for f in folds]


#Ana akış: tüm segmentleri bul, sınıflara grupla, fold'lara böl, JSON olarak kaydet
def main():

    #Sınıf -> şarkı -> segment yolları şeklinde sözlük oluştur
    by_class = defaultdict(lambda: defaultdict(list))
    for cls_dir in sorted(p for p in C.CACHE_DIR.iterdir() if p.is_dir()):
        for npz in sorted(cls_dir.glob("*.npz")):
            song_id = song_id_from_seg(npz)
            by_class[cls_dir.name][song_id].append(to_posix(npz))


    classes = sorted(by_class.keys())
    label_to_idx = {c: i for i, c in enumerate(classes)} #sınıf adı -> sayısal indeks

    #Her sınıftan şarkı listesi (segment yolu değil, şarkı ID'si)
    songs_per_class = {cls: list(by_class[cls].keys()) for cls in classes}

    #K-fold atamalarını yap (her fold için: hangi şarkı test'te?)
    test_assignments = stratified_kfold_song_ids(songs_per_class, C.N_FOLDS, C.SEED)

    folds = []

    print(f"\n{C.N_FOLDS}-fold stratifiye bolunme (sarki seviyesinde)")
    print(f"{'sinif':<22} {'sarki':>6} | " +
          " | ".join([f"f{k} test" for k in range(C.N_FOLDS)]))
    print("-" * (30 + 11 * C.N_FOLDS))

    #Her fold için train/test segment listelerini oluştur
    for k in range(C.N_FOLDS):
        train_segs, test_segs = [], []

        for cls in classes:
            test_songs = set(test_assignments[k].get(cls, [])) #bu fold için test setindeki şarkılar

            for song, segs in by_class[cls].items():
                #Şarkı test setindeyse segmentleri test'e, değilse train'e
                target = test_segs if song in test_songs else train_segs
                for seg_path in segs:
                    target.append({
                        "path":      seg_path,
                        "label":     cls,
                        "label_idx": label_to_idx[cls],
                        "song_id":   song,
                    })

        folds.append({"train": train_segs, "test": test_segs})

    #Bölünme tablosunu yazdır (kontrol için)
    for cls in classes:
        n_total = len(songs_per_class[cls])
        per_fold = [len(test_assignments[k].get(cls, [])) for k in range(C.N_FOLDS)]
        row = f"{cls:<22} {n_total:>6} | " + " | ".join(f"{c:>7}" for c in per_fold)
        print(row)

    print("-" * (30 + 11 * C.N_FOLDS))
    for k, fold in enumerate(folds):
        print(f"Fold {k}: train segment = {len(fold['train']):>6}, "
              f"test segment = {len(fold['test']):>6}")

    #Sonucu JSON olarak yaz
    out = {
        "label_to_idx": label_to_idx,
        "n_folds": C.N_FOLDS,
        "folds": folds,
    }
    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.FOLDS_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nKaydedildi -> {C.FOLDS_FILE}")


if __name__ == "__main__":
    main()