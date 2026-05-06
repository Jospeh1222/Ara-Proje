#Her metrik için: ortalama, std, min, max
#Per-class F1 ortalaması ve standart sapması


from pathlib import Path
import json
import numpy as np

import config as C


#Bir seviye (segment veya şarkı) için tüm foldları birleştir
def aggregate(level: str, results: list[dict], idx_to_label: dict[int, str]):
    #level: 'segment_level' veya 'song_level'

    metric_keys = ["accuracy", "precision", "recall", "f1", "kappa"]

    #Her metrik için fold başına değerleri topla
    rows = {k: [] for k in metric_keys}
    per_class_f1 = []

    for r in results:
        m = r[level]
        for k in metric_keys:
            rows[k].append(m[k])
        per_class_f1.append(m["per_class_f1"])

    #Tablo başlığı
    print(f"\n{'=' * 60}")
    print(f"  {level.upper().replace('_', ' ')} - {len(results)} fold ortalamasi")
    print(f"{'=' * 60}")
    print(f"  {'metrik':<10} {'ort.':>8} {'std':>8} {'min':>8} {'max':>8}")
    print(f"  {'-' * 46}")

    #Her metrik için istatistikler
    for k in metric_keys:
        arr = np.array(rows[k])
        print(f"  {k:<10} {arr.mean():>8.4f} {arr.std():>8.4f} "
              f"{arr.min():>8.4f} {arr.max():>8.4f}")

    #Sınıf bazında F1 ortalamaları (foldlar arası)
    pcf = np.array(per_class_f1) #boyut: (fold sayisi, sinif sayisi)

    print(f"\n  Sinif bazinda F1 (fold ortalamasi +/- std):")
    print(f"  {'sinif':<22} {'ort.':>8} {'std':>8}")
    print(f"  {'-' * 40}")
    for i in sorted(idx_to_label.keys()):
        col = pcf[:, i]
        print(f"  {idx_to_label[i]:<22} {col.mean():>8.4f} {col.std():>8.4f}")

    #JSON için özeti hazırla
    summary = {
        "n_folds": len(results),
        "level":   level,
    }
    for k in metric_keys:
        arr = np.array(rows[k])
        summary[f"{k}_mean"] = float(arr.mean())
        summary[f"{k}_std"]  = float(arr.std())
        summary[f"{k}_per_fold"] = [float(v) for v in arr]
    summary["per_class_f1_mean"] = pcf.mean(axis=0).tolist()
    summary["per_class_f1_std"]  = pcf.std(axis=0).tolist()

    return summary


def main():

    #Tüm fold sonuç dosyalarını yükle
    results = []
    missing = [] #eksik foldlar

    for k in range(C.N_FOLDS):
        p = C.fold_results_path(k)
        if not p.exists():
            missing.append(k)
            continue
        with open(p, encoding="utf-8") as f:
            results.append(json.load(f))

    #Eksik fold varsa uyar
    if missing:
        print(f"Eksik fold(lar): {missing}")
        print(f"Calistir: python train_fold.py {missing[0]}")
        return

    if not results:
        raise SystemExit("Fold sonucu bulunamadi. Once train_fold.py calistir.")

    #Sınıf etiketleri (ilk foldun bilgisinden al, hepsi aynı olmalı)
    idx_to_label = {int(v): k for k, v in results[0]["label_to_idx"].items()}

    #İki seviye için ayrı ayrı özet
    seg_summary  = aggregate("segment_level", results, idx_to_label)
    song_summary = aggregate("song_level",    results, idx_to_label)

    #Özeti JSON olarak kaydet (rapor için)
    out = {
        "segment_level": seg_summary,
        "song_level":    song_summary,
    }
    out_path = C.RESULTS_DIR / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nOzet kaydedildi -> {out_path}")


    #Rapor için kopyala-yapıştır kısa özet
    #*****************************************************************
    print(f"\n{'=' * 60}")
    print("  RAPOR ICIN OZET (sarki seviyesi - asil onemli olan)")
    print(f"{'=' * 60}")
    s = song_summary
    print(f"  {C.N_FOLDS}-fold capraz dogrulama, {len(idx_to_label)} sinif")
    print(f"  Accuracy : {s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f}")
    print(f"  Precision: {s['precision_mean']:.4f} +/- {s['precision_std']:.4f}  (macro)")
    print(f"  Recall   : {s['recall_mean']:.4f} +/- {s['recall_std']:.4f}  (macro)")
    print(f"  F1       : {s['f1_mean']:.4f} +/- {s['f1_std']:.4f}  (macro)")
    print(f"  Kappa    : {s['kappa_mean']:.4f} +/- {s['kappa_std']:.4f}")
    #*****************************************************************


if __name__ == "__main__":
    main()