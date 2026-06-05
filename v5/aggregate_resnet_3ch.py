#3-kanal ResNet deneyinin tum fold sonuclarini birlestirir

import json
import numpy as np

import config as C


def aggregate(level: str, results: list, idx_to_label: dict):
    metric_keys = ["accuracy", "precision", "recall", "f1", "kappa"]
    rows = {k: [] for k in metric_keys}
    per_class_f1 = []

    for r in results:
        m = r[level]
        for k in metric_keys:
            rows[k].append(m[k])
        per_class_f1.append(m["per_class_f1"])

    print(f"\n{'=' * 60}")
    print(f"  {level.upper().replace('_', ' ')} - {len(results)} fold ortalamasi")
    print(f"{'=' * 60}")
    print(f"  {'metrik':<10} {'ort.':>8} {'std':>8} {'min':>8} {'max':>8}")
    print(f"  {'-' * 46}")

    for k in metric_keys:
        arr = np.array(rows[k])
        print(f"  {k:<10} {arr.mean():>8.4f} {arr.std():>8.4f} "
              f"{arr.min():>8.4f} {arr.max():>8.4f}")

    pcf = np.array(per_class_f1)
    print(f"\n  Sinif bazinda F1:")
    print(f"  {'sinif':<22} {'ort.':>8} {'std':>8}")
    print(f"  {'-' * 40}")
    for i in sorted(idx_to_label.keys()):
        col = pcf[:, i]
        print(f"  {idx_to_label[i]:<22} {col.mean():>8.4f} {col.std():>8.4f}")

    summary = {"n_folds": len(results), "level": level}
    for k in metric_keys:
        arr = np.array(rows[k])
        summary[f"{k}_mean"] = float(arr.mean())
        summary[f"{k}_std"]  = float(arr.std())
        summary[f"{k}_per_fold"] = [float(v) for v in arr]
    summary["per_class_f1_mean"] = pcf.mean(axis=0).tolist()
    summary["per_class_f1_std"]  = pcf.std(axis=0).tolist()

    return summary


def main():
    results = []
    missing = []
    for k in range(C.N_FOLDS):
        p = C.resnet_fold_results_path(k)
        if not p.exists():
            missing.append(k)
            continue
        with open(p, encoding="utf-8") as f:
            results.append(json.load(f))

    if missing:
        print(f"Eksik fold(lar): {missing}")
        print(f"Calistir: python train_fold_resnet_3ch.py {missing[0]}")
        return

    if not results:
        raise SystemExit("Fold sonucu bulunamadi")

    idx_to_label = {int(v): k for k, v in results[0]["label_to_idx"].items()}

    print(f"\n{'=' * 60}")
    print("  ERKEN DURDURMA OZETI")
    print(f"{'=' * 60}")
    print(f"  {'fold':<6} {'en iyi epoch':>14} {'en iyi val_loss':>18}")
    print(f"  {'-' * 40}")
    for r in results:
        be = r.get("best_epoch", "?")
        bvl = r.get("best_val_loss", float("nan"))
        print(f"  {r['fold']:<6} {be:>14} {bvl:>18.4f}")

    seg_summary  = aggregate("segment_level", results, idx_to_label)
    song_summary = aggregate("song_level",    results, idx_to_label)

    out = {
        "experiment": "resnet18_3ch_mel_mfcc_chroma",
        "segment_level": seg_summary,
        "song_level":    song_summary,
    }
    out_path = C.RESNET_RESULTS_DIR / "resnet_3ch_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nOzet kaydedildi -> {out_path}")

    print(f"\n{'=' * 60}")
    print("  RAPOR ICIN OZET (Pretrained ResNet18 + 3-ch PNG - sarki seviyesi)")
    print(f"{'=' * 60}")
    s = song_summary
    print(f"  {C.N_FOLDS}-fold capraz dogrulama, {len(idx_to_label)} sinif")
    print(f"  Accuracy : {s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f}")
    print(f"  Precision: {s['precision_mean']:.4f} +/- {s['precision_std']:.4f}  (weighted)")
    print(f"  Recall   : {s['recall_mean']:.4f} +/- {s['recall_std']:.4f}  (weighted)")
    print(f"  F1       : {s['f1_mean']:.4f} +/- {s['f1_std']:.4f}  (weighted)")
    print(f"  Kappa    : {s['kappa_mean']:.4f} +/- {s['kappa_std']:.4f}")


if __name__ == "__main__":
    main()
