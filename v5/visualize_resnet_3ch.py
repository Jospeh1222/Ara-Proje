#K-fold ResNet sonuc

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C


def load_all_folds():
    results = []
    for k in range(C.N_FOLDS):
        p = C.resnet_fold_results_path(k)
        if not p.exists():
            raise SystemExit(f"Eksik fold sonucu: {p}")
        with open(p, encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def get_class_info(results):
    label_to_idx = results[0]["label_to_idx"]
    idx_to_label = {int(v): k for k, v in label_to_idx.items()}
    classes = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    return classes


def plot_confusion_matrix(results, classes, out_path, level="song_level"):
    cm_total = np.zeros((len(classes), len(classes)), dtype=int)
    for r in results:
        cm = np.array(r[level]["confusion_matrix"])
        cm_total += cm

    cm_norm = cm_total.astype(float) / cm_total.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            if val >= 0.05:
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center", color=color, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Tahmin Orani", rotation=270, labelpad=20)

    title_tr = "Sarki Seviyesi" if level == "song_level" else "Segment Seviyesi"
    ax.set_title(f"Karisiklik Matrisi - {title_tr} (ResNet18 3-ch, 5 fold)",
                 fontsize=13, pad=15)
    ax.set_xlabel("Tahmin Edilen Makam", fontsize=11)
    ax.set_ylabel("Gercek Makam", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi -> {out_path}")


def plot_per_class_f1(results, classes, out_path, level="song_level"):
    n_classes = len(classes)
    f1_per_fold = np.array([r[level]["per_class_f1"] for r in results])
    means = f1_per_fold.mean(axis=0)
    stds  = f1_per_fold.std(axis=0)

    sort_idx = np.argsort(means)[::-1]
    classes_sorted = [classes[i] for i in sort_idx]
    means_sorted = means[sort_idx]
    stds_sorted = stds[sort_idx]

    fig, ax = plt.subplots(figsize=(14, 7))
    x_pos = np.arange(n_classes)

    colors = []
    for v in means_sorted:
        if v >= 0.60:
            colors.append("#2ca02c")
        elif v >= 0.40:
            colors.append("#ff7f0e")
        else:
            colors.append("#d62728")

    bars = ax.bar(x_pos, means_sorted, yerr=stds_sorted, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.5,
                  error_kw={"linewidth": 1.2, "ecolor": "#333333"})

    for bar, mean, std in zip(bars, means_sorted, stds_sorted):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + std + 0.015,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes_sorted, rotation=45, ha="right")
    ax.set_ylabel("F1 Skoru (5 fold ortalamasi)", fontsize=11)
    ax.set_ylim(0, 1.0)

    title_tr = "Sarki Seviyesi" if level == "song_level" else "Segment Seviyesi"
    ax.set_title(f"Sinif Basina F1 - {title_tr} (ResNet18 3-ch)",
                 fontsize=13, pad=15)

    ax.axhline(y=0.60, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.40, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    overall_f1 = float(np.mean([r[level]["f1"] for r in results]))
    ax.axhline(y=overall_f1, color="blue", linestyle="-", linewidth=1.5,
               alpha=0.7, label=f"Genel F1 (weighted): {overall_f1:.3f}")
    ax.legend(loc="upper right", fontsize=10)

    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi -> {out_path}")


def plot_fold_consistency(results, out_path, level="song_level"):
    metric_keys = ["accuracy", "precision", "recall", "f1", "kappa"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "Kappa"]
    per_fold = {k: [r[level][k] for r in results] for k in metric_keys}

    fig, ax = plt.subplots(figsize=(12, 6))
    x_folds = np.arange(1, C.N_FOLDS + 1)
    markers = ["o", "s", "^", "D", "v"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        values = per_fold[key]
        ax.plot(x_folds, values, marker=markers[i], color=colors[i],
                linewidth=2, markersize=8, label=label)
        mean = np.mean(values)
        ax.axhline(y=mean, color=colors[i], linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xticks(x_folds)
    ax.set_xticklabels([f"Fold {k}" for k in range(C.N_FOLDS)])
    ax.set_ylabel("Metrik Degeri", fontsize=11)
    ax.set_ylim(0, 1.0)

    title_tr = "Sarki Seviyesi" if level == "song_level" else "Segment Seviyesi"
    ax.set_title(f"Foldlar Arasi Tutarlilik - {title_tr} (ResNet18 3-ch)",
                 fontsize=13, pad=15)
    ax.legend(loc="upper right", fontsize=10, ncol=5)
    ax.grid(linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi -> {out_path}")


def main():
    print("Sonuc dosyalari yukleniyor...")
    results = load_all_folds()
    classes = get_class_info(results)
    print(f"  {len(results)} fold yuklendi, {len(classes)} sinif")

    fig_dir = C.RESNET_RESULTS_DIR / "figures_resnet_3ch"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\nFigur uretiliyor...")

    plot_confusion_matrix(results, classes,
                         fig_dir / "confusion_matrix_song.png", "song_level")
    plot_per_class_f1(results, classes,
                     fig_dir / "per_class_f1_song.png", "song_level")
    plot_fold_consistency(results,
                         fig_dir / "fold_consistency_song.png", "song_level")
    plot_confusion_matrix(results, classes,
                         fig_dir / "confusion_matrix_segment.png", "segment_level")
    plot_per_class_f1(results, classes,
                     fig_dir / "per_class_f1_segment.png", "segment_level")

    print(f"\nTum figurler kaydedildi -> {fig_dir}")


if __name__ == "__main__":
    main()
