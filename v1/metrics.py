#Tum metrik hesaplamalari burada toplandi

from collections import defaultdict
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


#Bir tahmin-gercek vektor ciftinden 5 metrigi hesaplar
#Precision/recall/F1 icin
#Weighted: her sinif kendi ornek sayisiyla agirliklandirilir
def all_metrics(y_true, y_pred, n_classes):

    #Dogruluk (accuracy)
    acc = float((y_true == y_pred).mean())

    #Sinif bazinda precision, recall, f1 (her sinif icin ayri)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(n_classes)),
        average=None,        #once sinif basina al
        zero_division=0,
    )

    #Weighted ortalama: her sinifin metrigi, ornek sayisiyla agirliklandirilir
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(n_classes)),
        average="weighted",  #hocanin kodundaki gibi
        zero_division=0,
    )

    #Cohen's kappa: rastgele tahminden ne kadar iyi oldugumuzu gosterir
    kappa = float(cohen_kappa_score(y_true, y_pred,
                                    labels=list(range(n_classes))))

    #Karisiklik matrisi: hangi sinif hangi sinifla karisiyor
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    return {
        "accuracy":  acc,
        "precision": float(w_prec),
        "recall":    float(w_rec),
        "f1":        float(w_f1),
        "kappa":     kappa,
        "per_class_precision": prec.tolist(),
        "per_class_recall":    rec.tolist(),
        "per_class_f1":        f1.tolist(),
        "per_class_support":   support.tolist(),
        "confusion_matrix":    cm.tolist(),
    }


#Sarki seviyesinde tahmin: bir sarkinin tum segmentlerinin softmax'i ortalanir
#En yuksek olasiliga sahip sinif, o sarkinin tahmini olur (soft voting)
def song_level_predictions(probs, labels, song_ids):

    song_prob = defaultdict(lambda: np.zeros(probs.shape[1]))
    song_cnt  = defaultdict(int)
    song_lbl  = {}

    for p, l, sid in zip(probs, labels, song_ids):
        song_prob[sid] += p
        song_cnt[sid]  += 1
        song_lbl[sid]  = l

    s_preds, s_labels, sids = [], [], []
    for sid in sorted(song_prob.keys()):
        avg = song_prob[sid] / song_cnt[sid]
        s_preds.append(int(np.argmax(avg)))
        s_labels.append(int(song_lbl[sid]))
        sids.append(sid)

    return np.array(s_preds), np.array(s_labels), sids


#Metrikleri okunabilir bir tablo halinde yazdirir
def print_report(metrics, idx_to_label, title):

    print(f"\n--- {title} ---")
    print(f"  accuracy  : {metrics['accuracy']:.4f}")
    print(f"  precision : {metrics['precision']:.4f}  (weighted)")
    print(f"  recall    : {metrics['recall']:.4f}  (weighted)")
    print(f"  f1        : {metrics['f1']:.4f}  (weighted)")
    print(f"  kappa     : {metrics['kappa']:.4f}")

    #Sinif bazinda detayli tablo
    print(f"\n  {'sinif':<22} {'destek':>8} {'precision':>10} {'recall':>8} {'f1':>6}")
    print("  " + "-" * 56)
    for i in sorted(idx_to_label.keys()):
        name = idx_to_label[i]
        s   = metrics["per_class_support"][i]
        p   = metrics["per_class_precision"][i]
        r   = metrics["per_class_recall"][i]
        fsc = metrics["per_class_f1"][i]
        print(f"  {name:<22} {s:>8} {p:>10.3f} {r:>8.3f} {fsc:>6.3f}")