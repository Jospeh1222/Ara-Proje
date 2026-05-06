#Tüm metrik hesaplamaları burada toplandı

from collections import defaultdict
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


#Bir tahmin-gerçek vektör çiftinden 5 metriği hesaplar
#Precision/recall/F1 için "macro" ortalama
def all_metrics(y_true, y_pred, n_classes):

    #Doğruluk (accuracy): doğru tahmin / toplam tahmin
    acc = float((y_true == y_pred).mean())

    #Sınıf bazında precision, recall, f1 (her sınıf için ayrı)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(n_classes)), #16 sınıfın hepsini dahil et
        average=None,                  #ortalama almadan, sınıf başına
        zero_division=0,               #sıfır bölme durumunda 0 dön (uyarı vermesin)
    )

    #Macro ortalama: her sınıfın metriğinin ortalaması
    macro_prec = float(np.mean(prec))
    macro_rec  = float(np.mean(rec))
    macro_f1   = float(np.mean(f1))

    #Cohen's kappa: rastgele tahminden ne kadar iyi olduğumuzu gösterir
    #0 = rastgele kadar, 1 = mükemmel, dengesiz veride doğruluktan daha güvenilir
    kappa = float(cohen_kappa_score(y_true, y_pred,
                                    labels=list(range(n_classes))))

    #Karışıklık matrisi (confusion matrix): hangi sınıf hangi sınıfla karışıyor
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    return {
        "accuracy":  acc,
        "precision": macro_prec,
        "recall":    macro_rec,
        "f1":        macro_f1,
        "kappa":     kappa,
        "per_class_precision": prec.tolist(),
        "per_class_recall":    rec.tolist(),
        "per_class_f1":        f1.tolist(),
        "per_class_support":   support.tolist(),
        "confusion_matrix":    cm.tolist(),
    }


#Şarkı seviyesinde tahmin: bir şarkının tüm segmentlerinin softmax'ı toplanır
#En yüksek olasılığa sahip sınıf, o şarkının tahmini olur (soft voting)
#Bu yaklaşım çoğunluk oylamasından daha iyidir, güven seviyesini de hesaba katar
def song_level_predictions(probs, labels, song_ids):

    song_prob = defaultdict(lambda: np.zeros(probs.shape[1])) #her şarkı için olasılık birikimi
    song_cnt  = defaultdict(int)                              #segment sayısı
    song_lbl  = {}                                            #şarkının gerçek etiketi

    #Tüm segmentleri şarkıya göre grupla
    for p, l, sid in zip(probs, labels, song_ids):
        song_prob[sid] += p
        song_cnt[sid]  += 1
        song_lbl[sid]  = l

    s_preds, s_labels, sids = [], [], []
    for sid in sorted(song_prob.keys()):
        avg = song_prob[sid] / song_cnt[sid] #ortalama olasılık vektörü
        s_preds.append(int(np.argmax(avg)))   #en yüksek olanı seç
        s_labels.append(int(song_lbl[sid]))
        sids.append(sid)

    return np.array(s_preds), np.array(s_labels), sids


#Metrikleri okunabilir bir tablo halinde yazdırır
def print_report(metrics, idx_to_label, title):

    print(f"\n--- {title} ---")
    print(f"  accuracy  : {metrics['accuracy']:.4f}")
    print(f"  precision : {metrics['precision']:.4f}  (macro)")
    print(f"  recall    : {metrics['recall']:.4f}  (macro)")
    print(f"  f1        : {metrics['f1']:.4f}  (macro)")
    print(f"  kappa     : {metrics['kappa']:.4f}")

    #Sınıf bazında detaylı tablo
    print(f"\n  {'sinif':<22} {'destek':>8} {'precision':>10} {'recall':>8} {'f1':>6}")
    print("  " + "-" * 56)
    for i in sorted(idx_to_label.keys()):
        name = idx_to_label[i]
        s   = metrics["per_class_support"][i]    #o sınıfın test setindeki örnek sayısı
        p   = metrics["per_class_precision"][i]
        r   = metrics["per_class_recall"][i]
        fsc = metrics["per_class_f1"][i]
        print(f"  {name:<22} {s:>8} {p:>10.3f} {r:>8.3f} {fsc:>6.3f}")