#K-fold'un tek foldunu eğitir
#Tüm foldlar aynı süre eğitilir (40 epoch sabit)


#python train_fold.py 0     (sadece fold 0)
#python train_fold.py all    (tüm foldlar peş peşe)

#Drive otomatik kayıt: her epoch sonu Drive'a kopyalar (Colab kopması durumunda kayıp olmasın)
#Resume: last.pt varsa kaldığı yerden devam eder, otomatik tespit edilir

from pathlib import Path
import sys, os, shutil, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config as C
from dataset import MakamFoldDataset
from model import MakamCRNN
from metrics import all_metrics, song_level_predictions, print_report


#Dengesiz sınıflar için ağırlık hesapla
#Az örnekli sınıfa daha yüksek ağırlık verir, dengesizlik etkisini azaltır
def class_weights_from(items, num_classes):

    counts = np.zeros(num_classes)
    for it in items:
        counts[it["label_idx"]] += 1

    #Ters frekans, ortalama 1 olacak şekilde normalize et
    w = counts.sum() / (num_classes * counts.clip(min=1))
    return torch.tensor(w, dtype=torch.float32), counts


#Bir epoch çalıştır (eğitim veya doğrulama)
#train=True ise gradient hesaplar ve optimizer adımı atar
#train=False ise sadece tahmin üretir, sonuçları toplar
def run_epoch(model, loader, criterion, optimizer, device, train: bool):

    model.train() if train else model.eval() #BN ve Dropout için mod değiştir

    total = correct = 0
    loss_sum = 0.0
    all_preds, all_labels, all_songids, all_probs = [], [], [], []

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:

        for x, y, song_ids in loader:

            x = x.to(device, non_blocking=True) #GPU'ya gönder
            y = y.to(device, non_blocking=True)

            logits = model(x) #ileri yayılım
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()                                       #eski gradientleri sıfırla
                loss.backward()                                             #geri yayılım
                nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)   #gradient patlamasını önle
                optimizer.step()                                            #ağırlıkları güncelle

            #İstatistik topla
            preds = logits.argmax(dim=1)
            total    += y.size(0)
            correct  += (preds == y).sum().item()
            loss_sum += loss.item() * y.size(0)

            #Sonuçları sakla (test eval'i için)
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
            all_songids.extend(song_ids)
            all_probs.append(F.softmax(logits, dim=1).detach().cpu().numpy())

    return {
        "loss": loss_sum / total, "acc": correct / total,
        "preds":   np.concatenate(all_preds),
        "labels":  np.concatenate(all_labels),
        "songids": all_songids,
        "probs":   np.concatenate(all_probs),
    }


#Drive var mı kontrol et (Colab'da varsa true lokaldeyse false)
def drive_available():
    return os.path.exists(C.DRIVE_DIR)


#Drive'a kopyalama
def safe_drive_copy(src, dst):
    if not drive_available():
        return False
    try:
        shutil.copy(src, dst)
        return True
    except Exception as e:
        print(f"   (drive kopya hatasi: {e})")
        return False


#Tek bir foldun tam eğitim sürecini yürütür
def train_one_fold(fold: int):

    #Her fold için farklı seed (başlangıç koşulları farklı olsun)
    torch.manual_seed(C.SEED + fold)
    np.random.seed(C.SEED + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'#' * 60}")
    print(f"# FOLD {fold} / {C.N_FOLDS - 1}")
    print(f"{'#' * 60}")
    print(f"Cihaz: {device}")

    #Bu fold için stats dosyası var mı kontrol et
    if not C.stats_path(fold).exists():
        raise SystemExit(f"Eksik dosya: {C.stats_path(fold)}. "
                         f"Once calistir: python compute_stats.py {fold}")

    #Datasetleri ve loaderları oluştur
    train_ds = MakamFoldDataset(fold, "train", augment=True)  #augment sadece eğitimde
    test_ds  = MakamFoldDataset(fold, "test",  augment=False) #testte ham veri
    n_classes = train_ds.num_classes
    idx_to_label = {int(v): k for k, v in train_ds.label_to_idx.items()}
    print(f"train segment={len(train_ds)}  test segment={len(test_ds)}  sinif={n_classes}")

    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=C.NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=C.BATCH_SIZE, shuffle=False,
                              num_workers=C.NUM_WORKERS, pin_memory=True)

    #Model, parametre sayısını yazdır
    model = MakamCRNN(n_classes=n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parametre sayisi: {n_params:,}")

    #Sınıf ağırlıkları (dengesiz veri için)
    cw, counts = class_weights_from(train_ds.items, n_classes)
    print(f"Sinif basina segment: {counts.astype(int).tolist()}")

    #Loss + optimizer + scheduler
    criterion = nn.CrossEntropyLoss(weight=cw.to(device),
                                    label_smoothing=C.LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=C.EPOCHS) #cosine ile lr'i azalt

    C.CKPT_DIR.mkdir(exist_ok=True)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    start_epoch = 1

    #Resume mantığı: önce Drive'a bak, yoksa lokal'e bak, ikisi de yoksa baştan başla
    #*****************************************************
    last_path = C.ckpt_last_path(fold)
    drive_last = C.drive_last_path(fold)
    resume_src = None

    if drive_available() and os.path.exists(drive_last):
        try:
            shutil.copy(drive_last, last_path)
            resume_src = "Drive"
        except Exception as e:
            print(f"Drive'dan cekemedi: {e}")

    if resume_src is None and last_path.exists():
        resume_src = "lokal"

    if resume_src:
        try:
            ckpt = torch.load(last_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            print(f"{resume_src}'den devam: epoch {start_epoch}'den basliyor")

            #Eğer zaten tüm epochları bitirmişse tekrar eğitme
            if start_epoch > C.EPOCHS:
                print(f"Fold {fold} zaten tamamlanmis ({C.EPOCHS} epoch)")
                return
        except Exception as e:
            print(f"Resume basarisiz, sifirdan basliyor: {e}")
            start_epoch = 1
    #************************************************************************

    if drive_available():
        print(f"Drive kayit AKTIF")
    else:
        print(f"Drive bagli degil, sadece lokal kayit")


    #Ana eğitim döngüsü
    #*********************************************************
    for epoch in range(start_epoch, C.EPOCHS + 1):

        tr = run_epoch(model, train_loader, criterion, optimizer, device, True)
        scheduler.step()

        print(f"  Epoch {epoch:3d}/{C.EPOCHS} | "
              f"train loss={tr['loss']:.3f} acc={tr['acc']:.3f} | "
              f"lr={optimizer.param_groups[0]['lr']:.5f}")

        #Her epoch sonunda kayıt yap 
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "label_to_idx": train_ds.label_to_idx,
        }, last_path)
        safe_drive_copy(last_path, drive_last) #Drive'a da kopyala
    #***********************************************************************


    #Test seti üzerinde değerlendirme
    #***********************************************************************
    print(f"\nFold {fold} test setinde degerlendiriliyor...")
    te = run_epoch(model, test_loader, criterion, None, device, False)

    #Segment seviyesinde metrikler
    seg_metrics = all_metrics(te["labels"], te["preds"], n_classes)
    print_report(seg_metrics, idx_to_label, f"Fold {fold} - Segment seviyesi")

    #Şarkı seviyesinde metrikler (segment olasılıklarının ortalaması)
    s_preds, s_labels, _ = song_level_predictions(
        te["probs"], te["labels"], te["songids"]
    )
    song_metrics = all_metrics(s_labels, s_preds, n_classes)
    print_report(song_metrics, idx_to_label, f"Fold {fold} - Sarki seviyesi")

    #Final modeli ve sonuçları kaydet
    torch.save({
        "model": model.state_dict(),
        "label_to_idx": train_ds.label_to_idx,
        "fold": fold,
    }, C.ckpt_best_path(fold))
    safe_drive_copy(C.ckpt_best_path(fold), C.drive_best_path(fold))

    #Sonuçları JSON'a yaz
    fold_results = {
        "fold": fold,
        "n_classes": n_classes,
        "label_to_idx": train_ds.label_to_idx,
        "segment_level": seg_metrics,
        "song_level":    song_metrics,
    }
    with open(C.fold_results_path(fold), "w", encoding="utf-8") as f:
        json.dump(fold_results, f, indent=2, ensure_ascii=False)
    print(f"\nFold {fold} sonuclari kaydedildi -> {C.fold_results_path(fold)}")
    #*************************************************************


def main():
    if len(sys.argv) < 2:
        print("Kullanim: python train_fold.py <fold_no|all>")
        sys.exit(1)

    arg = sys.argv[1]
    if arg == "all":
        #Tüm foldları sırayla eğit
        for k in range(C.N_FOLDS):
            train_one_fold(k)
    else:
        train_one_fold(int(arg))


if __name__ == "__main__":
    main()