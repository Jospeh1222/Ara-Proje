#K-fold capraz dogrulamanin TEK FOLDUNU egitir

#Kullanim:
#python train_fold.py 0     -> sadece fold 0
#python train_fold.py all   -> tum foldlar pes pese

#Drive otomatik kayit: her epoch sonu Drive'a kopyalar
#Resume: last.pt varsa kaldigi yerden devam eder

from pathlib import Path
import sys, os, shutil, json, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

import config as C
from dataset import MakamFoldDataset
from model import MakamCRNN
from metrics import all_metrics, song_level_predictions, print_report


#Dengesiz siniflar icin agirlik hesapla (az olan sinif daha cok cezalansin)
def class_weights_from(items, num_classes):

    counts = np.zeros(num_classes)
    for it in items:
        counts[it["label_idx"]] += 1

    #Ters frekans, ortalama 1 olacak sekilde normalize et
    w = counts.sum() / (num_classes * counts.clip(min=1))
    return torch.tensor(w, dtype=torch.float32), counts


#Bir epoch calistir (egitim veya degerlendirme)
#train=True ise gradient hesaplar ve optimizer adimi atar
#train=False ise sadece tahmin uretir (val ve test icin)
def run_epoch(model, loader, criterion, optimizer, device, train: bool):

    model.train() if train else model.eval()

    total = correct = 0
    loss_sum = 0.0
    all_preds, all_labels, all_songids, all_probs = [], [], [], []

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:

        for x, y, song_ids in loader:

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                optimizer.step()

            preds = logits.argmax(dim=1)
            total    += y.size(0)
            correct  += (preds == y).sum().item()
            loss_sum += loss.item() * y.size(0)

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


#Drive var mi kontrol et (Colab'da varsa True, lokalde False)
def drive_available():
    return os.path.exists(C.DRIVE_DIR)


#Drive'a guvenli kopyalama, hata varsa egitimi kesmesin
def safe_drive_copy(src, dst):
    if not drive_available():
        return False
    try:
        shutil.copy(src, dst)
        return True
    except Exception as e:
        print(f"   (drive kopya hatasi: {e})")
        return False


#Tek bir foldun tam egitim surecini yurutur
def train_one_fold(fold: int):

    torch.manual_seed(C.SEED + fold)
    np.random.seed(C.SEED + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'#' * 60}")
    print(f"# FOLD {fold} / {C.N_FOLDS - 1}")
    print(f"{'#' * 60}")
    print(f"Cihaz: {device}")

    #Bu fold icin stats dosyasi var mi kontrol et
    if not C.stats_path(fold).exists():
        raise SystemExit(f"Eksik dosya: {C.stats_path(fold)}. "
                         f"Once calistir: python compute_stats.py {fold}")

    #Uc dataseti yukle: train (augmentli), val ve test (augmentsiz)
    train_ds = MakamFoldDataset(fold, "train", augment=True)
    val_ds   = MakamFoldDataset(fold, "val",   augment=False)
    test_ds  = MakamFoldDataset(fold, "test",  augment=False)
    n_classes = train_ds.num_classes
    idx_to_label = {int(v): k for k, v in train_ds.label_to_idx.items()}
    print(f"train={len(train_ds)} seg  val={len(val_ds)} seg  "
          f"test={len(test_ds)} seg  sinif={n_classes}")

    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=C.NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=C.BATCH_SIZE, shuffle=False,
                              num_workers=C.NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=C.BATCH_SIZE, shuffle=False,
                              num_workers=C.NUM_WORKERS, pin_memory=True)

    #Model
    model = MakamCRNN(n_classes=n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parametre sayisi: {n_params:,}")

    #Sinif agirliklari (dengesiz veri icin)
    cw, counts = class_weights_from(train_ds.items, n_classes)
    print(f"Sinif basina train segment: {counts.astype(int).tolist()}")

    #Loss + optimizer
    criterion = nn.CrossEntropyLoss(weight=cw.to(device),
                                    label_smoothing=C.LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)

    C.CKPT_DIR.mkdir(exist_ok=True)
    C.RESULTS_DIR.mkdir(exist_ok=True)

    #Erken durdurma durum degiskenleri
    start_epoch = 1
    best_val_loss = float("inf")          #simdiye kadarki en iyi (en dusuk) val loss
    best_epoch = 0                        #en iyi val loss'un goruldugu epoch
    best_state = None                     #en iyi epoch'un model agirliklari
    epochs_no_improve = 0                 #kac epoch'tur iyilesme yok

    #Resume mantigi: once Drive'a bak, yoksa lokal'e bak
    #*************************************************************************
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
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            best_epoch = ckpt.get("best_epoch", 0)
            epochs_no_improve = ckpt.get("epochs_no_improve", 0)
            best_state = ckpt.get("best_state", None)
            print(f"{resume_src}'den devam: epoch {start_epoch}'den basliyor "
                  f"(en iyi val_loss={best_val_loss:.4f} @ epoch {best_epoch})")
            if start_epoch > C.MAX_EPOCHS:
                print(f"Fold {fold} zaten max epoch'a ulasmis")
        except Exception as e:
            print(f"Resume basarisiz, sifirdan basliyor: {e}")
            start_epoch = 1
    #***********************************************************

    if drive_available():
        print(f"Drive kayit AKTIF")
    else:
        print(f"Drive bagli degil, sadece lokal kayit")
    print(f"Maksimum epoch: {C.MAX_EPOCHS}, erken durdurma sabri: {C.EARLY_STOP_PATIENCE}\n")


    #Ana egitim dongusu
    #*************************************************************************
    stopped_early = False
    for epoch in range(start_epoch, C.MAX_EPOCHS + 1):

        #Egitim adimi
        tr = run_epoch(model, train_loader, criterion, optimizer, device, train=True)

        #Validation adimi (gradient yok, sadece degerlendirme)
        va = run_epoch(model, val_loader, criterion, None, device, train=False)

        print(f"  Epoch {epoch:3d}/{C.MAX_EPOCHS} | "
              f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f} | "
              f"val loss={va['loss']:.4f} acc={va['acc']:.4f}", end="")

        #Erken durdurma kontrolu: val_loss iyilesti mi?
        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            best_epoch = epoch
            #En iyi agirliklari hafizada sakla (restore best weights icin)
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"  <- en iyi (val_loss dustu)")
        else:
            epochs_no_improve += 1
            print(f"  (iyilesme yok: {epochs_no_improve}/{C.EARLY_STOP_PATIENCE})")

        #Her epoch sonunda 'last' kaydi yap (resume icin)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "epochs_no_improve": epochs_no_improve,
            "label_to_idx": train_ds.label_to_idx,
        }, last_path)
        safe_drive_copy(last_path, drive_last)

        #Sabir doldu mu? Egitimi durdur
        if epochs_no_improve >= C.EARLY_STOP_PATIENCE:
            print(f"\n  Erken durdurma: val_loss {C.EARLY_STOP_PATIENCE} "
                  f"epoch boyunca iyilesmedi.")
            print(f"  En iyi model: epoch {best_epoch} (val_loss={best_val_loss:.4f})")
            stopped_early = True
            break
    #*******************************************************************************

    if not stopped_early:
        print(f"\n  Maksimum epoch ({C.MAX_EPOCHS}) tamamlandi.")
        print(f"  En iyi model: epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    #En iyi val_loss'a sahip agirliklari geri yukle (restore best weights)
    #Test degerlendirmesi bu model uzerinde yapilacak
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  En iyi agirliklar geri yuklendi (epoch {best_epoch})")


    #Test seti uzerinde degerlendirme (en iyi model ile)
    #**********************************************************
    print(f"\nFold {fold} test setinde degerlendiriliyor...")
    te = run_epoch(model, test_loader, criterion, None, device, train=False)

    #Segment seviyesinde metrikler
    seg_metrics = all_metrics(te["labels"], te["preds"], n_classes)
    print_report(seg_metrics, idx_to_label, f"Fold {fold} - Segment seviyesi")

    #Sarki seviyesinde metrikler
    s_preds, s_labels, _ = song_level_predictions(
        te["probs"], te["labels"], te["songids"]
    )
    song_metrics = all_metrics(s_labels, s_preds, n_classes)
    print_report(song_metrics, idx_to_label, f"Fold {fold} - Sarki seviyesi")

    #En iyi modeli kaydet
    torch.save({
        "model": model.state_dict(),
        "label_to_idx": train_ds.label_to_idx,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }, C.ckpt_best_path(fold))
    safe_drive_copy(C.ckpt_best_path(fold), C.drive_best_path(fold))

    #Sonuclari JSON'a yaz
    fold_results = {
        "fold": fold,
        "n_classes": n_classes,
        "label_to_idx": train_ds.label_to_idx,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "segment_level": seg_metrics,
        "song_level":    song_metrics,
    }
    with open(C.fold_results_path(fold), "w", encoding="utf-8") as f:
        json.dump(fold_results, f, indent=2, ensure_ascii=False)
    print(f"\nFold {fold} sonuclari kaydedildi -> {C.fold_results_path(fold)}")
    #****************************************************************************


def main():
    if len(sys.argv) < 2:
        print("Kullanim: python train_fold.py <fold_no|all>")
        sys.exit(1)

    arg = sys.argv[1]
    if arg == "all":
        for k in range(C.N_FOLDS):
            train_one_fold(k)
    else:
        train_one_fold(int(arg))


if __name__ == "__main__":
    main()