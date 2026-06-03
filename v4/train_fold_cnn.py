%%writefile /content/train_fold_cnn.py

from pathlib import Path
import sys, os, shutil, json, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn

import config as C
from dataset_cnn import MakamCNNDataset
from model_cnn import MakamCNN
from metrics import all_metrics, song_level_predictions, print_report
from torch.optim.lr_scheduler import CosineAnnealingLR

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)  # doğru sınıfın olasılığı
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()


def class_weights_from(items, num_classes):
    counts = np.zeros(num_classes)
    for it in items:
        counts[it["label_idx"]] += 1
    w = counts.sum() / (num_classes * counts.clip(min=1))
    return torch.tensor(w, dtype=torch.float32), counts


def run_epoch(model, loader, criterion, optimizer, device, train: bool, scheduler=None):
    model.train() if train else model.eval()

    total = correct = 0
    loss_sum = 0.0
    all_preds, all_labels, all_songids, all_probs = [], [], [], []

    with torch.enable_grad() if train else torch.no_grad():
        for x, y, song_ids in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if train:
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()

            total    += y.size(0)
            loss_sum += loss.item() * y.size(0)

            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
            all_songids.extend(song_ids)
            all_probs.append(F.softmax(logits, dim=1).detach().cpu().numpy())

    return {
        "loss":    loss_sum / total,
        "acc":     correct  / total,
        "preds":   np.concatenate(all_preds),
        "labels":  np.concatenate(all_labels),
        "songids": all_songids,
        "probs":   np.concatenate(all_probs),
    }

def drive_available():
    return os.path.exists(C.DRIVE_DIR)


def safe_drive_copy(src, dst):
    if not drive_available():
        return False
    try:
        shutil.copy(src, dst)
        return True
    except Exception as e:
        print(f"   (drive kopya hatasi: {e})")
        return False


def train_one_fold(fold: int):
    torch.manual_seed(C.SEED + fold)
    np.random.seed(C.SEED + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'#' * 60}")
    print(f"# CNN FOLD {fold} / {C.N_FOLDS - 1}")
    print(f"{'#' * 60}")
    print(f"Cihaz: {device}")

    if not C.stats_spec_path(fold).exists():
        raise SystemExit(f"Eksik: {C.stats_spec_path(fold)}")

    train_ds = MakamCNNDataset(fold, "train", augment=True)
    val_ds   = MakamCNNDataset(fold, "val",   augment=False)
    test_ds  = MakamCNNDataset(fold, "test",  augment=False)
    n_classes    = train_ds.num_classes
    idx_to_label = {int(v): k for k, v in train_ds.label_to_idx.items()}
    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  sinif={n_classes}")

    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=C.NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=C.BATCH_SIZE, shuffle=False,
                              num_workers=C.NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=C.BATCH_SIZE, shuffle=False,
                              num_workers=C.NUM_WORKERS, pin_memory=True)

    model    = MakamCNN(n_classes=n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parametre sayisi: {n_params:,}")

    cw, counts = class_weights_from(train_ds.items, n_classes)
    print(f"Sinif basina segment: {counts.astype(int).tolist()}")

    criterion = FocalLoss(weight=cw.to(device), gamma=2.0,
                          label_smoothing=C.LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
      optimizer,
      max_lr=C.LR,
      epochs=C.MAX_EPOCHS + 1,
      steps_per_epoch=len(train_loader),
      pct_start=0.3,
    )
    swa_model = AveragedModel(model)
    swa_start = 25  # bu epoch'tan sonra ağırlıkları biriktirmeye başla

    CNN_CKPT_DIR = C.CKPT_DIR / "cnn"
    CNN_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    C.RESULTS_DIR.mkdir(exist_ok=True)

    last_path  = CNN_CKPT_DIR / f"fold{fold}_last.pt"
    best_path  = CNN_CKPT_DIR / f"fold{fold}_best.pt"
    drive_last = f"{C.DRIVE_DIR}/checkpoints/cnn_fold{fold}_last.pt"
    drive_best = f"{C.DRIVE_DIR}/checkpoints/cnn_fold{fold}_best.pt"

    # Erken durdurma değişkenleri
    start_epoch      = 1
    best_val_loss    = float("inf")
    best_epoch       = 0
    best_state       = None
    epochs_no_improve = 0

    # Resume
    if drive_available() and os.path.exists(drive_last):
        try:
            shutil.copy(drive_last, last_path)
        except Exception as e:
            print(f"Drive'dan cekemedi: {e}")

    if last_path.exists():
        try:
            ckpt = torch.load(last_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            try:
                scheduler.load_state_dict(ckpt["scheduler"])  # ← buraya
            except:
                pass
            start_epoch       = ckpt["epoch"] + 1
            best_val_loss     = ckpt.get("best_val_loss", float("inf"))
            best_epoch        = ckpt.get("best_epoch", 0)
            epochs_no_improve = ckpt.get("epochs_no_improve", 0)
            best_state        = ckpt.get("best_state", None)
            print(f"Resume: epoch {start_epoch}'den basliyor "
                  f"(en iyi val_loss={best_val_loss:.4f} @ epoch {best_epoch})")
            if start_epoch > C.MAX_EPOCHS:
                print(f"Fold {fold} zaten tamamlanmis")
                return
        except Exception as e:
            print(f"Resume basarisiz, sifirdan basliyor: {e}")
            start_epoch = 1

    print(f"Maksimum epoch: {C.MAX_EPOCHS}  erken durdurma sabri: {C.EARLY_STOP_PATIENCE}\n")

    # Ana eğitim döngüsü
    stopped_early = False
    for epoch in range(start_epoch, C.MAX_EPOCHS + 1):

        tr = run_epoch(model, train_loader, criterion, optimizer, device, True, scheduler)
        va = run_epoch(model, val_loader,   criterion, None,      device, False)

        if epoch >= swa_start:
            swa_model.update_parameters(model)

        print(f"  Epoch {epoch:3d}/{C.MAX_EPOCHS} | "
              f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f} | "
              f"val loss={va['loss']:.4f} acc={va['acc']:.4f}", end="")

        if va["loss"] < best_val_loss:
            best_val_loss     = va["loss"]
            best_epoch        = epoch
            best_state        = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"  <- en iyi (val_loss dustu)")
        else:
            epochs_no_improve += 1
            print(f"  (iyilesme yok: {epochs_no_improve}/{C.EARLY_STOP_PATIENCE})")

        # Last checkpoint
        torch.save({
            "model":            model.state_dict(),
            "optimizer":        optimizer.state_dict(),
            "scheduler":        scheduler.state_dict(),
            "epoch":            epoch,
            "best_val_loss":    best_val_loss,
            "best_epoch":       best_epoch,
            "best_state":       best_state,
            "epochs_no_improve": epochs_no_improve,
            "label_to_idx":     train_ds.label_to_idx,
        }, last_path)
        safe_drive_copy(last_path, drive_last)

        # Early stopping
        if epoch >= 20 and epochs_no_improve >= C.EARLY_STOP_PATIENCE:
            print(f"\n  Erken durdurma: val_loss {C.EARLY_STOP_PATIENCE} "
                  f"epoch boyunca iyilesmedi.")
            print(f"  En iyi model: epoch {best_epoch} (val_loss={best_val_loss:.4f})")
            stopped_early = True
            break

    if not stopped_early:
        print(f"\n  Maksimum epoch ({C.MAX_EPOCHS}) tamamlandi.")
        print(f"  En iyi model: epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    # En iyi ağırlıkları geri yükle
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  En iyi agirliklar geri yuklendi (epoch {best_epoch})")

    # SWA BatchNorm güncelleme (manuel — loader 3'lü döndürdüğü için)
    swa_model.train()
    with torch.no_grad():
        for x, _, _ in train_loader:
            x = x.to(device, non_blocking=True)
            swa_model(x)

    model = swa_model.module
    model.eval()
    print("SWA ağırlıkları uygulandı")

    # Test değerlendirmesi **** değiştirildi TAA ile
    print(f"\nFold {fold} test setinde degerlendiriliyor (epoch {best_epoch})...")
    te = run_epoch(model, test_loader, criterion, None, device, False)
    seg_metrics = all_metrics(te["labels"], te["preds"], n_classes)
    print_report(seg_metrics, idx_to_label, f"CNN Fold {fold} - Segment")
    s_preds, s_labels, _ = song_level_predictions(te["probs"], te["labels"], te["songids"])
    song_metrics = all_metrics(s_labels, s_preds, n_classes)
    print_report(song_metrics, idx_to_label, f"CNN Fold {fold} - Sarki")

    # Sonuçları kaydet
    torch.save({
        "model":         model.state_dict(),
        "label_to_idx":  train_ds.label_to_idx,
        "fold":          fold,
        "best_epoch":    best_epoch,
        "best_val_loss": best_val_loss,
    }, best_path)
    safe_drive_copy(best_path, drive_best)

    fold_results = {
        "fold":          fold,
        "n_classes":     n_classes,
        "label_to_idx":  train_ds.label_to_idx,
        "best_epoch":    best_epoch,
        "best_val_loss": float(best_val_loss),
        "segment_level": seg_metrics,
        "song_level":    song_metrics,
    }
    out_path = C.RESULTS_DIR / f"cnn_fold{fold}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fold_results, f, indent=2, ensure_ascii=False)
    print(f"Sonuclar kaydedildi -> {out_path}")

def train_fold_ensemble(fold: int, seeds=[42, 123, 7]):
    """Aynı fold'u farklı seed'lerle eğit, test olasılıklarını ortala"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test verisini bir kez hazırla
    test_ds = MakamCNNDataset(fold, "test", augment=False)
    n_classes = test_ds.num_classes
    idx_to_label = {int(v): k for k, v in test_ds.label_to_idx.items()}
    test_loader = DataLoader(test_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                             num_workers=C.NUM_WORKERS, pin_memory=True)

    ens_probs = None
    labels = None
    songids = None

    for si, seed in enumerate(seeds):
        print(f"\n{'='*60}\n# FOLD {fold} - SEED {seed} ({si+1}/{len(seeds)})\n{'='*60}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_ds = MakamCNNDataset(fold, "train", augment=True)
        val_ds   = MakamCNNDataset(fold, "val",   augment=False)
        train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                                  num_workers=C.NUM_WORKERS, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                                  num_workers=C.NUM_WORKERS, pin_memory=True)

        model = MakamCNN(n_classes=n_classes).to(device)
        cw, _ = class_weights_from(train_ds.items, n_classes)
        criterion = nn.CrossEntropyLoss(weight=cw.to(device), label_smoothing=C.LABEL_SMOOTHING)
        optimizer = AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
        scheduler = OneCycleLR(optimizer, max_lr=C.LR, epochs=C.MAX_EPOCHS+1,
                               steps_per_epoch=len(train_loader), pct_start=0.3)
        swa_model = AveragedModel(model)
        swa_start = 25

        best_val_loss = float("inf"); best_state = None; epochs_no_improve = 0
        for epoch in range(1, C.MAX_EPOCHS + 1):
            tr = run_epoch(model, train_loader, criterion, optimizer, device, True, scheduler)
            va = run_epoch(model, val_loader, criterion, None, device, False)
            if epoch >= swa_start:
                swa_model.update_parameters(model)
            if va["loss"] < best_val_loss:
                best_val_loss = va["loss"]; best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            print(f"  Epoch {epoch:3d} | val loss={va['loss']:.4f} acc={va['acc']:.4f}"
                  f"{'  <-' if epochs_no_improve==0 else ''}")
            if epoch >= 20 and epochs_no_improve >= C.EARLY_STOP_PATIENCE:
                print(f"  Erken durdurma (epoch {epoch})")
                break

        # SWA BatchNorm güncelle
        swa_model.train()
        with torch.no_grad():
            for x, _, _ in train_loader:
                swa_model(x.to(device, non_blocking=True))
        eval_model = swa_model.module
        for m in eval_model.modules():
            if isinstance(m, nn.GRU):
                m.flatten_parameters()
        eval_model.eval()

        te = run_epoch(eval_model, test_loader, criterion, None, device, False)
        if ens_probs is None:
            ens_probs = te["probs"].copy()
            labels = te["labels"]; songids = te["songids"]
        else:
            ens_probs += te["probs"]

        # Tek seed sonucu (karşılaştırma için)
        sp, sl, _ = song_level_predictions(te["probs"], te["labels"], te["songids"])
        sm = all_metrics(sl, sp, n_classes)
        print(f"  Seed {seed} tek model Song F1: {sm['f1']:.3f}")

    # Ensemble: olasılık ortalaması
    ens_probs /= len(seeds)
    s_preds, s_labels, _ = song_level_predictions(ens_probs, labels, songids)
    song_metrics = all_metrics(s_labels, s_preds, n_classes)
    print_report(song_metrics, idx_to_label, f"CNN Fold {fold} - ENSEMBLE Sarki")

    fold_results = {
        "fold": fold, "n_classes": n_classes,
        "label_to_idx": test_ds.label_to_idx,
        "seeds": seeds, "song_level": song_metrics,
    }
    out_path = C.RESULTS_DIR / f"cnn_ensemble_fold{fold}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fold_results, f, indent=2, ensure_ascii=False)
    print(f"Kaydedildi -> {out_path}")
    return song_metrics


def main():
    user_args = [a for a in sys.argv[1:] if a == "all" or a.isdigit()]
    arg = user_args[0] if user_args else "all"

    if arg == "all":
        for k in range(C.N_FOLDS):
            train_one_fold(k)
    else:
        train_one_fold(int(arg))


if __name__ == "__main__":
    main()
