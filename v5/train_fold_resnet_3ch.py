#Val + early stopping, en iyi val_loss agirliklari geri yuklenir

#Kullanim:
#python train_fold_resnet_3ch.py 0
#python train_fold_resnet_3ch.py all

import sys, os, shutil, json, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

import config as C
from dataset_resnet_3ch import MakamResNet3chDataset
from model_resnet import MakamResNet
from metrics import all_metrics, song_level_predictions, print_report


def class_weights_from(items, num_classes):
    counts = np.zeros(num_classes)
    for it in items:
        counts[it["label_idx"]] += 1
    w = counts.sum() / (num_classes * counts.clip(min=1))
    return torch.tensor(w, dtype=torch.float32), counts


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
            loss   = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), C.RESNET_GRAD_CLIP)
                optimizer.step()

            preds     = logits.argmax(dim=1)
            total    += y.size(0)
            correct  += (preds == y).sum().item()
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


def train_one_fold(fold: int):
    torch.manual_seed(C.SEED + fold)
    np.random.seed(C.SEED + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'#' * 60}")
    print(f"# RESNET-3CH FOLD {fold} / {C.N_FOLDS - 1}")
    print(f"# Input: 3-kanal PNG (R=mel, G=MFCC, B=chroma)")
    print(f"{'#' * 60}")
    print(f"Cihaz: {device}")
    print(f"Fold dosyasi: {C.FOLDS_FILE}")

    train_ds = MakamResNet3chDataset(fold, "train", augment=True)
    val_ds   = MakamResNet3chDataset(fold, "val",   augment=False)
    test_ds  = MakamResNet3chDataset(fold, "test",  augment=False)
    n_classes    = train_ds.num_classes
    idx_to_label = {int(v): k for k, v in train_ds.label_to_idx.items()}
    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  sinif={n_classes}")

    train_loader = DataLoader(train_ds, batch_size=C.RESNET_BATCH_SIZE, shuffle=True,
                              num_workers=C.RESNET_NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=C.RESNET_BATCH_SIZE, shuffle=False,
                              num_workers=C.RESNET_NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=C.RESNET_BATCH_SIZE, shuffle=False,
                              num_workers=C.RESNET_NUM_WORKERS, pin_memory=True)

    model = MakamResNet(n_classes=n_classes,
                        freeze_early=C.RESNET_FREEZE_EARLY).to(device)
    total = model.get_total_params()
    trainable = model.get_trainable_params()
    print(f"Toplam parametre: {total:,}")
    print(f"Egitilebilir parametre: {trainable:,}  ({100*trainable/total:.1f}%)")
    print(f"Donduruluyor: conv1, bn1, layer1, layer2, layer3")
    print(f"Egitiliyor:   layer4, fc (yeni head)")

    cw, counts = class_weights_from(train_ds.items, n_classes)
    print(f"Sinif basina segment: {counts.astype(int).tolist()}")

    criterion = nn.CrossEntropyLoss(weight=cw.to(device),
                                    label_smoothing=C.RESNET_LABEL_SMOOTH)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=C.RESNET_LR,
                      weight_decay=C.RESNET_WEIGHT_DECAY)

    C.RESNET_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    C.RESNET_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    last_path = C.resnet_ckpt_last_path(fold)
    best_path = C.resnet_ckpt_best_path(fold)

    start_epoch       = 1
    best_val_loss     = float("inf")
    best_epoch        = 0
    best_state        = None
    epochs_no_improve = 0

    #Resume (Drive'da last.pt varsa devam et)
    if last_path.exists():
        try:
            ckpt = torch.load(last_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch       = ckpt["epoch"] + 1
            best_val_loss     = ckpt.get("best_val_loss", float("inf"))
            best_epoch        = ckpt.get("best_epoch", 0)
            epochs_no_improve = ckpt.get("epochs_no_improve", 0)
            best_state        = ckpt.get("best_state", None)
            print(f"Resume: epoch {start_epoch}'den basliyor "
                  f"(en iyi val_loss={best_val_loss:.4f} @ epoch {best_epoch})")
            if start_epoch > C.RESNET_MAX_EPOCHS:
                print(f"Fold {fold} zaten tamamlanmis")
                return
        except Exception as e:
            print(f"Resume basarisiz: {e}")
            start_epoch = 1

    print(f"Max epoch: {C.RESNET_MAX_EPOCHS}  patience: {C.RESNET_PATIENCE}  LR: {C.RESNET_LR}\n")

    stopped_early = False
    for epoch in range(start_epoch, C.RESNET_MAX_EPOCHS + 1):

        tr = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va = run_epoch(model, val_loader,   criterion, None,      device, False)

        print(f"  Epoch {epoch:3d}/{C.RESNET_MAX_EPOCHS} | "
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
            print(f"  (iyilesme yok: {epochs_no_improve}/{C.RESNET_PATIENCE})")

        torch.save({
            "model":             model.state_dict(),
            "optimizer":         optimizer.state_dict(),
            "epoch":             epoch,
            "best_val_loss":     best_val_loss,
            "best_epoch":        best_epoch,
            "best_state":        best_state,
            "epochs_no_improve": epochs_no_improve,
            "label_to_idx":      train_ds.label_to_idx,
        }, last_path)

        if epochs_no_improve >= C.RESNET_PATIENCE:
            print(f"\n  Erken durdurma: val_loss {C.RESNET_PATIENCE} epoch boyunca iyilesmedi.")
            print(f"  En iyi model: epoch {best_epoch} (val_loss={best_val_loss:.4f})")
            stopped_early = True
            break

    if not stopped_early:
        print(f"\n  Max epoch ({C.RESNET_MAX_EPOCHS}) tamamlandi.")
        print(f"  En iyi model: epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  En iyi agirliklar geri yuklendi (epoch {best_epoch})")

    print(f"\nFold {fold} test setinde degerlendiriliyor (epoch {best_epoch})...")
    te = run_epoch(model, test_loader, criterion, None, device, False)

    seg_metrics = all_metrics(te["labels"], te["preds"], n_classes)
    print_report(seg_metrics, idx_to_label, f"RESNET-3CH Fold {fold} - Segment")

    s_preds, s_labels, _ = song_level_predictions(
        te["probs"], te["labels"], te["songids"]
    )
    song_metrics = all_metrics(s_labels, s_preds, n_classes)
    print_report(song_metrics, idx_to_label, f"RESNET-3CH Fold {fold} - Sarki")

    torch.save({
        "model":         model.state_dict(),
        "label_to_idx":  train_ds.label_to_idx,
        "fold":          fold,
        "best_epoch":    best_epoch,
        "best_val_loss": best_val_loss,
    }, best_path)

    fold_results = {
        "fold":          fold,
        "experiment":    "resnet18_3ch_mel_mfcc_chroma",
        "n_classes":     n_classes,
        "label_to_idx":  train_ds.label_to_idx,
        "best_epoch":    best_epoch,
        "best_val_loss": float(best_val_loss),
        "segment_level": seg_metrics,
        "song_level":    song_metrics,
    }
    out_path = C.resnet_fold_results_path(fold)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fold_results, f, indent=2, ensure_ascii=False)
    print(f"\nSonuclar kaydedildi -> {out_path}")


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
