#3-kanal PNG + pretrained ResNet18

from pathlib import Path


#**************************************************************************
RAW_DIR = Path("/content/drive/MyDrive/makam_projesi1/makamlar")
PNG_3CH_DIR = Path("/content/cache_spec_3ch")            
FOLDS_FILE = Path("/content/folds_spec.json")            

RESNET_CKPT_DIR    = Path("/content/drive/MyDrive/makam_projesi1/checkpoints_resnet")
RESNET_RESULTS_DIR = Path("/content/drive/MyDrive/makam_projesi1/results_resnet")

DRIVE_DIR = "/content/drive/MyDrive/makam_projesi1"
#**********************************************************************


#K-fold
#*****************************************************************************
N_FOLDS = 5
SEED    = 42
#***********************************************************************


#ResNet egitim parametreleri
#******************************************************************
RESNET_BATCH_SIZE    = 32
RESNET_LR            = 1e-4
RESNET_MAX_EPOCHS    = 50
RESNET_WEIGHT_DECAY  = 5e-4
RESNET_LABEL_SMOOTH  = 0.05
RESNET_GRAD_CLIP     = 5.0
RESNET_NUM_WORKERS   = 2
RESNET_PATIENCE      = 7
RESNET_FREEZE_EARLY  = True
RESNET_DROPOUT       = 0.5
#*********************************************************


#SpecAugment
#***********************************************************************
FREQ_MASK_PARAM = 24
FREQ_MASK_N     = 2
TIME_MASK_PARAM = 50
TIME_MASK_N     = 2
#***************************************************************************


#ImageNet normalizasyon
#**********************************************************************
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
#**********************************************************************************


#Yardimci yollar
#************************************************************************
def resnet_ckpt_best_path(fold: int) -> Path:
    return RESNET_CKPT_DIR / f"resnet_3ch_fold{fold}_best.pt"

def resnet_ckpt_last_path(fold: int) -> Path:
    return RESNET_CKPT_DIR / f"resnet_3ch_fold{fold}_last.pt"

def resnet_fold_results_path(fold: int) -> Path:
    return RESNET_RESULTS_DIR / f"resnet_3ch_fold{fold}_results.json"
#*********************************************************************
