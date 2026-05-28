#Tum hiperparametreler ve yollar bu dosyada
#Ham veri Drive'da, oznitelikler de Drive'da, kod local Colab'da

from pathlib import Path


#DRIVE YOLLARI
#*****************************************************************************
#Ham ses dosyalari (kullanici buraya yukleyecek)
RAW_DIR = Path("/content/drive/MyDrive/data/raw")

#Cikti yollari (drive)
MEL_PNG_CACHE_DIR = Path("/content/drive/MyDrive/cache_mel_png")
SPLITS_DIR        = Path("/content/drive/MyDrive/splits")
CKPT_DIR          = Path("/content/drive/MyDrive/checkpoints")
RESULTS_DIR       = Path("/content/drive/MyDrive/results")

FOLDS_MEL_FILE = Path("/content/folds_mel_local.json")

DRIVE_DIR = "/content/drive/MyDrive"
#***********************************************************


#Ses ve segment parametreleri
#*************************************************************************
SAMPLE_RATE        = 22050
SEGMENT_SEC        = 20.0     #ekibin spec deneyiyle ayni
SEGMENT_HOP_SEC    = 20.0     #ortusmesiz
DROP_LAST_IF_SHORT = True
#*************************************************


#Oznitelik parametreleri
#*****************************************************************************
HOP_LENGTH = 512
N_FFT      = 2048
N_MELS     = 128
#***********************************************************************


#K-fold parametreleri
#**************************************************************************
N_FOLDS   = 5
VAL_RATIO = 0.15    
SEED      = 42      
#************************************************************


#Model parametreleri
#****************************************************************************
DROPOUT      = 0.4
CONV_DROPOUT = 0.1
#*********************************************************************


#Egitim parametreleri
#*************************************************************
BATCH_SIZE          = 32
MAX_EPOCHS          = 50
LR                  = 1e-3
WEIGHT_DECAY        = 2e-4
LABEL_SMOOTHING     = 0.05
GRAD_CLIP           = 5.0
NUM_WORKERS         = 2
EARLY_STOP_PATIENCE = 7
#************************************************************


#SpecAugment parametreleri
#*************************************************************
FREQ_MASK_PARAM = 16
FREQ_MASK_N     = 1
TIME_MASK_PARAM = 40
TIME_MASK_N     = 1
#*******************************************************


#Yardimci fonksiyonlar
#*************************************************************************
def stats_mel_path(fold: int) -> Path:
    return SPLITS_DIR / f"stats_mel_fold{fold}.json"

def ckpt_best_path(fold: int) -> Path:
    return CKPT_DIR / f"cnn_mel_fold{fold}_best.pt"

def ckpt_last_path(fold: int) -> Path:
    return CKPT_DIR / f"cnn_mel_fold{fold}_last.pt"

def fold_results_path(fold: int) -> Path:
    return RESULTS_DIR / f"cnn_mel_fold{fold}_results.json"
#***********************************************************************