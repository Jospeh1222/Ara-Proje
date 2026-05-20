#Tum hiperparametreler bu dosyada toplandi, baska yerde deger hardcode edilmemeli
#Bir deneyi degistirmek istediginde sadece buraya bak

from pathlib import Path

#Klasor ve dosya yollari
#***************************************************
RAW_DIR     = Path("data/raw")           #Ham ses dosyalari (data/raw/<makam>/*.mp3)
CACHE_DIR   = Path("cache/features")     #Cikarilan oznitelikler (.npz dosyalari)
SPLITS_DIR  = Path("splits")             #Fold ve istatistik dosyalari
FOLDS_FILE  = SPLITS_DIR / "folds.json"  #K-fold bolunme bilgisi (train/val/test)
CKPT_DIR    = Path("checkpoints")        #Egitilen modeller
RESULTS_DIR = Path("results")            #Her foldun test sonuclari

#Drive yollari (sadece Colab'da kullanilir, lokalde gormezden gelinir)
DRIVE_DIR    = "/content/drive/MyDrive"
DRIVE_PREFIX = f"{DRIVE_DIR}/makam_kfold"
#**********************************************************************


#Ses ve segment parametreleri
#***************************************************************************
SAMPLE_RATE        = 22050    #22050 Hz mono, muzik MIR'inde standart
SEGMENT_SEC        = 30.0     #Her segment 30 saniye
SEGMENT_HOP_SEC    = 30.0     #30 = ortusmesiz; 15 yaparsan %50 ortusme
DROP_LAST_IF_SHORT = True     #30sn'den kisa kuyruk atilir
#********************************************************


#Oznitelik parametreleri
#****************************************************************
HOP_LENGTH        = 512
N_FFT             = 2048
N_MELS            = 128
N_MFCC            = 20
BINS_PER_SEMITONE = 3
N_CHROMA          = 12 * BINS_PER_SEMITONE   #36 bant
#*************************************************************************


#K-fold capraz dogrulama parametreleri
#********************************************************************************
N_FOLDS   = 5      #5 fold
VAL_RATIO = 0.2    #Her foldun train havuzunun %20'si validation'a ayrilir
SEED      = 42     #Tekrar uretilebilirlik icin sabit seed
#**************************************************************


#Model mimarisi parametreleri
#************************************************************
LSTM_HIDDEN  = 96
DROPOUT      = 0.4
CONV_DROPOUT = 0.1
ATTN_DIM     = 128
#*******************************************************


#Egitim parametreleri
#********************************************************************
BATCH_SIZE       = 32
MAX_EPOCHS       = 50     #MAKSIMUM epoch; erken durdurma daha once durdurabilir
LR               = 1e-3
WEIGHT_DECAY     = 2e-4
LABEL_SMOOTHING  = 0.05
GRAD_CLIP        = 5.0
NUM_WORKERS      = 2


EARLY_STOP_PATIENCE = 3
#**********************************************************


#SpecAugment parametreleri (sadece egitim sirasinda uygulanir)
#*****************************************************************************
FREQ_MASK_PARAM = 16
FREQ_MASK_N     = 1
TIME_MASK_PARAM = 40
TIME_MASK_N     = 1
#**************************************************************************


#Yardimci fonksiyonlar
#********************************************************************************
def total_freq_bins() -> int:
    #Modele giren toplam frekans bandi sayisi (mel + mfcc + chroma)
    return N_MELS + N_MFCC + N_CHROMA


def stats_path(fold: int) -> Path:
    #Her foldun normalizasyon istatistikleri ayri dosyada
    return SPLITS_DIR / f"stats_fold{fold}.json"


def ckpt_best_path(fold: int) -> Path:
    return CKPT_DIR / f"fold{fold}_best.pt"


def ckpt_last_path(fold: int) -> Path:
    return CKPT_DIR / f"fold{fold}_last.pt"


def drive_best_path(fold: int) -> str:
    return f"{DRIVE_PREFIX}_fold{fold}_best.pt"


def drive_last_path(fold: int) -> str:
    return f"{DRIVE_PREFIX}_fold{fold}_last.pt"


def fold_results_path(fold: int) -> Path:
    return RESULTS_DIR / f"fold{fold}_results.json"
#************************************************************