%%writefile /content/config.py
from pathlib import Path

RAW_DIR    = Path("/content/drive/MyDrive/makam_projesi/makamlar")
CACHE_DIR  = Path("/content/cache_local")                          # lokal
SPLITS_DIR = Path("/content/drive/MyDrive/makam_projesi/splits_20sn")
FOLDS_FILE = Path("/content/drive/MyDrive/makam_projesi/splits1/folds_local.json")                     # lokal folds
CKPT_DIR   = Path("/content/checkpoints")
RESULTS_DIR= Path("/content/drive/MyDrive/makam_projesi/results")
DRIVE_DIR  = "/content/drive/MyDrive/makam_projesi"
DRIVE_PREFIX = "/content/drive/MyDrive/makam_projesi/checkpoints/fold"
SPEC_CACHE_DIR = Path("/content/drive/MyDrive/makam_projesi/cache_spec_png_20sn")
FOLDS_SPEC_FILE = Path("/content/folds_spec_local.json")
def stats_spec_path(fold: int) -> Path:
    return SPLITS_DIR / f"stats_spec_fold{fold}.json"

#Ses ve segment parametreleri
#***********************************************************
SAMPLE_RATE        = 22050    #22050 Hz mono müzik MIR'inde standart
SEGMENT_SEC        = 20.0     #Her segment 30 saniye
SEGMENT_HOP_SEC    = 20.0     #30 = örtüşmesiz 15 ile %50 örtüşme olur (2x veri ama yavaş)
DROP_LAST_IF_SHORT = True     #Şarkının sonunda 30sn'den kısa kalan kuyruk varsa atılır
#**************************************************************************


#Öznitelik parametreleri
#************************************************************************
HOP_LENGTH        = 512                       #STFT için pencere kayma adımı
N_FFT             = 2048                      #FFT pencere boyutu
N_MELS            = 128                       #Mel bandı sayısı
N_MFCC            = 20                        #MFCC katsayısı sayısı (mel'den türetiliyor)
BINS_PER_SEMITONE = 3                         #Yarım ton başına chroma bandı sayısı
N_CHROMA          = 12 * BINS_PER_SEMITONE    #Toplam chroma bandı = 36
#****************************************************************


#K-fold çapraz doğrulama parametreleri
#**********************************************************
N_FOLDS = 5    #5 fold, 10 daha hassas ama GPU süresi gerektirir
SEED    = 42   #Tekrar üretilebilirlik için sabit seed
VAL_RATIO = 0.15  # train setinin %15'i val'e ayrılır
#******************************************************************************


#Model mimarisi parametreleri
#*****************************************************************************
LSTM_HIDDEN  = 96    #BiLSTM gizli katman boyutu (çift yönlü olduğu için çıktı 192)
DROPOUT      = 0.4   #Sınıflandırıcıdan önce dropout
CONV_DROPOUT = 0.1   #CNN bloklarından sonra dropout
ATTN_DIM     = 128   #Attention mekanizmasının iç boyutu
#******************************************************


#Eğitim parametreleri
#*************************************************************************
BATCH_SIZE      = 32
EPOCHS          = 40     #Sabit epoch sayısı K-fold'da her fold aynı süre eğitilmeli
LR              = 1e-3   #Başlangıç öğrenme oranı
WEIGHT_DECAY    = 2e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP       = 5.0
NUM_WORKERS     = 2
MAX_EPOCHS      = 40
EARLY_STOP_PATIENCE = 7
#***************************************************************


#SpecAugment parametreleri
#************************************************************************
FREQ_MASK_PARAM = 16   #Frekans maskesi en fazla bu kadar bandı kapatır
FREQ_MASK_N     = 1    #Bir segmentte kaç adet frekans maskesi
TIME_MASK_PARAM = 40   #Zaman maskesi en fazla bu kadar frame'i kapatır
TIME_MASK_N     = 1    #Bir segmentte kaç adet zaman maskesi
#********************************************************************


#Yardımcı fonksiyonlar
#***********************************************************
def total_freq_bins() -> int:
    #Modele giren toplam frekans bandı sayısı (mel + mfcc + chroma)
    return N_MELS + N_MFCC + N_CHROMA


def stats_path(fold: int) -> Path:
    #Her foldun normalizasyon istatistikleri ayrı dosyada (test verisi sızmasın diye)
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
#*******************************************************
