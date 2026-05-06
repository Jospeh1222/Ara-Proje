#Tüm hiperparametreler bu dosyada

from pathlib import Path

#Klasör ve dosya yolları
#********************************************************************
RAW_DIR     = Path("data/raw")           #Ham ses dosyalarının bulunduğu klasör (data/raw/<makam>/*.mp3)
CACHE_DIR   = Path("cache/features")     #Çıkarılan özniteliklerin saklandığı klasör (.npz dosyaları)
SPLITS_DIR  = Path("splits")             #Fold ve istatistik dosyalarının olduğu klasör
FOLDS_FILE  = SPLITS_DIR / "folds.json"  #K-fold bölünme bilgisi
CKPT_DIR    = Path("checkpoints")        #Eğitim sırasında kaydedilen modeller
RESULTS_DIR = Path("results")            #Her foldun test sonuçları

#Drive yolları (sadece Colab'da kullanılır lokalde görmezden gelinir)
DRIVE_DIR    = "/content/drive/MyDrive"
DRIVE_PREFIX = f"{DRIVE_DIR}/makam_kfold"
#**************************************************************


#Ses ve segment parametreleri
#***********************************************************
SAMPLE_RATE        = 22050    #22050 Hz mono müzik MIR'inde standart
SEGMENT_SEC        = 30.0     #Her segment 30 saniye
SEGMENT_HOP_SEC    = 30.0     #30 = örtüşmesiz 15 ile %50 örtüşme olur (2x veri ama yavaş)
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