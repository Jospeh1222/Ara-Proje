import numpy as np
import librosa
from PIL import Image


#Egitim ile ayni parametreler (config_cnn.py'den)
#************************************************************
SAMPLE_RATE        = 22050
SEGMENT_SEC        = 30.0     #30 saniye
SEGMENT_HOP_SEC    = 30.0     #ortusmesiz
DROP_LAST_IF_SHORT = True

HOP_LENGTH        = 512
N_FFT             = 2048
N_MELS            = 128
N_MFCC            = 20
BINS_PER_SEMITONE = 3
N_CHROMA          = 12 * BINS_PER_SEMITONE   # 36

IMG_SIZE = 224
#************************************************************


def _normalize_0_1(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _resize_channel(x: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    img = Image.fromarray((_normalize_0_1(x) * 255).astype(np.uint8))
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def extract_segment_features(y_segment: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y_segment, sr=sr,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mfcc = librosa.feature.mfcc(
        y=y_segment, sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    chroma = librosa.feature.chroma_cqt(
        y=y_segment, sr=sr,
        hop_length=HOP_LENGTH,
        n_chroma=N_CHROMA,
        bins_per_octave=N_CHROMA
    )

    r_channel = _resize_channel(mel_db)
    g_channel = _resize_channel(mfcc)
    b_channel = _resize_channel(chroma)

    rgb = np.stack([r_channel, g_channel, b_channel], axis=-1)
    return rgb


def split_audio_to_segments(y: np.ndarray, sr: int):
    seg_samples = int(SEGMENT_SEC * sr)
    hop_samples = int(SEGMENT_HOP_SEC * sr)

    if len(y) < seg_samples:
        return

    n_full = (len(y) - seg_samples) // hop_samples + 1

    for i in range(n_full):
        start = i * hop_samples
        end = start + seg_samples
        yield i, y[start:end]

    if not DROP_LAST_IF_SHORT:
        last_start = n_full * hop_samples
        if last_start < len(y):
            tail = y[last_start:]
            padded = np.pad(tail, (0, seg_samples - len(tail)))
            yield n_full, padded


def load_audio(file_path: str) -> tuple:
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return y, sr


def count_segments(audio_duration_sec: float) -> int:
    if audio_duration_sec < SEGMENT_SEC:
        return 0
    return int((audio_duration_sec - SEGMENT_SEC) // SEGMENT_HOP_SEC) + 1