#Ham ses dosyalarini 20 saniyelik segmentlere boler
#Her segment icin MEL spektrogram PNG olarak kaydeder

from pathlib import Path
import warnings
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import config as C

warnings.filterwarnings("ignore", message=".*Trying to estimate tuning.*")

AUDIO_EXTS  = (".mp3", ".wav", ".flac", ".m4a", ".ogg")
SEG_SAMPLES = int(C.SEGMENT_SEC * C.SAMPLE_RATE)
HOP_SAMPLES = int(C.SEGMENT_HOP_SEC * C.SAMPLE_RATE)

IMG_SIZE = 224


def save_melspectrogram_png(y: np.ndarray, sr: int, out_path: Path):
    #Mel spektrogram hesapla
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH,
        n_mels=C.N_MELS, power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    #Viridis colormap ile PNG kaydet, eksensiz
    fig, ax = plt.subplots(figsize=(IMG_SIZE / 100, IMG_SIZE / 100), dpi=100)
    librosa.display.specshow(
        mel_db, sr=sr, hop_length=C.HOP_LENGTH,
        x_axis=None, y_axis=None, cmap="viridis", ax=ax,
    )
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def process_file(audio_path: Path, out_dir: Path) -> int:
    try:
        y, sr = librosa.load(audio_path, sr=C.SAMPLE_RATE, mono=True)
    except Exception as e:
        raise RuntimeError(f"yukleme hatasi: {e}")

    if len(y) < SEG_SAMPLES:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    written = 0

    for i, start in enumerate(range(0, len(y) - SEG_SAMPLES + 1, HOP_SAMPLES)):
        out_path = out_dir / f"{stem}_seg{i:02d}.png"

        #Zaten varsa atla (resume guvenli)
        if out_path.exists():
            written += 1
            continue

        seg = y[start:start + SEG_SAMPLES]
        save_melspectrogram_png(seg, sr, out_path)
        written += 1

    if not C.DROP_LAST_IF_SHORT:
        last_start = (len(y) // HOP_SAMPLES) * HOP_SAMPLES
        if last_start + SEG_SAMPLES > len(y) and last_start < len(y):
            seg = np.pad(y[last_start:], (0, SEG_SAMPLES - len(y[last_start:])))
            out_path = out_dir / f"{stem}_seg{written:02d}.png"
            save_melspectrogram_png(seg, sr, out_path)
            written += 1

    return written


def main():
    classes = sorted([d for d in C.RAW_DIR.iterdir() if d.is_dir()])
    if not classes:
        raise SystemExit(f"Sinif klasoru bulunamadi: {C.RAW_DIR}")

    print(f"{len(classes)} sinif bulundu")
    print(f"Ornekleme: {C.SAMPLE_RATE} Hz | Segment: {C.SEGMENT_SEC}s")
    print(f"PNG: {IMG_SIZE}x{IMG_SIZE} viridis | Mel: {C.N_MELS} bant")
    print(f"Cikti: {C.MEL_PNG_CACHE_DIR}\n")

    total_segments = 0
    skipped = []

    for cls_dir in classes:
        files = sorted([p for p in cls_dir.iterdir()
                        if p.suffix.lower() in AUDIO_EXTS])
        if not files:
            print(f"  ! {cls_dir.name}: ses dosyasi yok")
            continue

        out_cls = C.MEL_PNG_CACHE_DIR / cls_dir.name
        for audio in tqdm(files, desc=cls_dir.name):
            try:
                n = process_file(audio, out_cls)
                if n == 0:
                    skipped.append(f"{audio.name} (cok kisa)")
                else:
                    total_segments += n
            except Exception as e:
                skipped.append(f"{audio.name}: {e}")

    print(f"\nToplam PNG yazildi: {total_segments}")
    if skipped:
        print(f"\nAtlanan {len(skipped)} dosya:")
        for s in skipped[:30]:
            print(f"  - {s}")
        if len(skipped) > 30:
            print(f"  ... ve {len(skipped) - 30} daha")


if __name__ == "__main__":
    main()