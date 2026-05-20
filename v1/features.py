#Ses dosyalarini (MP3/WAV/FLAC/M4A) 30 saniyelik segmentlere boler
#Her segment icin mel spektrogram + MFCC + chroma cikarir ve .npz olarak kaydeder
#Bozuk veya cok kisa dosyalar atlanir, egitim cokmez


from pathlib import Path
import warnings
import numpy as np
import librosa
from tqdm import tqdm

import config as C

#Chroma bazen sessiz kisimlarda "tuning estimate" uyarisi veriyor, bizi etkilemiyor
warnings.filterwarnings("ignore", message=".*Trying to estimate tuning.*")

#Desteklenen ses formatlari
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")

#Bir segmentteki ornek sayisi (30sn * 22050 Hz)
SEG_SAMPLES = int(C.SEGMENT_SEC * C.SAMPLE_RATE)
HOP_SAMPLES = int(C.SEGMENT_HOP_SEC * C.SAMPLE_RATE)


#Bir segmentin ses sinyalinden mel, mfcc ve chroma cikarir
def extract_features(y: np.ndarray, sr: int):

    #Mel spektrogram tini bilgisini icerir
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH,
        n_mels=C.N_MELS, power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    #MFCC mel'den turetilir
    mfcc = librosa.feature.mfcc(S=mel_db, sr=sr, n_mfcc=C.N_MFCC)

    #Chroma - bant sayisi config'ten gelir (36 veya 53)
    chroma = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=C.HOP_LENGTH,
        n_chroma=C.N_CHROMA, bins_per_octave=C.N_CHROMA,
    )

    #Uc matrisin zaman ekseni 1 frame fark edebilir, esitle
    T = min(mel_db.shape[1], mfcc.shape[1], chroma.shape[1])
    return (mel_db[:, :T].astype(np.float32),
            mfcc[:, :T].astype(np.float32),
            chroma[:, :T].astype(np.float32))


#Bir ses dosyasini segmentlere boler ve her segment icin .npz kaydeder
def process_file(audio_path: Path, out_dir: Path) -> int:

    try:
        y, sr = librosa.load(audio_path, sr=C.SAMPLE_RATE, mono=True)
    except Exception as e:
        raise RuntimeError(f"yukleme hatasi: {e}")

    #Sarki 30sn'den kisaysa hic segment cikmaz, atla
    if len(y) < SEG_SAMPLES:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    written = 0

    #Sarkiyi 30sn'lik parcalara bol
    for i, start in enumerate(range(0, len(y) - SEG_SAMPLES + 1, HOP_SAMPLES)):
        seg = y[start:start + SEG_SAMPLES]
        mel_db, mfcc, chroma = extract_features(seg, sr)
        out_path = out_dir / f"{stem}_seg{i:02d}.npz"
        np.savez_compressed(
            out_path,
            mel=mel_db, mfcc=mfcc, chroma=chroma,
            song_id=stem, seg_idx=i,
        )
        written += 1

    #Config'te tail kaybedilmesin denmisse son kisa kuyruk da kaydedilir
    if not C.DROP_LAST_IF_SHORT:
        last_start = (len(y) // HOP_SAMPLES) * HOP_SAMPLES
        if last_start + SEG_SAMPLES > len(y) and last_start < len(y):
            seg = y[last_start:]
            seg = np.pad(seg, (0, SEG_SAMPLES - len(seg)))
            mel_db, mfcc, chroma = extract_features(seg, sr)
            out_path = out_dir / f"{stem}_seg{written:02d}.npz"
            np.savez_compressed(
                out_path,
                mel=mel_db, mfcc=mfcc, chroma=chroma,
                song_id=stem, seg_idx=written,
            )
            written += 1

    return written


#Ana akis: tum sinif klasorlerini gez, her ses dosyasi icin process_file calistir
def main():

    classes = sorted([d for d in C.RAW_DIR.iterdir() if d.is_dir()])
    if not classes:
        raise SystemExit(f"Sinif klasoru bulunamadi: {C.RAW_DIR}")

    print(f"{len(classes)} sinif bulundu")
    print(f"Ornekleme: {C.SAMPLE_RATE} Hz, segment: {C.SEGMENT_SEC}s, hop: {C.SEGMENT_HOP_SEC}s")
    print(f"Oznitelikler: mel{C.N_MELS} + mfcc{C.N_MFCC} + chroma{C.N_CHROMA}\n")

    total_segments = 0
    skipped = []

    for cls_dir in classes:
        files = sorted([p for p in cls_dir.iterdir()
                        if p.suffix.lower() in AUDIO_EXTS])
        if not files:
            print(f"  ! {cls_dir.name}: ses dosyasi yok")
            continue

        out_cls = C.CACHE_DIR / cls_dir.name
        for audio in tqdm(files, desc=cls_dir.name):
            try:
                n = process_file(audio, out_cls)
                if n == 0:
                    skipped.append(f"{audio.name} (cok kisa)")
                else:
                    total_segments += n
            except Exception as e:
                skipped.append(f"{audio.name}: {e}")

    print(f"\nToplam segment yazildi: {total_segments}")
    if skipped:
        print(f"\nAtlanan {len(skipped)} dosya:")
        for s in skipped[:30]:
            print(f"  - {s}")
        if len(skipped) > 30:
            print(f"  ... ve {len(skipped) - 30} daha")


if __name__ == "__main__":
    main()