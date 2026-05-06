#Ses dosyalarını 30 saniyelik segmentlere böler
#Her segment için mel spektrogram + MFCC + chroma çıkarır ve .npz olarak kaydeder
#Bozuk veya çok kısa dosyalar atlanır

from pathlib import Path
import warnings
import numpy as np
import librosa
from tqdm import tqdm

import config as C

#Chroma bazen sessiz kısımlarda "tuning estimate" uyarısı veriyor ignore et
warnings.filterwarnings("ignore", message=".*Trying to estimate tuning.*")

#ses formatları
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")

#Bir segmentteki örnek sayısı (30sn * 22050 Hz)
SEG_SAMPLES = int(C.SEGMENT_SEC * C.SAMPLE_RATE)
HOP_SAMPLES = int(C.SEGMENT_HOP_SEC * C.SAMPLE_RATE) #segmentler arası kayma


#Bir segmentin ses sinyalinden mel, mfcc ve chroma çıkarır
def extract_features(y: np.ndarray, sr: int):

    #Mel spektrogram tını bilgisini içerir (hangi enstrüman, ses karakteri)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH,
        n_mels=C.N_MELS, power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max) #güç değerleri dB'ye çevrilir

    #MFCC mel'den türetilirs
    mfcc = librosa.feature.mfcc(S=mel_db, sr=sr, n_mfcc=C.N_MFCC)

    #Chroma 36 banda
    chroma = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=C.HOP_LENGTH,
        n_chroma=C.N_CHROMA, bins_per_octave=C.N_CHROMA,
    )

    #Üç matrisin zaman ekseni 1 frame fark edebilir, eşitle
    T = min(mel_db.shape[1], mfcc.shape[1], chroma.shape[1])
    return (mel_db[:, :T].astype(np.float32),
            mfcc[:, :T].astype(np.float32),
            chroma[:, :T].astype(np.float32))


#Bir ses dosyasını segmentlere böler ve her segment için .npz dosyası kaydeder
#Dönüş: kaç tane segment yazıldığı
def process_file(audio_path: Path, out_dir: Path) -> int:

    try:
        #Tüm dosyalar 22050 Hz mono'ya çevrilerek yüklenir
        y, sr = librosa.load(audio_path, sr=C.SAMPLE_RATE, mono=True)
    except Exception as e:
        raise RuntimeError(f"yukleme hatasi: {e}")

    #Şarkı 30sn'den kısaysa hiç segment çıkmaz, atla
    if len(y) < SEG_SAMPLES:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem #dosya adı uzantısız
    written = 0

    #Şarkıyı 30sn'lik parçalara böl
    for i, start in enumerate(range(0, len(y) - SEG_SAMPLES + 1, HOP_SAMPLES)):

        seg = y[start:start + SEG_SAMPLES] #segmenti al
        mel_db, mfcc, chroma = extract_features(seg, sr) #öznitelikleri çıkar

        out_path = out_dir / f"{stem}_seg{i:02d}.npz"

        #Sıkıştırılmış formatta kaydet
        np.savez_compressed(
            out_path,
            mel=mel_db, mfcc=mfcc, chroma=chroma,
            song_id=stem, seg_idx=i,
        )
        written += 1

    #Eğer config'te tail (son segment) kaybedilmesin denmişse, son kalan kısa kuyruk da kaydedilsin
    if not C.DROP_LAST_IF_SHORT:
        last_start = (len(y) // HOP_SAMPLES) * HOP_SAMPLES
        if last_start + SEG_SAMPLES > len(y) and last_start < len(y):
            seg = y[last_start:]
            seg = np.pad(seg, (0, SEG_SAMPLES - len(seg))) #eksiği sıfırla doldur
            mel_db, mfcc, chroma = extract_features(seg, sr)
            out_path = out_dir / f"{stem}_seg{written:02d}.npz"
            np.savez_compressed(
                out_path,
                mel=mel_db, mfcc=mfcc, chroma=chroma,
                song_id=stem, seg_idx=written,
            )
            written += 1

    return written


#Ana akış: tüm sınıf klasörlerini gez, her ses dosyası için process_file çalıştır
def main():

    #Sınıf klasörlerini bul (data/raw/Hicaz, data/raw/Saba gibi)
    classes = sorted([d for d in C.RAW_DIR.iterdir() if d.is_dir()])
    if not classes:
        raise SystemExit(f"Sinif klasoru bulunamadi: {C.RAW_DIR}")

    print(f"{len(classes)} sinif bulundu")
    print(f"Ornekleme: {C.SAMPLE_RATE} Hz, segment: {C.SEGMENT_SEC}s, hop: {C.SEGMENT_HOP_SEC}s")
    print(f"Oznitelikler: mel{C.N_MELS} + mfcc{C.N_MFCC} + chroma{C.N_CHROMA}\n")

    total_segments = 0
    skipped = [] #atlanan dosyalar

    for cls_dir in classes:

        #Bu sınıftaki ses dosyalarını topla
        files = sorted([p for p in cls_dir.iterdir()
                        if p.suffix.lower() in AUDIO_EXTS])

        if not files:
            print(f"  ! {cls_dir.name}: ses dosyasi yok")
            continue

        out_cls = C.CACHE_DIR / cls_dir.name

        #Her dosyayı sırayla işle, hata varsa logla ama devam et
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

    #Atlanan dosyaları göster
    if skipped:
        print(f"\nAtlanan {len(skipped)} dosya:")
        for s in skipped[:30]:
            print(f"  - {s}")
        if len(skipped) > 30:
            print(f"  ... ve {len(skipped) - 30} daha")


if __name__ == "__main__":
    main()