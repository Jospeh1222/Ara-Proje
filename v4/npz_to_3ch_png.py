%%writefile /content/npz_to_3ch_png.py

# NPZ'den gerçek 3 kanallı PNG üret
# R = mel, G = MFCC, B = chroma
# Her kanal kendi içinde normalize edilir, 224x224'e yeniden boyutlandırılır

from pathlib import Path
import numpy as np
from PIL import Image
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

NPZ_DIR = Path("/content/drive/MyDrive/makam_projesi/cache")
OUT_DIR = Path("/content/cache_spec_3ch")
SIZE = (224, 224)  # (H, W)


def normalize(x):
    # 0-1 aralığına çek
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def resize_channel(x, size):
    # x: (F, T) → PIL ile (H, W)'ye boyutlandır
    img = Image.fromarray((normalize(x) * 255).astype(np.uint8))
    img = img.resize((size[1], size[0]), Image.BILINEAR)  # (W, H)
    return np.array(img, dtype=np.uint8)


def process_file(npz_path):
    npz_path = Path(npz_path)
    rel = npz_path.relative_to(NPZ_DIR)
    out_path = OUT_DIR / rel.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        return

    try:
        z = np.load(npz_path)
        mel    = resize_channel(z["mel"],    SIZE)  # R
        mfcc   = resize_channel(z["mfcc"],   SIZE)  # G
        chroma = resize_channel(z["chroma"], SIZE)  # B

        rgb = np.stack([mel, mfcc, chroma], axis=-1)  # (H, W, 3)
        Image.fromarray(rgb, mode="RGB").save(out_path)
    except Exception as e:
        print(f"Hata {npz_path}: {e}")


OUT_DIR.mkdir(exist_ok=True)
files = glob.glob(str(NPZ_DIR / "**/*.npz"), recursive=True)
print(f"İşlenecek: {len(files)} NPZ")

with ThreadPoolExecutor(max_workers=8) as ex:
    list(tqdm(ex.map(process_file, files), total=len(files)))

print("Tamamlandı!")
n_out = len(glob.glob(str(OUT_DIR / "**/*.png"), recursive=True))
print(f"Üretilen PNG: {n_out}")
