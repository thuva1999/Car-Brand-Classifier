import random
import scipy.io as sio
from pathlib import Path
from PIL import Image

root = Path(r"C:\Users\i7\Downloads\CompCars\data")
out = Path(r"C:\Users\i7\Documents\Car-Brand-Classifier\data\brand_dataset")
MAX_PER_BRAND = 80
MIN_SIDE = 120

# CompCars name (as printed in makes.txt) -> project folder name
BRANDS = {
    "BYD": "BYD",
    "Chrey": "Chery",
    "Haval": "Haval",
    "Changan": "Changan",
    "Geely": "Geely",
    "HongQi": "Hongqi",
    "Jianghuai": "JAC",
}

m = sio.loadmat(root / "misc" / "make_model_name.mat")
name_to_id = {str(x[0]).lower(): i for i, x in enumerate(m["make_names"].ravel(), start=1) if x.size}

random.seed(0)
for src_name, dst_name in BRANDS.items():
    make_id = name_to_id.get(src_name.lower())
    if make_id is None:
        print(f"NOT FOUND: {src_name}")
        continue
    files = list((root / "image" / str(make_id)).rglob("*.jpg"))
    random.shuffle(files)
    dst = out / dst_name
    dst.mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in files:
        if saved >= MAX_PER_BRAND:
            break
        lab = root / "label" / f.relative_to(root / "image").with_suffix(".txt")
        try:
            x1, y1, x2, y2 = map(int, lab.read_text().split("\n")[2].split())
            img = Image.open(f).convert("RGB").crop((x1, y1, x2, y2))
        except Exception:
            continue
        if min(img.size) < MIN_SIDE:
            continue
        img.save(dst / f"compcars_{dst_name}_{saved:03d}.jpg", quality=92)
        saved += 1
    print(f"{dst_name}: {saved} crops")