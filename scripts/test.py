# paste vào terminal, chạy từ thư mục project
import numpy as np
from PIL import Image
import glob

masks = sorted(glob.glob("data/cholec_raw/**/*_endo_color_mask.png", recursive=True))[:5]
for m in masks:
    img = np.array(Image.open(m).convert("RGB"))
    pixels = img.reshape(-1, 3)
    unique = np.unique(pixels, axis=0)
    print(f"\n{m}")
    print(f"  Unique colors ({len(unique)}):")
    for c in unique:
        count = np.sum(np.all(pixels == c, axis=1))
        pct = count / len(pixels) * 100
        if pct > 0.1:
            print(f"    RGB({c[0]:3d}, {c[1]:3d}, {c[2]:3d}) = {pct:.1f}%")