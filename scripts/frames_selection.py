# Script tự động chọn 20 frames theo tỷ lệ trên
import glob, os
import numpy as np
from PIL import Image

print(os.getcwd())
raw_dir = "data/cholec_raw"
out_dir = "data/frames"
os.makedirs(out_dir, exist_ok=True)

# Load tất cả pairs (frame, mask)
frame_files = sorted(glob.glob(f"{raw_dir}/**/*_endo.png", recursive=True))
frame_files = [f for f in frame_files if not any(
    x in f for x in ["_mask", "_watershed", "_color"]
)]

def get_complexity(frame_path):
    mask_path = frame_path.replace("_endo.png", "_endo_color_mask.png")
    if not os.path.exists(mask_path):
        return "medium"
    mask = np.array(Image.open(mask_path).convert("RGB"))
    pixels = mask.reshape(-1, 3)
    unique = len(np.unique(pixels, axis=0))
    if unique <= 4:
        return "easy"
    elif unique <= 8:
        return "medium"
    else:
        return "hard"

# Phân loại
by_complexity = {"easy": [], "medium": [], "hard": []}
print("Classifying frames...")
for i, f in enumerate(frame_files):
    if i % 500 == 0:
        print(f"  {i}/{len(frame_files)}")
    c = get_complexity(f)
    by_complexity[c].append(f)

print(f"Easy: {len(by_complexity['easy'])}")
print(f"Medium: {len(by_complexity['medium'])}")
print(f"Hard: {len(by_complexity['hard'])}")

# Chọn ngẫu nhiên theo tỷ lệ
import random
random.seed(42)

selected = (
    random.sample(by_complexity["easy"], min(6, len(by_complexity["easy"]))) +
    random.sample(by_complexity["medium"], min(10, len(by_complexity["medium"]))) +
    random.sample(by_complexity["hard"], min(4, len(by_complexity["hard"])))
)

# Lưu và tạo metadata
import json
metadata = []
for i, src in enumerate(selected[:20]):
    complexity = get_complexity(src)
    dst = f"{out_dir}/frame_{i+1:02d}.png"
    
    from shutil import copy
    copy(src, dst)
    
    metadata.append({
        "frame_id": f"frame_{i+1:02d}",
        "source": src,
        "complexity": complexity,
        "suggested_question_types": {
            "easy": ["recognition", "workflow_phase"],
            "medium": ["anatomy_landmark", "safety_verification"],
            "hard": ["defer", "risk_pitfall"]
        }[complexity]
    })

with open("data/frames/frame_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n20 frames saved to data/frames/")
print("Metadata saved to data/frames/frame_metadata.json")