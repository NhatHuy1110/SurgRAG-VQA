"""
frames_selection_v3.py — Smart 100-frame selector for SurgRAG-VQA
=================================================================
FIX from previous version:
  ROOT CAUSE: Color mask RGB values were guessed wrong → only 1-2 classes
  detected per frame → everything classified "easy" → only 43/100 frames.

  SOLUTION: Use **watershed mask** instead of color mask.
  Paper states: "each annotated pixel has the same class ID value for
  three color channels" — so class 2 (liver) → pixel RGB(2,2,2).
  This is deterministic and requires NO color guessing.

Improvements over v2:
  1. Watershed-based class parsing: reliable class detection via class IDs.
  2. Anatomy-driven question type scoring using actual class presence.
  3. Improved defer logic with ambiguous anatomy detection.
  4. Near-duplicate rejection via perceptual hash.
  5. Post-selection validation with sanity checks.
  6. Copies both frame + color mask for downstream use.

Requires: numpy, Pillow
Usage:    python scripts/frames_selection_v3.py
"""

import glob
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from shutil import copy2
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

# ════════════════════════════════════════════════════════════════════
# 0. CONFIG
# ════════════════════════════════════════════════════════════════════

RAW_DIR = Path("data/cholec_raw")
OUT_DIR = Path("data/frames_v3")

TOTAL_FRAMES = 100
RANDOM_SEED = 42
MAX_PER_VIDEO = 8
MIN_GAP_SAME_VIDEO = 50
RELAXED_GAP = 20
PHASH_HAMMING_THRESHOLD = 8

# ── CholecSeg8k class IDs (from paper Table I) ──
# In watershed mask: pixel value == class ID on all 3 channels
# e.g. class 2 (liver) → pixel RGB(2, 2, 2)
CLASS_ID_TO_NAME = {
    0:  "black_background",
    1:  "abdominal_wall",
    2:  "liver",
    3:  "gastrointestinal_tract",
    4:  "fat",
    5:  "grasper",
    6:  "connective_tissue",
    7:  "blood",
    8:  "cystic_duct",
    9:  "l_hook_electrocautery",
    10: "gallbladder",
    11: "hepatic_vein",
    12: "liver_ligament",
}

# Classes grouped by surgical role
ANATOMY_CLASSES = {"liver", "gallbladder", "cystic_duct", "hepatic_vein",
                   "gastrointestinal_tract", "liver_ligament"}
TISSUE_CLASSES  = {"fat", "connective_tissue", "abdominal_wall"}
TOOL_CLASSES    = {"grasper", "l_hook_electrocautery"}
RISK_CLASSES    = {"blood", "hepatic_vein", "cystic_duct"}

# ── Targets ──
QTYPE_TARGETS = {
    "recognition": 15,
    "workflow_phase": 20,
    "anatomy_landmark": 25,
    "safety_verification": 25,
    "risk_pitfall": 15,
}
DIFFICULTY_TARGETS = {"easy": 30, "medium": 40, "hard": 30}

BUCKET_TARGETS = {
    ("easy", "recognition"): 12,
    ("easy", "workflow_phase"): 10,
    ("easy", "anatomy_landmark"): 4,
    ("easy", "safety_verification"): 4,
    ("easy", "risk_pitfall"): 0,
    ("medium", "recognition"): 3,
    ("medium", "workflow_phase"): 7,
    ("medium", "anatomy_landmark"): 15,
    ("medium", "safety_verification"): 12,
    ("medium", "risk_pitfall"): 3,
    ("hard", "recognition"): 0,
    ("hard", "workflow_phase"): 3,
    ("hard", "anatomy_landmark"): 6,
    ("hard", "safety_verification"): 9,
    ("hard", "risk_pitfall"): 12,
}

DEFER_TARGET = 25
QUESTION_TYPES = list(QTYPE_TARGETS.keys())
DIFFICULTIES = ["easy", "medium", "hard"]

np.random.seed(RANDOM_SEED)

# ════════════════════════════════════════════════════════════════════
# 1. UTILITY HELPERS
# ════════════════════════════════════════════════════════════════════

def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def norm(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return clip01((x - lo) / (hi - lo))

def inv_norm(x: float, lo: float, hi: float) -> float:
    return 1.0 - norm(x, lo, hi)

def extract_video_id(frame_path: Path) -> str:
    for part in reversed(frame_path.parts):
        if re.fullmatch(r"video\d+", part.lower()):
            return part
    return frame_path.parent.name

def extract_frame_index(frame_path: Path) -> int:
    nums = re.findall(r"\d+", frame_path.stem)
    return int(nums[-1]) if nums else -1


def find_watershed_mask(frame_path: Path) -> Optional[Path]:
    """Find the watershed mask (class-ID encoded) for a frame."""
    candidates = [
        Path(str(frame_path).replace("_endo.png", "_endo_watershed_mask.png")),
        Path(str(frame_path).replace("_endo.png", "_watershed_mask.png")),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_color_mask(frame_path: Path) -> Optional[Path]:
    """Find the color mask (for copying to output, visualization)."""
    candidates = [
        Path(str(frame_path).replace("_endo.png", "_endo_color_mask.png")),
        Path(str(frame_path).replace("_endo.png", "_color_mask.png")),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


# ════════════════════════════════════════════════════════════════════
# 2. PERCEPTUAL HASH — near-duplicate detection
# ════════════════════════════════════════════════════════════════════

def average_hash(img_rgb: np.ndarray, hash_size: int = 16) -> int:
    pil = Image.fromarray(img_rgb).convert("L").resize(
        (hash_size, hash_size), Image.BILINEAR
    )
    arr = np.array(pil, dtype=np.float32)
    bits = (arr > arr.mean()).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h

def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


# ════════════════════════════════════════════════════════════════════
# 3. WATERSHED MASK PARSING — reliable class detection
# ════════════════════════════════════════════════════════════════════

def parse_watershed_classes(watershed_path: Path) -> Dict[str, float]:
    """
    Parse watershed mask where pixel value == class ID on all channels.
    Returns {class_name: pixel_fraction} for each detected class.
    """
    mask = np.array(Image.open(watershed_path).convert("L"))
    total = mask.size

    class_fractions = {}
    unique_vals, counts = np.unique(mask, return_counts=True)

    for val, count in zip(unique_vals, counts):
        val = int(val)
        if val in CLASS_ID_TO_NAME:
            name = CLASS_ID_TO_NAME[val]
            class_fractions[name] = round(count / total, 5)

    return class_fractions


def semantic_features(class_fractions: Dict[str, float]) -> Dict[str, float]:
    """Derive high-level surgical features from class fractions."""
    anatomy_area = sum(class_fractions.get(c, 0) for c in ANATOMY_CLASSES)
    tissue_area  = sum(class_fractions.get(c, 0) for c in TISSUE_CLASSES)
    tool_area    = sum(class_fractions.get(c, 0) for c in TOOL_CLASSES)
    risk_area    = sum(class_fractions.get(c, 0) for c in RISK_CLASSES)
    bg_area      = class_fractions.get("black_background", 0)

    n_anatomy = sum(1 for c in ANATOMY_CLASSES if class_fractions.get(c, 0) > 0.003)
    n_tools   = sum(1 for c in TOOL_CLASSES if class_fractions.get(c, 0) > 0.001)
    n_total   = sum(1 for c, v in class_fractions.items()
                    if v > 0.003 and c != "black_background")

    has_gallbladder  = class_fractions.get("gallbladder", 0) > 0.005
    has_liver        = class_fractions.get("liver", 0) > 0.005
    has_cystic_duct  = class_fractions.get("cystic_duct", 0) > 0.002
    has_blood        = class_fractions.get("blood", 0) > 0.003
    has_hepatic_vein = class_fractions.get("hepatic_vein", 0) > 0.002

    cvs_relevant = has_gallbladder and has_liver and has_cystic_duct

    return {
        "anatomy_area": round(anatomy_area, 5),
        "tissue_area": round(tissue_area, 5),
        "tool_area": round(tool_area, 5),
        "risk_area": round(risk_area, 5),
        "bg_area": round(bg_area, 5),
        "n_anatomy": n_anatomy,
        "n_tools": n_tools,
        "n_classes": n_total,
        "has_gallbladder": has_gallbladder,
        "has_liver": has_liver,
        "has_cystic_duct": has_cystic_duct,
        "has_blood": has_blood,
        "has_hepatic_vein": has_hepatic_vein,
        "cvs_relevant": cvs_relevant,
    }


# ════════════════════════════════════════════════════════════════════
# 4. IMAGE QUALITY STATS
# ════════════════════════════════════════════════════════════════════

def image_stats(img_rgb: np.ndarray) -> Dict[str, float]:
    gray = img_rgb.mean(axis=2).astype(np.float32)
    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)

    brightness = float(gray.mean())
    contrast = float(gray.std())

    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx**2 + gy**2)
    sharpness = float((grad_mag**2).mean())
    edge_density = float((grad_mag > 20).mean())

    maxc = img_rgb.max(axis=2).astype(np.float32)
    minc = img_rgb.min(axis=2).astype(np.float32)
    saturation = float(((maxc - minc) / np.maximum(maxc, 1.0)).mean())

    bright_fraction = float((gray > 220).mean())
    dark_fraction = float((gray < 35).mean())
    specular_fraction = float(((r > 240) & (g > 240) & (b > 240)).mean())
    red_dominance = float((r - 0.5 * (g + b)).mean())

    exposure_center = 125.0
    exposure_score = max(0.0, 1.0 - abs(brightness - exposure_center) / exposure_center)
    quality_score = (
        12.0 * exposure_score
        + 0.018 * contrast
        + 0.0030 * sharpness
        + 18.0 * edge_density
    )

    return {
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "sharpness": round(sharpness, 4),
        "edge_density": round(edge_density, 4),
        "saturation": round(saturation, 4),
        "bright_fraction": round(bright_fraction, 4),
        "dark_fraction": round(dark_fraction, 4),
        "specular_fraction": round(specular_fraction, 4),
        "red_dominance": round(red_dominance, 4),
        "quality_score": round(float(quality_score), 4),
    }


# ════════════════════════════════════════════════════════════════════
# 5. DIFFICULTY INFERENCE
# ════════════════════════════════════════════════════════════════════

def infer_difficulty(img_stats: Dict, sem_feats: Dict) -> str:
    q = img_stats["quality_score"]
    bright_f = img_stats["bright_fraction"]
    dark_f = img_stats["dark_fraction"]
    specular = img_stats["specular_fraction"]
    n_classes = sem_feats["n_classes"]
    has_blood = sem_feats["has_blood"]
    risk_area = sem_feats["risk_area"]

    hard_conditions = [
        has_blood and risk_area > 0.03 and n_classes >= 5,
        n_classes >= 8,
        sem_feats["cvs_relevant"] and has_blood,
        specular > 0.08 and n_classes >= 6,
        bright_f > 0.15 and dark_f > 0.15 and n_classes >= 5,
        has_blood and n_classes >= 6,
    ]

    easy_conditions = [
        n_classes <= 3 and q >= 12.0,
        n_classes <= 2 and sem_feats["anatomy_area"] < 0.25,
        bright_f > 0.25 and n_classes <= 3,
        dark_f > 0.50 and n_classes <= 3,
    ]

    if any(hard_conditions):
        return "hard"
    if any(easy_conditions):
        return "easy"
    return "medium"


# ════════════════════════════════════════════════════════════════════
# 6. QUESTION TYPE SCORING
# ════════════════════════════════════════════════════════════════════

def infer_question_type_scores(
    img_stats: Dict, sem_feats: Dict, difficulty: str
) -> Dict[str, float]:
    q = norm(img_stats["quality_score"], 8, 24)
    bright_f = img_stats["bright_fraction"]
    dark_f = img_stats["dark_fraction"]
    contrast = img_stats["contrast"]
    sharpness = img_stats["sharpness"]
    specular = img_stats["specular_fraction"]

    n_anatomy = sem_feats["n_anatomy"]
    n_tools = sem_feats["n_tools"]
    n_classes = sem_feats["n_classes"]
    anatomy_area = sem_feats["anatomy_area"]
    tool_area = sem_feats["tool_area"]
    risk_area = sem_feats["risk_area"]
    has_blood = sem_feats["has_blood"]
    cvs = sem_feats["cvs_relevant"]

    clear_vis = (
        0.35 * inv_norm(bright_f, 0.10, 0.35)
        + 0.25 * inv_norm(dark_f, 0.15, 0.50)
        + 0.20 * norm(contrast, 15, 45)
        + 0.20 * norm(sharpness, 40, 500)
    )
    vis_issue = (
        0.45 * norm(bright_f, 0.10, 0.40)
        + 0.25 * norm(dark_f, 0.20, 0.55)
        + 0.15 * inv_norm(contrast, 10, 25)
        + 0.15 * norm(specular, 0.03, 0.12)
    )

    recognition = (
        0.25 * q
        + 0.25 * clear_vis
        + 0.20 * norm(tool_area, 0.005, 0.10)
        + 0.15 * inv_norm(n_classes, 3, 8)
        + 0.15 * (1.0 if n_tools >= 1 else 0.3)
    )

    workflow_phase = (
        0.25 * norm(anatomy_area, 0.10, 0.60)
        + 0.20 * vis_issue
        + 0.20 * inv_norm(n_classes, 5, 10)
        + 0.20 * norm(float(n_tools >= 1), 0, 1)
        + 0.15 * norm(img_stats["brightness"], 50, 210)
    )

    anatomy_landmark = (
        0.30 * norm(n_anatomy, 2, 5)
        + 0.25 * norm(anatomy_area, 0.15, 0.55)
        + 0.20 * clear_vis
        + 0.15 * q
        + 0.10 * (0.8 if sem_feats["has_gallbladder"] and sem_feats["has_liver"] else 0.2)
    )

    safety_verification = (
        0.25 * (1.0 if cvs else 0.15)
        + 0.20 * norm(n_anatomy, 2, 5)
        + 0.20 * norm(tool_area, 0.003, 0.08)
        + 0.15 * clear_vis
        + 0.10 * norm(anatomy_area, 0.20, 0.60)
        + 0.10 * (0.7 if sem_feats["has_cystic_duct"] else 0.1)
    )

    risk_pitfall = (
        0.25 * norm(risk_area, 0.005, 0.08)
        + 0.20 * (1.0 if has_blood else 0.1)
        + 0.20 * vis_issue
        + 0.15 * norm(n_classes, 5, 10)
        + 0.10 * inv_norm(q, 6, 18)
        + 0.10 * norm(specular, 0.02, 0.12)
    )

    if difficulty == "easy":
        recognition += 0.06
        workflow_phase += 0.06
    elif difficulty == "medium":
        anatomy_landmark += 0.06
        safety_verification += 0.06
    else:
        risk_pitfall += 0.08
        safety_verification += 0.04

    return {
        "recognition": round(clip01(recognition), 4),
        "workflow_phase": round(clip01(workflow_phase), 4),
        "anatomy_landmark": round(clip01(anatomy_landmark), 4),
        "safety_verification": round(clip01(safety_verification), 4),
        "risk_pitfall": round(clip01(risk_pitfall), 4),
    }


# ════════════════════════════════════════════════════════════════════
# 7. DEFER SCORE
# ════════════════════════════════════════════════════════════════════

def infer_defer_score(
    img_stats: Dict, sem_feats: Dict, difficulty: str, q_scores: Dict[str, float]
) -> float:
    vis_issue = (
        0.50 * norm(img_stats["bright_fraction"], 0.10, 0.40)
        + 0.20 * norm(img_stats["dark_fraction"], 0.20, 0.55)
        + 0.15 * inv_norm(img_stats["contrast"], 10, 25)
        + 0.15 * norm(img_stats["specular_fraction"], 0.03, 0.12)
    )
    risk_obscure = norm(sem_feats["risk_area"], 0.01, 0.10)

    ambiguous_anatomy = 0.0
    if sem_feats["n_anatomy"] >= 3 and sem_feats["anatomy_area"] < 0.20:
        ambiguous_anatomy = 0.6
    elif sem_feats["n_classes"] >= 6 and sem_feats["anatomy_area"] < 0.25:
        ambiguous_anatomy = 0.4

    poor_quality = inv_norm(img_stats["quality_score"], 10, 24)

    score = (
        0.30 * vis_issue
        + 0.25 * risk_obscure
        + 0.20 * ambiguous_anatomy
        + 0.15 * poor_quality
        + 0.10 * q_scores["risk_pitfall"]
    )

    if difficulty == "hard":
        score += 0.08
    if sem_feats["has_blood"] and vis_issue > 0.40:
        score += 0.06

    return round(clip01(score), 4)


# ════════════════════════════════════════════════════════════════════
# 8. BUILD CANDIDATES
# ════════════════════════════════════════════════════════════════════

def build_candidates() -> List[Dict]:
    frame_files = sorted(glob.glob(str(RAW_DIR / "**/*_endo.png"), recursive=True))
    frame_files = [
        Path(f) for f in frame_files
        if not any(x in f for x in ["_mask", "_watershed", "_color"])
    ]

    print(f"[INFO] Found {len(frame_files)} endoscopic frames")
    if len(frame_files) == 0:
        raise FileNotFoundError(f"No *_endo.png frames in {RAW_DIR}")

    ws_found = sum(1 for f in frame_files if find_watershed_mask(f) is not None)
    cm_found = sum(1 for f in frame_files if find_color_mask(f) is not None)
    print(f"[INFO] Watershed masks: {ws_found}/{len(frame_files)}")
    print(f"[INFO] Color masks: {cm_found}/{len(frame_files)}")

    candidates: List[Dict] = []
    for i, frame_path in enumerate(frame_files):
        if i % 500 == 0:
            print(f"  processing {i}/{len(frame_files)}")

        ws_mask_path = find_watershed_mask(frame_path)
        color_mask_path = find_color_mask(frame_path)
        rgb = load_rgb(frame_path)
        istats = image_stats(rgb)
        phash = average_hash(rgb)

        if ws_mask_path is not None:
            class_fracs = parse_watershed_classes(ws_mask_path)
        else:
            class_fracs = {}

        sem = semantic_features(class_fracs)
        difficulty = infer_difficulty(istats, sem)
        q_scores = infer_question_type_scores(istats, sem, difficulty)
        defer_score = infer_defer_score(istats, sem, difficulty, q_scores)
        q_sorted = sorted(q_scores.items(), key=lambda kv: kv[1], reverse=True)

        candidates.append({
            "source": str(frame_path),
            "video_id": extract_video_id(frame_path),
            "frame_index": extract_frame_index(frame_path),
            "watershed_mask_path": str(ws_mask_path) if ws_mask_path else None,
            "color_mask_path": str(color_mask_path) if color_mask_path else None,
            "has_mask": ws_mask_path is not None,
            "phash": phash,
            "difficulty": difficulty,
            "primary_question_type": q_sorted[0][0],
            "question_type_scores": q_scores,
            "question_type_ranking": [k for k, _ in q_sorted],
            "defer_score": defer_score,
            "suggested_should_defer": defer_score >= 0.50,
            "classes_detected": {
                k: v for k, v in class_fracs.items()
                if v > 0.003 and k != "black_background"
            },
            "n_classes": sem["n_classes"],
            "n_anatomy": sem["n_anatomy"],
            "cvs_relevant": sem["cvs_relevant"],
            "has_blood": sem["has_blood"],
            **istats,
            "anatomy_area": sem["anatomy_area"],
            "tissue_area": sem["tissue_area"],
            "tool_area": sem["tool_area"],
            "risk_area": sem["risk_area"],
        })

    return candidates


def print_candidate_stats(candidates: List[Dict]) -> None:
    q_counter = Counter(c["primary_question_type"] for c in candidates)
    d_counter = Counter(c["difficulty"] for c in candidates)
    defer_counter = sum(1 for c in candidates if c["suggested_should_defer"])
    mask_counter = sum(1 for c in candidates if c["has_mask"])

    all_classes = Counter()
    for c in candidates:
        for cls in c["classes_detected"]:
            all_classes[cls] += 1

    print(f"\n[INFO] Candidates: {len(candidates)} total, {mask_counter} with masks")
    print("[INFO] Primary question type distribution:")
    for k in QUESTION_TYPES:
        print(f"  - {k}: {q_counter.get(k, 0)}")
    print("[INFO] Difficulty distribution:")
    for k in DIFFICULTIES:
        print(f"  - {k}: {d_counter.get(k, 0)}")
    print(f"[INFO] Defer suggestions: {defer_counter}/{len(candidates)}")
    print("[INFO] Class detection frequency (frames containing class):")
    for cls, count in sorted(all_classes.items(), key=lambda x: -x[1]):
        print(f"  - {cls}: {count} frames ({100*count/len(candidates):.1f}%)")


# ════════════════════════════════════════════════════════════════════
# 9. SELECTION ENGINE
# ════════════════════════════════════════════════════════════════════

def candidate_rank_for_qtype(item: Dict, qtype: str) -> int:
    try:
        return item["question_type_ranking"].index(qtype)
    except ValueError:
        return 999

def bucket_priority(item: Dict, qtype: str) -> Tuple[float, float, float, int]:
    score = item["question_type_scores"][qtype]
    defer_bonus = item["defer_score"] if qtype in {
        "workflow_phase", "safety_verification", "risk_pitfall"
    } else 0.0
    return (score, defer_bonus, item["quality_score"], item["frame_index"])

def is_near_duplicate(item: Dict, selected_hashes: List[int]) -> bool:
    for h in selected_hashes:
        if hamming_distance(item["phash"], h) < PHASH_HAMMING_THRESHOLD:
            return True
    return False

def can_take(item, used, selected_indices_by_video, count_by_video,
             selected_hashes, min_gap):
    if item["source"] in used:
        return False
    if count_by_video[item["video_id"]] >= MAX_PER_VIDEO:
        return False
    idx = item["frame_index"]
    for prev in selected_indices_by_video[item["video_id"]]:
        if abs(idx - prev) < min_gap:
            return False
    if is_near_duplicate(item, selected_hashes):
        return False
    return True

def clone_with_assignment(item, difficulty, qtype):
    assigned = dict(item)
    assigned["assigned_question_type"] = qtype
    assigned["assigned_difficulty"] = difficulty
    return assigned

def add_selected_item(item, difficulty, qtype, selected, used,
                      selected_indices_by_video, count_by_video, selected_hashes):
    assigned = clone_with_assignment(item, difficulty, qtype)
    used.add(assigned["source"])
    selected.append(assigned)
    count_by_video[assigned["video_id"]] += 1
    selected_indices_by_video[assigned["video_id"]].append(assigned["frame_index"])
    selected_hashes.append(assigned["phash"])
    return assigned

def select_for_bucket(candidates, selected, used, selected_indices_by_video,
                      count_by_video, selected_hashes, bucket, target_n):
    difficulty, qtype = bucket
    chosen = []
    phases = [
        {"exact_diff": True,  "max_rank": 0, "min_gap": MIN_GAP_SAME_VIDEO},
        {"exact_diff": True,  "max_rank": 1, "min_gap": MIN_GAP_SAME_VIDEO},
        {"exact_diff": True,  "max_rank": 2, "min_gap": RELAXED_GAP},
        {"exact_diff": True,  "max_rank": 4, "min_gap": RELAXED_GAP},
        {"exact_diff": False, "max_rank": 1, "min_gap": RELAXED_GAP},
        {"exact_diff": False, "max_rank": 4, "min_gap": 0},
    ]

    for phase in phases:
        if len(chosen) >= target_n:
            break
        pool = []
        for item in candidates:
            if phase["exact_diff"] and item["difficulty"] != difficulty:
                continue
            if candidate_rank_for_qtype(item, qtype) > phase["max_rank"]:
                continue
            if item["question_type_scores"][qtype] < 0.32:
                continue
            if not can_take(item, used, selected_indices_by_video,
                           count_by_video, selected_hashes, phase["min_gap"]):
                continue
            pool.append(item)

        pool.sort(key=lambda x: bucket_priority(x, qtype), reverse=True)
        for item in pool:
            if len(chosen) >= target_n:
                break
            if not can_take(item, used, selected_indices_by_video,
                           count_by_video, selected_hashes, phase["min_gap"]):
                continue
            chosen.append(
                add_selected_item(item, difficulty, qtype, selected, used,
                                  selected_indices_by_video, count_by_video,
                                  selected_hashes)
            )
    return chosen

def bucket_counts(selected):
    return Counter(
        (x["assigned_difficulty"], x["assigned_question_type"]) for x in selected
    )

def fill_bucket_deficits(candidates, selected, used, selected_indices_by_video,
                         count_by_video, selected_hashes):
    phases = [
        {"exact_diff": True,  "max_rank": 1, "min_gap": MIN_GAP_SAME_VIDEO, "min_score": 0.32},
        {"exact_diff": True,  "max_rank": 2, "min_gap": RELAXED_GAP,        "min_score": 0.30},
        {"exact_diff": True,  "max_rank": 4, "min_gap": RELAXED_GAP,        "min_score": 0.28},
        {"exact_diff": False, "max_rank": 2, "min_gap": RELAXED_GAP,        "min_score": 0.30},
        {"exact_diff": False, "max_rank": 4, "min_gap": 0,                  "min_score": 0.25},
    ]

    for phase in phases:
        counts = bucket_counts(selected)
        deficits = [
            (bucket, target - counts.get(bucket, 0))
            for bucket, target in BUCKET_TARGETS.items()
            if target > counts.get(bucket, 0)
        ]
        if not deficits:
            break
        deficits.sort(key=lambda kv: (DIFFICULTIES.index(kv[0][0]), -kv[1]))
        for bucket, missing in deficits:
            difficulty, qtype = bucket
            while missing > 0 and len(selected) < TOTAL_FRAMES:
                pool = [
                    item for item in candidates
                    if (not phase["exact_diff"] or item["difficulty"] == difficulty)
                    and candidate_rank_for_qtype(item, qtype) <= phase["max_rank"]
                    and item["question_type_scores"][qtype] >= phase["min_score"]
                    and can_take(item, used, selected_indices_by_video,
                                count_by_video, selected_hashes, phase["min_gap"])
                ]
                if not pool:
                    break
                pool.sort(key=lambda x: bucket_priority(x, qtype), reverse=True)
                add_selected_item(pool[0], difficulty, qtype, selected, used,
                                  selected_indices_by_video, count_by_video,
                                  selected_hashes)
                missing -= 1

def enforce_defer_coverage(candidates, selected, used, selected_indices_by_video,
                           count_by_video, selected_hashes):
    current_defer = sum(1 for x in selected if x["suggested_should_defer"])
    if current_defer >= DEFER_TARGET:
        return
    needed = DEFER_TARGET - current_defer
    print(f"[INFO] Boosting defer coverage by {needed} frames")

    selected_by_bucket = defaultdict(list)
    for item in selected:
        selected_by_bucket[
            (item["assigned_difficulty"], item["assigned_question_type"])
        ].append(item)

    replacement_buckets = [
        ("hard", "risk_pitfall"),
        ("hard", "safety_verification"),
        ("medium", "safety_verification"),
        ("medium", "risk_pitfall"),
        ("easy", "workflow_phase"),
        ("medium", "workflow_phase"),
        ("hard", "workflow_phase"),
    ]

    for bucket in replacement_buckets:
        if needed <= 0:
            break
        difficulty, qtype = bucket
        incoming = [
            cand for cand in candidates
            if cand["source"] not in used
            and cand["difficulty"] == difficulty
            and cand["suggested_should_defer"]
            and candidate_rank_for_qtype(cand, qtype) <= 2
            and cand["question_type_scores"][qtype] >= 0.35
            and count_by_video[cand["video_id"]] < MAX_PER_VIDEO
            and not any(abs(cand["frame_index"] - p) < RELAXED_GAP
                        for p in selected_indices_by_video[cand["video_id"]])
        ]
        incoming.sort(
            key=lambda x: (x["defer_score"], x["question_type_scores"][qtype]),
            reverse=True,
        )

        replaceable = [
            x for x in selected_by_bucket[bucket]
            if not x["suggested_should_defer"]
        ]
        replaceable.sort(
            key=lambda x: (x["question_type_scores"][qtype], -x["quality_score"])
        )

        while needed > 0 and incoming and replaceable:
            new_item = incoming.pop(0)
            old_item = replaceable.pop(0)

            used.remove(old_item["source"])
            selected.remove(old_item)
            count_by_video[old_item["video_id"]] -= 1
            selected_indices_by_video[old_item["video_id"]].remove(old_item["frame_index"])
            if old_item["phash"] in selected_hashes:
                selected_hashes.remove(old_item["phash"])

            assigned_new = add_selected_item(
                new_item, difficulty, qtype, selected, used,
                selected_indices_by_video, count_by_video, selected_hashes,
            )
            selected_by_bucket[bucket].append(assigned_new)
            needed -= 1

    final_defer = sum(1 for x in selected if x["suggested_should_defer"])
    print(f"[INFO] Defer coverage after adjustment: {final_defer}/{len(selected)}")

def trim_to_bucket_targets(selected):
    trimmed = {}
    for bucket, target in BUCKET_TARGETS.items():
        items = [x for x in selected
                 if (x["assigned_difficulty"], x["assigned_question_type"]) == bucket]
        qtype = bucket[1]
        items.sort(key=lambda x: (x["question_type_scores"][qtype],
                                   x["defer_score"], x["quality_score"]),
                   reverse=True)
        trimmed[bucket] = items[:target]
    return [item for bucket in BUCKET_TARGETS for item in trimmed[bucket]]

def rebuild_selection_state(selected):
    used = set()
    idx_by_video = defaultdict(list)
    count_by_video = defaultdict(int)
    hashes = []
    for item in selected:
        used.add(item["source"])
        count_by_video[item["video_id"]] += 1
        idx_by_video[item["video_id"]].append(item["frame_index"])
        hashes.append(item["phash"])
    return used, idx_by_video, count_by_video, hashes


# ════════════════════════════════════════════════════════════════════
# 10. VALIDATION
# ════════════════════════════════════════════════════════════════════

def validate_selection(selected):
    issues = []

    for item in selected:
        try:
            img = Image.open(item["source"])
            img.verify()
        except Exception as e:
            issues.append(f"CORRUPT: {item['source']} — {e}")

    no_mask = [item for item in selected if not item["has_mask"]]
    if no_mask:
        issues.append(f"NO_MASK: {len(no_mask)} frames have no watershed mask")

    hashes = [item["phash"] for item in selected]
    dup_pairs = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hamming_distance(hashes[i], hashes[j]) < PHASH_HAMMING_THRESHOLD:
                dup_pairs.append((selected[i]["source"], selected[j]["source"],
                                  hamming_distance(hashes[i], hashes[j])))
    if dup_pairs:
        issues.append(f"NEAR_DUPLICATES: {len(dup_pairs)} pairs")
        for a, b, d in dup_pairs[:5]:
            issues.append(f"  hamming={d}: {Path(a).name} <-> {Path(b).name}")

    vids = Counter(item["video_id"] for item in selected)
    if len(vids) < 5:
        issues.append(f"LOW_VIDEO_DIVERSITY: only {len(vids)} videos")
    over = {v: c for v, c in vids.items() if c > MAX_PER_VIDEO}
    if over:
        issues.append(f"OVER_QUOTA: {over}")

    counts = bucket_counts(selected)
    misses = {}
    for bucket, target in BUCKET_TARGETS.items():
        actual = counts.get(bucket, 0)
        if actual != target and target > 0:
            misses[f"{bucket[0]}|{bucket[1]}"] = f"{actual}/{target}"
    if misses:
        issues.append(f"BUCKET_MISMATCH: {misses}")

    defer_count = sum(1 for x in selected if x["suggested_should_defer"])
    if defer_count < DEFER_TARGET:
        issues.append(f"DEFER_SHORT: {defer_count}/{DEFER_TARGET}")

    return {
        "total_frames": len(selected),
        "total_issues": len(issues),
        "issues": issues,
        "videos_used": len(vids),
        "defer_count": defer_count,
        "frames_with_mask": sum(1 for x in selected if x["has_mask"]),
        "near_duplicate_pairs": len(dup_pairs),
    }


# ════════════════════════════════════════════════════════════════════
# 11. SAVE OUTPUTS
# ════════════════════════════════════════════════════════════════════

def save_outputs(selected):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qtype_order = {q: i for i, q in enumerate(QUESTION_TYPES)}
    diff_order = {d: i for i, d in enumerate(DIFFICULTIES)}
    selected.sort(key=lambda x: (
        diff_order.get(x["assigned_difficulty"], 99),
        qtype_order.get(x["assigned_question_type"], 99),
        x["video_id"], x["frame_index"],
    ))

    metadata = []
    blueprint = []
    for i, item in enumerate(selected, start=1):
        src = Path(item["source"])
        dst_name = f"frame_{i:03d}.png"
        dst = OUT_DIR / dst_name
        copy2(src, dst)

        if item.get("color_mask_path"):
            try:
                copy2(item["color_mask_path"], OUT_DIR / f"frame_{i:03d}_mask.png")
            except Exception:
                pass

        top3 = sorted(item["question_type_scores"].items(),
                       key=lambda kv: kv[1], reverse=True)[:3]
        note = (
            f"Auto-selected for {item['assigned_question_type']} | "
            f"{item['assigned_difficulty']}; top3={top3}; "
            f"defer={item['defer_score']:.2f}; "
            f"classes={list(item.get('classes_detected', {}).keys())}"
        )

        metadata.append({
            "frame_id": f"frame_{i:03d}",
            "file_name": dst_name,
            "source": item["source"],
            "video_id": item["video_id"],
            "frame_index": item["frame_index"],
            "assigned_question_type": item["assigned_question_type"],
            "assigned_difficulty": item["assigned_difficulty"],
            "primary_question_type": item["primary_question_type"],
            "question_type_ranking": item["question_type_ranking"],
            "question_type_scores": item["question_type_scores"],
            "suggested_should_defer": item["suggested_should_defer"],
            "defer_score": item["defer_score"],
            "classes_detected": item.get("classes_detected", {}),
            "n_classes": item["n_classes"],
            "n_anatomy": item["n_anatomy"],
            "cvs_relevant": item["cvs_relevant"],
            "has_blood": item["has_blood"],
            "quality_score": item["quality_score"],
            "brightness": item["brightness"],
            "contrast": item["contrast"],
            "sharpness": item["sharpness"],
            "edge_density": item["edge_density"],
            "notes": note,
        })

        blueprint.append({
            "qid": f"Q{i:03d}",
            "frame": dst_name,
            "question_type": item["assigned_question_type"],
            "difficulty": item["assigned_difficulty"],
            "should_defer": item["suggested_should_defer"],
            "classes_detected": list(item.get("classes_detected", {}).keys()),
            "notes": note,
        })

    with open(OUT_DIR / "frame_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "question_blueprint_v3.json", "w", encoding="utf-8") as f:
        json.dump(blueprint, f, indent=2, ensure_ascii=False)

    q_counter = Counter(x["assigned_question_type"] for x in metadata)
    d_counter = Counter(x["assigned_difficulty"] for x in metadata)
    defer_count = sum(1 for x in metadata if x["suggested_should_defer"])
    frames_per_video = Counter(x["video_id"] for x in metadata)

    summary = {
        "total_selected": len(metadata),
        "qtype_distribution": dict(sorted(q_counter.items())),
        "difficulty_distribution": dict(sorted(d_counter.items())),
        "defer_count": defer_count,
        "videos_used": sorted({x["video_id"] for x in metadata}),
        "frames_per_video": dict(sorted(frames_per_video.items())),
        "bucket_targets": {f"{d}|{q}": n for (d, q), n in BUCKET_TARGETS.items()},
        "bucket_actuals": {
            f"{d}|{q}": sum(1 for x in metadata
                           if x["assigned_difficulty"] == d
                           and x["assigned_question_type"] == q)
            for (d, q) in BUCKET_TARGETS
        },
        "qtype_targets": QTYPE_TARGETS,
        "difficulty_targets": DIFFICULTY_TARGETS,
    }
    with open(OUT_DIR / "selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Saved {len(metadata)} frames to {OUT_DIR}")
    print(f"[DONE] Q-type: {summary['qtype_distribution']}")
    print(f"[DONE] Difficulty: {summary['difficulty_distribution']}")
    print(f"[DONE] Defer: {defer_count}/{len(metadata)}")
    print(f"[DONE] Videos: {len(summary['videos_used'])}")


# ════════════════════════════════════════════════════════════════════
# 12. MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    print(f"[INFO] cwd = {os.getcwd()}")
    print(f"[INFO] RAW_DIR = {RAW_DIR.resolve()}")
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR}")

    candidates = build_candidates()
    print_candidate_stats(candidates)

    selected = []
    used = set()
    count_by_video = defaultdict(int)
    selected_indices_by_video = defaultdict(list)
    selected_hashes = []

    bucket_order = [
        ("hard", "risk_pitfall"),
        ("hard", "safety_verification"),
        ("hard", "anatomy_landmark"),
        ("hard", "workflow_phase"),
        ("medium", "anatomy_landmark"),
        ("medium", "safety_verification"),
        ("medium", "workflow_phase"),
        ("medium", "risk_pitfall"),
        ("easy", "recognition"),
        ("easy", "workflow_phase"),
        ("easy", "anatomy_landmark"),
        ("easy", "safety_verification"),
        ("medium", "recognition"),
    ]

    for bucket in bucket_order:
        target_n = BUCKET_TARGETS[bucket]
        chosen = select_for_bucket(
            candidates, selected, used, selected_indices_by_video,
            count_by_video, selected_hashes, bucket, target_n,
        )
        print(f"[INFO] Bucket {bucket}: {len(chosen)}/{target_n}")

    fill_bucket_deficits(candidates, selected, used, selected_indices_by_video,
                         count_by_video, selected_hashes)

    if len(selected) < TOTAL_FRAMES:
        print(f"[WARN] Only {len(selected)} frames before defer adjustment")

    enforce_defer_coverage(candidates, selected, used, selected_indices_by_video,
                           count_by_video, selected_hashes)

    if len(selected) > TOTAL_FRAMES:
        print(f"[INFO] Trimming from {len(selected)} to {TOTAL_FRAMES}")
        selected = trim_to_bucket_targets(selected)
        used, selected_indices_by_video, count_by_video, selected_hashes = \
            rebuild_selection_state(selected)

    if len(selected) < TOTAL_FRAMES:
        fill_bucket_deficits(candidates, selected, used, selected_indices_by_video,
                             count_by_video, selected_hashes)

    final_defer = sum(1 for x in selected if x["suggested_should_defer"])
    if final_defer < DEFER_TARGET:
        print(f"[WARN] Defer at {final_defer}, re-enforcing")
        enforce_defer_coverage(candidates, selected, used, selected_indices_by_video,
                               count_by_video, selected_hashes)

    final_counts = bucket_counts(selected)
    print("\n[INFO] Final bucket counts:")
    for bucket in BUCKET_TARGETS:
        print(f"  {bucket}: {final_counts.get(bucket, 0)}/{BUCKET_TARGETS[bucket]}")
    print(f"\n[INFO] Total selected: {len(selected)} frames")

    print("\n[VALIDATE] Running post-selection checks...")
    report = validate_selection(selected)
    if report["total_issues"] == 0:
        print("[VALIDATE] All checks passed!")
    else:
        print(f"[VALIDATE] {report['total_issues']} issue(s):")
        for issue in report["issues"]:
            print(f"  - {issue}")

    save_outputs(selected)

    with open(OUT_DIR / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[DONE] Validation report: {OUT_DIR / 'validation_report.json'}")


if __name__ == "__main__":
    main()