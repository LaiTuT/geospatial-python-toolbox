#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split part of YOLO train set into val set, with optional density-aware stratification.

Expected structure (modifiable):
  train_images_dir/
  train_labels_dir/
  val_images_dir/
  val_labels_dir/

Label format:
  One object instance per line (YOLO detect/seg both satisfy this assumption).
  For seg: cls x1 y1 x2 y2 ... (polygon points)
  For det: cls x y w h
"""

from __future__ import annotations
import os
import shutil
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# =========================
# User-configurable variables
# =========================
TRAIN_IMAGES_DIR = r"D:\YOLOV8\Extract_RV\YOLO-RV\ultralytics\data\images\train"
TRAIN_LABELS_DIR = r"D:\YOLOV8\Extract_RV\YOLO-RV\ultralytics\data\labels\train"

VAL_IMAGES_DIR   = r"D:\YOLOV8\Extract_RV\YOLO-RV\ultralytics\data\images\val"
VAL_LABELS_DIR   = r"D:\YOLOV8\Extract_RV\YOLO-RV\ultralytics\data\labels\val"

# Choose how many go to val:
VAL_RATIO = 0.20       # e.g., 0.20 -> 20% (recommended for ~400 images)
# Or use fixed count instead of ratio (set to None to use ratio):
VAL_COUNT: Optional[int] = None

# How to measure "density":
# "lines" -> number of lines in label txt (instances)
# "bytes" -> label file size in bytes (proxy, often useful for seg polygon complexity)
DENSITY_MODE = "lines"  # "lines" or "bytes"

# Stratify by density?
# If True: split into 3 bins (low/med/high) then sample val proportionally from each bin.
# If False: pure random split.
STRATIFY_BY_DENSITY = True
NUM_BINS = 3  # low/med/high

# File handling:
ACTION = "copy"  # "copy" or "move"
DRY_RUN = False  # True -> print actions, do not actually copy/move

# Reproducibility:
SEED = 42

# Image extensions to look for:
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


# =========================
# Implementation
# =========================

@dataclass
class Sample:
    stem: str
    img_path: Path
    lbl_path: Path
    density: int


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_density(lbl_path: Path, mode: str) -> int:
    if mode == "bytes":
        return lbl_path.stat().st_size
    if mode == "lines":
        # count non-empty lines
        cnt = 0
        with lbl_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    cnt += 1
        return cnt
    raise ValueError(f"Unknown DENSITY_MODE: {mode}")


def find_image_for_stem(images_dir: Path, stem: str) -> Optional[Path]:
    # Find first matching extension
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def collect_samples(train_images: Path, train_labels: Path, density_mode: str) -> List[Sample]:
    samples: List[Sample] = []
    label_files = sorted(train_labels.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No .txt labels found in: {train_labels}")

    missing_images = 0
    for lbl in label_files:
        stem = lbl.stem
        img = find_image_for_stem(train_images, stem)
        if img is None:
            missing_images += 1
            continue
        density = read_density(lbl, density_mode)
        samples.append(Sample(stem=stem, img_path=img, lbl_path=lbl, density=density))

    if not samples:
        raise RuntimeError("No matched (image,label) pairs found. Check paths and naming.")
    if missing_images:
        print(f"[WARN] {missing_images} label files have no matching image (skipped).")
    return samples


def stratified_split(samples: List[Sample], val_n: int, num_bins: int, seed: int) -> Tuple[List[Sample], List[Sample]]:
    # Sort by density then cut into bins of (almost) equal size
    rng = random.Random(seed)
    samples_sorted = sorted(samples, key=lambda s: s.density)
    n = len(samples_sorted)

    # Create bins by index ranges
    bins: List[List[Sample]] = []
    for b in range(num_bins):
        start = (n * b) // num_bins
        end = (n * (b + 1)) // num_bins
        bins.append(samples_sorted[start:end])

    # Allocate val quota per bin proportional to bin sizes
    bin_sizes = [len(b) for b in bins]
    if sum(bin_sizes) != n:
        raise RuntimeError("Bin sizing error.")

    # Initial quotas
    quotas = [int(round(val_n * (sz / n))) for sz in bin_sizes]

    # Fix rounding to match exactly val_n
    diff = val_n - sum(quotas)
    # distribute remaining (+/-) one by one
    idx = 0
    while diff != 0 and num_bins > 0:
        i = idx % num_bins
        if diff > 0:
            if quotas[i] < bin_sizes[i]:
                quotas[i] += 1
                diff -= 1
        else:  # diff < 0
            if quotas[i] > 0:
                quotas[i] -= 1
                diff += 1
        idx += 1

    # Sample within each bin
    val_samples: List[Sample] = []
    train_samples: List[Sample] = []

    for b, q in zip(bins, quotas):
        b_copy = b[:]
        rng.shuffle(b_copy)
        val_part = b_copy[:q]
        train_part = b_copy[q:]
        val_samples.extend(val_part)
        train_samples.extend(train_part)

    # Final shuffle (optional)
    rng.shuffle(val_samples)
    rng.shuffle(train_samples)

    if len(val_samples) != val_n:
        raise RuntimeError(f"Split error: expected val {val_n}, got {len(val_samples)}")

    return train_samples, val_samples


def random_split(samples: List[Sample], val_n: int, seed: int) -> Tuple[List[Sample], List[Sample]]:
    rng = random.Random(seed)
    s = samples[:]
    rng.shuffle(s)
    val_samples = s[:val_n]
    train_samples = s[val_n:]
    return train_samples, val_samples


def transfer_file(src: Path, dst: Path, action: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY_RUN] {action.upper()} {src} -> {dst}")
        return
    ensure_dir(dst.parent)
    if action == "copy":
        shutil.copy2(src, dst)
    elif action == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"Unknown ACTION: {action}")


def main():
    train_images = Path(TRAIN_IMAGES_DIR)
    train_labels = Path(TRAIN_LABELS_DIR)
    val_images = Path(VAL_IMAGES_DIR)
    val_labels = Path(VAL_LABELS_DIR)

    if not train_images.exists():
        raise FileNotFoundError(f"TRAIN_IMAGES_DIR not found: {train_images}")
    if not train_labels.exists():
        raise FileNotFoundError(f"TRAIN_LABELS_DIR not found: {train_labels}")

    ensure_dir(val_images)
    ensure_dir(val_labels)

    random.seed(SEED)

    samples = collect_samples(train_images, train_labels, DENSITY_MODE)
    total = len(samples)

    if VAL_COUNT is not None:
        val_n = int(VAL_COUNT)
    else:
        val_n = int(round(total * VAL_RATIO))

    if val_n <= 0 or val_n >= total:
        raise ValueError(f"Invalid val size: {val_n}, total: {total}")

    print(f"Found {total} matched samples.")
    print(f"Target val size = {val_n} ({val_n/total:.1%}), mode={DENSITY_MODE}, stratify={STRATIFY_BY_DENSITY}")

    if STRATIFY_BY_DENSITY:
        train_set, val_set = stratified_split(samples, val_n, NUM_BINS, SEED)
    else:
        train_set, val_set = random_split(samples, val_n, SEED)

    # Move/copy selected val samples to val folders
    moved = 0
    for s in val_set:
        # destination paths
        dst_img = val_images / s.img_path.name
        dst_lbl = val_labels / s.lbl_path.name

        transfer_file(s.img_path, dst_img, ACTION, DRY_RUN)
        transfer_file(s.lbl_path, dst_lbl, ACTION, DRY_RUN)
        moved += 1

    print(f"Done. {moved} samples transferred to val.")
    if ACTION == "copy":
        print("[NOTE] You copied to val. Train still contains originals (duplicate).")
        print("       If you want train/val disjoint, set ACTION='move' OR delete the copied ones from train.")
    else:
        print("[NOTE] You moved to val. Train/val are now disjoint.")

    # Quick density summary
    def summary(ss: List[Sample]) -> Tuple[int, float, int, int]:
        dens = [x.density for x in ss]
        return (len(dens), sum(dens)/len(dens), min(dens), max(dens))

    tr_n, tr_avg, tr_min, tr_max = summary(train_set)
    va_n, va_avg, va_min, va_max = summary(val_set)
    print(f"Train: n={tr_n}, density avg={tr_avg:.2f}, min={tr_min}, max={tr_max}")
    print(f"Val:   n={va_n}, density avg={va_avg:.2f}, min={va_min}, max={va_max}")


if __name__ == "__main__":
    main()
