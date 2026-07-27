#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tile-based YOLO segmentation validation/inference for float32 GeoTIFF.

This script is designed for your float RGB 0-1 YOLO segmentation workflow:
    1. Read one GeoTIFF with GDAL.
    2. Keep float32 values in [0, 1] without converting to uint8.
    3. Cut the image into 1024x1024 tiles.
    4. Run YOLO segmentation on each tile.
    5. Merge tile predictions back into full-size GeoTIFF outputs.

Important:
    The tile is passed to YOLO as a torch tensor with shape [1, 3, H, W].
    This avoids the normal numpy/PIL/OpenCV image path that may flip channels.

Outputs:
    - <name>_pred_cls.tif:
        uint8 class map. 0=background, 1=class0/ridge, 2=class1/valley by default.
    - <name>_best_conf.tif:
        float32 best instance confidence per pixel.
    - <name>_ridge_conf.tif:
        float32 best confidence for class 0.
    - <name>_valley_conf.tif:
        float32 best confidence for class 1.

Requirements:
    pip install numpy opencv-python gdal torch ultralytics

Example:
    python infer_yolo_geotiff_tiles.py ^
        --input D:\\path\\image.tif ^
        --model D:\\YOLOV8\\Extract_RV\\YOLO-RV\\ultralytics\\runs\\segment\\xxx\\weights\\best.pt ^
        --output D:\\path\\out ^
        --tile-size 1024 ^
        --overlap 0 ^
        --device 0
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

# Work around duplicate OpenMP runtimes on some Windows conda/pip mixed environments.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from osgeo import gdal


# -------------------------------------------------------------------------
# Defaults for quick edits without changing CLI arguments.
# -------------------------------------------------------------------------
DEFAULT_ULTRALYTICS_ROOT = r"D:\YOLOV8\Extract_RV\YOLO-RV\ultralytics"
DEFAULT_MODEL_PATH = (
    r"D:\YOLOV8\Extract_RV\YOLO-RV\ultralytics\runs\segment"
    r"\yolopm_segment_pscolor\weights\best.pt"
)
DEFAULT_INPUT_PATH = r""
DEFAULT_OUTPUT_DIR = r"D:\YOLOV8\Extract_RV\YOLO-RV\ultralytics\data\test_out"


def add_ultralytics_to_path(ultralytics_root: str) -> None:
    """Prefer the local modified Ultralytics package."""
    if ultralytics_root:
        root = str(Path(ultralytics_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)


def read_geotiff_float_hwc(path: str, channels: int = 3, clip_01: bool = True) -> tuple[np.ndarray, gdal.Dataset]:
    """
    Read a GeoTIFF as HWC float32.

    Args:
        path: Input GeoTIFF.
        channels: Number of channels expected by YOLO. For 1-band input, it is repeated.
        clip_01: Clip finite values into [0,1].

    Returns:
        image: HWC float32 array.
        ds: GDAL dataset, kept alive for geo metadata.
    """
    ds = gdal.Open(path)
    if ds is None:
        raise ValueError(f"Cannot open GeoTIFF: {path}")

    arr = ds.ReadAsArray()
    if arr is None:
        raise ValueError(f"Cannot read raster array: {path}")

    arr = arr.astype(np.float32, copy=False)

    if arr.ndim == 2:
        # H, W -> H, W, 1
        arr = arr[..., None]
    elif arr.ndim == 3:
        # GDAL returns C, H, W for multiband rasters.
        arr = np.moveaxis(arr, 0, -1)
    else:
        raise ValueError(f"Unsupported raster shape {arr.shape}; expected HxW or CxHxW")

    # Replace nodata/NaN/Inf with 0.
    invalid = ~np.isfinite(arr)
    for bi in range(min(ds.RasterCount, arr.shape[-1])):
        nodata = ds.GetRasterBand(bi + 1).GetNoDataValue()
        if nodata is not None:
            invalid[..., bi] |= arr[..., bi] == nodata
    arr = arr.copy()
    arr[invalid] = 0.0

    if arr.shape[-1] == 1 and channels == 3:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] >= channels:
        arr = arr[..., :channels]
    elif arr.shape[-1] < channels:
        # Pad missing channels with zeros.
        pad = np.zeros((*arr.shape[:2], channels - arr.shape[-1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=-1)

    if clip_01:
        arr = np.clip(arr, 0.0, 1.0)

    return np.ascontiguousarray(arr.astype(np.float32, copy=False)), ds


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Return tile starts that cover [0, length)."""
    if length <= tile_size:
        return [0]

    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def iter_tiles(height: int, width: int, tile_size: int, overlap: int) -> Iterable[tuple[int, int, int, int]]:
    """Yield y, x, valid_h, valid_w for all tiles."""
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("--overlap must satisfy 0 <= overlap < tile_size")

    stride = tile_size - overlap
    ys = tile_starts(height, tile_size, stride)
    xs = tile_starts(width, tile_size, stride)
    for y in ys:
        for x in xs:
            yield y, x, min(tile_size, height - y), min(tile_size, width - x)


def make_tile(image: np.ndarray, y: int, x: int, valid_h: int, valid_w: int, tile_size: int) -> np.ndarray:
    """Create a tile_size x tile_size x C tile, zero-padded on bottom/right if needed."""
    tile = np.zeros((tile_size, tile_size, image.shape[-1]), dtype=np.float32)
    tile[:valid_h, :valid_w] = image[y : y + valid_h, x : x + valid_w]
    return tile


def tile_to_tensor(tile_hwc: np.ndarray, device: str | int) -> torch.Tensor:
    """Convert HWC float32 [0,1] tile to BCHW tensor without channel flipping."""
    tensor = torch.from_numpy(np.ascontiguousarray(tile_hwc.transpose(2, 0, 1))).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def write_geotiff(
    path: str,
    array: np.ndarray,
    ref_ds: gdal.Dataset,
    dtype,
    nodata=None,
    compress: bool = True,
) -> None:
    """Write a single-band GeoTIFF preserving projection/geotransform."""
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    h, w = array.shape
    options = ["TILED=YES", "BIGTIFF=IF_SAFER"]
    if compress:
        options += ["COMPRESS=LZW"]

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(path, w, h, 1, dtype, options=options)
    if out_ds is None:
        raise RuntimeError(f"Failed to create output GeoTIFF: {path}")

    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())

    band = out_ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(array)
    band.FlushCache()
    out_ds = None
    print(f"Saved: {path}")


def merge_result_masks(
    result,
    y: int,
    x: int,
    valid_h: int,
    valid_w: int,
    class_map: np.ndarray,
    best_conf: np.ndarray,
    class_conf: list[np.ndarray],
    class_offset: int = 1,
    mask_threshold: float = 0.5,
) -> int:
    """
    Merge one tile's YOLO Result masks into full-size output arrays.

    Returns:
        Number of instances merged.
    """
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return 0

    masks = result.masks.data.detach().float().cpu().numpy()  # N, mh, mw
    cls = result.boxes.cls.detach().long().cpu().numpy()
    conf = result.boxes.conf.detach().float().cpu().numpy()

    merged = 0
    for mi, ci, cf in zip(masks, cls, conf):
        # Ultralytics returns masks at image/tile resolution for tensor inference in this setup.
        # If not, resize by nearest/bilinear through torch for safety.
        if mi.shape != (result.orig_shape[0], result.orig_shape[1]):
            mt = torch.from_numpy(mi)[None, None].float()
            mt = torch.nn.functional.interpolate(
                mt,
                size=(result.orig_shape[0], result.orig_shape[1]),
                mode="bilinear",
                align_corners=False,
            )
            mi = mt[0, 0].numpy()

        tile_mask = mi[:valid_h, :valid_w] > mask_threshold
        if not tile_mask.any():
            continue

        yy = slice(y, y + valid_h)
        xx = slice(x, x + valid_w)
        region_best = best_conf[yy, xx]
        update = tile_mask & (cf > region_best)

        if update.any():
            region_class = class_map[yy, xx]
            region_class[update] = int(ci) + class_offset
            region_best[update] = float(cf)
            class_map[yy, xx] = region_class
            best_conf[yy, xx] = region_best

        if 0 <= int(ci) < len(class_conf):
            cc = class_conf[int(ci)][yy, xx]
            cc[tile_mask] = np.maximum(cc[tile_mask], float(cf))
            class_conf[int(ci)][yy, xx] = cc

        merged += 1
    return merged


def run(args: argparse.Namespace) -> None:
    add_ultralytics_to_path(args.ultralytics_root)

    from ultralytics import YOLO

    input_path = Path(args.input)
    model_path = Path(args.model)
    output_dir = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input GeoTIFF not found: {input_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Tile size: {args.tile_size}, overlap: {args.overlap}")
    print(f"Device: {args.device}, conf: {args.conf}, iou: {args.iou}")

    image, ref_ds = read_geotiff_float_hwc(str(input_path), channels=args.channels, clip_01=not args.no_clip)
    h, w, c = image.shape
    print(f"Image: shape={image.shape}, dtype={image.dtype}, min={float(image.min()):.6g}, max={float(image.max()):.6g}")

    model = YOLO(str(model_path), task="segment")

    class_map = np.zeros((h, w), dtype=np.uint8)
    best_conf = np.zeros((h, w), dtype=np.float32)
    class_conf = [np.zeros((h, w), dtype=np.float32) for _ in range(args.num_classes)]

    tiles = list(iter_tiles(h, w, args.tile_size, args.overlap))
    print(f"Tiles: {len(tiles)}")

    total_instances = 0
    with torch.no_grad():
        for ti, (y, x, valid_h, valid_w) in enumerate(tiles, 1):
            tile = make_tile(image, y, x, valid_h, valid_w, args.tile_size)
            tensor = tile_to_tensor(tile, args.device)

            results = model.predict(
                source=tensor,
                imgsz=args.tile_size,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=args.device,
                verbose=False,
                retina_masks=args.retina_masks,
            )

            merged = merge_result_masks(
                results[0],
                y,
                x,
                valid_h,
                valid_w,
                class_map,
                best_conf,
                class_conf,
                class_offset=args.class_offset,
                mask_threshold=args.mask_threshold,
            )
            total_instances += merged

            if ti == 1 or ti == len(tiles) or ti % args.print_every == 0:
                print(f"[{ti}/{len(tiles)}] tile x={x}, y={y}, valid={valid_w}x{valid_h}, instances={merged}")

    stem = input_path.stem
    write_geotiff(output_dir / f"{stem}_pred_cls.tif", class_map, ref_ds, gdal.GDT_Byte, nodata=0)
    write_geotiff(output_dir / f"{stem}_best_conf.tif", best_conf, ref_ds, gdal.GDT_Float32, nodata=0.0)

    # For your current dataset names: class 0=ridge, class 1=valley.
    class_names = args.class_names.split(",") if args.class_names else [f"class{i}" for i in range(args.num_classes)]
    for ci, arr in enumerate(class_conf):
        name = class_names[ci].strip() if ci < len(class_names) and class_names[ci].strip() else f"class{ci}"
        write_geotiff(output_dir / f"{stem}_{name}_conf.tif", arr, ref_ds, gdal.GDT_Float32, nodata=0.0)

    print(f"Done. Merged instances: {total_instances}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO segmentation on a large float32 GeoTIFF by 1024x1024 tiles.")
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT_PATH, help="Input float32 GeoTIFF.")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL_PATH, help="YOLO segmentation .pt model.")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_DIR, help="Output folder.")
    parser.add_argument("--ultralytics-root", default=DEFAULT_ULTRALYTICS_ROOT, help="Local Ultralytics repo root.")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size for inference.")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap between adjacent tiles.")
    parser.add_argument("--device", default="0", help="Device for YOLO, e.g. 0 or cpu.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per tile.")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Binary threshold for predicted masks.")
    parser.add_argument("--channels", type=int, default=3, help="Number of model input channels.")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of segmentation classes.")
    parser.add_argument("--class-offset", type=int, default=1, help="Value offset in pred_cls.tif; 1 means bg=0.")
    parser.add_argument("--class-names", default="ridge,valley", help="Comma-separated class names for conf outputs.")
    parser.add_argument("--retina-masks", action="store_true", help="Use high-resolution masks from Ultralytics.")
    parser.add_argument("--no-clip", action="store_true", help="Do not clip input values to [0,1].")
    parser.add_argument("--print-every", type=int, default=20, help="Progress print interval in tiles.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input:
        print("Error: --input is required, or set DEFAULT_INPUT_PATH in the script.")
        sys.exit(1)
    run(args)


if __name__ == "__main__":
    main()
