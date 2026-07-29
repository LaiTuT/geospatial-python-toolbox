#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create 3-channel topo images from DSM for YOLO training.

The output channels are derived from DSM only:
    - dx (Sobel X)
    - dy (Sobel Y)
    - curvature (Laplacian)

Input can be a single GeoTIFF or a folder containing GeoTIFFs (no recursion).

Requirements:
    pip install numpy opencv-python gdal

Example:
    # Single file
    python create_pseudo_color_for_yolo.py --input dsm.tif --output out.tif

    # Batch folder (no subdirectories)
    python create_pseudo_color_for_yolo.py --input ./dsms --output ./out
"""

import argparse
import sys
import os
import glob
from osgeo import gdal
import cv2
import numpy as np

#请给我一篇关于“多任务标签”的最新论文的相关研究报告（24年之后），用于提供我参考相关工作模块，最好有包含实例分割相关任务（yolo-seg相关也可）-不强求
# Default paths for quick edits without changing CLI arguments.
DEFAULT_INPUT_PATH = r"D:\YOLOV8\Extract_RV\YOLO-RV\ultralytics\data\images\tf5m"
DEFAULT_OUTPUT_PATH = r"D:\YOLOV8\Extract_RV\YOLO-RV\ultralytics\data\images\pscolor"


def read_raster(path: str):
    """Read a raster and return metadata + array."""
    ds = gdal.Open(path)
    if ds is None:
        raise ValueError(f"Cannot open raster: {path}")

    bands = ds.RasterCount
    width = ds.RasterXSize
    height = ds.RasterYSize

    if bands == 1:
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray().astype(np.float32)
        nodata = band.GetNoDataValue()
    else:
        # Multi-band: shape (bands, height, width)
        arr = ds.ReadAsArray().astype(np.float32)
        nodata = ds.GetRasterBand(1).GetNoDataValue()

    return {
        "data": arr,
        "nodata": nodata,
        "shape": (height, width),
        "bands": bands,
        "ds": ds,
        "path": path,
    }


def stretch_01(arr: np.ndarray, nodata=None, low_pct=2.0, high_pct=98.0, mask=None) -> np.ndarray:
    """Linearly stretch a float32 array to float32 [0,1] using percentile cuts."""
    if mask is None:
        mask = np.isnan(arr) | np.isinf(arr)
        if nodata is not None:
            mask |= (arr == nodata)
    else:
        mask = mask | np.isnan(arr) | np.isinf(arr)

    valid = arr[~mask]
    if valid.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)

    low, high = np.percentile(valid, [low_pct, high_pct])
    if high <= low:
        high = low + 1e-6

    scaled = (arr - low) / (high - low)
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled[mask] = 0
    return scaled.astype(np.float32)


def resize_band(arr: np.ndarray, target_shape: tuple, interp=cv2.INTER_LINEAR) -> np.ndarray:
    """Resize a single-band 2D array to target (H, W)."""
    if arr.shape == target_shape:
        return arr
    return cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=interp)


def get_single_channel_01(raster_dict: dict, target_shape: tuple = None) -> np.ndarray:
    """Extract a single-band 2D float32 image in [0,1] from a raster dict."""
    arr = raster_dict["data"]
    nodata = raster_dict["nodata"]

    if raster_dict["bands"] > 1:
        # If multi-band, average to grayscale for generic use
        arr = np.mean(arr, axis=0)
    else:
        arr = arr

    if target_shape is not None and arr.shape != target_shape:
        arr = resize_band(arr, target_shape)

    return stretch_01(arr, nodata)


def method_topo(inputs: dict) -> np.ndarray:
    """
    Create a 3-channel image from DSM using dx, dy, and curvature (Laplacian).
    This mirrors the directional gradient + curvature idea used in superpixels.
    """
    dsm = inputs.get("dsm")
    if dsm is None:
        raise ValueError("Missing DSM input for topo method")

    arr = dsm["data"]
    nodata = dsm["nodata"]

    if dsm["bands"] > 1:
        arr = np.mean(arr, axis=0)

    arr = arr.astype(np.float32)
    mask = np.isnan(arr) | np.isinf(arr)
    if nodata is not None:
        mask |= (arr == nodata)

    valid = arr[~mask]
    fill_value = float(np.median(valid)) if valid.size > 0 else 0.0
    arr_filled = arr.copy()
    arr_filled[mask] = fill_value

    dx = cv2.Sobel(arr_filled, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(arr_filled, cv2.CV_32F, 0, 1, ksize=3)
    curvature = cv2.Laplacian(arr_filled, cv2.CV_32F, ksize=3)

    ch_dx = stretch_01(dx, mask=mask)
    ch_dy = stretch_01(dy, mask=mask)
    ch_curv = stretch_01(curvature, mask=mask)

    return np.stack([ch_dx, ch_dy, ch_curv], axis=-1)


def save_image(rgb_array: np.ndarray, output_path: str, ref_ds=None):
    """Save the composite. Preserve geo-info if output is GeoTIFF."""
    print(f"Output shape: {rgb_array.shape}, dtype: {rgb_array.dtype}")

    ext = output_path.lower()
    if ext.endswith((".tif", ".tiff")):
        driver = gdal.GetDriverByName("GTiff")
        h, w, _ = rgb_array.shape
        out_ds = driver.Create(output_path, w, h, 3, gdal.GDT_Float32)
        if out_ds is None:
            raise RuntimeError(f"Failed to create GeoTIFF: {output_path}")

        if ref_ds is not None:
            out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
            out_ds.SetProjection(ref_ds.GetProjection())

        rgb_float = np.clip(rgb_array.astype(np.float32), 0.0, 1.0)
        for i in range(3):
            out_ds.GetRasterBand(i + 1).WriteArray(rgb_float[:, :, i])
        out_ds = None
    else:
        raise ValueError("Float RGB output requires .tif/.tiff to preserve 0-1 values")

    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create 3-channel topo images from DSM for YOLO training.")
    parser.add_argument(
        "-i",
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Input DSM GeoTIFF or folder",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output file or folder",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Read inputs
    # ------------------------------------------------------------------
    input_path = args.input
    output_path = args.output
    if not input_path or not output_path:
        print("Error: Input and output paths are required. Set defaults or pass -i/-o.")
        sys.exit(1)

    if os.path.isdir(input_path):
        if os.path.isfile(output_path):
            print("Error: Output must be a folder when input is a folder.")
            sys.exit(1)
        os.makedirs(output_path, exist_ok=True)

        patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(input_path, pattern)))
        files = sorted(set(files))

        if not files:
            print("Error: No GeoTIFF files found in input folder.")
            sys.exit(1)

        for in_file in files:
            base = os.path.splitext(os.path.basename(in_file))[0]
            out_file = os.path.join(output_path, f"{base}.tif")
            print(f"Reading DSM: {in_file}")
            inputs = {"dsm": read_raster(in_file)}
            composite = method_topo(inputs)
            ref_ds = inputs["dsm"]["ds"]
            save_image(composite, out_file, ref_ds)
        return

    if not os.path.isfile(input_path):
        print("Error: Input path does not exist.")
        sys.exit(1)

    if os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_path, f"{base}.tif")

    # ------------------------------------------------------------------
    # Composite
    # ------------------------------------------------------------------
    print(f"Method: {args.method}")
    print(f"Reading DSM: {input_path}")
    inputs = {"dsm": read_raster(input_path)}
    composite = method_topo(inputs)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    ref_ds = inputs["dsm"]["ds"]
    save_image(composite, output_path, ref_ds)


if __name__ == "__main__":
    main()
