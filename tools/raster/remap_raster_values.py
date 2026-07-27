"""Remap two pixel values in every GeoTIFF in a directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def remap_rasters(
    input_dir: Path,
    first_value: float,
    second_value: float,
    first_target: int = 128,
    second_target: int = 255,
    output_suffix: str = "_remap",
) -> tuple[int, int]:
    """Write byte GeoTIFF copies with two source values remapped."""
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise RuntimeError(
            "GDAL is required. Install it with: conda install -c conda-forge gdal"
        ) from exc

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    inputs = sorted((*input_dir.glob("*.tif"), *input_dir.glob("*.tiff")))
    driver = gdal.GetDriverByName("GTiff")
    completed = 0

    for input_path in inputs:
        dataset = gdal.Open(str(input_path), gdal.GA_ReadOnly)
        if dataset is None:
            print(f"[error] Could not open: {input_path}")
            continue

        array = dataset.ReadAsArray()
        if array.ndim == 2:
            array = array[np.newaxis, ...]
        array = np.where(array == first_value, first_target, array)
        array = np.where(array == second_value, second_target, array)
        if np.any((array < 0) | (array > 255)):
            print(f"[error] Values outside byte range remain in: {input_path}")
            dataset = None
            continue

        bands, rows, columns = array.shape
        output_path = input_path.with_name(f"{input_path.stem}{output_suffix}.tif")
        output = driver.Create(str(output_path), columns, rows, bands, gdal.GDT_Byte)
        if output is None:
            print(f"[error] Could not create: {output_path}")
            dataset = None
            continue

        output.SetGeoTransform(dataset.GetGeoTransform())
        output.SetProjection(dataset.GetProjection())
        output.SetMetadata(dataset.GetMetadata())

        for index in range(bands):
            source_band = dataset.GetRasterBand(index + 1)
            target_band = output.GetRasterBand(index + 1)
            nodata = source_band.GetNoDataValue()
            if nodata is not None and 0 <= nodata <= 255:
                target_band.SetNoDataValue(nodata)
            target_band.WriteArray(array[index].astype(np.uint8))

        dataset = None
        output = None
        completed += 1
        print(f"[ok] {input_path.name} -> {output_path.name}")

    return completed, len(inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("first_value", type=float, help="Value to replace first")
    parser.add_argument("second_value", type=float, help="Value to replace second")
    parser.add_argument("--first-target", type=int, default=128)
    parser.add_argument("--second-target", type=int, default=255)
    parser.add_argument("--output-suffix", default="_remap")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    done, total = remap_rasters(
        args.input_dir,
        args.first_value,
        args.second_value,
        args.first_target,
        args.second_target,
        args.output_suffix,
    )
    print(f"Completed {done} of {total} file(s).")
