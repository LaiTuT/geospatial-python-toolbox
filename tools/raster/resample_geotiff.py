"""Batch-resample GeoTIFF files to a target pixel size with GDAL."""

from __future__ import annotations

import argparse
from pathlib import Path

def resample_directory(
    input_dir: Path,
    output_dir: Path,
    x_resolution: float,
    y_resolution: float,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Resample every GeoTIFF directly inside *input_dir*."""
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise RuntimeError(
            "GDAL is required. Install it with: conda install -c conda-forge gdal"
        ) from exc

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted((*input_dir.glob("*.tif"), *input_dir.glob("*.tiff")))
    completed = 0

    for input_path in inputs:
        output_path = output_dir / input_path.name
        if output_path.exists() and not overwrite:
            print(f"[skip] Output exists: {output_path}")
            continue

        options = gdal.WarpOptions(
            format="GTiff",
            xRes=x_resolution,
            yRes=y_resolution,
            resampleAlg=gdal.GRA_Bilinear,
            targetAlignedPixels=True,
        )
        result = gdal.Warp(str(output_path), str(input_path), options=options)
        if result is None:
            print(f"[error] GDAL could not resample: {input_path}")
            continue
        result = None
        completed += 1
        print(f"[ok] {input_path.name}")

    return completed, len(inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing GeoTIFF files")
    parser.add_argument("output_dir", type=Path, help="Directory for resampled files")
    parser.add_argument("--x-resolution", type=float, default=2.0, help="Output X pixel size")
    parser.add_argument("--y-resolution", type=float, default=2.0, help="Output Y pixel size")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    completed, total = resample_directory(
        args.input_dir,
        args.output_dir,
        args.x_resolution,
        args.y_resolution,
        args.overwrite,
    )
    print(f"Completed {completed} of {total} file(s).")
    return 0 if completed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
