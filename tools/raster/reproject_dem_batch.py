from pathlib import Path
import csv

from osgeo import gdal, osr


# ====== User settings ======
inputpath = r"D:\Dataproces\10m\mou_washinton"
outpath = r"D:\Dataproces\10m\mou_washinton_reprojected"

# Source CRS from the USGS 3DEP 1/3 arc-second DEM metadata:
# horizontal NAD83 geographic, units degree.
SOURCE_EPSG = 4269

# Set to None to auto-pick NAD83 / UTM zone by each tile center longitude.
# Or set manually, for example: TARGET_EPSG = 26911
TARGET_EPSG = None

# DEM is continuous bare-earth elevation data, so bilinear is usually appropriate.
RESAMPLING = "bilinear"

# Output resolution in projected units, meters for UTM.
X_RES = 10
Y_RES = 10

# File types to process.
FILE_EXTS = (".tif", ".tiff")

# Search subfolders too.
RECURSIVE = False

# Overwrite existing outputs.
OVERWRITE = True

# Optional processing log. Set to None if you do not want a CSV log.
LOG_CSV = "reproject_log.csv"
# ===========================


gdal.UseExceptions()


def nad83_utm_epsg_from_lon(lon: float) -> int:
    """Return EPSG code for NAD83 / UTM zone in the northern hemisphere."""
    zone = int((lon + 180) // 6) + 1
    if zone < 1 or zone > 60:
        raise ValueError(f"Longitude is outside valid UTM range: {lon}")
    return 26900 + zone


def raster_center_lon_lat(dataset: gdal.Dataset) -> tuple[float, float]:
    gt = dataset.GetGeoTransform()
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    center_px = width / 2
    center_py = height / 2
    x = gt[0] + center_px * gt[1] + center_py * gt[2]
    y = gt[3] + center_px * gt[4] + center_py * gt[5]

    src_wkt = dataset.GetProjection()
    if not src_wkt:
        src_srs = osr.SpatialReference()
        src_srs.ImportFromEPSG(SOURCE_EPSG)
    else:
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(src_wkt)

    geo_srs = src_srs.CloneGeogCS()
    transform = osr.CoordinateTransformation(src_srs, geo_srs)
    lon, lat, _ = transform.TransformPoint(x, y)
    return lon, lat


def iter_rasters(folder: Path):
    iterator = folder.rglob("*") if RECURSIVE else folder.glob("*")
    for path in iterator:
        if path.is_file() and path.suffix.lower() in FILE_EXTS:
            yield path


def reproject_one(src_path: Path, dst_path: Path) -> dict:
    src_ds = gdal.Open(str(src_path), gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"Could not open input raster: {src_path}")

    lon, lat = raster_center_lon_lat(src_ds)
    target_epsg = TARGET_EPSG or nad83_utm_epsg_from_lon(lon)
    dst_srs = f"EPSG:{target_epsg}"
    src_srs = src_ds.GetProjection() or f"EPSG:{SOURCE_EPSG}"

    if dst_path.exists() and OVERWRITE:
        dst_path.unlink()
    elif dst_path.exists():
        print(f"Skip existing: {dst_path}")
        src_ds = None
        return {
            "input": str(src_path),
            "output": str(dst_path),
            "center_lon": lon,
            "center_lat": lat,
            "source_epsg": SOURCE_EPSG,
            "target_epsg": target_epsg,
            "status": "skipped_existing",
        }

    warp_options = gdal.WarpOptions(
        srcSRS=src_srs,
        dstSRS=dst_srs,
        xRes=X_RES,
        yRes=Y_RES,
        resampleAlg=RESAMPLING,
        targetAlignedPixels=True,
        multithread=True,
        format="GTiff",
        creationOptions=[
            "TILED=YES",
            "COMPRESS=LZW",
            "BIGTIFF=IF_SAFER",
        ],
    )

    print(f"Processing: {src_path.name} -> {dst_path.name} ({dst_srs})")
    result = gdal.Warp(str(dst_path), src_ds, options=warp_options)
    if result is None:
        raise RuntimeError(f"GDAL warp failed: {src_path}")

    result.FlushCache()
    result = None
    src_ds = None

    return {
        "input": str(src_path),
        "output": str(dst_path),
        "center_lon": lon,
        "center_lat": lat,
        "source_epsg": SOURCE_EPSG,
        "target_epsg": target_epsg,
        "status": "processed",
    }


def write_log(rows: list[dict], log_path: Path) -> None:
    if not rows:
        return

    fieldnames = [
        "input",
        "output",
        "center_lon",
        "center_lat",
        "source_epsg",
        "target_epsg",
        "status",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    src_dir = Path(inputpath)
    dst_dir = Path(outpath)

    if not src_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    rasters = list(iter_rasters(src_dir))
    if not rasters:
        print(f"No raster files found in: {src_dir}")
        return

    log_rows = []
    for src_path in rasters:
        relative = src_path.relative_to(src_dir)
        dst_path = dst_dir / relative
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        log_rows.append(reproject_one(src_path, dst_path))

    if LOG_CSV:
        write_log(log_rows, dst_dir / LOG_CSV)

    print(f"Done. Processed {len(rasters)} raster(s).")


if __name__ == "__main__":
    main()
