import json
import os
import glob
import subprocess
from pathlib import Path
from osgeo import gdal, osr


EPT_JSON = r"https://usgs-lidar-public.s3.us-west-2.amazonaws.com/USGS_LPC_AL_25Co_B1_2017/ept.json"

DEM_DIR = r"D:\Dataproces\1m\AL_25Co_B1_2017\TIFF2"
DSM_DIR = r"D:\Dataproces\1m\AL_25Co_B1_2017\DSM2"

PDAL_EXE = "pdal"

RUN_ONLY_ONE = False

gdal.UseExceptions()
os.makedirs(DSM_DIR, exist_ok=True)


def get_epsg_from_wkt(wkt: str) -> str:
    """从 WKT 字符串动态获取 EPSG 代码"""
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    code = srs.GetAuthorityCode(None)
    if code:
        return f"EPSG:{code}"
    # 如果无法获取 Authority Code，尝试从 PROJCS 或 GEOGCS 获取
    code = srs.GetAuthorityCode("PROJCS") or srs.GetAuthorityCode("GEOGCS")
    if code:
        return f"EPSG:{code}"
    raise RuntimeError(f"无法从 WKT 获取 EPSG 代码")


def get_dem_info(dem_path: str):
    """读取 DEM 的网格与投影信息"""
    ds = gdal.Open(dem_path)
    if ds is None:
        raise RuntimeError(f"GDAL 打不开 DEM: {dem_path}")

    gt = ds.GetGeoTransform()
    x0, px_w, _, y0, _, px_h = gt
    w = ds.RasterXSize
    h = ds.RasterYSize

    # DEM extent（注意 px_h 通常为负）
    x1 = x0 + w * px_w
    y1 = y0 + h * px_h
    xmin, xmax = (x0, x1) if x0 < x1 else (x1, x0)
    ymin, ymax = (y1, y0) if y1 < y0 else (y0, y1)

    proj_wkt = ds.GetProjection()
    resx = abs(px_w)
    resy = abs(px_h)
    ds = None
    return xmin, ymin, xmax, ymax, w, h, resx, resy, gt, proj_wkt


def bounds_dem_to_3857(xmin, ymin, xmax, ymax, dem_proj_wkt: str):
    """把 DEM 边界框(DEM CRS) 转成 EPSG:3857 边界框，供 readers.ept bounds 用"""
    srs_dem = osr.SpatialReference()
    srs_dem.ImportFromWkt(dem_proj_wkt)

    srs_3857 = osr.SpatialReference()
    srs_3857.ImportFromEPSG(3857)

    tx = osr.CoordinateTransformation(srs_dem, srs_3857)

    corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    xs, ys = [], []
    for x, y in corners:
        X, Y, _ = tx.TransformPoint(x, y)
        xs.append(X)
        ys.append(Y)

    return min(xs), min(ys), max(xs), max(ys)


def create_empty_dsm_like_dem(dem_path: str, out_dsm: str, nodata=-9999.0):
    """如果该瓦片范围内没点，输出一张全 NoData 的 DSM（网格完全等同 DEM）"""
    xmin, ymin, xmax, ymax, w, h, resx, resy, gt, proj = get_dem_info(dem_path)

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        out_dsm, w, h, 1, gdal.GDT_Float32,
        options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"]
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)

    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.Fill(nodata)
    band.FlushCache()
    ds.FlushCache()
    ds = None


def warp_to_dem_grid(dem_path: str, src_dsm: str, out_dsm: str, nodata=-9999.0):
    """把 raw DSM 强制 warp 到 DEM 的网格（extent + 分辨率 + 对齐）"""
    xmin, ymin, xmax, ymax, w, h, resx, resy, gt, proj = get_dem_info(dem_path)

    warp_opts = gdal.WarpOptions(
        format="GTiff",
        dstSRS=proj,
        outputBounds=(xmin, ymin, xmax, ymax),
        xRes=resx,
        yRes=resy,
        targetAlignedPixels=True,  # 等价 -tap：网格对齐
        resampleAlg="near",        # DSM 用 near 更安全（避免平滑）
        dstNodata=nodata,
        creationOptions=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"]
    )

    out = gdal.Warp(out_dsm, src_dsm, options=warp_opts)
    if out is None:
        raise RuntimeError("gdal.Warp 失败")
    out.FlushCache()
    out = None


def run_one(dem_path: str):
    dem_name = Path(dem_path).name
    stem = Path(dem_path).stem

    out_dsm = os.path.join(DSM_DIR, f"{stem}_DSM.tif")
    raw_dsm = os.path.join(DSM_DIR, f"{stem}_DSM_raw.tif")
    tmp_pipeline = os.path.join(DSM_DIR, "_tmp_pipeline.json")

    if os.path.exists(out_dsm) and os.path.getsize(out_dsm) > 0:
        print(f"[SKIP] {dem_name} -> 已存在")
        return

    # 读 DEM 网格 + DEM->3857 bounds
    xmin, ymin, xmax, ymax, w, h, resx, resy, gt, dem_proj = get_dem_info(dem_path)
    xmin3857, ymin3857, xmax3857, ymax3857 = bounds_dem_to_3857(xmin, ymin, xmax, ymax, dem_proj)

    # 动态获取 DEM 的 EPSG 代码
    dem_epsg = get_epsg_from_wkt(dem_proj)
    print(f"[INFO] DEM 坐标系: {dem_epsg}")

    # 组 pipeline：读 EPT(3857) + 裁剪(3857 bounds) + 投回 DEM CRS + 写 raw DSM(不强行对齐)
    pipeline = {
        "pipeline": [
            {
                "type": "readers.ept",
                "filename": EPT_JSON,
                "resolution": resx,
                "bounds": f"([{xmin3857},{xmax3857}],[{ymin3857},{ymax3857}])"
            },
            {
                "type": "filters.reprojection",
                "out_srs": dem_epsg
            },
            {
                "type": "writers.gdal",
                "filename": raw_dsm,
                "gdaldriver": "GTiff",
                "output_type": "max",
                "data_type": "float32",
                "nodata": -9999,
                "resolution": resx
            }
        ]
    }

    with open(tmp_pipeline, "w", encoding="utf-8") as f:
        json.dump(pipeline, f)

    print(f"[RUN ] {dem_name}")
    result = subprocess.run([PDAL_EXE, "pipeline", tmp_pipeline], text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # 如果该瓦片没点：输出一张空 DSM（对齐 DEM）
    if "no points for output" in (result.stderr or "").lower():
        print(f"[SKIP] {dem_name} -> 该范围内无点，生成全 NoData DSM")
        create_empty_dsm_like_dem(dem_path, out_dsm, nodata=-9999)
        # 清理 raw
        if os.path.exists(raw_dsm):
            try:
                os.remove(raw_dsm)
            except OSError:
                pass
        return

    result.check_returncode()

    # 关键：warp 到 DEM 网格，确保 Origin/PixelSize/Size 完全一致
    warp_to_dem_grid(dem_path, raw_dsm, out_dsm, nodata=-9999)

    # 可选：删除 raw
    try:
        os.remove(raw_dsm)
    except OSError:
        pass

    print(f"[DONE] -> {out_dsm}")


def main():
    dem_files = sorted(glob.glob(os.path.join(DEM_DIR, "*.tif")))
    if not dem_files:
        raise SystemExit(f"没在 {DEM_DIR} 找到任何 .tif DEM 文件")

    if RUN_ONLY_ONE:
        dem_files = dem_files[:1]

    for dem in dem_files:
        run_one(dem)

    print("全部完成。")


if __name__ == "__main__":
    main()
