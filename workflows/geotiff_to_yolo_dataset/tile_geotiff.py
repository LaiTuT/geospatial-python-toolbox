#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
from typing import List, Optional, Tuple
from osgeo import gdal, gdal_array
import numpy as np

def compute_tile_geotransform(base_gt: Tuple[float, float, float, float, float, float], x_off: int, y_off: int):
    originX = base_gt[0] + x_off * base_gt[1] + y_off * base_gt[2]
    originY = base_gt[3] + x_off * base_gt[4] + y_off * base_gt[5]
    return originX, base_gt[1], base_gt[2], originY, base_gt[4], base_gt[5]

def list_tifs_in_dir(dir_path: str) -> List[str]:
    exts = (".tif", ".tiff", ".TIF", ".TIFF")
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(exts)]

def tile_single_tif(input_path: str, output_dir: str, tile_size: int = 1024, overlap: int = 0,
                    nodata: Optional[float] = None, compress: str = "LZW",
                    bigtiff: str = "IF_SAFER", only_valid: bool = False) -> int:
    gdal.UseExceptions()
    ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"无法打开影像: {input_path}")
        return 0

    width = ds.RasterXSize
    height = ds.RasterYSize
    bands = ds.RasterCount
    proj = ds.GetProjection()
    gt = ds.GetGeoTransform(can_return_null=True)

    os.makedirs(output_dir, exist_ok=True)

    step = tile_size - overlap if overlap < tile_size else 1
    x_tiles = math.ceil((width - overlap) / step)
    y_tiles = math.ceil((height - overlap) / step)

    driver = gdal.GetDriverByName("GTiff")
    creation_options = ["TILED=YES"]
    if compress:
        creation_options.append(f"COMPRESS={compress}")
    if bigtiff:
        creation_options.append(f"BIGTIFF={bigtiff}")

    print(f"源影像: {input_path}")
    print(f"尺寸: {width}x{height}, 波段: {bands}")
    print(f"瓦片大小: {tile_size}x{tile_size}, 重叠: {overlap}")
    print(f"预计瓦片数量: {x_tiles * y_tiles}")
    print(f"输出目录: {output_dir}")

    total = 0
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    for ty in range(y_tiles):
        # 对于最后一个瓦片，向后退缩以确保大小为 tile_size
        if ty == y_tiles - 1:
            y_off = max(0, height - tile_size)
        else:
            y_off = ty * step
        tile_h = tile_size

        for tx in range(x_tiles):
            # 对于最后一个瓦片，向后退缩以确保大小为 tile_size
            if tx == x_tiles - 1:
                x_off = max(0, width - tile_size)
            else:
                x_off = tx * step
            tile_w = tile_size

            if only_valid and nodata is not None:
                # 粗略检查是否全为 NoData（逐波段），使用更稳健的 ReadRaster + NumPy 转换
                all_nodata = True
                for b in range(1, bands + 1):
                    band = ds.GetRasterBand(b)
                    try:
                        buf = band.ReadRaster(
                            x_off, y_off, tile_w, tile_h,
                            buf_xsize=tile_w, buf_ysize=tile_h,
                            buf_type=band.DataType
                        )
                        if buf is None:
                            # 若读取失败，认为该波段非全 NoData，避免误删
                            all_nodata = False
                            break
                        np_type = gdal_array.GDALTypeCodeToNumericTypeCode(band.DataType)
                        arr = np.frombuffer(buf, dtype=np.dtype(np_type))
                        # 读取为行优先，形状为 (tile_h, tile_w)
                        if arr.size != tile_w * tile_h:
                            # 尺寸异常，保守处理为非全 NoData
                            all_nodata = False
                            break
                        arr = arr.reshape((tile_h, tile_w))
                        # 浮点比较容忍度：若数据是浮点类型，则用近似比较
                        if np.issubdtype(arr.dtype, np.floating):
                            # 允许极小误差
                            if not np.allclose(arr, nodata, atol=1e-8, rtol=0):
                                all_nodata = False
                                break
                        else:
                            if not (arr == nodata).all():
                                all_nodata = False
                                break
                    except Exception:
                        # 读取/转换异常时，不跳过该瓦片
                        all_nodata = False
                        break
                if all_nodata:
                    continue

            tile_name = f"{base_name}_x{tx}_y{ty}.tif"
            out_path = os.path.join(output_dir, tile_name)

            out_ds = driver.Create(out_path, tile_size, tile_size, bands, ds.GetRasterBand(1).DataType, options=creation_options)
            if out_ds is None:
                print(f"创建瓦片失败: {out_path}")
                continue

            # 写入数据（逐波段）
            for b in range(1, bands + 1):
                band = ds.GetRasterBand(b)
                out_band = out_ds.GetRasterBand(b)
                buf = band.ReadRaster(x_off, y_off, tile_w, tile_h, buf_xsize=tile_w, buf_ysize=tile_h)
                out_band.WriteRaster(0, 0, tile_w, tile_h, buf)
                if nodata is not None:
                    out_band.SetNoDataValue(nodata)

            # 复制空间参考
            if proj:
                out_ds.SetProjection(proj)
            if gt:
                out_gt = compute_tile_geotransform(gt, x_off, y_off)
                out_ds.SetGeoTransform(out_gt)

            out_ds.FlushCache()
            out_ds = None
            total += 1

    print(f"完成，生成瓦片数: {total}")
    return total

def tile_inputs(inputs: List[str], output_root: str, tile_size: int = 1024, overlap: int = 0,
                nodata: Optional[float] = None, compress: str = "LZW",
                bigtiff: str = "IF_SAFER", only_valid: bool = False) -> int:
    total = 0
    for inp in inputs:
        if os.path.isdir(inp):
            tifs = list_tifs_in_dir(inp)
            if not tifs:
                print(f"目录无 TIF 文件: {inp}")
                continue
            # 为每个源文件创建子目录，避免重名
            for tif in tifs:
                subdir = os.path.join(output_root, os.path.splitext(os.path.basename(tif))[0])
                total += tile_single_tif(
                    tif, subdir, tile_size, overlap, nodata, compress, bigtiff, only_valid
                )
        elif os.path.isfile(inp):
            subdir = os.path.join(output_root, os.path.splitext(os.path.basename(inp))[0])
            total += tile_single_tif(
                inp, subdir, tile_size, overlap, nodata, compress, bigtiff, only_valid
            )
        else:
            print(f"路径不存在: {inp}")
    print(f"总计生成瓦片数: {total}")
    return total

def main():
    """
    写死运行配置：
    - 修改 `INPUT_MODE` 为 'file' 或 'dir'
    - 修改 `INPUT_PATHS` 为一个或多个路径（文件或目录）
    - 修改 `OUTPUT_ROOT` 为输出根目录
    - 其他参数也在此处直接调整
    """
    INPUT_MODE = 'dir'  # 可选 'file' 或 'dir'；当为 'dir' 时，会批量遍历该目录下所有 TIF
    INPUT_PATHS = r"D:\Dataproces\10m\moutain_reprojected",
            # 文件示例（当 INPUT_MODE='file' 生效）
        # r"d:\data\imagery\big_image.tif",  # 单文件示例（当 INPUT_MODE='file' 生效）

    OUTPUT_ROOT = r"D:\Dataproces\10m\moutain_reprojected\tiles"
    TILE_SIZE = 1024
    OVERLAP = 0
    NODATA = -9999  # 如 0 或 255 或 -9999
    COMPRESS = "LZW"  # 可选 LZW/DEFLATE/JPEG 等
    BIGTIFF = "IF_SAFER"  # YES/NO/IF_NEEDED/IF_SAFER
    ONLY_VALID = True    # True 时跳过全 NoData 的瓦片

    if INPUT_MODE not in ('file', 'dir'):
        raise ValueError("INPUT_MODE 只能为 'file' 或 'dir'")

    # 将模式应用到路径：如果是 'dir'，就只取存在的目录；如果是 'file'，只取存在的文件
    normalized_inputs = []
    for p in INPUT_PATHS:
        if INPUT_MODE == 'dir' and os.path.isdir(p):
            normalized_inputs.append(p)
        elif INPUT_MODE == 'file' and os.path.isfile(p):
            normalized_inputs.append(p)
        else:
            print(f"跳过无效路径（或模式不匹配）: {p}")

    if not normalized_inputs:
        print("未找到可用输入，请检查 INPUT_MODE 与 INPUT_PATHS")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    tile_inputs(
        inputs=normalized_inputs,
        output_root=OUTPUT_ROOT,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        nodata=NODATA,
        compress=COMPRESS,
        bigtiff=BIGTIFF,
        only_valid=ONLY_VALID
    )

if __name__ == "__main__":
    main()
