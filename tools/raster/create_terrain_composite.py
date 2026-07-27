"""
使用 GDAL DEMProcessing 生成坡度和山体阴影
"""

import numpy as np
import os
import glob
from osgeo import gdal
from PIL import Image


def process_dem_gdal_demprocessing(tif_path, output_tif_path):
    """
    使用 GDAL DEMProcessing 生成地形指标，输出为 3 波段 TIF
    """
    temp_dir = os.path.dirname(output_tif_path)
    basename = os.path.splitext(os.path.basename(tif_path))[0]

    # 临时文件
    slope_tif = os.path.join(temp_dir, f"{basename}_slope.tif")
    hillshade_tif = os.path.join(temp_dir, f"{basename}_hillshade.tif")

    try:
        print(f"  处理: {os.path.basename(tif_path)}")

        # ========== 1. 读取原始 DEM ==========
        dem_ds = gdal.Open(tif_path)
        dem_array = dem_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        dem_ds = None

        print(f"    DEM 大小: {dem_array.shape}")
        print(f"    DEM 范围: [{dem_array.min():.2f}, {dem_array.max():.2f}]")

        # ========== 2. 标准化高程 ==========
        dem_mean = dem_array.mean()
        dem_std = dem_array.std()
        if dem_std > 0:
            elevation = (dem_array - dem_mean) / dem_std
        else:
            elevation = np.zeros_like(dem_array)
        elevation = elevation.astype(np.float32)

        # ========== 3. 使用 GDAL 生成坡度 ==========
        print(f"    生成坡度...")
        gdal.DEMProcessing(
            slope_tif,
            tif_path,
            'slope',              # 坡度（度）
            azimuth=0,
            altitude=90,
            zFactor=1.0
        )

        # 读取坡度
        slope_ds = gdal.Open(slope_tif)
        slope_array = slope_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        slope_ds = None

        # 坡度归一化到 0-1
        slope_min = slope_array.min()
        slope_max = slope_array.max()
        if slope_max > slope_min:
            slope = (slope_array - slope_min) / (slope_max - slope_min)
        else:
            slope = np.zeros_like(slope_array)
        slope = slope.astype(np.float32)
        print(f"    坡度范围: [{slope.min():.3f}, {slope.max():.3f}]")

        # ========== 4. 使用 GDAL 生成山体阴影 ==========
        print(f"    生成山体阴影...")
        gdal.DEMProcessing(
            hillshade_tif,
            tif_path,
            'hillshade',
            azimuth=315,          # 光源方向（NW）
            altitude=45           # 光源仰角
        )

        # 读取山体阴影
        hillshade_ds = gdal.Open(hillshade_tif)
        hillshade_array = hillshade_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        hillshade_ds = None

        # 山体阴影已经是 0-255
        hillshade = (hillshade_array / 255.0).astype(np.float32)
        print(f"    阴影范围: [{hillshade.min():.3f}, {hillshade.max():.3f}]")

        # ========== 5. 合成 3 通道并保存为 TIF ==========
        height, width = elevation.shape

        # 创建输出 TIF
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_tif_path, width, height, 3, gdal.GDT_Float32)

        if out_ds is None:
            raise RuntimeError(f"无法创建输出文件: {output_tif_path}")

        # 写入三个波段
        out_ds.GetRasterBand(1).WriteArray(elevation)
        out_ds.GetRasterBand(2).WriteArray(slope)
        out_ds.GetRasterBand(3).WriteArray(hillshade)

        # 复制地理参考信息
        dem_ds_ref = gdal.Open(tif_path)
        if dem_ds_ref:
            out_ds.SetProjection(dem_ds_ref.GetProjection())
            out_ds.SetGeoTransform(dem_ds_ref.GetGeoTransform())
            dem_ds_ref = None

        out_ds.FlushCache()
        out_ds = None

        print(f"  ✅ 保存: {os.path.basename(output_tif_path)}")

        # 清理临时文件
        if os.path.exists(slope_tif):
            os.remove(slope_tif)
        if os.path.exists(hillshade_tif):
            os.remove(hillshade_tif)

        return True

    except Exception as e:
        print(f"  ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_process(input_dir, output_dir):
    """批量处理"""
    os.makedirs(output_dir, exist_ok=True)

    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    tif_files += glob.glob(os.path.join(input_dir, "*.TIF"))

    if not tif_files:
        print(f"⚠️  未找到 TIF 文件")
        return 0, 0

    print(f"发现 {len(tif_files)} 个文件\n")

    success, fail = 0, 0
    for i, tif_path in enumerate(tif_files, 1):
        print(f"[{i}/{len(tif_files)}]")

        output_name = os.path.basename(tif_path).replace('.tif', '.tif').replace('.TIF', '.tif')
        output_path = os.path.join(output_dir, output_name)

        if process_dem_gdal_demprocessing(tif_path, output_path):
            success += 1
        else:
            fail += 1
        print()

    return success, fail


if __name__ == '__main__':
    print("=" * 70)
    print("GDAL DEMProcessing: 生成地形指标 (3波段TIF)")
    print("=" * 70)

    train_input = r"D:\YOLOV8\Extract_RV\ultralytics\data\images\train"
    train_output = r"D:\YOLOV8\Extract_RV\ultralytics\data\images3\train"

    val_input = r"D:\YOLOV8\Extract_RV\ultralytics\data\images\val"
    val_output = r"D:\YOLOV8\Extract_RV\ultralytics\data\images3\val"

    print("\n【训练集】")
    print("-" * 70)
    s1, f1 = batch_process(train_input, train_output)

    print("\n【验证集】")
    print("-" * 70)
    s2, f2 = batch_process(val_input, val_output)

    print("\n" + "=" * 70)
    print(f"总计: ✅ {s1+s2} 成功, ❌ {f1+f2} 失败")
    print("=" * 70)
