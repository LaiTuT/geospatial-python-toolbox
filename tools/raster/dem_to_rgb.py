#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEM to RGB Image Converter
将 DEM（数字高程模型）转换为带山体阴影效果的 RGB 图像

原理（与 LInSAR 完全一致）：
1. 高程 → 颜色映射（21级颜色表线性插值）
2. Sobel 梯度 → 坡度/坡向计算
3. 山体阴影计算（根据光源位置）
4. Lab 色彩空间合成（调整亮度，保持色调）

依赖：
    pip install numpy opencv-python gdal

使用：
    python dem_to_rgb.py input_dem.tif output_rgb.png --azimuth -45 --altitude 45
"""

import argparse
import os
import glob
import json
import numpy as np
import cv2
from osgeo import gdal


# ==================== 配置文件读取 ====================

def load_config(config_path):
    """
    加载配置文件

    Args:
        config_path: 配置文件路径（JSON 格式）

    Returns:
        config: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def get_default_config_path():
    """获取默认配置文件路径（与脚本同目录）"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'dem_to_rgb_config.json')


# ==================== 预定义颜色表 ====================

def get_color_table(name='linsar'):
    """
    获取预定义的颜色表

    Args:
        name: 颜色表名称
            - 'linsar': LInSAR 默认配色（21级，浅黄绿→绿→黄→橙→棕→白→蓝）
            - 'terrain': 经典地形配色（绿-黄-棕-白）
            - 'elevation': 高程配色（蓝-绿-黄-红）
            - 'dem': DEM 标准配色（类似 ArcGIS）

    Returns:
        numpy array of shape (N, 3), RGB values in [0, 255]
    """
    color_tables = {
        # LInSAR 默认21级颜色表，从低到高：
        # 浅黄绿 → 绿 → 黄 → 橙 → 棕 → 浅棕 → 白 → 浅蓝 → 蓝
        'linsar': np.array([
            [200, 215, 133],   # 浅黄绿（低地）
            [171, 217, 177],   # 浅绿
            [124, 196, 120],   # 绿色
            [117, 193, 120],   # 绿色
            [175, 204, 166],   # 浅绿
            [219, 208, 78],    # 黄绿
            [241, 207, 14],    # 亮黄
            [242, 167, 0],     # 橙黄
            [192, 159, 13],    # 棕黄
            [211, 192, 112],   # 浅棕
            [240, 219, 161],   # 浅棕
            [252, 236, 192],   # 米黄
            [248, 250, 234],   # 近白
            [229, 254, 250],   # 浅青
            [219, 255, 253],   # 浅青
            [214, 251, 252],   # 浅蓝
            [183, 244, 247],   # 浅蓝
            [115, 224, 241],   # 天蓝
            [29, 188, 239],    # 蓝色
            [0, 137, 245],     # 深蓝
            [0, 80, 250],      # 深蓝（最高）
        ], dtype=np.float32),

        'terrain': np.array([
            [0, 128, 0],       # 深绿（低地）
            [0, 200, 0],       # 绿色
            [128, 200, 0],     # 黄绿
            [200, 200, 0],     # 黄色
            [200, 150, 0],     # 橙黄
            [180, 120, 60],    # 棕色
            [150, 100, 50],    # 深棕
            [200, 180, 140],   # 浅棕
            [230, 220, 200],   # 浅灰
            [255, 255, 255],   # 白色（雪线）
        ], dtype=np.float32),

        'elevation': np.array([
            [0, 0, 200],       # 深蓝（最低）
            [0, 100, 255],     # 蓝色
            [0, 200, 200],     # 青色
            [0, 255, 0],       # 绿色
            [128, 255, 0],     # 黄绿
            [255, 255, 0],     # 黄色
            [255, 128, 0],     # 橙色
            [255, 0, 0],       # 红色
            [128, 0, 0],       # 深红
            [255, 255, 255],   # 白色（最高）
        ], dtype=np.float32),

        'dem': np.array([
            [0, 100, 0],       # 深绿
            [50, 150, 50],     # 绿色
            [100, 180, 80],    # 浅绿
            [150, 200, 100],   # 黄绿
            [200, 200, 100],   # 黄色
            [220, 180, 80],    # 土黄
            [200, 150, 80],    # 棕黄
            [180, 120, 80],    # 棕色
            [160, 100, 80],    # 深棕
            [200, 200, 200],   # 灰色
            [255, 255, 255],   # 白色
        ], dtype=np.float32),
    }

    return color_tables.get(name, color_tables['linsar'])


# ==================== 核心算法 ====================

def map_elevation_to_color(dem, mask, color_table, min_z=None, max_z=None):
    """
    将高程映射为颜色

    Args:
        dem: 高程数据 (H, W)
        mask: 有效值掩码 (H, W), True 表示有效
        color_table: 颜色表 (N, 3)
        min_z: 最小高程（None 则自动计算）
        max_z: 最大高程（None 则自动计算）

    Returns:
        color_image: RGB 图像 (H, W, 3), 范围 [0, 1]
    """
    if min_z is None:
        min_z = dem[mask].min()
    if max_z is None:
        max_z = dem[mask].max()

    # 归一化到 [0, N-1]
    n_colors = len(color_table)

    # 与 LInSAR 一致：对高程值做 clip
    z = np.clip(dem, min_z, max_z)

    normalized = np.zeros_like(dem, dtype=np.float32)
    normalized[mask] = (z[mask] - min_z) / (max_z - min_z) * (n_colors - 1)

    # 线性插值
    lower = np.floor(normalized).astype(np.int32)
    upper = np.ceil(normalized).astype(np.int32)

    # 边界处理
    lower = np.clip(lower, 0, n_colors - 1)
    upper = np.clip(upper, 0, n_colors - 1)

    # 插值因子
    t = normalized - lower

    # 线性插值
    color_image = np.zeros((dem.shape[0], dem.shape[1], 3), dtype=np.float32)
    for c in range(3):
        color_image[:, :, c] = color_table[lower, c] * (1 - t) + color_table[upper, c] * t

    # 归一化到 [0, 1]
    color_image = color_image / 255.0

    return color_image


def compute_hillshade(dem, mask, cell_size, azimuth=-45.0, altitude=45.0, z_scale=1.0):
    """
    计算山体阴影

    参考: http://edndoc.esri.com/arcobjects/9.2/net/shared/geoprocessing/spatial_analyst_tools/how_hillshade_works.htm

    Args:
        dem: 高程数据 (H, W)
        mask: 有效值掩码 (H, W)
        cell_size: 像元大小（分辨率）
        azimuth: 光源方位角（度），从北顺时针，默认 -45（西北）
        altitude: 光源高度角（度），从地平线向上，默认 45
        z_scale: Z 方向缩放因子，用于增强/减弱地形起伏

    Returns:
        hillshade: 阴影值 (H, W), 范围 [0, 1]
    """
    DEG2RAD = np.pi / 180.0
    RAD2DEG = 180.0 / np.pi

    # 转换光源参数
    zenith = 90.0 - altitude  # 天顶角
    azimuth_rad = (360.0 - azimuth + 90.0) * DEG2RAD  # 转换为数学角度
    if azimuth_rad > 2 * np.pi:
        azimuth_rad -= 2 * np.pi

    # 计算 Sobel 梯度
    # 注意：OpenCV 的 Sobel 输出需要缩放
    grad_x = cv2.Sobel(dem, cv2.CV_32F, 1, 0, ksize=3) / (8.0 * cell_size)
    grad_y = cv2.Sobel(dem, cv2.CV_32F, 0, 1, ksize=3) / (8.0 * cell_size)

    # 计算坡度（度）
    slope = np.arctan(z_scale * np.sqrt(grad_x**2 + grad_y**2)) * RAD2DEG

    # 计算坡向（度）
    aspect = np.arctan2(grad_y, -grad_x) * RAD2DEG
    aspect = np.where(aspect < 0, aspect + 360, aspect)

    # 计算山体阴影
    cos_zenith = np.cos(zenith * DEG2RAD)
    sin_zenith = np.sin(zenith * DEG2RAD)

    hillshade = (cos_zenith * np.cos(slope * DEG2RAD) +
                 sin_zenith * np.sin(slope * DEG2RAD) *
                 np.cos((azimuth_rad - aspect * DEG2RAD)))

    # 限制范围 [0, 1]
    hillshade = np.clip(hillshade, 0, 1)

    # 无效值区域设为 0
    hillshade[~mask] = 0

    return hillshade


def blend_color_and_shade(color_image, hillshade, mask, shade_boost=1.0):
    """
    在 Lab 色彩空间中合成颜色和阴影

    在 Lab 空间中，L 通道代表亮度，a、b 通道代表颜色。
    只调整 L 通道可以改变明暗而不改变色调。

    Args:
        color_image: RGB 颜色图像 (H, W, 3), 范围 [0, 1]
        hillshade: 阴影值 (H, W), 范围 [0, 1]
        mask: 有效值掩码 (H, W)
        shade_boost: 阴影增强因子，>1 增强阴影对比度，<1 减弱

    Returns:
        result: RGBA 图像 (H, W, 4), 范围 [0, 255]
    """
    # RGB → Lab（与 LInSAR 一致：使用 float32，L 范围 0-100）
    # 注意：OpenCV 对 float32 输入，Lab 的 L 通道范围是 0-100
    color_32f = color_image.astype(np.float32)
    lab = cv2.cvtColor(color_32f, cv2.COLOR_RGB2Lab)

    # 增强阴影对比度（可选）
    if shade_boost != 1.0:
        # 将 hillshade 从 [0,1] 调整到增强后的范围
        # shade_boost > 1: 暗的更暗，亮的更亮
        hillshade_adjusted = np.power(hillshade, 1.0 / shade_boost)
    else:
        hillshade_adjusted = hillshade

    # 调整 L 通道（亮度）
    # L 通道范围是 [0, 100]（float32 输入时）
    lab[:, :, 0] = lab[:, :, 0] * hillshade_adjusted
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 100)

    # Lab → RGB
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    # 添加 Alpha 通道
    result = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    result[:, :, :3] = rgb
    result[:, :, 3] = (mask * 255).astype(np.uint8)  # 有效区域不透明

    return result


# ==================== 文件读写 ====================

def read_geotiff(filepath):
    """
    读取 GeoTIFF 文件

    Args:
        filepath: 文件路径

    Returns:
        dem: 高程数据 (H, W)
        transform: 仿射变换参数 (6,)
        projection: 投影信息 (WKT)
        nodata: 无效值
    """
    ds = gdal.Open(filepath)
    if ds is None:
        raise ValueError(f"无法打开文件: {filepath}")

    band = ds.GetRasterBand(1)
    dem = band.ReadAsArray().astype(np.float32)

    transform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    nodata = band.GetNoDataValue()

    return dem, transform, projection, nodata


def write_geotiff(filepath, image, transform, projection, nodata=None):
    """
    写入 GeoTIFF 文件（保留坐标信息）

    Args:
        filepath: 输出路径
        image: 图像数据 (H, W, C)
        transform: 仿射变换参数
        projection: 投影信息
        nodata: 无效值
    """
    height, width = image.shape[:2]
    bands = image.shape[2] if len(image.shape) == 3 else 1

    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(filepath, width, height, bands, gdal.GDT_Byte)

    ds.SetGeoTransform(transform)
    ds.SetProjection(projection)

    if bands == 1:
        ds.GetRasterBand(1).WriteArray(image)
        if nodata is not None:
            ds.GetRasterBand(1).SetNoDataValue(nodata)
    else:
        for i in range(bands):
            ds.GetRasterBand(i + 1).WriteArray(image[:, :, i])

    ds.FlushCache()


def dem_to_rgb(dem, mask, cell_size, color_table='linsar',
               azimuth=-45.0, altitude=45.0, z_scale=1.0,
               min_z=None, max_z=None, shade_boost=1.0):
    """
    将 DEM 转换为 RGB 图像（核心函数）

    Args:
        dem: 高程数据 (H, W)
        mask: 有效值掩码 (H, W)
        cell_size: 像元大小（分辨率）
        color_table: 颜色表名称或数组
        azimuth: 光源方位角（度）
        altitude: 光源高度角（度）
        z_scale: Z 方向缩放
        min_z: 最小高程
        max_z: 最大高程
        shade_boost: 阴影增强因子，>1 增强对比度

    Returns:
        rgb_image: RGBA 图像 (H, W, 4), 范围 [0, 255]
    """
    # 获取颜色表
    if isinstance(color_table, str):
        colors = get_color_table(color_table)
    else:
        colors = np.array(color_table, dtype=np.float32)

    # Step 1: 高程 → 颜色
    color_image = map_elevation_to_color(dem, mask, colors, min_z, max_z)

    # Step 2: 计算山体阴影
    hillshade = compute_hillshade(dem, mask, cell_size, azimuth, altitude, z_scale)

    # Step 3: 合成颜色和阴影
    rgb_image = blend_color_and_shade(color_image, hillshade, mask, shade_boost)

    return rgb_image


# ==================== 单文件处理 ====================

def process_single_file(input_path, output_path, args):
    """
    处理单个 DEM 文件

    Args:
        input_path: 输入 DEM 文件路径
        output_path: 输出 RGB 文件路径
        args: 命令行参数
    """
    print(f"读取 DEM 文件: {input_path}")
    dem, transform, projection, nodata = read_geotiff(input_path)

    # 处理无效值
    nodata_val = args.nodata if args.nodata is not None else nodata

    if nodata_val is not None:
        mask = (dem != nodata_val) & (~np.isnan(dem))
    else:
        mask = ~np.isnan(dem)

    # 与 LInSAR 一致：对 mask 做 3x3 腐蚀，避免边界梯度计算错误
    mask_uint8 = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_eroded = cv2.erode(mask_uint8, kernel)
    mask = mask_eroded == 255  # 转回 bool 类型

    # 获取像元大小
    cell_size = abs(transform[1])  # 假设 x 和 y 方向分辨率相同

    print(f"DEM 大小: {dem.shape[1]} x {dem.shape[0]}")
    print(f"像元大小: {cell_size}")
    print(f"高程范围: {dem[mask].min():.2f} ~ {dem[mask].max():.2f}")

    # 转换
    print("正在生成 RGB 图像...")
    rgb_image = dem_to_rgb(
        dem, mask, cell_size,
        color_table=args.colortable,
        azimuth=args.azimuth,
        altitude=args.altitude,
        z_scale=args.zscale,
        min_z=args.minz,
        max_z=args.maxz,
        shade_boost=args.shade_boost
    )

    # 输出
    print(f"保存输出文件: {output_path}")
    ext = output_path.lower().split('.')[-1]

    if ext in ['tif', 'tiff']:
        # GeoTIFF（保留坐标）
        write_geotiff(output_path, rgb_image, transform, projection)
    else:
        # 普通图像格式
        if ext == 'png':
            cv2.imwrite(output_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2BGRA))
        else:
            # JPEG 不支持 Alpha 通道
            rgb_no_alpha = rgb_image[:, :, :3]
            cv2.imwrite(output_path, cv2.cvtColor(rgb_no_alpha, cv2.COLOR_RGB2BGR))

    print(f"完成: {output_path}")


# ==================== 批量处理 ====================

def process_batch(input_dir, output_dir, args):
    """
    批量处理文件夹下的所有 TIF 文件

    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
        args: 命令行参数
    """
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有 .tif 和 .tiff 文件（不递归子文件夹）
    tif_files = glob.glob(os.path.join(input_dir, '*.tif')) + \
                glob.glob(os.path.join(input_dir, '*.tiff')) + \
                glob.glob(os.path.join(input_dir, '*.TIF')) + \
                glob.glob(os.path.join(input_dir, '*.TIFF'))

    # 去重（处理大小写重复的情况）
    tif_files = list(set(tif_files))
    tif_files.sort()  # 按文件名排序

    if not tif_files:
        print(f"警告: 在 {input_dir} 中未找到任何 TIF 文件")
        return

    print(f"找到 {len(tif_files)} 个 TIF 文件")
    print(f"输出文件夹: {output_dir}")
    print("-" * 50)

    success_count = 0
    fail_count = 0

    for i, input_path in enumerate(tif_files):
        # 生成输出文件名
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{basename}.png")

        print(f"\n[{i+1}/{len(tif_files)}] 处理: {os.path.basename(input_path)}")

        try:
            process_single_file(input_path, output_path, args)
            success_count += 1
        except Exception as e:
            print(f"错误: 处理失败 - {e}")
            fail_count += 1

    print("\n" + "=" * 50)
    print(f"批量处理完成！成功: {success_count}, 失败: {fail_count}")


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(
        description='将 DEM 转换为带山体阴影效果的 RGB 图像',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 单文件模式（使用默认的LInSAR配色）
  python dem_to_rgb.py input_dem.tif output_rgb.png

  # 批量模式 - 使用配置文件
  python dem_to_rgb.py --batch

  # 批量模式 - 指定配置文件
  python dem_to_rgb.py --batch --config my_config.json

  # 批量模式 - 指定文件夹（覆盖配置文件）
  python dem_to_rgb.py --batch --input-dir /path/to/tif_folder --output-dir /path/to/output

  # 带参数处理（增强地形起伏）
  python dem_to_rgb.py input_dem.tif output_rgb.png --zscale 2.0

  # 使用其他配色方案
  python dem_to_rgb.py input_dem.tif output_rgb.png --colortable terrain
        '''
    )

    # 配置文件参数
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径（默认为脚本同目录下的 dem_to_rgb_config.json）')

    # 位置参数（单文件模式）
    parser.add_argument('input', nargs='?', help='输入 DEM 文件（GeoTIFF）')
    parser.add_argument('output', nargs='?', help='输出 RGB 文件')

    # 批量处理参数
    parser.add_argument('--batch', action='store_true',
                        help='启用批量模式')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='批量模式：输入文件夹路径（覆盖配置文件）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='批量模式：输出文件夹路径（覆盖配置文件）')

    # 光照参数
    parser.add_argument('--azimuth', type=float, default=None,
                        help='光源方位角（度），从北顺时针，默认 -45（西北，与LInSAR一致）')
    parser.add_argument('--altitude', type=float, default=None,
                        help='光源高度角（度），从地平线向上，默认 45')

    # 地形参数
    parser.add_argument('--zscale', type=float, default=None,
                        help='Z 方向缩放因子，用于增强/减弱地形起伏，默认 1.0')
    parser.add_argument('--shade-boost', type=float, default=None,
                        help='阴影增强因子，>1 增强明暗对比（推荐1.5-2.0用于YOLO），默认 1.0')

    # 颜色参数
    parser.add_argument('--colortable', type=str, default=None,
                        choices=['linsar', 'terrain', 'elevation', 'dem'],
                        help='颜色表名称，默认 linsar（与LInSAR一致）')
    parser.add_argument('--minz', type=float, default=None,
                        help='最小高程（默认自动计算）')
    parser.add_argument('--maxz', type=float, default=None,
                        help='最大高程（默认自动计算）')

    # 无效值
    parser.add_argument('--nodata', type=float, default=None,
                        help='无效值（默认从文件读取）')

    args = parser.parse_args()

    # ==================== 加载配置 ====================
    config = {}

    # 确定配置文件路径
    if args.config:
        config_path = args.config
    else:
        config_path = get_default_config_path()

    # 尝试加载配置文件
    if os.path.exists(config_path):
        print(f"加载配置文件: {config_path}")
        config = load_config(config_path)
    else:
        print(f"未找到配置文件: {config_path}，使用默认值")

    # ==================== 合并参数（命令行优先） ====================

    # 批量模式参数
    input_dir = args.input_dir or config.get('input_dir')
    output_dir = args.output_dir or config.get('output_dir')

    # 渲染参数
    render_params = config.get('render_params', {})

    azimuth = args.azimuth if args.azimuth is not None else render_params.get('azimuth', -45.0)
    altitude = args.altitude if args.altitude is not None else render_params.get('altitude', 45.0)
    zscale = args.zscale if args.zscale is not None else render_params.get('zscale', 1.0)
    shade_boost = args.shade_boost if args.shade_boost is not None else render_params.get('shade_boost', 1.0)
    colortable = args.colortable if args.colortable is not None else render_params.get('colortable', 'linsar')
    minz = args.minz if args.minz is not None else render_params.get('minz')
    maxz = args.maxz if args.maxz is not None else render_params.get('maxz')
    nodata = args.nodata if args.nodata is not None else render_params.get('nodata')

    # 创建一个简单的 args 对象用于传递参数
    class Args:
        pass

    merged_args = Args()
    merged_args.azimuth = azimuth
    merged_args.altitude = altitude
    merged_args.zscale = zscale
    merged_args.shade_boost = shade_boost
    merged_args.colortable = colortable
    merged_args.minz = minz
    merged_args.maxz = maxz
    merged_args.nodata = nodata

    # ==================== 判断运行模式 ====================

    if args.batch:
        # 批量模式
        if not input_dir:
            parser.error("批量模式需要指定输入文件夹\n"
                        "方式1: 在配置文件中设置 input_dir\n"
                        "方式2: 使用 --input-dir 参数")

        # 默认输出目录为输入目录下的 output 子目录
        if not output_dir:
            output_dir = os.path.join(input_dir, 'output')

        print(f"输入文件夹: {input_dir}")
        print(f"输出文件夹: {output_dir}")
        print(f"方位角: {azimuth}°, 高度角: {altitude}°, Z缩放: {zscale}")
        print(f"颜色表: {colortable}")
        print("-" * 50)

        process_batch(input_dir, output_dir, merged_args)

    else:
        # 单文件模式
        if not args.input or not args.output:
            parser.error("单文件模式需要指定 input 和 output 参数\n"
                        "示例: python dem_to_rgb.py input_dem.tif output_rgb.png")
        process_single_file(args.input, args.output, merged_args)


if __name__ == '__main__':
    main()
