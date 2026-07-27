#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEM to RGB Image Converter + 坐标映射保存
将 DEM（数字高程模型）转换为带山体阴影效果的 RGB 图像，并自动保存坐标映射信息

原理（与 LInSAR 完全一致）：
1. 高程 → 颜色映射（21级颜色表线性插值）
2. Sobel 梯度 → 坡度/坡向计算
3. 山体阴影计算（根据光源位置）
4. Lab 色彩空间合成（调整亮度，保持色调）
5. 保存坐标映射信息（Geotransform + 投影）用于后续转换

改进点：
  ✓ 所有功能集成（无需依赖其他模块）
  ✓ 自动保存 *_mapper.json 文件（包含坐标映射信息）
  ✓ 保留所有原有参数和功能

依赖：
    pip install numpy opencv-python gdal

使用示例：
    python dem_to_rgb_with_metadata.py input_dem.tif output_rgb.png
    # 输出：output_rgb.png + output_rgb_mapper.json

    python dem_to_rgb_with_metadata.py --batch
    # 批量处理所有 TIF 文件，为每个文件生成 PNG + mapper.json
"""

import argparse
import os
import glob
import json
import numpy as np
import cv2
from osgeo import gdal


# ==================== 坐标映射工具 ====================

class CoordinateMapper(object):
    """坐标映射器：完成像素坐标 ↔ 地理坐标转换"""

    def __init__(self, geotransform, projection, image_shape):
        """
        初始化坐标映射器

        Args:
            geotransform: GDAL Geotransform 元组 (6,)
                (x0, px_width, 0, y0, 0, py_height)
            projection: 投影信息 WKT 字符串
            image_shape: 图像尺寸 (height, width)
        """
        self.geotransform = geotransform
        self.projection = projection
        self.height, self.width = image_shape

        self.x0, self.px_width, _, self.y0, _, self.py_height = geotransform

    def pixel_to_geo(self, pixel_coords):
        """像素坐标 → 地理坐标"""
        pixel_coords = np.asarray(pixel_coords, dtype=np.float64)

        if pixel_coords.ndim == 1 and len(pixel_coords) == 2:
            col, row = pixel_coords
            x = self.x0 + col * self.px_width
            y = self.y0 + row * self.py_height
            return np.array([x, y])

        elif pixel_coords.ndim == 1 and len(pixel_coords) == 4:
            col_min, row_min, col_max, row_max = pixel_coords
            x_min = self.x0 + col_min * self.px_width
            y_min = self.y0 + row_min * self.py_height
            x_max = self.x0 + col_max * self.px_width
            y_max = self.y0 + row_max * self.py_height

            x_min, x_max = min(x_min, x_max), max(x_min, x_max)
            y_min, y_max = min(y_min, y_max), max(y_min, y_max)

            return np.array([x_min, y_min, x_max, y_max])

    def get_geo_extent(self):
        """获取图像的地理范围"""
        tl = self.pixel_to_geo([0, 0])
        br = self.pixel_to_geo([self.width - 1, self.height - 1])

        x_min = min(tl[0], br[0])
        x_max = max(tl[0], br[0])
        y_min = min(tl[1], br[1])
        y_max = max(tl[1], br[1])

        return {'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}


def save_mapper_metadata(mapper, output_path):
    """保存映射器元数据到 JSON 文件"""
    data = {
        'geotransform': list(mapper.geotransform),
        'projection': mapper.projection,
        'image_shape': [mapper.height, mapper.width],
        'geo_extent': mapper.get_geo_extent()
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  坐标映射信息已保存: {output_path}")


# ==================== 配置文件读取 ====================

def load_config(config_path):
    """加载配置文件"""
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
            - 'linsar': LInSAR 默认配色（21级）
            - 'terrain': 经典地形配色
            - 'elevation': 高程配色
            - 'dem': DEM 标准配色

    Returns:
        numpy array of shape (N, 3), RGB values in [0, 255]
    """
    color_tables = {
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
            [0, 128, 0],       # 深绿
            [0, 200, 0],       # 绿色
            [128, 200, 0],     # 黄绿
            [200, 200, 0],     # 黄色
            [200, 150, 0],     # 橙黄
            [180, 120, 60],    # 棕色
            [150, 100, 50],    # 深棕
            [200, 180, 140],   # 浅棕
            [230, 220, 200],   # 浅灰
            [255, 255, 255],   # 白色
        ], dtype=np.float32),

        'elevation': np.array([
            [0, 0, 200],       # 深蓝
            [0, 100, 255],     # 蓝色
            [0, 200, 200],     # 青色
            [0, 255, 0],       # 绿色
            [128, 255, 0],     # 黄绿
            [255, 255, 0],     # 黄色
            [255, 128, 0],     # 橙色
            [255, 0, 0],       # 红色
            [128, 0, 0],       # 深红
            [255, 255, 255],   # 白色
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
    """将高程映射为颜色"""
    if min_z is None:
        min_z = dem[mask].min()
    if max_z is None:
        max_z = dem[mask].max()

    n_colors = len(color_table)
    z = np.clip(dem, min_z, max_z)

    normalized = np.zeros_like(dem, dtype=np.float32)
    normalized[mask] = (z[mask] - min_z) / (max_z - min_z) * (n_colors - 1)

    lower = np.floor(normalized).astype(np.int32)
    upper = np.ceil(normalized).astype(np.int32)

    lower = np.clip(lower, 0, n_colors - 1)
    upper = np.clip(upper, 0, n_colors - 1)

    t = normalized - lower

    color_image = np.zeros((dem.shape[0], dem.shape[1], 3), dtype=np.float32)
    for c in range(3):
        color_image[:, :, c] = color_table[lower, c] * (1 - t) + color_table[upper, c] * t

    color_image = color_image / 255.0

    return color_image


def compute_hillshade(dem, mask, cell_size, azimuth=-45.0, altitude=45.0, z_scale=1.0):
    """计算山体阴影"""
    DEG2RAD = np.pi / 180.0
    RAD2DEG = 180.0 / np.pi

    zenith = 90.0 - altitude
    azimuth_rad = (360.0 - azimuth + 90.0) * DEG2RAD
    if azimuth_rad > 2 * np.pi:
        azimuth_rad -= 2 * np.pi

    grad_x = cv2.Sobel(dem, cv2.CV_32F, 1, 0, ksize=3) / (8.0 * cell_size)
    grad_y = cv2.Sobel(dem, cv2.CV_32F, 0, 1, ksize=3) / (8.0 * cell_size)

    slope = np.arctan(z_scale * np.sqrt(grad_x**2 + grad_y**2)) * RAD2DEG

    aspect = np.arctan2(grad_y, -grad_x) * RAD2DEG
    aspect = np.where(aspect < 0, aspect + 360, aspect)

    cos_zenith = np.cos(zenith * DEG2RAD)
    sin_zenith = np.sin(zenith * DEG2RAD)

    hillshade = (cos_zenith * np.cos(slope * DEG2RAD) +
                 sin_zenith * np.sin(slope * DEG2RAD) *
                 np.cos((azimuth_rad - aspect * DEG2RAD)))

    hillshade = np.clip(hillshade, 0, 1)
    hillshade[~mask] = 0

    return hillshade


def blend_color_and_shade(color_image, hillshade, mask, shade_boost=1.0):
    """在 Lab 色彩空间中合成颜色和阴影"""
    color_32f = color_image.astype(np.float32)
    lab = cv2.cvtColor(color_32f, cv2.COLOR_RGB2Lab)

    if shade_boost != 1.0:
        hillshade_adjusted = np.power(hillshade, 1.0 / shade_boost)
    else:
        hillshade_adjusted = hillshade

    lab[:, :, 0] = lab[:, :, 0] * hillshade_adjusted
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 100)

    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    result = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    result[:, :, :3] = rgb
    result[:, :, 3] = (mask * 255).astype(np.uint8)

    return result


# ==================== 文件读写 ====================

def read_geotiff(filepath):
    """读取 GeoTIFF 文件"""
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
    """写入 GeoTIFF 文件（保留坐标信息）"""
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
    """将 DEM 转换为 RGB 图像（核心函数）"""
    if isinstance(color_table, str):
        colors = get_color_table(color_table)
    else:
        colors = np.array(color_table, dtype=np.float32)

    color_image = map_elevation_to_color(dem, mask, colors, min_z, max_z)
    hillshade = compute_hillshade(dem, mask, cell_size, azimuth, altitude, z_scale)
    rgb_image = blend_color_and_shade(color_image, hillshade, mask, shade_boost)

    return rgb_image


# ==================== 单文件处理 ====================

def process_single_file(input_path, output_path, args):
    """
    处理单个 DEM 文件，并保存坐标映射信息

    Args:
        input_path: 输入 DEM 文件路径
        output_path: 输出 RGB 文件路径
        args: 命令行参数
    """
    print(f"  读取 DEM 文件: {input_path}")
    dem, transform, projection, nodata = read_geotiff(input_path)

    # 处理无效值
    nodata_val = args.nodata if args.nodata is not None else nodata

    if nodata_val is not None:
        mask = (dem != nodata_val) & (~np.isnan(dem))
    else:
        mask = ~np.isnan(dem)

    # 与 LInSAR 一致：对 mask 做 3x3 腐蚀
    mask_uint8 = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_eroded = cv2.erode(mask_uint8, kernel)
    mask = mask_eroded == 255

    cell_size = abs(transform[1])

    print(f"  DEM 大小: {dem.shape[1]} x {dem.shape[0]}")
    print(f"  像元大小: {cell_size}")
    print(f"  高程范围: {dem[mask].min():.2f} ~ {dem[mask].max():.2f}")

    # 转换
    print(f"  正在生成 RGB 图像...")
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

    # 输出 RGB 图像
    print(f"  保存输出文件: {output_path}")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    ext = output_path.lower().split('.')[-1]

    if ext in ['tif', 'tiff']:
        write_geotiff(output_path, rgb_image, transform, projection)
    else:
        if ext == 'png':
            cv2.imwrite(output_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2BGRA))
        else:
            rgb_no_alpha = rgb_image[:, :, :3]
            cv2.imwrite(output_path, cv2.cvtColor(rgb_no_alpha, cv2.COLOR_RGB2BGR))

    # ✨ 核心改进：保存坐标映射信息
    mapper = CoordinateMapper(transform, projection, (dem.shape[0], dem.shape[1]))
    mapper_output_path = os.path.splitext(output_path)[0] + '_mapper.json'
    save_mapper_metadata(mapper, mapper_output_path)

    print(f"  完成: {output_path}")


# ==================== 批量处理 ====================

def process_batch(input_dir, output_dir, args):
    """
    批量处理文件夹下的所有 TIF 文件

    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
        args: 命令行参数
    """
    os.makedirs(output_dir, exist_ok=True)

    tif_files = glob.glob(os.path.join(input_dir, '*.tif')) + \
                glob.glob(os.path.join(input_dir, '*.tiff')) + \
                glob.glob(os.path.join(input_dir, '*.TIF')) + \
                glob.glob(os.path.join(input_dir, '*.TIFF'))

    tif_files = list(set(tif_files))
    tif_files.sort()

    if not tif_files:
        print(f"警告: 在 {input_dir} 中未找到任何 TIF 文件")
        return

    print(f"找到 {len(tif_files)} 个 TIF 文件")
    print(f"输出文件夹: {output_dir}")
    print("-" * 60)

    success_count = 0
    fail_count = 0

    for i, input_path in enumerate(tif_files):
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{basename}.png")

        print(f"\n[{i+1}/{len(tif_files)}] 处理: {os.path.basename(input_path)}")

        try:
            process_single_file(input_path, output_path, args)
            success_count += 1
        except Exception as e:
            print(f"  错误: {e}")
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"批量处理完成！成功: {success_count}/{len(tif_files)}")
    if fail_count > 0:
        print(f"失败: {fail_count} 个文件")


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(
        description='DEM to RGB 转换 + 坐标映射保存',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 单文件模式（使用默认配色）
  python dem_to_rgb_with_metadata.py input_dem.tif output_rgb.png

  # 批量模式 - 使用配置文件
  python dem_to_rgb_with_metadata.py --batch

  # 批量模式 - 指定文件夹（覆盖配置文件）
  python dem_to_rgb_with_metadata.py --batch --input-dir /path/to/tif_folder --output-dir /path/to/output

  # 带参数处理（增强地形起伏）
  python dem_to_rgb_with_metadata.py input_dem.tif output_rgb.png --zscale 2.0 --shade-boost 1.5

  输出说明：
  ✓ image.png - RGB 彩色图像
  ✓ image_mapper.json - 坐标映射信息（YOLO 后续使用）
        '''
    )

    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径（默认为脚本同目录下的 dem_to_rgb_config.json）')

    parser.add_argument('input', nargs='?', help='输入 DEM 文件（GeoTIFF）')
    parser.add_argument('output', nargs='?', help='输出 RGB 文件')

    parser.add_argument('--batch', action='store_true', help='启用批量模式')
    parser.add_argument('--input-dir', type=str, default=None, help='批量模式：输入文件夹路径')
    parser.add_argument('--output-dir', type=str, default=None, help='批量模式：输出文件夹路径')

    parser.add_argument('--azimuth', type=float, default=None,
                        help='光源方位角（度），默认 -45')
    parser.add_argument('--altitude', type=float, default=None,
                        help='光源高度角（度），默认 45')

    parser.add_argument('--zscale', type=float, default=None,
                        help='Z 方向缩放因子，默认 1.0')
    parser.add_argument('--shade-boost', type=float, default=None,
                        help='阴影增强因子，默认 1.0')

    parser.add_argument('--colortable', type=str, default=None,
                        choices=['linsar', 'terrain', 'elevation', 'dem'],
                        help='颜色表名称，默认 linsar')
    parser.add_argument('--minz', type=float, default=None, help='最小高程')
    parser.add_argument('--maxz', type=float, default=None, help='最大高程')
    parser.add_argument('--nodata', type=float, default=None, help='无效值')

    args = parser.parse_args()

    # ==================== 加载配置 ====================
    config = {}

    if args.config:
        config_path = args.config
    else:
        config_path = get_default_config_path()

    if os.path.exists(config_path):
        print(f"加载配置文件: {config_path}")
        config = load_config(config_path)
    else:
        print(f"未找到配置文件: {config_path}，使用默认值")

    # ==================== 合并参数 ====================

    input_dir = args.input_dir or config.get('input_dir')
    output_dir = args.output_dir or config.get('output_dir')

    render_params = config.get('render_params', {})

    azimuth = args.azimuth if args.azimuth is not None else render_params.get('azimuth', -45.0)
    altitude = args.altitude if args.altitude is not None else render_params.get('altitude', 45.0)
    zscale = args.zscale if args.zscale is not None else render_params.get('zscale', 1.0)
    shade_boost = args.shade_boost if args.shade_boost is not None else render_params.get('shade_boost', 1.0)
    colortable = args.colortable if args.colortable is not None else render_params.get('colortable', 'linsar')
    minz = args.minz if args.minz is not None else render_params.get('minz')
    maxz = args.maxz if args.maxz is not None else render_params.get('maxz')
    nodata = args.nodata if args.nodata is not None else render_params.get('nodata')

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
        if not input_dir:
            parser.error("批量模式需要指定输入文件夹\n"
                        "方式1: 在配置文件中设置 input_dir\n"
                        "方式2: 使用 --input-dir 参数")

        if not output_dir:
            output_dir = os.path.join(input_dir, 'output')

        print(f"输入文件夹: {input_dir}")
        print(f"输出文件夹: {output_dir}")
        print(f"参数: 方位角={azimuth}°, 高度角={altitude}°, Z缩放={zscale}, 颜色表={colortable}")
        print("-" * 60)

        process_batch(input_dir, output_dir, merged_args)

    else:
        if not args.input or not args.output:
            parser.error("单文件模式需要指定 input 和 output 参数\n"
                        "示例: python dem_to_rgb_with_metadata.py input_dem.tif output_rgb.png")

        print(f"输入文件: {args.input}")
        print(f"输出文件: {args.output}")
        print(f"参数: 方位角={azimuth}°, 高度角={altitude}°, Z缩放={zscale}, 颜色表={colortable}")
        print("-" * 60)

        process_single_file(args.input, args.output, merged_args)


if __name__ == '__main__':
    main()
