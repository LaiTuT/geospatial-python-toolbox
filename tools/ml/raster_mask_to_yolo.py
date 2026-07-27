#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import cv2
import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

# =========================
# 辅助：保存 GDAL Dataset 为 PNG（反转颜色）
# =========================
def save_gdal_ds_as_png(ds, output_path, invert=True):
    """
    将 GDAL 内存数据集保存为 PNG
    invert: 是否反转颜色（True=白底黑线，False=黑底白线）
    """
    if ds is None:
        print(f"[Warn] 数据集为空，跳过保存: {output_path}")
        return

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()

    if arr is None:
        print(f"[Warn] 无法读取数据: {output_path}")
        return

    # 可选：反转像素值
    if invert:
        arr = 255 - arr

    # 创建新的内存数据集
    mem_drv = gdal.GetDriverByName('MEM')
    ds_out = mem_drv.Create('', ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
    ds_out.GetRasterBand(1).WriteArray(arr)

    # 保存为 PNG
    drv = gdal.GetDriverByName("PNG")
    if drv is None:
        print("[Err] GDAL 缺少 PNG 驱动")
        return

    try:
        dst_ds = drv.CreateCopy(output_path, ds_out, strict=0)
        dst_ds = None
        ds_out = None
        print(f"[OK] PNG 已保存: {output_path}")
    except Exception as e:
        print(f"[Err] PNG 保存失败 {output_path}: {e}")

# =========================
# 核心：从大 Mask 裁剪 -> PNG可视化 -> TXT提取
# =========================
def widen_single_pixel_mask(binary: np.ndarray, target_width_pixels: float = 2.5) -> np.ndarray:
    """
    将单像素中心线拓宽为 YOLO-seg 更容易识别的二值区域。

    说明：
    - `target_width_pixels=2.5` 表示目标线宽约为 2.5 像素；
    - 由于最终仍是 0/255 栅格掩膜，2.5 像素会被近似到像素网格上
      （水平/垂直线通常表现为约 3 像素宽）。
    """
    if binary is None or target_width_pixels is None or target_width_pixels <= 1:
        return binary

    fg = (binary > 0).astype(np.uint8)
    if cv2.countNonZero(fg) == 0:
        return binary

    # 把原始单像素视为中心线，按目标线宽的一半向外扩张。
    half_width = float(target_width_pixels) / 2.0

    # distanceTransform 计算每个背景像素到最近前景像素中心的距离。
    # 输入中 0 被视为目标点，所以这里让前景=0、背景=1。
    background = (fg == 0).astype(np.uint8)
    dist = cv2.distanceTransform(background, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    widened = ((fg > 0) | (dist <= half_width)).astype(np.uint8) * 255
    return widened


def process_tile_from_large_mask(
    tile_path: str,
    large_mask_ds: gdal.Dataset,
    out_txt_path: str,
    class_mapping: dict,
    min_area: float = 10.0,
    simplify_factor: float = 0.005,
    single_pixel_width: float = 2.5,
    use_morphology: bool = True,
    debug_png_path: str = None,
    debug_mode: bool = False  # 新增：是否输出调试信息
):
    # 1. 打开参考 TIF (Tile)
    try:
        ds_tile = gdal.Open(tile_path, gdal.GA_ReadOnly)
        if ds_tile is None:
            print(f"[Err] 无法打开瓦片: {tile_path}")
            return
    except Exception as e:
        print(f"[Err] 打开瓦片异常 {tile_path}: {e}")
        return

    w_tile = ds_tile.RasterXSize
    h_tile = ds_tile.RasterYSize
    gt_tile = ds_tile.GetGeoTransform()
    proj_tile = ds_tile.GetProjection()

    # 2. 创建内存栅格
    mem_drv = gdal.GetDriverByName('MEM')
    ds_cropped = mem_drv.Create('', w_tile, h_tile, 1, gdal.GDT_Byte)
    ds_cropped.SetGeoTransform(gt_tile)
    ds_cropped.SetProjection(proj_tile)

    band_out = ds_cropped.GetRasterBand(1)
    band_out.SetNoDataValue(0)
    band_out.Fill(0)

    # 3. 投影裁剪
    try:
        gdal.ReprojectImage(
            large_mask_ds,
            ds_cropped,
            large_mask_ds.GetProjection(),
            proj_tile,
            gdal.GRA_NearestNeighbour
        )
    except Exception as e:
        print(f"[Err] 裁剪失败 {tile_path}: {e}")
        ds_tile = None
        ds_cropped = None
        return

    # 4. 【可选】输出 PNG
    if debug_png_path:
        save_gdal_ds_as_png(ds_cropped, debug_png_path, invert=True)  # 反转颜色

    # 5. 读取数据
    mask_arr = band_out.ReadAsArray()

    # === 调试信息 ===
    if debug_mode and mask_arr is not None:
        unique_vals = np.unique(mask_arr)
        print(f"[Debug] {os.path.basename(tile_path)}")
        print(f"        像素值: {unique_vals}")
        print(f"        统计: {[(v, np.sum(mask_arr == v)) for v in unique_vals if v > 0]}")
    # ===============

    ds_tile = None
    ds_cropped = None

    if mask_arr is None or np.count_nonzero(mask_arr) == 0:
        # 裁剪区域为空，生成空 txt 文件（保证 txt 和瓦片一一对应）
        os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
        with open(out_txt_path, "w", encoding="utf-8") as f:
            pass  # 创建空文件
        if debug_mode:
            print(f"[OK] 空标签已生成: {out_txt_path} (裁剪区域为空)")
        return

    # 6. 轮廓提取
    txt_lines = []
    for pixel_val, class_id in class_mapping.items():
        binary = np.zeros_like(mask_arr, dtype=np.uint8)
        binary[mask_arr == pixel_val] = 255

        if cv2.countNonZero(binary) == 0:
            continue

        # 在提取轮廓 / 转成 YOLO-seg 多边形前，先把单像素线拓宽到指定宽度。
        binary = widen_single_pixel_mask(binary, target_width_pixels=single_pixel_width)

        if use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 使用 CHAIN_APPROX_NONE 保留所有轮廓点，提高精细度
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue

            # 简化轮廓：simplify_factor 越小越精细，0 表示不简化
            if simplify_factor > 0:
                epsilon = simplify_factor * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                pts = approx.reshape(-1, 2)
            else:
                # 不简化，直接使用原始轮廓
                pts = cnt.reshape(-1, 2)

            if len(pts) < 3:
                continue
            norm_pts = []
            for x, y in pts:
                nx = max(0.0, min(1.0, x / w_tile))
                ny = max(0.0, min(1.0, y / h_tile))
                norm_pts.extend([nx, ny])

            line_str = f"{class_id} " + " ".join(f"{v:.6f}" for v in norm_pts)
            txt_lines.append(line_str)

    # 7. 保存 TXT
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    if txt_lines:
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))
        if debug_mode:
            print(f"[OK] 标签已生成: {out_txt_path} ({len(txt_lines)} 个对象)")
    else:
        # 生成空的 txt 文件（YOLO 训练需要）
        with open(out_txt_path, "w", encoding="utf-8") as f:
            pass  # 创建空文件
        if debug_mode:
            print(f"[OK] 空标签已生成: {out_txt_path} (背景图，无实例)")


# =========================
# 批处理逻辑
# =========================
def batch_geo_mask_to_yolo_single_source(
    tif_dir: str,
    large_mask_path: str,
    out_dir: str,
    class_mapping: dict,
    debug_mask_dir: str = None,
    tif_ext=".tif",
    recursive=True,
    min_area: float = 100.0,  # 最小实例面积阈值（像素）
    simplify_factor: float = 0.001,  # 轮廓简化因子，越小越精细，0表示不简化
    single_pixel_width: float = 2.5,  # 单像素线拓宽后的目标宽度（像素）
    debug_mode=False  # 新增参数
):
    if not os.path.exists(large_mask_path):
        print(f"[Err] 大 Mask 文件不存在: {large_mask_path}")
        return

    try:
        large_mask_ds = gdal.Open(large_mask_path, gdal.GA_ReadOnly)
        if large_mask_ds is None:
            print(f"[Err] 无法打开大 Mask: {large_mask_path}")
            return
    except Exception as e:
        print(f"[Err] 打开大 Mask 异常: {e}")
        return

    print(f"[OK] 已加载大 Mask: {large_mask_path}")
    print(f"    尺寸: {large_mask_ds.RasterXSize} x {large_mask_ds.RasterYSize}")

    tif_files = []
    if recursive:
        tif_files = glob.glob(os.path.join(tif_dir, "**", f"*{tif_ext}"), recursive=True)
    else:
        tif_files = glob.glob(os.path.join(tif_dir, f"*{tif_ext}"))

    print(f"\n开始处理，共 {len(tif_files)} 个瓦片...")

    count = 0
    success = 0

    for tif_path in tif_files:
        rel_path = os.path.relpath(tif_path, tif_dir)
        rel_name = os.path.splitext(rel_path)[0]

        out_txt_path = os.path.join(out_dir, rel_name + ".txt")

        png_out_path = None
        if debug_mask_dir:
            png_out_path = os.path.join(debug_mask_dir, rel_name + ".png")

        try:
            process_tile_from_large_mask(
                tif_path,
                large_mask_ds,
                out_txt_path,
                class_mapping,
                min_area=min_area,
                simplify_factor=simplify_factor,
                single_pixel_width=single_pixel_width,
                debug_png_path=png_out_path,
                debug_mode=debug_mode  # 传递调试标志
            )

            if os.path.exists(out_txt_path):
                success += 1

            count += 1
            if count % 50 == 0:
                print(f"进度: {count}/{len(tif_files)} ({success} 成功)")

        except Exception as e:
            print(f"[Err] 处理失败 {rel_name}: {e}")

    large_mask_ds = None

    print(f"\n=== 处理完成 ===")
    print(f"总瓦片数: {len(tif_files)}")
    print(f"成功生成标签: {success}")
    print(f"Labels 目录: {out_dir}")
    if debug_mask_dir:
        print(f"Mask PNG 目录: {debug_mask_dir}")


# =========================
# 多 Mask 批处理：mask 文件名 → 瓦片子目录
# =========================
def batch_geo_mask_to_yolo_multi_source(
    mask_dir: str,
    tif_root_dir: str,
    out_root_dir: str,
    class_mapping: dict,
    debug_mask_root_dir: str = None,
    tif_ext=".tif",
    recursive=True,
    min_area: float = 100.0,
    simplify_factor: float = 0.001,
    single_pixel_width: float = 2.5,
    debug_mode=False
):
    """
    mask_dir 下所有 .tif mask，每个 mask 的文件名（不含扩展名）对应
    tif_root_dir 下的同名子目录，从中读取瓦片处理。

    目录结构示例:
      mask_dir/
        A.tif
        B.tif
      tif_root_dir/
        A/
          tile_0_0.tif ...
        B/
          tile_0_0.tif ...

    输出:
      out_root_dir/
        A/
          tile_0_0.txt ...
        B/
          tile_0_0.txt ...
    """
    mask_files = glob.glob(os.path.join(mask_dir, "*.tif"))
    mask_files += glob.glob(os.path.join(mask_dir, "*.TIF"))
    mask_files = sorted(set(mask_files))

    if not mask_files:
        print(f"[Err] mask 目录下未找到 .tif 文件: {mask_dir}")
        return

    print(f"找到 {len(mask_files)} 个 Mask 文件\n")

    for mask_path in mask_files:
        mask_name = os.path.splitext(os.path.basename(mask_path))[0]
        tile_subdir = os.path.join(tif_root_dir, mask_name)

        if not os.path.isdir(tile_subdir):
            print(f"[Warn] 跳过 {mask_name}: 瓦片子目录不存在 → {tile_subdir}")
            continue

        print(f"{'='*50}")
        print(f"处理: Mask={mask_name}")
        print(f"  瓦片目录: {tile_subdir}")
        print(f"  输出目录: {out_root_dir}")

        batch_geo_mask_to_yolo_single_source(
            tif_dir=tile_subdir,
            large_mask_path=mask_path,
            out_dir=out_root_dir,
            class_mapping=class_mapping,
            debug_mask_dir=debug_mask_root_dir,
            tif_ext=tif_ext,
            recursive=recursive,
            min_area=min_area,
            simplify_factor=simplify_factor,
            single_pixel_width=single_pixel_width,
            debug_mode=debug_mode
        )

    print(f"\n{'='*50}")
    print("全部 Mask 处理完成！")


# =========================
# 可视化：根据 YOLO TXT 生成检查图
# =========================
def visualize_yolo_txt(
    image_path: str,
    txt_path: str,
    output_path: str,
    class_colors: dict = None,
    line_thickness: int = 2,
    grayscale_mode: bool = False,
    class_grayscale_values: dict = None
):
    """
    根据 YOLO 分割标签生成可视化检查图

    Args:
        image_path: 原始图像路径（支持 TIFF/PNG 等）
        txt_path: YOLO 标签文件路径
        output_path: 输出可视化图像路径
        class_colors: 类别颜色映射 {class_id: (B, G, R)}，默认自动生成
        line_thickness: 轮廓线宽度
        grayscale_mode: 是否生成灰度图（黑底，填充灰度值）
        class_grayscale_values: 灰度模式下各类别的灰度值 {class_id: gray_value}
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"[Warn] 图像不存在: {image_path}")
        return
    if not os.path.exists(txt_path):
        print(f"[Warn] 标签不存在: {txt_path}")
        return

    # 读取图像获取尺寸
    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"[Err] 无法打开图像: {image_path}")
        return

    h, w = ds.RasterYSize, ds.RasterXSize
    ds = None

    # 灰度模式：生成黑底填充图
    if grayscale_mode:
        if class_grayscale_values is None:
            class_grayscale_values = {
                0: 255,  # 白色
                1: 128,  # 灰色
                2: 200,
                3: 100,
                4: 180,
                5: 80,
            }

        # 创建黑色背景
        img = np.zeros((h, w), dtype=np.uint8)

        # 读取 YOLO 标签
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        # 填充每个实例
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            # 转换归一化坐标为像素坐标
            pts = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i + 1] * h)
                pts.append([x, y])

            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

            # 获取灰度值
            gray_val = class_grayscale_values.get(class_id, 255)

            # 填充多边形
            cv2.fillPoly(img, [pts], gray_val)

        # 保存灰度图
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"[OK] 灰度可视化已保存: {output_path}")
        return

    # 彩色模式：在原图上绘制轮廓
    # 读取图像
    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    # 读取为RGB（取前3个波段）
    img = np.zeros((ds.RasterYSize, ds.RasterXSize, 3), dtype=np.uint8)
    for i in range(min(3, ds.RasterCount)):
        band = ds.GetRasterBand(i + 1)
        arr = band.ReadAsArray()
        # 归一化到 0-255
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
        img[:, :, i] = arr
    ds = None

    # 默认颜色
    if class_colors is None:
        class_colors = {
            0: (0, 255, 0),      # 绿色
            1: (0, 0, 255),      # 红色
            2: (255, 0, 0),      # 蓝色
            3: (255, 255, 0),    # 青色
            4: (255, 0, 255),    # 紫色
            5: (0, 255, 255),    # 黄色
        }

    # 读取 YOLO 标签
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # 绘制每个实例
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:  # class_id + 至少3个点(6个坐标)
            continue

        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        # 转换归一化坐标为像素坐标
        pts = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)
            pts.append([x, y])

        pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

        # 获取颜色
        color = class_colors.get(class_id, (255, 255, 255))

        # 绘制多边形轮廓
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=line_thickness)

        # 绘制类别标签
        label = f"C{class_id}"
        x_center = int(np.mean([p[0] for p in pts.reshape(-1, 2)]))
        y_center = int(np.mean([p[1] for p in pts.reshape(-1, 2)]))
        cv2.putText(img, label, (x_center - 10, y_center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[OK] 可视化已保存: {output_path}")


def batch_visualize_yolo_labels(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    image_ext: str = ".tif",
    class_colors: dict = None,
    grayscale_mode: bool = False,
    class_grayscale_values: dict = None
):
    """
    批量生成 YOLO 标签可视化检查图

    Args:
        image_dir: 图像目录
        label_dir: 标签目录
        output_dir: 输出目录
        image_ext: 图像扩展名
        class_colors: 类别颜色映射（彩色模式）
        grayscale_mode: 是否生成灰度图（黑底，填充灰度值）
        class_grayscale_values: 灰度模式下各类别的灰度值 {class_id: gray_value}
    """
    # 查找所有标签文件
    txt_files = glob.glob(os.path.join(label_dir, "**", "*.txt"), recursive=True)

    print(f"找到 {len(txt_files)} 个标签文件")
    mode_str = "灰度" if grayscale_mode else "彩色"
    print(f"输出模式: {mode_str}")

    success = 0
    for txt_path in txt_files:
        # 获取对应的图像路径
        rel_path = os.path.relpath(txt_path, label_dir)
        rel_name = os.path.splitext(rel_path)[0]

        image_path = os.path.join(image_dir, rel_name + image_ext)
        output_path = os.path.join(output_dir, rel_name + "_vis.png")

        try:
            visualize_yolo_txt(
                image_path, txt_path, output_path, class_colors,
                grayscale_mode=grayscale_mode,
                class_grayscale_values=class_grayscale_values
            )
            success += 1
        except Exception as e:
            print(f"[Err] 处理失败 {rel_name}: {e}")

    print(f"\n=== 可视化完成 ===")
    print(f"成功: {success}/{len(txt_files)}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    MASK_DIR = r"D:\Dataproces\1m\AL_25Co_B2_2017\mask"        # 存放多张 .tif mask 的目录
    TIF_DIR = r"D:\Dataproces\1m\AL_25Co_B2_2017\all_tifs"        # 瓦片根目录，子目录名与 mask 同名

    OUT_TXT_DIR = r"D:\Dataproces\1m\AL_25Co_B2_2017\labels"
    OUT_PNG_DIR = r"D:\Dataproces\1m\AL_25Co_B2_2017\masks"
    VIS_DIR = r"D:\Dataproces\1m\AL_25Co_B2_2017\vis_from_txt"

    CLASS_MAP = {
        255: 0,
        128: 1
    }

    CLASS_GRAYSCALE = {
        0: 255,
        1: 128,
    }

    # Step 1: 批量多 Mask → YOLO 标签
    batch_geo_mask_to_yolo_multi_source(
        mask_dir=MASK_DIR,
        tif_root_dir=TIF_DIR,
        out_root_dir=OUT_TXT_DIR,
        class_mapping=CLASS_MAP,
        debug_mask_root_dir=OUT_PNG_DIR,
        tif_ext=".tif",
        min_area=100,
        simplify_factor=0.0,
        single_pixel_width=2.5,
        debug_mode=True
    )

    # Step 2: 可视化（灰度填充）
    print("\n" + "="*50)
    print("开始生成灰度可视化检查图...")
    print("="*50)
    batch_visualize_yolo_labels(
        image_dir=TIF_DIR,
        label_dir=OUT_TXT_DIR,
        output_dir=VIS_DIR,
        image_ext=".tif",
        grayscale_mode=True,
        class_grayscale_values=CLASS_GRAYSCALE
    )
