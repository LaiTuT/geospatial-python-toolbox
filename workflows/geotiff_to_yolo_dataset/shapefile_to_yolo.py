#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：
- 遍历指定 tif 文件夹（可递归）
- 对每张 tif：裁切 shp 相交要素，按需要对线要素做像素 buffer（线转面，适配 YOLO-seg）
- 输出 YOLO 分割 txt（归一化坐标，cls x1 y1 x2 y2 ...）
- 可选输出 shp->mask 可视化（PNG/GTiff，背景 0，类别写 cls+1）
- 可选把生成的 YOLO txt 再“反渲染”为 mask（验证 txt 是否正确）

依赖：
- GDAL(osgeo)  (>=3.0 建议；>=3.11 会提示 OGR Memory 弃用，本脚本已规避)
- shapely
"""

import os
import glob
import json
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
from osgeo import gdal, ogr, osr
from shapely.geometry import shape, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.affinity import affine_transform

gdal.UseExceptions()


# =========================
# 基础工具
# =========================
def open_raster(path: str):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"无法打开影像: {path}")
    gt = ds.GetGeoTransform(can_return_null=True)
    if gt is None:
        raise RuntimeError(f"影像缺少 GeoTransform: {path}")
    proj_wkt = ds.GetProjection()
    width, height = ds.RasterXSize, ds.RasterYSize
    return ds, gt, proj_wkt, width, height


def _make_srs(wkt: Optional[str]) -> Optional[osr.SpatialReference]:
    """创建 SRS，并强制传统 GIS 轴顺序（规避 GDAL3+ EPSG:4326 轴顺序坑）"""
    if not wkt:
        return None
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    try:
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except Exception:
        pass
    return srs


def _build_ct(src_wkt: Optional[str], dst_wkt: Optional[str]) -> Optional[osr.CoordinateTransformation]:
    if not src_wkt or not dst_wkt or src_wkt == dst_wkt:
        return None
    src = _make_srs(src_wkt)
    dst = _make_srs(dst_wkt)
    if src is None or dst is None:
        return None
    return osr.CoordinateTransformation(src, dst)


def _transform_bbox(minx, miny, maxx, maxy, src_wkt: Optional[str], dst_wkt: Optional[str]):
    """
    bbox 从 src_wkt -> dst_wkt
    返回变换后 bbox (minx, miny, maxx, maxy)
    """
    ct = _build_ct(src_wkt, dst_wkt)
    if ct is None:
        return minx, miny, maxx, maxy

    corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    xs, ys = [], []
    for x, y in corners:
        x2, y2, *_ = ct.TransformPoint(x, y)
        xs.append(x2)
        ys.append(y2)
    return min(xs), min(ys), max(xs), max(ys)


def ogr_to_shp(geom: ogr.Geometry):
    return shape(json.loads(geom.ExportToJson()))


def affine_geo_to_pix(gt: Tuple[float, float, float, float, float, float]):
    """GeoTransform 反解（地理坐标 -> 像素坐标）的 shapely affine 参数"""
    det = gt[1] * gt[5] - gt[2] * gt[4]
    if det == 0:
        raise RuntimeError("GeoTransform 不可逆")
    a = gt[5] / det
    b = -gt[2] / det
    d = -gt[4] / det
    e = gt[1] / det
    xoff = -a * gt[0] - b * gt[3]
    yoff = -d * gt[0] - e * gt[3]
    return (a, b, d, e, xoff, yoff)


def clip_to_img(geom, w: int, h: int):
    rect = Polygon([(0, 0), (w, 0), (w, h), (0, h)])
    c = geom.intersection(rect)
    return None if c.is_empty else c


def exterior_only(geom) -> List[List[Tuple[float, float]]]:
    """只输出外环（YOLO seg 常用）；MultiPolygon 会输出多个实例外环"""
    if isinstance(geom, Polygon):
        return [list(geom.exterior.coords)]
    if isinstance(geom, MultiPolygon):
        return [list(g.exterior.coords) for g in geom.geoms]
    return []


def norm_coords(coords: List[Tuple[float, float]], w: int, h: int) -> List[Tuple[float, float]]:
    out = []
    for x, y in coords:
        xn = max(0.0, min(1.0, x / float(w)))
        yn = max(0.0, min(1.0, y / float(h)))
        out.append((xn, yn))
    return out


# =========================
# 分类字段读取
# =========================
def read_class(feat: ogr.Feature, field: str) -> Optional[int]:
    raw = feat.GetField(field)
    val = None
    try:
        val = int(raw)
    except Exception:
        try:
            val = int(float(raw))
        except Exception:
            val = None

    if val is None:
        return None

    # 255 -> 0 (ridge), 128 -> 1 (valley)
    if val == 255:
        return 0
    if val == 128:
        return 1
    return None


# =========================
# shp -> mask（规避 OGR 'Memory' 弃用；PNG 用 CreateCopy）
# =========================
def _write_mem_raster_to_file(mem_ds: gdal.Dataset, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    drv = gdal.GetDriverByName("PNG" if ext == ".png" else "GTiff")
    if drv is None:
        raise RuntimeError(f"找不到输出驱动: {ext}")
    out_ds = drv.CreateCopy(out_path, mem_ds, strict=0)
    if out_ds is None:
        raise RuntimeError(f"写出失败: {out_path}")
    out_ds.FlushCache()
    out_ds = None



def rasterize_mask_from_polys(polys, w: int, h: int, out_mask: str, all_touched: bool = True):
    """
    polys: list[(cls, shapely_geom)]，cls 为 0/1；写入 cls+1 便于可视化，背景 0
    兼容旧 GDAL：使用 RasterizeLayer（避免 RasterizeGeometries 不存在）
    PNG 输出使用 CreateCopy
    """
    if not polys:
        return

    # 1) 构建内存矢量数据源（用 gdal 的 Memory 驱动）
    vdrv = gdal.GetDriverByName("Memory")
    if vdrv is None:
        raise RuntimeError("GDAL 不支持 Memory 驱动（矢量）")

    vds = vdrv.Create("", 0, 0, 0, gdal.GDT_Unknown)
    lyr = vds.CreateLayer("poly", srs=None, geom_type=ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("val", ogr.OFTInteger))

    for cls, geom in polys:
        if geom.is_empty:
            continue
        og = ogr.CreateGeometryFromWkb(geom.wkb)
        if og is None:
            continue
        feat = ogr.Feature(lyr.GetLayerDefn())
        feat.SetField("val", DEFAULT_CLASS_MASK_VALUES.get(int(cls), int(cls) + 1))
        feat.SetGeometry(og)
        lyr.CreateFeature(feat)
        feat = None

    # 2) 内存栅格（MEM）用于 rasterize
    mem_ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    mem_ds.SetGeoTransform((0, 1, 0, 0, 0, 1))
    band = mem_ds.GetRasterBand(1)
    band.Fill(0)

    opts = ["ATTRIBUTE=val"]
    if all_touched:
        opts.append("ALL_TOUCHED=TRUE")

    # 3) RasterizeLayer（几乎所有 GDAL 都有）
    gdal.RasterizeLayer(mem_ds, [1], lyr, options=opts)
    mem_ds.FlushCache()

    # 4) 写出文件（PNG/GTiff）
    _write_mem_raster_to_file(mem_ds, out_mask)

    mem_ds = None
    vds = None


# =========================
# YOLO txt -> mask 可视化（验证 txt 是否正确）
# =========================

DEFAULT_CLASS_MASK_VALUES = {
    0: 255,
    1: 128,
}


def rasterize_polys_to_mask_array(
    polys,
    w: int,
    h: int,
    class_mask_values: Optional[Dict[int, int]] = None,
    all_touched: bool = True,
) -> np.ndarray:
    """
    ?????????????????????/?? mask ???

    polys: list[(cls, shapely_geom)]
    class_mask_values: YOLO ?? -> mask ??????? raster_mask_to_yolo.py ?????
                       class 0 ? 255?class 1 ? 128?
    """
    class_mask_values = class_mask_values or DEFAULT_CLASS_MASK_VALUES
    mask = np.zeros((h, w), dtype=np.uint8)

    if not polys:
        return mask

    vdrv = gdal.GetDriverByName("Memory")
    if vdrv is None:
        raise RuntimeError("GDAL ??? Memory ??????")

    vds = vdrv.Create("", 0, 0, 0, gdal.GDT_Unknown)
    lyr = vds.CreateLayer("poly", srs=None, geom_type=ogr.wkbUnknown)
    lyr.CreateField(ogr.FieldDefn("val", ogr.OFTInteger))

    for cls, geom in polys:
        if geom is None or geom.is_empty:
            continue
        val = int(class_mask_values.get(int(cls), int(cls) + 1))
        og = ogr.CreateGeometryFromWkb(geom.wkb)
        if og is None:
            continue
        feat = ogr.Feature(lyr.GetLayerDefn())
        feat.SetField("val", val)
        feat.SetGeometry(og)
        lyr.CreateFeature(feat)
        feat = None

    mem_ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    mem_ds.SetGeoTransform((0, 1, 0, 0, 0, 1))
    band = mem_ds.GetRasterBand(1)
    band.Fill(0)

    opts = ["ATTRIBUTE=val"]
    if all_touched:
        opts.append("ALL_TOUCHED=TRUE")

    gdal.RasterizeLayer(mem_ds, [1], lyr, options=opts)
    mem_ds.FlushCache()

    arr = band.ReadAsArray()
    if arr is None:
        arr = mask
    else:
        arr = arr.astype(np.uint8, copy=False)

    mem_ds = None
    vds = None
    return arr


def write_mask_array(mask_arr: np.ndarray, out_mask: str):
    """? mask ????? PNG/GTiff?"""
    if out_mask is None:
        return
    h, w = mask_arr.shape[:2]
    mem_ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    mem_ds.SetGeoTransform((0, 1, 0, 0, 0, 1))
    mem_ds.GetRasterBand(1).WriteArray(mask_arr.astype(np.uint8, copy=False))
    mem_ds.FlushCache()
    _write_mem_raster_to_file(mem_ds, out_mask)
    mem_ds = None


def mask_array_to_yolo_lines(
    mask_arr: np.ndarray,
    class_mapping: Dict[int, int],
    min_area: float = 0.0,
    simplify_factor: float = 0.0,
    use_morphology: bool = False,
) -> List[str]:
    """
    ?? raster_mask_to_yolo.py?? mask ????????????????????? YOLO-seg ???
    class_mapping: mask ??? -> YOLO ?? id??? {255: 0, 128: 1}
    """
    if mask_arr is None:
        return []

    h, w = mask_arr.shape[:2]
    txt_lines: List[str] = []

    for pixel_val, class_id in class_mapping.items():
        binary = np.zeros_like(mask_arr, dtype=np.uint8)
        binary[mask_arr == pixel_val] = 255

        if cv2.countNonZero(binary) == 0:
            continue

        if use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue

            if simplify_factor > 0:
                epsilon = simplify_factor * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                pts = approx.reshape(-1, 2)
            else:
                pts = cnt.reshape(-1, 2)

            if len(pts) < 3:
                continue

            norm_pts = []
            for x, y in pts:
                nx = max(0.0, min(1.0, float(x) / float(w)))
                ny = max(0.0, min(1.0, float(y) / float(h)))
                norm_pts.extend([nx, ny])

            line_str = f"{class_id} " + " ".join(f"{v:.6f}" for v in norm_pts)
            txt_lines.append(line_str)

    return txt_lines


def yolo_txt_to_mask(txt_path: str, w: int, h: int, out_mask: str, all_touched: bool = True):
    """
    YOLO-seg txt -> mask（兼容旧 GDAL：RasterizeLayer）
    背景 0；类别写 cls+1
    """
    # 内存矢量 DS
    vdrv = gdal.GetDriverByName("Memory")
    vds = vdrv.Create("", 0, 0, 0, gdal.GDT_Unknown)
    lyr = vds.CreateLayer("poly", srs=None, geom_type=ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("val", ogr.OFTInteger))

    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 1 + 6:
                    continue
                cls = int(float(parts[0]))
                nums = list(map(float, parts[1:]))
                if len(nums) % 2 != 0:
                    continue

                ring = ogr.Geometry(ogr.wkbLinearRing)
                for i in range(0, len(nums), 2):
                    xn, yn = nums[i], nums[i + 1]
                    x = xn * w
                    y = yn * h
                    if x < 0: x = 0
                    if x > w - 1: x = w - 1
                    if y < 0: y = 0
                    if y > h - 1: y = h - 1
                    ring.AddPoint(float(x), float(y))
                if ring.GetPointCount() < 3:
                    continue
                ring.CloseRings()

                poly = ogr.Geometry(ogr.wkbPolygon)
                poly.AddGeometry(ring)

                feat = ogr.Feature(lyr.GetLayerDefn())
                feat.SetField("val", DEFAULT_CLASS_MASK_VALUES.get(int(cls), int(cls) + 1))
                feat.SetGeometry(poly)
                lyr.CreateFeature(feat)
                feat = None

    # 内存栅格
    mem_ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    mem_ds.SetGeoTransform((0, 1, 0, 0, 0, 1))
    band = mem_ds.GetRasterBand(1)
    band.Fill(0)

    opts = ["ATTRIBUTE=val"]
    if all_touched:
        opts.append("ALL_TOUCHED=TRUE")

    gdal.RasterizeLayer(mem_ds, [1], lyr, options=opts)
    mem_ds.FlushCache()

    _write_mem_raster_to_file(mem_ds, out_mask)

    mem_ds = None
    vds = None



# =========================
# 单瓦片处理
# =========================
def process_tile(
    tile_path: str,
    shp_path: str,
    field: str,
    out_txt: str,
    buffer_pixels: float = 5.0,
    simplify_tol: float = 0.0,
    enable_buffer: bool = True,
    out_shp_mask: Optional[str] = None,
    min_area: float = 0.0,
    mask_simplify_factor: float = 0.0,
    mask_class_values: Optional[Dict[int, int]] = None,
    warn_on_line_without_buffer: bool = True,
):
    ds, gt, dst_wkt, w, h = open_raster(tile_path)

    drv = ogr.GetDriverByName("ESRI Shapefile")
    vds = drv.Open(shp_path, 0)
    if vds is None:
        raise RuntimeError(f"无法打开 Shapefile: {shp_path}")

    lyr = vds.GetLayer()
    if lyr is None:
        raise RuntimeError("Shapefile 无图层")

    src_srs = lyr.GetSpatialRef()
    src_wkt = src_srs.ExportToWkt() if src_srs else None

    # ---- 影像 bbox（在影像 CRS：dst_wkt）
    def pix2geo(x, y):
        return (gt[0] + x * gt[1] + y * gt[2], gt[3] + x * gt[4] + y * gt[5])

    corners = [pix2geo(0, 0), pix2geo(w, 0), pix2geo(w, h), pix2geo(0, h)]
    minx = min(p[0] for p in corners)
    miny = min(p[1] for p in corners)
    maxx = max(p[0] for p in corners)
    maxy = max(p[1] for p in corners)

    # ✅ 关键：SetSpatialFilterRect 使用图层坐标系，需把影像 bbox 从 dst_wkt -> src_wkt
    fminx, fminy, fmaxx, fmaxy = _transform_bbox(minx, miny, maxx, maxy, dst_wkt, src_wkt)
    lyr.SetSpatialFilterRect(fminx, fminy, fmaxx, fmaxy)
    lyr.ResetReading()

    # 像素仿射
    aff = affine_geo_to_pix(gt)

    # ✅ 复用坐标变换：src -> dst
    ct_src_to_dst = _build_ct(src_wkt, dst_wkt)

    # 新流程：
    # 1) shp 要素投影到瓦片像素坐标；
    # 2) 线要素先 buffer 拓宽成面；
    # 3) 将拓宽后的面栅格化成 mask；
    # 4) 仿照 raster_mask_to_yolo.py 从 mask 连通轮廓生成 YOLO-seg txt。
    mask_polys = []  # (cls, shapely_poly)

    for feat in lyr:
        cls = read_class(feat, field)
        if cls is None:
            continue

        og = feat.GetGeometryRef()
        if og is None:
            continue

        og2 = og.Clone()
        if ct_src_to_dst is not None:
            og2.Transform(ct_src_to_dst)

        shp_geom = ogr_to_shp(og2)

        # 地理 -> 像素
        pix = affine_transform(shp_geom, aff)

        # 线要素：YOLO-seg 需要面；不 buffer 就无法输出分割
        is_line = isinstance(pix, (LineString, MultiLineString))
        if is_line and not enable_buffer:
            if warn_on_line_without_buffer:
                # 不抛异常，只跳过这条
                # 如果你想“自动保证有输出”，可以把这里改成强制 buffer 一个最小值
                pass
            continue

        if enable_buffer and is_line:
            pix = pix.buffer(buffer_pixels, cap_style=1, join_style=2)

        pix = clip_to_img(pix, w, h)
        if pix is None:
            continue

        if simplify_tol > 0:
            pix = pix.simplify(simplify_tol, preserve_topology=True)

        # 只记录面；线必须先 buffer 成面，之后统一 rasterize -> mask -> yolo。
        if isinstance(pix, (Polygon, MultiPolygon)):
            mask_polys.append((cls, pix))

    lyr.SetSpatialFilter(None)

    class_values = mask_class_values or DEFAULT_CLASS_MASK_VALUES
    mask_arr = rasterize_polys_to_mask_array(
        mask_polys,
        w,
        h,
        class_mask_values=class_values,
        all_touched=True,
    )

    # 写 shp->mask：现在输出的是“拓宽后要素”的 mask；
    # 即使没有要素，也会输出一张全 0 mask，方便与 labels 一一对应。
    if out_shp_mask is not None:
        try:
            write_mask_array(mask_arr, out_shp_mask)
        except Exception as e:
            print(f"[WARN] mask 输出失败 {out_shp_mask}: {e}")

    class_mapping = {int(v): int(k) for k, v in class_values.items()}
    for cls, _ in mask_polys:
        cls = int(cls)
        if cls not in class_values:
            class_mapping[int(cls) + 1] = cls

    lines = mask_array_to_yolo_lines(
        mask_arr,
        class_mapping=class_mapping,
        min_area=min_area,
        simplify_factor=mask_simplify_factor,
        use_morphology=False,
    )

    # 写 txt
    out_dir = os.path.dirname(out_txt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    ds = None
    vds = None


# =========================
# 批处理：tif -> yolo txt (+ 可选 shp->mask)
# =========================
def batch_convert_from_tif_folder(
    shp_path: str,
    field: str,
    tif_dir: str,
    labels_dir: str,
    patterns=(".tif", ".tiff"),
    recursive: bool = True,
    buffer_pixels: float = 5.0,
    simplify_tol: float = 0.0,
    keep_subdirs: bool = True,
    enable_buffer: bool = True,
    masks_dir: Optional[str] = None,
    mask_ext: str = ".png",
    min_area: float = 0.0,
    mask_simplify_factor: float = 0.0,
    mask_class_values: Optional[Dict[int, int]] = None,
):
    tif_dir = os.path.abspath(tif_dir)
    labels_dir = os.path.abspath(labels_dir)
    masks_dir = os.path.abspath(masks_dir) if masks_dir else None

    files = []
    for ext in patterns:
        if recursive:
            files += glob.glob(os.path.join(tif_dir, "**", f"*{ext}"), recursive=True)
        else:
            files += glob.glob(os.path.join(tif_dir, f"*{ext}"))
    files.sort()

    total_ok = 0
    total_warn = 0

    for tile in files:
        if keep_subdirs:
            rel = os.path.relpath(tile, tif_dir)
            rel_noext = os.path.splitext(rel)[0]
            out_txt = os.path.join(labels_dir, rel_noext + ".txt")
            out_mask = os.path.join(masks_dir, rel_noext + mask_ext) if masks_dir else None
        else:
            name = os.path.splitext(os.path.basename(tile))[0]
            out_txt = os.path.join(labels_dir, name + ".txt")
            out_mask = os.path.join(masks_dir, name + mask_ext) if masks_dir else None

        try:
            process_tile(
                tile,
                shp_path,
                field,
                out_txt,
                buffer_pixels=buffer_pixels,
                simplify_tol=simplify_tol,
                enable_buffer=enable_buffer,
                out_shp_mask=out_mask,
                min_area=min_area,
                mask_simplify_factor=mask_simplify_factor,
                mask_class_values=mask_class_values,
            )
            total_ok += 1
        except Exception as e:
            total_warn += 1
            print(f"[WARN] {tile}: {e}")

    print(
        f"完成：{total_ok} 张 tif（WARN: {total_warn}），标签输出到：{labels_dir}"
        + (f"；shp->mask 输出到：{masks_dir}" if masks_dir else "")
    )


# =========================
# 批处理：shp 文件夹 -> 每个 shp 独立生成 yolo txt / mask / 可视化
# =========================
def _strip_suffixes(name: str, suffixes=("_buffer",)) -> str:
    """去掉常见后缀，用于 shp 名称匹配瓦片子目录。"""
    out = name
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if suffix and out.lower().endswith(suffix.lower()):
                out = out[: -len(suffix)]
                changed = True
                break
    return out


def _resolve_tif_dir_for_shp(
    tif_root_dir: str,
    shp_base_name: str,
    match_tif_subdir_by_shp_name: bool = True,
    strip_name_suffixes=("_buffer",),
) -> Tuple[Optional[str], str]:
    """
    根据 shp 文件名寻找对应瓦片目录。

    match_tif_subdir_by_shp_name=True 时会优先尝试：
      1) tif_root_dir / 去后缀后的 shp 名称
      2) tif_root_dir / 原始 shp 名称
    例如：
      USGS_xxx_buffer.shp -> tiles/USGS_xxx

    返回：(tile_dir, output_name)
    """
    output_name = _strip_suffixes(shp_base_name, strip_name_suffixes)

    if not match_tif_subdir_by_shp_name:
        return tif_root_dir, output_name

    candidates = [
        os.path.join(tif_root_dir, output_name),
        os.path.join(tif_root_dir, shp_base_name),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate, output_name

    return None, output_name


def batch_convert_from_shp_folder(
    shp_dir: str,
    field: str,
    tif_root_dir: str,
    labels_root_dir: str,
    patterns=(".tif", ".tiff"),
    recursive: bool = True,
    buffer_pixels: float = 5.0,
    simplify_tol: float = 0.0,
    keep_subdirs: bool = True,
    enable_buffer: bool = True,
    masks_root_dir: Optional[str] = None,
    mask_ext: str = ".png",
    vis_from_txt_root_dir: Optional[str] = None,
    vis_ext: str = ".png",
    min_area: float = 0.0,
    mask_simplify_factor: float = 0.0,
    mask_class_values: Optional[Dict[int, int]] = None,
    match_tif_subdir_by_shp_name: bool = True,
    strip_name_suffixes=("_buffer",),
):
    """
    批量处理 shp_dir 下所有 .shp。

    默认目录约定：
      shp_dir/
        A.shp
        B_buffer.shp
      tif_root_dir/
        A/
          tile_0_0.tif ...
        B/
          tile_0_0.tif ...
      labels_root_dir/
        A/
        B/

    如果所有 shp 都要处理同一个 tif_dir，则把 match_tif_subdir_by_shp_name=False，
    此时 tif_root_dir 会被直接当作瓦片目录。
    """
    shp_dir = os.path.abspath(shp_dir)
    tif_root_dir = os.path.abspath(tif_root_dir)
    labels_root_dir = os.path.abspath(labels_root_dir)
    masks_root_dir = os.path.abspath(masks_root_dir) if masks_root_dir else None
    vis_from_txt_root_dir = os.path.abspath(vis_from_txt_root_dir) if vis_from_txt_root_dir else None

    shp_files = glob.glob(os.path.join(shp_dir, "*.shp"))
    shp_files += glob.glob(os.path.join(shp_dir, "*.SHP"))
    shp_files = sorted(set(shp_files))

    if not shp_files:
        print(f"[Err] shp 目录下未找到 .shp 文件: {shp_dir}")
        return

    print(f"找到 {len(shp_files)} 个 shp 文件")

    ok = 0
    warn = 0

    for shp_path in shp_files:
        shp_base = os.path.splitext(os.path.basename(shp_path))[0]
        tile_dir, out_name = _resolve_tif_dir_for_shp(
            tif_root_dir,
            shp_base,
            match_tif_subdir_by_shp_name=match_tif_subdir_by_shp_name,
            strip_name_suffixes=strip_name_suffixes,
        )

        if tile_dir is None or not os.path.isdir(tile_dir):
            warn += 1
            print(f"[Warn] 跳过 {shp_base}: 找不到对应瓦片目录，已尝试 {tif_root_dir}\\{out_name} 和 {tif_root_dir}\\{shp_base}")
            continue

        labels_dir = os.path.join(labels_root_dir, out_name)
        masks_dir = os.path.join(masks_root_dir, out_name) if masks_root_dir else None
        vis_dir = os.path.join(vis_from_txt_root_dir, out_name) if vis_from_txt_root_dir else None

        print(f"\n{'=' * 60}")
        print(f"处理 shp: {shp_base}")
        print(f"  瓦片目录: {tile_dir}")
        print(f"  标签目录: {labels_dir}")
        if masks_dir:
            print(f"  mask 目录: {masks_dir}")

        try:
            batch_convert_from_tif_folder(
                shp_path,
                field,
                tile_dir,
                labels_dir,
                patterns=patterns,
                recursive=recursive,
                buffer_pixels=buffer_pixels,
                simplify_tol=simplify_tol,
                keep_subdirs=keep_subdirs,
                enable_buffer=enable_buffer,
                masks_dir=masks_dir,
                mask_ext=mask_ext,
                min_area=min_area,
                mask_simplify_factor=mask_simplify_factor,
                mask_class_values=mask_class_values,
            )

            if vis_dir:
                batch_visualize_yolo_txt_as_mask(
                    tile_dir,
                    labels_dir,
                    vis_dir,
                    patterns=patterns,
                    recursive=recursive,
                    keep_subdirs=keep_subdirs,
                    vis_ext=vis_ext,
                )

            ok += 1
        except Exception as e:
            warn += 1
            print(f"[WARN] 处理 shp 失败 {shp_path}: {e}")

    print(f"\n全部 shp 处理完成：成功 {ok}/{len(shp_files)}，WARN {warn}")


# =========================
# 批处理：YOLO txt -> mask（用于可视化验证）
# =========================
def batch_visualize_yolo_txt_as_mask(
    tif_dir: str,
    labels_dir: str,
    out_vis_dir: str,
    patterns=(".tif", ".tiff"),
    recursive: bool = True,
    keep_subdirs: bool = True,
    vis_ext: str = ".png",
):
    tif_dir = os.path.abspath(tif_dir)
    labels_dir = os.path.abspath(labels_dir)
    out_vis_dir = os.path.abspath(out_vis_dir)

    files = []
    for ext in patterns:
        if recursive:
            files += glob.glob(os.path.join(tif_dir, "**", f"*{ext}"), recursive=True)
        else:
            files += glob.glob(os.path.join(tif_dir, f"*{ext}"))
    files.sort()

    ok = 0
    warn = 0

    for tile in files:
        ds = gdal.Open(tile, gdal.GA_ReadOnly)
        if ds is None:
            warn += 1
            print(f"[WARN] 无法打开 tif: {tile}")
            continue
        w, h = ds.RasterXSize, ds.RasterYSize
        ds = None

        if keep_subdirs:
            rel = os.path.relpath(tile, tif_dir)
            rel_noext = os.path.splitext(rel)[0]
            txt_path = os.path.join(labels_dir, rel_noext + ".txt")
            out_mask = os.path.join(out_vis_dir, rel_noext + vis_ext)
        else:
            name = os.path.splitext(os.path.basename(tile))[0]
            txt_path = os.path.join(labels_dir, name + ".txt")
            out_mask = os.path.join(out_vis_dir, name + vis_ext)

        try:
            yolo_txt_to_mask(txt_path, w, h, out_mask, all_touched=True)
            ok += 1
        except Exception as e:
            warn += 1
            print(f"[WARN] YOLO 可视化失败 {tile}: {e}")

    print(f"YOLO 可视化完成：{ok} 张（WARN: {warn}），输出到：{out_vis_dir}")


# =========================
# 入口（配置区）
# =========================
if __name__ == "__main__":
    SHP_DIR = r"D:\Dataproces\1m\AL_25Co_B2_2017\shp"
    FIELD = "grid_code"  # 255→0，128→1

    # 瓦片根目录。默认会按 shp 文件名查找同名子目录：
    #   shp/USGS_xxx_buffer.shp -> tiles/USGS_xxx/
    # 如所有 shp 都共用同一个瓦片目录，可在下面把 MATCH_TIF_SUBDIR_BY_SHP_NAME 改为 False。
    TIF_ROOT_DIR = r"D:\Dataproces\1m\AL_25Co_B2_2017\tiles"
    LABELS_OUT = r"D:\Dataproces\1m\AL_25Co_B2_2017\labels"

    # shp->mask（拓宽后的矢量先栅格化）可视化输出；设为 None 则不输出
    MASKS_OUT = None
    MASK_EXT = ".png"  # ".png" 或 ".tif"

    # 线要素 buffer（像素单位）向两边各拓宽，例如2.5，总宽5
    BUFFER_PIXELS = 1.25
    ENABLE_BUFFER = True  # 线做 YOLO-seg 建议 True；False 时线不会产生分割 txt

    SIMPLIFY_TOL = 0.0
    MIN_AREA = 0.0
    MASK_SIMPLIFY_FACTOR = 0.0
    MASK_CLASS_VALUES = {
        0: 255,
        1: 128,
    }
    RECURSIVE = True
    KEEP_SUBDIRS = False
    MATCH_TIF_SUBDIR_BY_SHP_NAME = True
    STRIP_NAME_SUFFIXES = ("",)
    VIS_FROM_TXT_OUT = r"D:\Dataproces\1m\AL_25Co_B2_2017\vis_from_txt"

    # 批量处理 shp 文件夹：
    # 每个 shp：线要素拓宽 -> mask -> YOLO txt（+ 可选输出 mask / txt反渲染验证图）
    batch_convert_from_shp_folder(
        SHP_DIR,
        FIELD,
        TIF_ROOT_DIR,
        LABELS_OUT,
        patterns=(".tif", ".tiff"),
        recursive=RECURSIVE,
        buffer_pixels=BUFFER_PIXELS,
        simplify_tol=SIMPLIFY_TOL,
        keep_subdirs=KEEP_SUBDIRS,
        enable_buffer=ENABLE_BUFFER,
        masks_root_dir=MASKS_OUT,
        mask_ext=MASK_EXT,
        min_area=MIN_AREA,
        mask_simplify_factor=MASK_SIMPLIFY_FACTOR,
        mask_class_values=MASK_CLASS_VALUES,
        vis_from_txt_root_dir=VIS_FROM_TXT_OUT,
        vis_ext=".png",
        match_tif_subdir_by_shp_name=MATCH_TIF_SUBDIR_BY_SHP_NAME,
        strip_name_suffixes=STRIP_NAME_SUFFIXES,
    )
