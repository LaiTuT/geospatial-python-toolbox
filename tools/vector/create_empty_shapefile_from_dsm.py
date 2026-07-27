#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于一个 DSM 栅格创建新的空 Shapefile。

用途：
- 从 DSM 读取坐标系；
- 创建空 shp，字段结构兼容 shapefile_to_yolo.py；
- 默认创建线要素图层，后续可直接手工/程序写入线，再由 shapefile_to_yolo.py 拓宽并转 YOLO-seg。
"""

import os
from typing import Iterable, Tuple

from osgeo import gdal, ogr, osr

gdal.UseExceptions()


# =========================
# 配置区
# =========================
DSM_PATH = r"D:\Dataproces\1m\AL_25Co_B2_2017\dsm.tif"
OUT_SHP_PATH = r"D:\Dataproces\1m\AL_25Co_B2_2017\shp\empty_template.shp"

# shapefile_to_yolo.py 默认读取 grid_code：255 -> 类别0，128 -> 类别1
CLASS_FIELD = "grid_code"

# 可选：LINESTRING / POLYGON / POINT
GEOMETRY_TYPE = "LINESTRING"


def _make_srs_from_raster(ds: gdal.Dataset) -> osr.SpatialReference:
    proj_wkt = ds.GetProjection()
    srs = osr.SpatialReference()
    if proj_wkt:
        srs.ImportFromWkt(proj_wkt)
    try:
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except Exception:
        pass
    return srs


def _ogr_geom_type(geometry_type: str) -> int:
    geom = geometry_type.strip().upper()
    mapping = {
        "POINT": ogr.wkbPoint,
        "LINE": ogr.wkbLineString,
        "LINESTRING": ogr.wkbLineString,
        "POLYGON": ogr.wkbPolygon,
    }
    if geom not in mapping:
        raise ValueError(f"不支持的 GEOMETRY_TYPE: {geometry_type}")
    return mapping[geom]


def _delete_existing_shapefile(shp_path: str):
    base, _ = os.path.splitext(shp_path)
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj"):
        path = base + ext
        if os.path.exists(path):
            os.remove(path)


def create_empty_shp_from_dsm(
    dsm_path: str,
    out_shp_path: str,
    geometry_type: str = "LINESTRING",
    fields: Iterable[Tuple[str, int, int, int]] = None,
    overwrite: bool = True,
):
    """
    fields: 迭代字段定义：
      (字段名, OGR字段类型, 宽度, 精度)
      例如 ("grid_code", ogr.OFTInteger, 10, 0)
    """
    if not os.path.exists(dsm_path):
        raise FileNotFoundError(f"DSM 文件不存在: {dsm_path}")

    ds = gdal.Open(dsm_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"无法打开 DSM: {dsm_path}")

    srs = _make_srs_from_raster(ds)
    ds = None

    out_dir = os.path.dirname(out_shp_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if overwrite:
        _delete_existing_shapefile(out_shp_path)
    elif os.path.exists(out_shp_path):
        raise FileExistsError(f"输出 shp 已存在: {out_shp_path}")

    drv = ogr.GetDriverByName("ESRI Shapefile")
    if drv is None:
        raise RuntimeError("找不到 ESRI Shapefile 驱动")

    vds = drv.CreateDataSource(out_shp_path)
    if vds is None:
        raise RuntimeError(f"无法创建 shp: {out_shp_path}")

    layer_name = os.path.splitext(os.path.basename(out_shp_path))[0]
    lyr = vds.CreateLayer(layer_name, srs=srs, geom_type=_ogr_geom_type(geometry_type))
    if lyr is None:
        raise RuntimeError(f"无法创建图层: {layer_name}")

    if fields is None:
        fields = (
            (CLASS_FIELD, ogr.OFTInteger, 10, 0),
            ("class_name", ogr.OFTString, 32, 0),
        )

    for name, field_type, width, precision in fields:
        field_def = ogr.FieldDefn(name, field_type)
        if width:
            field_def.SetWidth(width)
        if precision:
            field_def.SetPrecision(precision)
        if lyr.CreateField(field_def) != 0:
            raise RuntimeError(f"字段创建失败: {name}")

    vds.FlushCache()
    vds = None

    print(f"[OK] 已创建空 shp: {out_shp_path}")
    print(f"     几何类型: {geometry_type}")
    print(f"     分类字段: {CLASS_FIELD}，填 255 表示类别0，填 128 表示类别1")


if __name__ == "__main__":
    create_empty_shp_from_dsm(
        dsm_path=DSM_PATH,
        out_shp_path=OUT_SHP_PATH,
        geometry_type=GEOMETRY_TYPE,
        overwrite=True,
    )
