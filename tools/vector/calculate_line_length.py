# -*- coding: utf-8 -*-
"""
功能：
    计算 shp 中每一个折线要素的长度，并写入新字段。

依赖：
    GDAL / OGR

适用：
    LineString / MultiLineString 折线 shp

注意：
    如果 shp 是投影坐标系，例如 EPSG:26904，结果单位通常是米。
    如果 shp 是经纬度坐标系，例如 EPSG:4326 / EPSG:4269，结果单位是“度”，不能直接当米用。
"""

from osgeo import ogr, gdal
import os


# =========================
# 输入参数：按需修改这里
# =========================

INPUT_SHP = r"D:\Dataproces\10m\hills_alleghenymous_reprojected\shp\USGS_13_n37w082.shp"

# 是否直接修改原始 shp
# True  = 直接在原 shp 上新增字段
# False = 复制一个新 shp，再在新 shp 上新增字段
UPDATE_IN_PLACE = True

# 输出 shp，仅当 UPDATE_IN_PLACE = False 时使用
OUTPUT_SHP = r"D:\Dataproces\10m\hills_alleghenymous_reprojected\shp\USGS_13_n37w082_length.shp"

# 新字段名
# Shapefile 字段名建议不超过 10 个字符
LENGTH_FIELD = "Len_m"

# 长度换算系数
# EPSG:26904 下坐标单位是米，所以这里用 1.0
# 如果想写入公里，可改成 0.001，并把字段名改为 Len_km
LENGTH_FACTOR = 1.0


def delete_shapefile(shp_path):
    """
    删除一个 shp 及其配套文件。
    """
    base, _ = os.path.splitext(shp_path)
    exts = [
        ".shp", ".shx", ".dbf", ".prj", ".cpg",
        ".qpj", ".sbn", ".sbx", ".fix", ".shp.xml"
    ]

    for ext in exts:
        path = base + ext
        if os.path.exists(path):
            os.remove(path)


def copy_shapefile(input_shp, output_shp):
    """
    复制 shp 数据源。
    """
    if os.path.exists(output_shp):
        delete_shapefile(output_shp)

    driver = ogr.GetDriverByName("ESRI Shapefile")

    src_ds = ogr.Open(input_shp, 0)
    if src_ds is None:
        raise RuntimeError(f"无法打开输入 shp：{input_shp}")

    out_ds = driver.CopyDataSource(src_ds, output_shp)
    if out_ds is None:
        raise RuntimeError(f"无法复制 shp 到：{output_shp}")

    src_ds = None
    out_ds = None


def field_exists(layer, field_name):
    """
    判断字段是否存在。
    """
    layer_defn = layer.GetLayerDefn()

    for i in range(layer_defn.GetFieldCount()):
        if layer_defn.GetFieldDefn(i).GetName().lower() == field_name.lower():
            return True

    return False


def create_length_field(layer, field_name):
    """
    创建长度字段。
    """
    if field_exists(layer, field_name):
        print(f"字段已存在，将直接覆盖字段值：{field_name}")
        return

    field_defn = ogr.FieldDefn(field_name, ogr.OFTReal)
    field_defn.SetWidth(18)
    field_defn.SetPrecision(3)

    result = layer.CreateField(field_defn)

    if result != 0:
        raise RuntimeError(f"创建字段失败：{field_name}")


def is_line_geometry(geom):
    """
    判断是否为折线或多折线。
    """
    if geom is None:
        return False

    geom_type = ogr.GT_Flatten(geom.GetGeometryType())

    return geom_type in (
        ogr.wkbLineString,
        ogr.wkbMultiLineString
    )


def calculate_length_to_field(shp_path, field_name, length_factor=1.0):
    """
    计算每个要素长度，并写入字段。
    """
    gdal.UseExceptions()

    ds = ogr.Open(shp_path, 1)  # 1 表示更新模式
    if ds is None:
        raise RuntimeError(f"无法以更新模式打开 shp：{shp_path}")

    layer = ds.GetLayer(0)
    if layer is None:
        raise RuntimeError("无法读取图层。")

    spatial_ref = layer.GetSpatialRef()
    if spatial_ref is not None:
        print("图层坐标系：")
        print(spatial_ref.GetName())
    else:
        print("警告：图层没有坐标系信息，长度单位取决于坐标本身。")

    create_length_field(layer, field_name)

    total_length = 0.0
    valid_count = 0
    skipped_count = 0

    layer.ResetReading()

    for feature in layer:
        geom = feature.GetGeometryRef()

        if not is_line_geometry(geom):
            feature.SetField(field_name, 0.0)
            layer.SetFeature(feature)
            skipped_count += 1
            continue

        length = geom.Length() * length_factor

        feature.SetField(field_name, length)
        layer.SetFeature(feature)

        total_length += length
        valid_count += 1

    ds = None

    print("处理完成。")
    print(f"有效折线要素数量：{valid_count}")
    print(f"跳过非折线或空几何数量：{skipped_count}")
    print(f"总长度：{total_length:.3f}")


def main():
    if UPDATE_IN_PLACE:
        target_shp = INPUT_SHP
    else:
        copy_shapefile(INPUT_SHP, OUTPUT_SHP)
        target_shp = OUTPUT_SHP

    calculate_length_to_field(
        shp_path=target_shp,
        field_name=LENGTH_FIELD,
        length_factor=LENGTH_FACTOR
    )

    print(f"结果 shp：{target_shp}")


if __name__ == "__main__":
    main()
