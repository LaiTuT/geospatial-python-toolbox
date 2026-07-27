from osgeo import gdal, ogr
import os
import re


# 需要删除的中间字段
DROP_EXACT_FIELDS = {
    "len_m",
    "len_m_1",
    "s_deg",
    "e_deg",
    "rm_flag",
    "rm_code",
    "rm_code_1",
    "loop_rm",

    "LP_LEN",
    "LP_SDEG",
    "LP_EDEG",
    "LP_RM",
    "LP_CODE",

    "LOOP_LEN",
    "LOOP_RM",
    "LOOP_CODE",
}

# 需要删除的字段前缀
DROP_PREFIXES = (
    "LP_",
    "LOOP_",
)

# 需要删除的字段名模式
DROP_PATTERNS = (
    r"^len_m(_\d+)?$",
    r"^rm_code(_\d+)?$",
)


def should_drop_field(field_name):
    """
    判断字段是否属于中间诊断字段。
    """
    name = field_name.strip()
    name_upper = name.upper()

    if name_upper in {f.upper() for f in DROP_EXACT_FIELDS}:
        return True

    for prefix in DROP_PREFIXES:
        if name_upper.startswith(prefix.upper()):
            return True

    for pattern in DROP_PATTERNS:
        if re.match(pattern, name, flags=re.IGNORECASE):
            return True

    return False


def create_clean_layer(driver, output_path, src_layer, geom_type):
    if os.path.exists(output_path):
        driver.DeleteDataSource(output_path)

    out_ds = driver.CreateDataSource(output_path)
    if out_ds is None:
        raise RuntimeError(f"无法创建输出文件: {output_path}")

    srs = src_layer.GetSpatialRef()
    layer_name = os.path.splitext(os.path.basename(output_path))[0]

    out_layer = out_ds.CreateLayer(
        layer_name,
        srs,
        geom_type
    )

    src_defn = src_layer.GetLayerDefn()

    kept_fields = []

    for i in range(src_defn.GetFieldCount()):
        field_defn = src_defn.GetFieldDefn(i)
        field_name = field_defn.GetName()

        if should_drop_field(field_name):
            continue

        out_layer.CreateField(field_defn)
        kept_fields.append(field_name)

    return out_ds, out_layer, kept_fields


def fieldpruner(
    input_shp,
    output_shp,
    keep_fields=None
):
    """
    FieldPruner

    功能：
    - 删除 LinePruner / LoopPurger 产生的中间字段
    - 输出一个字段干净的新 SHP

    keep_fields:
        None 表示保留所有非中间字段。
        如果传入列表，则只保留列表中的字段。
    """

    ds = gdal.OpenEx(input_shp, gdal.OF_VECTOR)
    if ds is None:
        raise RuntimeError(f"无法打开输入文件: {input_shp}")

    src_layer = ds.GetLayer(0)
    geom_type = src_layer.GetGeomType()
    src_defn = src_layer.GetLayerDefn()

    driver = ogr.GetDriverByName("ESRI Shapefile")

    if os.path.exists(output_shp):
        driver.DeleteDataSource(output_shp)

    out_ds = driver.CreateDataSource(output_shp)
    if out_ds is None:
        raise RuntimeError(f"无法创建输出文件: {output_shp}")

    srs = src_layer.GetSpatialRef()
    layer_name = os.path.splitext(os.path.basename(output_shp))[0]

    out_layer = out_ds.CreateLayer(
        layer_name,
        srs,
        geom_type
    )

    # 建立输出字段
    output_field_names = []

    for i in range(src_defn.GetFieldCount()):
        field_defn = src_defn.GetFieldDefn(i)
        field_name = field_defn.GetName()

        if should_drop_field(field_name):
            continue

        if keep_fields is not None:
            if field_name not in keep_fields:
                continue

        out_layer.CreateField(field_defn)
        output_field_names.append(field_name)

    out_defn = out_layer.GetLayerDefn()

    input_count = 0
    output_count = 0

    for src_feat in src_layer:
        input_count += 1

        geom_ref = src_feat.GetGeometryRef()
        if geom_ref is None:
            continue

        out_feat = ogr.Feature(out_defn)
        out_feat.SetGeometry(geom_ref.Clone())

        for field_name in output_field_names:
            src_idx = src_defn.GetFieldIndex(field_name)
            out_idx = out_defn.GetFieldIndex(field_name)

            if src_idx >= 0 and out_idx >= 0:
                out_feat.SetField(out_idx, src_feat.GetField(src_idx))

        out_layer.CreateFeature(out_feat)
        out_feat = None

        output_count += 1

    out_ds = None
    ds = None

    print("\n========== FieldPruner Report ==========")
    print(f"Input features  : {input_count}")
    print(f"Output features : {output_count}")
    print("----------------------------------------")
    print("Kept fields:")
    for f in output_field_names:
        print(f"  - {f}")
    print("----------------------------------------")
    print(f"Output file     : {output_shp}")
    print("========================================\n")


if __name__ == "__main__":
    fieldpruner(
        input_shp=r"D:\Dataproces\10m\rep\mountains_westcanada_reprojected\shp\USGS_13_n61w154_kk.shp",
        output_shp=r"D:\Dataproces\10m\rep\mountains_westcanada_reprojected\shp\USGS_13_n61w154.shp",
        keep_fields=[
            "arcid",
            "grid_code",
        ]
    )
