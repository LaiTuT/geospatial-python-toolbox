from osgeo import gdal, ogr
from collections import defaultdict
import os


SKIP_FIELDS = {
    "len_m", "len_m_1",
    "s_deg", "e_deg",
    "rm_flag", "rm_code", "rm_code_1",
    "loop_rm",
    "LP_LEN", "LP_SDEG", "LP_EDEG", "LP_RM", "LP_CODE",
    "LOOP_LEN", "LOOP_RM", "LOOP_CODE"
}


def should_skip_field(field_name):
    return field_name.upper() in {f.upper() for f in SKIP_FIELDS}


def create_output_layer(driver, out_path, src_layer, geom_type):
    if os.path.exists(out_path):
        driver.DeleteDataSource(out_path)

    out_ds = driver.CreateDataSource(out_path)
    if out_ds is None:
        raise RuntimeError(f"无法创建输出文件: {out_path}")

    srs = src_layer.GetSpatialRef()
    layer_name = os.path.splitext(os.path.basename(out_path))[0]
    out_layer = out_ds.CreateLayer(layer_name, srs, geom_type)

    src_defn = src_layer.GetLayerDefn()

    # 复制原始字段，但跳过旧的诊断字段
    for i in range(src_defn.GetFieldCount()):
        field_defn = src_defn.GetFieldDefn(i)
        field_name = field_defn.GetName()

        if should_skip_field(field_name):
            continue

        out_layer.CreateField(field_defn)

    # 新增 LinePruner 诊断字段
    f_len = ogr.FieldDefn("LP_LEN", ogr.OFTReal)
    f_len.SetWidth(18)
    f_len.SetPrecision(3)
    out_layer.CreateField(f_len)

    out_layer.CreateField(ogr.FieldDefn("LP_SDEG", ogr.OFTInteger))
    out_layer.CreateField(ogr.FieldDefn("LP_EDEG", ogr.OFTInteger))
    out_layer.CreateField(ogr.FieldDefn("LP_RM", ogr.OFTInteger))

    f_code = ogr.FieldDefn("LP_CODE", ogr.OFTString)
    f_code.SetWidth(32)
    out_layer.CreateField(f_code)

    return out_ds, out_layer


def copy_feature(src_feat, out_layer, geom, length, s_deg, e_deg, rm_flag, rm_code):
    out_defn = out_layer.GetLayerDefn()
    out_feat = ogr.Feature(out_defn)

    src_defn = src_feat.GetDefnRef()

    # 按字段名复制，避免跳过字段后索引错位
    for i in range(src_defn.GetFieldCount()):
        src_field_name = src_defn.GetFieldDefn(i).GetName()

        if should_skip_field(src_field_name):
            continue

        out_index = out_defn.GetFieldIndex(src_field_name)

        if out_index >= 0:
            out_feat.SetField(out_index, src_feat.GetField(i))

    out_feat.SetGeometry(geom.Clone())
    out_feat.SetField("LP_LEN", float(length))
    out_feat.SetField("LP_SDEG", int(s_deg))
    out_feat.SetField("LP_EDEG", int(e_deg))
    out_feat.SetField("LP_RM", int(rm_flag))
    out_feat.SetField("LP_CODE", rm_code)

    out_layer.CreateFeature(out_feat)
    out_feat = None


def linepruner_node_topology_v2(
    input_shp,
    kept_shp,
    removed_shp,
    short_len=20.0,
    isolated_len=20.0,
    from_field="from_node",
    to_field="to_node"
):
    """
    LinePruner-NodeTopo v2

    删除：
    - 短悬挂线
    - 短孤立线

    保留：
    - 正常连接线
    - 长悬挂线
    - 长孤立线
    """

    ds = gdal.OpenEx(input_shp, gdal.OF_VECTOR)
    if ds is None:
        raise RuntimeError(f"无法打开输入文件: {input_shp}")

    src_layer = ds.GetLayer(0)
    geom_type = src_layer.GetGeomType()

    records = []
    node_to_fids = defaultdict(set)

    # 读取线要素
    for feat in src_layer:
        geom_ref = feat.GetGeometryRef()
        if geom_ref is None:
            continue

        fid = feat.GetFID()
        from_node = feat.GetField(from_field)
        to_node = feat.GetField(to_field)

        if from_node is None or to_node is None:
            continue

        geom = geom_ref.Clone()
        length = geom.Length()

        records.append({
            "fid": fid,
            "feature": feat.Clone(),
            "geometry": geom,
            "from_node": from_node,
            "to_node": to_node,
            "length": length
        })

        node_to_fids[from_node].add(fid)
        node_to_fids[to_node].add(fid)

    driver = ogr.GetDriverByName("ESRI Shapefile")

    kept_ds, kept_layer = create_output_layer(
        driver, kept_shp, src_layer, geom_type
    )

    removed_ds, removed_layer = create_output_layer(
        driver, removed_shp, src_layer, geom_type
    )

    kept_count = 0
    removed_count = 0

    normal_count = 0
    short_dangle_count = 0
    long_dangle_count = 0
    short_isolated_count = 0
    long_isolated_count = 0

    for rec in records:
        fid = rec["fid"]
        from_node = rec["from_node"]
        to_node = rec["to_node"]
        length = rec["length"]

        s_deg = len(node_to_fids[from_node] - {fid})
        e_deg = len(node_to_fids[to_node] - {fid})

        start_connected = s_deg > 0
        end_connected = e_deg > 0

        is_normal = start_connected and end_connected
        is_isolated = not start_connected and not end_connected
        is_one_end_dangle = (
            (start_connected and not end_connected)
            or (not start_connected and end_connected)
        )

        rm_flag = 0
        rm_code = "KEEP"

        if is_normal:
            normal_count += 1
            rm_code = "NORMAL_KEEP"

        elif is_isolated:
            if length <= isolated_len:
                rm_flag = 1
                rm_code = "SHORT_ISOLATED"
                short_isolated_count += 1
            else:
                rm_flag = 0
                rm_code = "LONG_ISO_KEEP"
                long_isolated_count += 1

        elif is_one_end_dangle:
            if length <= short_len:
                rm_flag = 1
                rm_code = "SHORT_DANGLE"
                short_dangle_count += 1
            else:
                rm_flag = 0
                rm_code = "LONG_DANGLE_KEEP"
                long_dangle_count += 1

        else:
            # 理论上不会进入，但保留兜底
            rm_flag = 0
            rm_code = "UNKNOWN_KEEP"

        if rm_flag == 1:
            copy_feature(
                rec["feature"],
                removed_layer,
                rec["geometry"],
                length,
                s_deg,
                e_deg,
                rm_flag,
                rm_code
            )
            removed_count += 1
        else:
            copy_feature(
                rec["feature"],
                kept_layer,
                rec["geometry"],
                length,
                s_deg,
                e_deg,
                rm_flag,
                rm_code
            )
            kept_count += 1

    kept_ds = None
    removed_ds = None
    ds = None

    print("\n========== LinePruner-NodeTopo v2 Report ==========")
    print(f"Input features          : {len(records)}")
    print(f"Kept features           : {kept_count}")
    print(f"Removed features        : {removed_count}")
    print("--------------------------------------------------")
    print(f"Normal connected kept   : {normal_count}")
    print(f"Short dangle removed    : {short_dangle_count}")
    print(f"Long dangle kept        : {long_dangle_count}")
    print(f"Short isolated removed  : {short_isolated_count}")
    print(f"Long isolated kept      : {long_isolated_count}")
    print("--------------------------------------------------")
    print(f"Short dangle threshold  : {short_len} m")
    print(f"Short isolated threshold: {isolated_len} m")
    print(f"Kept output             : {kept_shp}")
    print(f"Removed output          : {removed_shp}")
    print("==================================================\n")


if __name__ == "__main__":
    linepruner_node_topology_v2(
        input_shp=r"D:\Dataproces\10m\rep\mountains_westcanada_reprojected\shp\USGS_13_n61w154_keep.shp",
        kept_shp=r"D:\Dataproces\10m\rep\mountains_westcanada_reprojected\shp\USGS_13_n61w154_kk.shp",
        removed_shp=r"D:\Dataproces\10m\rep\mountains_westcanada_reprojected\shp\USGS_13_n61w154_kkremoved.shp",
        short_len=300.0,
        isolated_len=300.0,
        from_field="from_node",
        to_field="to_node"
    )
