from osgeo import gdal, ogr
from collections import defaultdict
import os
import sys


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

    # 复制原字段
    for i in range(src_defn.GetFieldCount()):
        out_layer.CreateField(src_defn.GetFieldDefn(i))

    # 新增诊断字段
    f_len = ogr.FieldDefn("len_m", ogr.OFTReal)
    f_len.SetWidth(18)
    f_len.SetPrecision(3)
    out_layer.CreateField(f_len)

    out_layer.CreateField(ogr.FieldDefn("loop_rm", ogr.OFTInteger))

    f_code = ogr.FieldDefn("rm_code", ogr.OFTString)
    f_code.SetWidth(32)
    out_layer.CreateField(f_code)

    return out_ds, out_layer


def copy_feature(src_feat, out_layer, geom, length, loop_rm, rm_code):
    out_defn = out_layer.GetLayerDefn()
    out_feat = ogr.Feature(out_defn)

    for i in range(src_feat.GetFieldCount()):
        out_feat.SetField(i, src_feat.GetField(i))

    out_feat.SetGeometry(geom.Clone())
    out_feat.SetField("len_m", float(length))
    out_feat.SetField("loop_rm", int(loop_rm))
    out_feat.SetField("rm_code", rm_code)

    out_layer.CreateFeature(out_feat)
    out_feat = None


def find_bridge_edges(records):
    """
    使用 Tarjan bridge detection。
    返回 bridge edge id 集合。

    注意：
    - 这是无向图算法。
    - 使用 edge_id，而不是只使用节点对，因此可以正确处理平行边。
    - self-loop 不可能是 bridge，后续会被作为环线删除。
    """

    adjacency = defaultdict(list)

    for rec in records:
        edge_id = rec["fid"]
        u = rec["from_node"]
        v = rec["to_node"]

        # 自环不加入桥边搜索；自环直接属于环
        if u == v:
            continue

        adjacency[u].append((v, edge_id))
        adjacency[v].append((u, edge_id))

    timer = 0
    tin = {}
    low = {}
    visited = set()
    bridges = set()

    sys.setrecursionlimit(max(1000000, len(records) * 4))

    def dfs(v, parent_edge_id=None):
        nonlocal timer

        visited.add(v)
        tin[v] = timer
        low[v] = timer
        timer += 1

        for to, edge_id in adjacency[v]:
            if edge_id == parent_edge_id:
                continue

            if to in visited:
                low[v] = min(low[v], tin[to])
            else:
                dfs(to, edge_id)
                low[v] = min(low[v], low[to])

                if low[to] > tin[v]:
                    bridges.add(edge_id)

    for node in adjacency.keys():
        if node not in visited:
            dfs(node)

    return bridges


def looppurger_node_topology(
    input_shp,
    kept_shp,
    removed_shp,
    from_field="from_node",
    to_field="to_node",
    delete_all_cycle_edges=True
):
    """
    删除所有成环线段。

    判定规则：
    - self-loop: from_node == to_node，删除
    - non-bridge edge: 属于至少一个环，删除
    - bridge edge: 不属于环，保留
    """

    if not delete_all_cycle_edges:
        raise ValueError("LoopPurger 的目标是删除全部环线，delete_all_cycle_edges 应保持 True。")

    ds = gdal.OpenEx(input_shp, gdal.OF_VECTOR)
    if ds is None:
        raise RuntimeError(f"无法打开输入文件: {input_shp}")

    src_layer = ds.GetLayer(0)
    geom_type = src_layer.GetGeomType()

    records = []

    for feat in src_layer:
        geom_ref = feat.GetGeometryRef()
        if geom_ref is None:
            continue

        from_node = feat.GetField(from_field)
        to_node = feat.GetField(to_field)

        if from_node is None or to_node is None:
            continue

        geom = geom_ref.Clone()
        length = geom.Length()

        records.append({
            "fid": feat.GetFID(),
            "feature": feat.Clone(),
            "geometry": geom,
            "from_node": from_node,
            "to_node": to_node,
            "length": length
        })

    bridge_fids = find_bridge_edges(records)

    removed_fids = set()
    self_loop_count = 0
    cycle_edge_count = 0

    for rec in records:
        fid = rec["fid"]
        u = rec["from_node"]
        v = rec["to_node"]

        if u == v:
            removed_fids.add(fid)
            self_loop_count += 1
            continue

        # 不是桥边，就说明它属于至少一个环
        if fid not in bridge_fids:
            removed_fids.add(fid)
            cycle_edge_count += 1

    driver = ogr.GetDriverByName("ESRI Shapefile")

    kept_ds, kept_layer = create_output_layer(
        driver,
        kept_shp,
        src_layer,
        geom_type
    )

    removed_ds, removed_layer = create_output_layer(
        driver,
        removed_shp,
        src_layer,
        geom_type
    )

    kept_count = 0
    removed_count = 0

    for rec in records:
        fid = rec["fid"]

        if fid in removed_fids:
            if rec["from_node"] == rec["to_node"]:
                code = "SELF_LOOP"
            else:
                code = "CYCLE_EDGE"

            copy_feature(
                rec["feature"],
                removed_layer,
                rec["geometry"],
                rec["length"],
                1,
                code
            )
            removed_count += 1
        else:
            copy_feature(
                rec["feature"],
                kept_layer,
                rec["geometry"],
                rec["length"],
                0,
                "BRIDGE_KEEP"
            )
            kept_count += 1

    kept_ds = None
    removed_ds = None
    ds = None

    print("\n========== LoopPurger-NodeTopo Report ==========")
    print(f"Input features       : {len(records)}")
    print(f"Kept bridge edges    : {kept_count}")
    print(f"Removed loop edges   : {removed_count}")
    print("-----------------------------------------------")
    print(f"Self-loop removed    : {self_loop_count}")
    print(f"Cycle edges removed  : {cycle_edge_count}")
    print("-----------------------------------------------")
    print(f"Kept output          : {kept_shp}")
    print(f"Removed output       : {removed_shp}")
    print("================================================\n")


if __name__ == "__main__":
    looppurger_node_topology(
        input_shp=r"D:\Dataproces\10m\rep\mountains_westcanada_reprojected\shp\USGS_13_n61w154_keep.shp",
        kept_shp=r"D:\Dataproces\10m\rep\mountains_westcanada_reprojected\shp\USGS_13_n61w154_loop_keep.shp",
        removed_shp=r"D:\Dataproces\10m\rep\mountains_westcanada_reprojected\shp\USGS_13_n61w154_loop_removed.shp",
        from_field="from_node",
        to_field="to_node"
    )
