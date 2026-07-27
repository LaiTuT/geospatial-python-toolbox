from osgeo import ogr, osr
import os

# 输入输出文件路径
input_shapefile = r"D:\Dataproces\TFM\396000\39600.shp"
output_shapefile = r"D:\Dataproces\TFM\396000\39600_merge.shp"

# 打开输入的 Shapefile
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(input_shapefile, 0)  # 0表示以只读方式打开
if not dataSource:
    print("打开Shapefile失败!")
    exit(1)

# 获取图层
layer = dataSource.GetLayer()

# 根据grid_code分为两类
grid_code_values = [255, 128]
groups = {255: [], 128: []}

for feature in layer:
    grid_code = feature.GetField("grid_code")
    if grid_code in grid_code_values:
        groups[grid_code].append(feature)

# 创建一个新的 Shapefile 用于存储结果
if os.path.exists(output_shapefile):
    driver.DeleteDataSource(output_shapefile)

outDataSource = driver.CreateDataSource(output_shapefile)
outLayer = outDataSource.CreateLayer("merged", geom_type=ogr.wkbLineString)

# 获取图层字段定义
layerDefn = layer.GetLayerDefn()

# 添加字段（与原shp文件保持一致）
for i in range(layerDefn.GetFieldCount()):
    fieldDefn = layerDefn.GetFieldDefn(i)
    outLayer.CreateField(fieldDefn)

# 功能：合并相交的线
def merge_intersecting_lines(features):
    merged_lines = []
    merged_attributes = []

    while features:
        base_feature = features.pop(0)
        base_geom = base_feature.GetGeometryRef()
        base_fid = base_feature.GetFID()

        # 获取字段定义并保留属性
        base_attrs = {}
        for i in range(base_feature.GetFieldCount()):
            field_defn = layerDefn.GetFieldDefn(i)
            field_name = field_defn.GetName()
            field_value = base_feature.GetField(i)
            base_attrs[field_name] = field_value

        # 找出所有相交的要素
        intersecting = [base_feature]
        to_remove = []
        for other_feature in features:
            other_geom = other_feature.GetGeometryRef()
            if base_geom.Intersects(other_geom):
                intersecting.append(other_feature)
                to_remove.append(other_feature)

        # 合并所有相交的要素
        merged_geom = base_geom
        for feature in intersecting[1:]:
            merged_geom = merged_geom.Union(feature.GetGeometryRef())

        # 保留FID最小的要素的属性
        min_fid_feature = min(intersecting, key=lambda f: f.GetFID())

        # 获取最小FID的要素的属性
        merged_attrs = {}
        for i in range(min_fid_feature.GetFieldCount()):
            field_defn = layerDefn.GetFieldDefn(i)
            field_name = field_defn.GetName()
            field_value = min_fid_feature.GetField(i)
            merged_attrs[field_name] = field_value

        # 创建合并后的新要素并添加到输出
        new_feature = ogr.Feature(outLayer.GetLayerDefn())
        new_feature.SetGeometry(merged_geom)
        for field_name, value in merged_attrs.items():
            new_feature.SetField(field_name, value)

        outLayer.CreateFeature(new_feature)

        # 移除已经处理的要素
        for feature in to_remove:
            features.remove(feature)

# 对每个组处理
for grid_code in groups:
    features_to_process = groups[grid_code]
    merge_intersecting_lines(features_to_process)

# 关闭文件
dataSource = None
outDataSource = None

print("处理完成，输出文件为:", output_shapefile)
