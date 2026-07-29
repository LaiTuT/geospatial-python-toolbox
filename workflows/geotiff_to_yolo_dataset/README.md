# GeoTIFF 到 YOLO 分割数据集工作流

这套流程把整幅 DSM/DEM GeoTIFF 与同名的 Shapefile 或 mask GeoTIFF 标注转换为可训练的 YOLO 分割数据集。

## 核对后的完整流程

1. 对照源 TIF 与 SHP/mask TIF 的主文件名和数量。
2. 将每张源 TIF 切成带地理参考的固定尺寸瓦片。
3. 按瓦片地理范围裁切对应标注：
   - SHP 分支：筛选相交要素、转换坐标、将线按像素宽度拓宽成面。
   - mask 分支：使用最近邻重投影，把整幅 mask 裁到每个瓦片网格。
4. 从逐瓦片 mask/面轮廓生成 YOLO segmentation TXT，空白瓦片也创建空 TXT。
5. 从原始 DSM/DEM 瓦片计算 `dx`、`dy`、曲率三个通道，生成同名三通道 GeoTIFF 训练图像。
6. 再次对照训练图像与 TXT 的名称和数量，并检查类别、点数和归一化坐标。
7. 复制配对数据到 `train/val` 目录，生成 Ultralytics 可读取的 `dataset.yaml`。

SHP 分支不会额外保存大量碎片 Shapefile。它在内存中按每个瓦片的地理范围裁切要素，并可输出逐瓦片 PNG mask；这与“先拆 SHP 再转标签”的结果等价，目录更简洁。

## 目录内容

| 文件 | 用途 |
|---|---|
| `run_workflow.py` | 推荐使用的统一入口 |
| `config.example.json` | 两种标注分支共用的配置模板 |
| `tile_geotiff.py` | GeoTIFF 切片原工具副本 |
| `shapefile_to_yolo.py` | SHP 裁切、栅格化与 YOLO 转换原工具副本 |
| `raster_mask_to_yolo.py` | mask TIF 裁切与 YOLO 转换原工具副本 |
| `create_yolo_terrain_composite.py` | 三通道地形图生成原工具副本 |
| `check_shapefile_raster_pairs.py` | 源文件配对检查参考工具 |
| `sync_yolo_images_labels.py` | 图像/标签同步检查参考工具，原入口可能删除文件 |
| `split_yolo_train_val.py` | 密度分层拆分参考工具 |
| `validate_yolo_dataset.py` | YOLO 数据检查参考工具 |

统一入口内置了非破坏性的名称核对、标签检查和随机拆分，不会调用参考工具中的删除入口。

## 输入目录

源 TIF 与标注按归一化后的主名一一对应：

```text
input/
  rasters/
    area_a.tif
    area_b.tif
  annotations/
    area_a_cleaned.shp
    area_a_cleaned.dbf
    area_a_cleaned.shx
    area_a_cleaned.prj
    area_b_cleaned.shp
    ...
```

默认会从标注名末尾反复移除 `_buffer`、`_cleaned`、`_mask`，因此 `area_a.tif` 可以匹配 `area_a_cleaned.shp` 或 `area_a_mask.tif`。可通过 `annotation_name_suffixes` 修改规则。

### SHP 标注要求

- `annotation_type` 设置为 `"shp"`。
- 图层必须有分类字段，默认是 `grid_code`。
- 当前原转换器固定使用 `255 → class 0`、`128 → class 1`。
- 线要素会按 `buffer_pixels` 向两侧拓宽；默认 1.25 像素，即约 2.5 像素总宽。
- SHP 与源 TIF 可以使用不同坐标系，脚本会进行坐标变换。

### Mask TIF 标注要求

- `annotation_type` 设置为 `"mask"`。
- mask 文件应有有效投影和地理变换。
- 默认像元映射为 `255 → class 0`、`128 → class 1`。
- mask 与源 TIF 可以使用不同网格，裁切时使用最近邻重投影。

## 使用方法

先修改 [config.example.json](config.example.json) 中的输入、标注和输出路径。相对路径以配置文件所在目录为基准。

```powershell
python workflows/geotiff_to_yolo_dataset/run_workflow.py workflows/geotiff_to_yolo_dataset/config.example.json
```

首次输出目录必须为空。需要重新生成时，将配置中的 `overwrite` 改为 `true`；此设置会删除并重建整个 `output_dir`，因此输出目录不能与源数据目录互相包含。

## 主要参数

| 参数 | 含义 |
|---|---|
| `annotation_type` | `shp` 或 `mask` |
| `tile_size` | 方形瓦片边长，默认 1024 像素 |
| `overlap` | 相邻瓦片重叠像素数 |
| `nodata` | 源 TIF 的 NoData 值 |
| `only_valid_tiles` | 是否跳过全 NoData 瓦片 |
| `minimum_area` | 小于该像素面积的轮廓不写入标签 |
| `simplify_factor` | 轮廓简化比例，0 表示不简化 |
| `write_debug_masks` | 是否保存逐瓦片标注 mask |
| `write_label_previews` | SHP 分支是否反渲染 TXT 供检查 |
| `validation_ratio` | 验证集比例，默认 0.2 |
| `class_names` | `dataset.yaml` 中的类别顺序 |

`tile_size` 必须小于或等于每张源栅格的宽、高。`overlap` 必须小于 `tile_size`。训练图像是 0～1 浮点三波段 GeoTIFF，不是普通 8 位 PNG。

## 输出结构

```text
output/
  01_tiles/             保留地理参考的原始瓦片
  02_tile_masks/        逐瓦片标注检查图，可关闭
  03_labels/            按源区域组织的 YOLO TXT
  04_label_previews/    SHP 分支的 TXT 反渲染图，可关闭
  05_rgb_images/        同名三通道地形训练图
  dataset/
    images/train/
    images/val/
    labels/train/
    labels/val/
    dataset.yaml
  reports/
    workflow_report.json
```

最终训练可使用：

```powershell
yolo segment train data=output/dataset/dataset.yaml model=yolo11n-seg.pt
```

## 审计与安全

- 源 TIF 没有同名标注时，流程会在切片前停止。
- RGB 与 TXT 数量或主名不一致时，不会生成训练集，并在报告中列出差异。
- 空 TXT 是合法的背景标签，不会被当作错误。
- 所有训练集文件均为复制，源文件和阶段文件不会被移动。
- `workflow_report.json` 记录源文件配对、瓦片配对、标签问题和 train/val 数量。

## 依赖

需要 Python 3.10+、GDAL/OGR、NumPy、OpenCV 和 Shapely。建议在 Conda 环境中安装 GDAL。
