# Geospatial Python Tools

一组用于 GeoTIFF、Shapefile、DEM/DSM、LiDAR EPT 与 YOLO 地理数据工作流的独立 Python 工具。脚本按领域组织，文件名统一使用小写 `snake_case`，可按需单独使用。

> 当前版本完成了仓库与命名规范化。所有 Python/JSON 文件均通过静态语法检查，但部分历史脚本仍使用文件顶部的路径常量；运行前请先阅读对应脚本。涉及覆盖、删除或移动数据的工具应先在数据副本上验证。

## 目录结构

```text
tools/
  raster/   GeoTIFF、DEM/DSM 栅格处理
  vector/   Shapefile 与线要素处理
  ml/       YOLO/COCO/ONNX 数据与推理
  lidar/    USGS EPT 检索与 DSM 生成
  misc/     非 GIS 专用辅助工具
docs/       工作流草案与命名迁移表
scripts/    仓库维护脚本
workflows/  可直接串行运行的固定处理流程
```

## 安装

建议使用 Python 3.10+ 和 Conda 安装 GDAL，避免 Windows 下 `pip` 编译 GDAL 的常见问题：

```bash
conda create -n geospatial-tools python=3.11
conda activate geospatial-tools
conda install -c conda-forge gdal
pip install -r requirements.txt
```

只有 `raster_mask_to_shapefile_arcpy.py` 依赖 ArcPy，需在 ArcGIS Pro 自带的 Python 环境中运行。深度学习工具还需要 PyTorch、Ultralytics 或 ONNX Runtime，按具体脚本选择安装即可。

## 栅格工具

| 工具 | 作用 | 入口 |
|---|---|---|
| `resample_geotiff.py` | 批量按目标像元大小重采样 GeoTIFF | CLI |
| `create_terrain_composite.py` | 从 DEM 计算坡度和山体阴影并生成三波段地形合成图 | 修改脚本配置 |
| `reproject_dem_batch.py` | 批量重投影 DEM，可按中心经度自动选择 NAD83 UTM 分区 | 修改脚本配置 |
| `remap_raster_values.py` | 批量将栅格中的两个像元值映射为新的字节值 | CLI |
| `dem_to_rgb.py` | 将 DEM 渲染为带色彩和阴影的 RGB 图像 | CLI/JSON |
| `dem_to_rgb_with_metadata.py` | DEM 转 RGB，并输出用于坐标回映的元数据 JSON | CLI/JSON |
| `fix_raster_nodata_batch.py` | 调用 GDAL 工具批量修复 NoData 与旁车文件 | 修改脚本配置 |
| `flatten_raster_files.py` | 递归收集指定扩展名文件并平铺复制，自动处理重名 | CLI |
| `tile_geotiff.py` | 将大幅 GeoTIFF 切片并保持地理参考 | 修改脚本配置 |

示例：

```bash
python tools/raster/resample_geotiff.py data/input data/resampled --x-resolution 2 --y-resolution 2
python tools/raster/remap_raster_values.py data/masks 1 2 --first-target 128 --second-target 255
python tools/raster/dem_to_rgb.py input.tif output.png
python tools/raster/flatten_raster_files.py data/tree data/flat --extension .tif
```

## 矢量工具

| 工具 | 作用 | 入口 |
|---|---|---|
| `calculate_line_length.py` | 为线 Shapefile 创建并计算长度字段 | 修改脚本配置 |
| `check_shapefile_raster_pairs.py` | 按文件名检查 SHP 与 TIF 是否成对及组件是否完整 | 修改脚本配置 |
| `check_shapefile_schemas.py` | 对比目录中多个 Shapefile 的字段结构 | 函数/修改配置 |
| `create_empty_shapefile_from_dsm.py` | 继承 DSM 坐标系创建空 Shapefile 标注模板 | 修改脚本配置 |
| `delete_files_by_name.py` | 按名称清单批量删除文件 | 修改脚本配置，破坏性 |
| `dissolve_intersecting_lines.py` | 合并相交线要素并输出新的 Shapefile | 修改脚本配置 |
| `raster_mask_to_shapefile_arcpy.py` | 使用 ArcPy 将栅格掩膜批量转为 Shapefile | ArcGIS/修改配置 |
| `prune_shapefile_fields.py` | 清理 Shapefile 中的诊断或临时字段 | 函数/修改配置 |
| `prune_dangling_lines.py` | 按节点拓扑和长度移除短悬挂线 | 函数/修改配置 |
| `prune_line_loops.py` | 使用桥边检测识别并移除线网中的环 | 函数/修改配置 |

## 机器学习工具

| 工具 | 作用 | 入口 |
|---|---|---|
| `create_yolo_terrain_composite.py` | 将 DSM 生成适合 YOLO 训练的三通道地形影像 | CLI |
| `raster_mask_to_yolo.py` | 将地理栅格掩膜转为 YOLO 分割标签并可视化 | 修改脚本配置 |
| `shapefile_to_yolo.py` | 将 Shapefile 标注栅格化并转换为 YOLO 分割标签 | 修改脚本配置 |
| `validate_yolo_dataset.py` | 检查 YOLO 图像、标签、类别和坐标合法性 | 修改脚本配置 |
| `sync_yolo_images_labels.py` | 查找或清理 YOLO 图像与标签中的孤立文件 | 修改脚本配置，可能删除 |
| `split_yolo_train_val.py` | 随机或按标签密度拆分 YOLO 训练集和验证集 | 修改脚本配置，可能移动 |
| `rename_png_suffixes.py` | 批量统一 PNG 文件名后缀，支持预览 | CLI |
| `yolo_to_coco.py` | 将 YOLO 检测/分割标签转换为 COCO JSON | CLI |
| `run_onnx_segmentation.py` | 使用滑窗和 NMS 运行 ONNX 分割模型 | 修改脚本配置 |
| `infer_yolo_geotiff_tiles.py` | 对浮点 GeoTIFF 分块执行 Ultralytics YOLO 推理并合并掩膜 | CLI |

## LiDAR 与其他工具

| 工具 | 作用 | 入口 |
|---|---|---|
| `find_usgs_ept.py` | 根据 DEM 范围扫描并匹配 USGS 公开 EPT 数据集 | 修改脚本配置/网络 |
| `generate_dsm_from_ept.py` | 通过 EPT/PDAL 流程生成与目标 DEM 网格对齐的 DSM | 修改脚本配置 |
| `sort_excel_rows.py` | 按指定列对 Excel 表格排序 | 修改脚本配置 |
| `extract_wechat_images.py` | 提取微信公众号历史或单篇文章中的图片 | CLI/网络 |

## 固定工作流

山脊/山谷自动生成矢量线的完整清理流程已单独整理到 [`workflows/ridge_valley_vector_cleanup`](workflows/ridge_valley_vector_cleanup/README.md)。统一入口会按“短尾线 → 环 → 再次短尾线 → 字段清理”执行，并保留各阶段删除结果以便 GIS 复核。

## 仓库检查

无需安装 GDAL 即可执行基础检查：

```bash
python scripts/check_repository.py
```

旧文件名与新文件名的完整对应关系见 [docs/renaming.md](docs/renaming.md)。英文简介见 [README.en.md](README.en.md)。

## 许可证

本项目采用 [MIT License](LICENSE)。第三方软件（GDAL、ArcPy、PyTorch、Ultralytics 等）仍受各自许可证约束。
