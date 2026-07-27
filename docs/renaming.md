# 文件重命名对照表

所有 Python 文件统一使用小写 `snake_case`，名称优先表达“动作 + 对象”。目录按主要用途划分；此变更只调整组织和命名，两个注明的栅格工具同时完成了参数化修正。

| 原文件 | 新文件 |
|---|---|
| `1to2.py` | `tools/raster/resample_geotiff.py` |
| `band_mix.py` | `tools/raster/create_terrain_composite.py` |
| `batch_reproject_dem_gdal.py` | `tools/raster/reproject_dem_batch.py` |
| `dem_to_rgb.py` | `tools/raster/dem_to_rgb.py` |
| `dem_to_rgb_with_mapper.py` | `tools/raster/dem_to_rgb_with_metadata.py` |
| `fix_nodata_batch.py` | `tools/raster/fix_raster_nodata_batch.py` |
| `merge_tif.py` | `tools/raster/flatten_raster_files.py` |
| `tile_tif.py` | `tools/raster/tile_geotiff.py` |
| `RV_seprate` | `tools/raster/remap_raster_values.py` |
| `cal_lenth.py` | `tools/vector/calculate_line_length.py` |
| `checkname_shp_tif.py` | `tools/vector/check_shapefile_raster_pairs.py` |
| `checkshp.py` | `tools/vector/check_shapefile_schemas.py` |
| `create_empty_shp_from_dsm.py` | `tools/vector/create_empty_shapefile_from_dsm.py` |
| `deletebyname.py` | `tools/vector/delete_files_by_name.py` |
| `dissolve_lines.py` | `tools/vector/dissolve_intersecting_lines.py` |
| `esri_mask2shp.py` | `tools/vector/raster_mask_to_shapefile_arcpy.py` |
| `ogr_filedpruner.py` | `tools/vector/prune_shapefile_fields.py` |
| `ogr_LinePruner.py` | `tools/vector/prune_dangling_lines.py` |
| `ogr_Looppruner.py` | `tools/vector/prune_line_loops.py` |
| `check_dataset.py` | `tools/ml/validate_yolo_dataset.py` |
| `checkname.py` | `tools/ml/sync_yolo_images_labels.py` |
| `create_pseudo_color_for_yolo.py` | `tools/ml/create_yolo_terrain_composite.py` |
| `Mask2yolo.py` | `tools/ml/raster_mask_to_yolo.py` |
| `onnx_test.py` | `tools/ml/run_onnx_segmentation.py` |
| `rename_png_suffix.py` | `tools/ml/rename_png_suffixes.py` |
| `shp2yolo.py` | `tools/ml/shapefile_to_yolo.py` |
| `split_train_val.py` | `tools/ml/split_yolo_train_val.py` |
| `validate_yolo_tif_tiles.py` | `tools/ml/infer_yolo_geotiff_tiles.py` |
| `yolo2coco.py` | `tools/ml/yolo_to_coco.py` |
| `find_ept.py` | `tools/lidar/find_usgs_ept.py` |
| `generate_dsm.py` | `tools/lidar/generate_dsm_from_ept.py` |
| `excel_sort.py` | `tools/misc/sort_excel_rows.py` |
| `wechat_image_extractor.py` | `tools/misc/extract_wechat_images.py` |

`resample_geotiff.py` 已从绑定本机路径的一次性脚本改为 CLI。`flatten_raster_files.py` 修正了原脚本默认查找 `.txt` 而非 `.tif` 的问题，并允许通过 `--extension` 处理其他类型。
