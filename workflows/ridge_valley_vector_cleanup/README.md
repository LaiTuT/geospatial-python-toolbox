# 山脊山谷矢量线清理流程

此目录是山脊/山谷自动提取线的独立清理流程，包含原有三个 `prune` 工具的副本和统一入口 `run_cleanup.py`。

## 核对后的处理顺序

1. `prune_dangling_lines.py`：删除长度不超过阈值的短悬挂线和短孤立线。
2. `prune_line_loops.py`：使用 Tarjan 桥边检测删除自环及所有属于环的边。
3. `prune_dangling_lines.py`：再次清理删环后新暴露出的短悬挂线或短孤立线。
4. `prune_shapefile_fields.py`：删除前三步生成的 `LP_*`、`LOOP_*`、`len_m`、`rm_code` 等诊断字段。

因此，这里实际是“三个算法文件、四个处理阶段”。你记得的“先删除尾端小线，再删除环”是正确的；补充的第四步很可能是再次清理尾线，而不是另一个遗失的源文件。仓库中没有发现负责生成节点字段的第四个旧脚本。

## 输入要求

- 输入必须是线 Shapefile。
- 必须已有 `from_node` 和 `to_node` 字段；字段值代表每条线两端的拓扑节点 ID。
- 默认长度阈值按米解释，因此输入应使用米制投影坐标系。
- 节点字段必须来自同一套拓扑构建规则。仅靠端点坐标相同但节点 ID 不同，算法不会认为线条相连。
- 建议先修复无效几何并确认线已在交点处分段，否则拓扑判断可能与视觉连接关系不一致。

## 一次运行完整流程

```bash
python workflows/ridge_valley_vector_cleanup/run_cleanup.py input/ridge_valley.shp output/cleanup
```

指定阈值和最终保留字段：

```bash
python workflows/ridge_valley_vector_cleanup/run_cleanup.py input/ridge_valley.shp output/cleanup \
  --short-length 300 \
  --isolated-length 300 \
  --keep-fields arcid grid_code from_node to_node
```

Windows PowerShell 中可写成一行，或使用反引号续行。

## 输出文件

假设输入为 `ridge_valley.shp`，输出目录会包含：

| 文件 | 内容 |
|---|---|
| `ridge_valley_01_dangle_kept.shp` | 第一次尾线清理后的保留线 |
| `ridge_valley_01_dangle_removed.shp` | 第一次删除的短尾线/短孤立线 |
| `ridge_valley_02_loop_kept.shp` | 删环后的保留线 |
| `ridge_valley_02_loop_removed.shp` | 被判定为环的线 |
| `ridge_valley_03_dangle_kept.shp` | 第二次尾线清理后的保留线 |
| `ridge_valley_03_dangle_removed.shp` | 删环后暴露并被删除的短线 |
| `ridge_valley_cleaned.shp` | 删除诊断字段后的最终结果 |

Shapefile 实际还会同时生成 `.shx`、`.dbf`、`.prj`、`.cpg` 等配套文件。所有阶段结果都保留是为了在 QGIS/ArcGIS 中复核误删；确认阈值合适后再使用 `*_cleaned.shp`。

## 重要行为

- 再次使用同一个输出目录时，已有同名阶段文件会被覆盖。
- `prune_line_loops.py` 会删除环上的全部边，不是只断开环中的一条边。
- 当前尾线清理每次只按该次输入的拓扑度数判断。统一入口执行两次，但不会无限递归删除整条短支链。
- `--keep-fields` 会限制最终属性字段；不传该参数时，仅删除工作流的诊断字段。
- `--allow-non-metric` 只关闭单位保护，不会把米阈值自动转换为英尺或度。

## 依赖

建议通过 Conda 安装 GDAL：

```bash
conda install -c conda-forge gdal
```
