# DSM → YOLO 完整工作流方案（历史草案）

> 状态：此文档记录早期设计思路，包含尚未纳入仓库的计划模块，不能作为当前工具清单。当前可用脚本及名称以根目录 `README.md` 为准。

## 一、核心问题分析

### 你的需求
```
DSM 原始数据
    ↓
转换为 RGB 图像（以便用于 YOLO 训练）
    ↓
YOLO 模型训练（检测 ridge/valley）
    ↓
在新 DSM 上进行推理
    ↓
YOLO 输出检测框（RGB 像素坐标）
    ↓
???????? 如何映射回原始 DSM 坐标系 ????????
    ↓
生成与 DSM 完全对应的 mask
```

### 关键难点
1. **坐标映射丢失**：RGB 图像转为 PNG 后，地理参考信息被丢弃
2. **像素坐标 ↔ 地理坐标转换**：YOLO 输出的是像素坐标，需要转回地理坐标
3. **结果验证**：最终的 mask 是否与原始 DSM 真正配准

---

## 二、我的解决方案（3 个核心模块）

### 模块 1：坐标映射工具 (`coordinate_mapper.py`)

**目的**：建立像素坐标 ↔ 地理坐标的双向转换

**关键概念**：GDAL Geotransform
```
Geotransform = (x0, px_width, 0, y0, 0, py_height)
    x0, y0        : 左上角地理坐标
    px_width      : 每个像素的宽度（通常为正，如 10m）
    py_height     : 每个像素的高度（通常为负，如 -10m）

转换公式：
    x_geo = x0 + col * px_width
    y_geo = y0 + row * py_height
```

**关键方法**：
- `pixel_to_geo(pixel_coord)` - 像素坐标转地理坐标
- `geo_to_pixel(geo_coord)` - 地理坐标转像素坐标
- `get_geo_extent()` - 获取图像的地理范围

**输入**：从 GeoTIFF 读取的 Geotransform + 投影信息

---

### 模块 2：DEM 转 RGB + 保存坐标信息 (`tools/raster/dem_to_rgb_with_metadata.py`)

**改进点**：原来的 `dem_to_rgb.py` 只输出 PNG，这里自动保存坐标信息

**处理流程**：
```
DEM (GeoTIFF)
    ↓
读取 Geotransform + 投影信息
    ↓
转换为 RGB 图像（原有逻辑）
输出 RGB.png + RGB_mapper.json ← 新增！
    ↓
mapper.json 包含：
{
    "geotransform": [x0, px_width, 0, y0, 0, py_height],
    "projection": "EPSG:32629",  // 或其他投影
    "image_shape": [height, width],
    "geo_extent": {"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ...}
}
```

**关键特点**：
- 输出尺寸 = 原始 DEM 尺寸（像素数不变）
- 坐标映射完全保留
- mapper.json 文件很小（几 KB）

---

### 模块 3：YOLO 结果转 DSM Mask (`yolo_to_dsm_mask.py`)

**工作流**：
```
YOLO 检测结果 (txt)
    ↓
读取 YOLO 输出格式：
    class_id x_center_norm y_center_norm w_norm h_norm
    （坐标已归一化到 [0, 1]）
    ↓
使用 mapper.json 进行坐标转换
    像素坐标 → 地理坐标
    ↓
输出多种格式：
    ├─ GeoJSON      （地理坐标，可用 QGIS 打开）
    ├─ PNG mask     （像素 mask，便于查看）
    ├─ GeoTIFF mask （与原 DEM 完全配准！）
    └─ JSON         （原始数据，含所有信息）
```

---

## 三、完整工作流程（按时间顺序）

### 阶段 A：数据准备与模型训练

```
Step A1: 准备原始 DSM 数据
├─ 输入：多个 DEM TIF 文件
└─ 要求：需要包含地理参考（带 Geotransform）

Step A2: 转换 DEM → RGB（保存坐标信息）
├─ 命令：python tools/raster/dem_to_rgb_with_metadata.py --batch
├─ 输入：DEM TIF 文件
├─ 输出：
│   ├─ image_001.png
│   ├─ image_001_mapper.json  ← 关键！
│   ├─ image_002.png
│   ├─ image_002_mapper.json
│   └─ ...
└─ 说明：mapper.json 保存了每个 RGB 图像的坐标映射

Step A3: 制作 YOLO 训练数据集
├─ 在生成的 RGB 图像上进行标注
├─ 工具：使用 Roboflow、LabelImg 或其他标注工具
└─ 输出：标注的 txt 文件（YOLO 格式）

Step A4: 训练 YOLO 模型
├─ 输入：标注好的 RGB 图像 + 对应的 txt 文件
├─ 工具：YOLOv8 官方库
└─ 输出：训练好的 .pt 模型文件
```

### 阶段 B：新数据推理与结果转换

```
Step B1: 对新 DSM 进行转换 + 推理
├─ 输入：新的 DEM 文件（demo_new.tif）
├─ 命令 1：python tools/raster/dem_to_rgb_with_metadata.py demo_new.tif demo_new_rgb.png
│         输出：demo_new_rgb.png + demo_new_rgb_mapper.json
├─ 命令 2：yolo predict model=trained.pt source=demo_new_rgb.png save_txt=True
│         输出：检测结果 txt 文件（在 runs/detect/predict/labels/ 中）
└─ 说明：YOLO 输出的 txt 中坐标是像素的归一化值

Step B2: 将 YOLO 结果转换回 DSM 坐标
├─ 命令：python yolo_to_dsm_mask.py \
│          --yolo-dir ./runs/detect/predict/labels/ \
│          --mapper demo_new_rgb_mapper.json \
│          --dem demo_new.tif \
│          --output ./final_masks \
│          --formats geojson,pixel_mask,geotiff_mask,json \
│          --class-names ridge,valley
├─ 输出：
│   ├─ demo_new_rgb.geojson        （地理坐标检测框）
│   ├─ demo_new_rgb_mask.png       （可视化）
│   ├─ demo_new_rgb_mask.tif       ← 最终结果！（与 DEM 完全配准）
│   └─ demo_new_rgb_results.json   （详细数据）
└─ 说明：GeoTIFF mask 可以直接在 GIS 中与原 DEM 叠加

Step B3: 验证结果
├─ 在 QGIS 中打开：
│   ├─ 原始 DEM
│   ├─ 生成的 RGB 图像（可选）
│   └─ 最终的 mask.tif
├─ 检查内容：
│   ├─ 坐标是否对齐
│   ├─ 检测结果是否合理
│   └─ mask 栅格值是否正确
└─ 说明：如果对齐，说明方案成功
```

---

## 四、技术细节说明

### 坐标映射的工作原理

```
原始 DEM (GeoTIFF)
├─ 尺寸：1000 × 1000 像素
├─ Geotransform：(500000, 10, 0, 2000000, 0, -10)
│                 x0=500000 (左上角 x)
│                 px_width=10 (每像素 10m)
│                 y0=2000000 (左上角 y)
│                 py_height=-10 (每像素 -10m，y 向下)
└─ 投影：EPSG:32629 (UTM Zone 29N)

↓ (DEM → RGB 转换)

RGB 图像 (PNG)
├─ 尺寸：1000 × 1000 像素（完全相同！）
└─ 内容：颜色 + 阴影（只有渲染效果改变，尺寸不变）

↓ (保存 mapper.json)

mapper.json
{
    "geotransform": [500000, 10, 0, 2000000, 0, -10],
    "projection": "EPSG:32629",
    "image_shape": [1000, 1000],
    "geo_extent": {
        "x_min": 500000,
        "y_min": 1990000,
        "x_max": 510000,
        "y_max": 2000000
    }
}

↓ (YOLO 推理)

YOLO 检测结果（txt）
class_id x_center_norm y_center_norm w_norm h_norm
0        0.5           0.3           0.1   0.2
         ↑             ↑
    像素 col=500      像素 row=300

↓ (使用 mapper 转换)

地理坐标
x = 500000 + 500 * 10 = 505000
y = 2000000 + 300 * (-10) = 1997000

↓

最终 mask.tif
├─ 栅格值：1=ridge, 2=valley, 0=background
├─ Geotransform：与原 DEM 完全相同！
└─ 投影：与原 DEM 完全相同！

结果：可直接在 GIS 中与原 DEM 叠加！
```

### 为什么要保存 mapper.json？

| 需求 | 方案 | 优缺点 |
|------|------|--------|
| 保留地理参考 | RGB 输出为 GeoTIFF | ✓ 完整 ✗ 文件大（MB 级） |
| 保留地理参考 | RGB 输出为 PNG + mapper.json | ✓ 轻量级 ✓ 灵活 ✗ 需额外管理 JSON |
| 不保留地理参考 | RGB 输出为 PNG | ✓ 最小化 ✗ 无法回映 |

**选择**：第二种方案最优
- 节省空间：PNG (几 MB) + JSON (几 KB) < GeoTIFF (十几 MB)
- 灵活性：可用任何格式的 RGB（JPEG、PNG 等）
- 可维护性：JSON 易于查看和修改

---

## 五、文件对应关系

### 文件清单

| 文件名 | 用途 | 何时创建 |
|--------|------|---------|
| `coordinate_mapper.py` | 坐标映射工具库 | 已创建 ✓ |
| `tools/raster/dem_to_rgb_with_metadata.py` | DEM 转 RGB（自动保存 mapper）| 已创建 ✓ |
| `yolo_to_dsm_mask.py` | YOLO 结果转 DSM mask | 已创建 ✓ |
| `dem_to_rgb.py` | 原始 DEM 转 RGB（保留不动）| 已有 |
| `tools/ml/yolo_to_coco.py` | YOLO 转 COCO 格式（保留不动）| 已有 |

### 数据文件流转

```
训练阶段：
dem_001.tif → tools/raster/dem_to_rgb_with_metadata.py →
    ├─ image_001.png
    ├─ image_001_mapper.json
    └─ (已标注) image_001.txt
                        ↓
              YOLO 训练 → model.pt

推理阶段：
new_dem.tif → tools/raster/dem_to_rgb_with_metadata.py →
    ├─ new_rgb.png
    └─ new_rgb_mapper.json

new_rgb.png → YOLO 推理 → new_rgb.txt (检测结果)

new_rgb.txt ← yolo_to_dsm_mask.py ← new_rgb_mapper.json + new_dem.tif
    → final_mask.tif (最终结果，与 new_dem 配准)
```

---

## 六、边界情况处理

### 1. 坐标边界处理
```python
# YOLO 检测框可能超出图像边界
x_min = max(0, x_min)
x_max = min(width - 1, x_max)
y_min = max(0, y_min)
y_max = min(height - 1, y_max)

# 在 yolo_to_dsm_mask.py 中已处理
```

### 2. NoData 值处理
```python
# DEM 中可能有无效值
if dem == nodata_value:
    mask[valid_area] = 0  # 背景
# 在 dem_to_rgb.py 中已处理，mapper 继承该 mask
```

### 3. 投影不同的情况
```python
# 如果需要转换投影
# 可使用 GDAL 的 gdalwarp 或 osr 模块
# 当前假设所有数据使用相同投影
```

---

## 七、验证步骤

### 快速验证（局部测试）

```bash
# 1. 准备测试数据
ls dem_*.tif  # 应该有输入 DEM

# 2. 转换测试
python tools/raster/dem_to_rgb_with_metadata.py test_dem.tif test_rgb.png

# 3. 检查输出
ls test_rgb.*
# 应该看到：
# - test_rgb.png
# - test_rgb_mapper.json

# 4. 查看 mapper 内容
cat test_rgb_mapper.json

# 5. 验证坐标转换
python -c "
from coordinate_mapper import load_mapper_metadata
mapper = load_mapper_metadata('test_rgb_mapper.json')
extent = mapper.get_geo_extent()
print('地理范围:', extent)
# 应该输出: {'x_min': ..., 'y_min': ..., 'x_max': ..., 'y_max': ...}
"
```

### 完整验证（端到端）

```bash
# 在 QGIS 中
1. 打开原始 DEM
2. 打开生成的 test_rgb.png 为栅格图层
   → 应该能看到 RGB 图像与 DEM 完全对齐
3. 如果有检测结果，打开 mask.tif
   → 应该与 DEM 像素完全对齐
```

---

## 八、下一步建议

### 立即行动（需要你确认）

1. **确认工作流程**：
   - [ ] 阶段 A（训练）和阶段 B（推理）流程是否符合你的需求？
   - [ ] 输出格式（GeoJSON、PNG mask、GeoTIFF mask）是否都需要？

2. **确认参数配置**：
   - [ ] 你的 DEM 的投影是什么？（EPSG code）
   - [ ] 像素大小是多少？（1m、5m、10m？）
   - [ ] NoData 值是什么？

3. **准备测试数据**：
   - [ ] 准备 1-2 个小的测试 DEM 文件（用于验证工作流）
   - [ ] 运行 `tools/raster/dem_to_rgb_with_metadata.py` 生成测试数据

### 后续优化（可选）

- [ ] 添加 morphology 操作优化 mask（膨胀/腐蚀）
- [ ] 支持多波段数据
- [ ] 添加精度评估指标
- [ ] 性能优化（大文件处理）

---

## 九、总结

### 解决的核心问题
✓ DSM → RGB 坐标信息丢失 → 使用 mapper.json 保存
✓ YOLO 输出像素坐标 → 使用 CoordinateMapper 转换
✓ 最终 mask 与原 DSM 坐标对应 → 通过 GeoTIFF 保存配准信息

### 工作流程简图
```
DSM (GeoTIFF)
    ↓ tools/raster/dem_to_rgb_with_metadata.py
RGB (PNG) + mapper.json
    ↓ YOLO 训练 + 推理
检测结果 (txt)
    ↓ yolo_to_dsm_mask.py
最终 Mask (GeoTIFF) ← 与原 DSM 完全配准！
```

### 关键文件
- `coordinate_mapper.py` - 坐标转换的数学基础
- `tools/raster/dem_to_rgb_with_metadata.py` - 保证坐标信息不丢失
- `yolo_to_dsm_mask.py` - 完成最后的回映

---

**现在，请告诉我：**
1. 这个工作流程思路是否正确？
2. 有没有需要调整的地方？
3. 是否需要修改某些步骤或输出格式？
