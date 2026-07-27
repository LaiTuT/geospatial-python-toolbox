"""Run the ridge/valley vector cleanup workflow from start to finish."""

from __future__ import annotations

import argparse
from pathlib import Path

def validate_input(
    input_shp: Path,
    from_field: str,
    to_field: str,
    allow_non_metric: bool,
) -> None:
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise RuntimeError(
            "需要 GDAL/OGR。建议运行: conda install -c conda-forge gdal"
        ) from exc

    if not input_shp.is_file():
        raise FileNotFoundError(f"输入 Shapefile 不存在: {input_shp}")

    dataset = gdal.OpenEx(str(input_shp), gdal.OF_VECTOR)
    if dataset is None:
        raise RuntimeError(f"GDAL 无法打开输入 Shapefile: {input_shp}")

    layer = dataset.GetLayer(0)
    definition = layer.GetLayerDefn()
    missing = [
        name
        for name in (from_field, to_field)
        if definition.GetFieldIndex(name) < 0
    ]
    if missing:
        raise ValueError(
            "输入缺少节点拓扑字段: "
            + ", ".join(missing)
            + "。这两个字段应由上游矢量生成流程提供。"
        )

    spatial_ref = layer.GetSpatialRef()
    if not allow_non_metric:
        if spatial_ref is None or not spatial_ref.IsProjected():
            raise ValueError("输入必须使用米制投影坐标系，不能直接使用经纬度坐标系。")
        metres_per_unit = spatial_ref.GetLinearUnits()
        if abs(metres_per_unit - 1.0) > 1e-9:
            unit_name = spatial_ref.GetLinearUnitsName() or "unknown"
            raise ValueError(
                f"输入坐标单位不是米（{unit_name}, 1 unit={metres_per_unit} m）。"
                "请先重投影，或明确使用 --allow-non-metric。"
            )

    dataset = None


def run_cleanup(
    input_shp: Path,
    output_dir: Path,
    short_length: float = 300.0,
    isolated_length: float = 300.0,
    from_field: str = "from_node",
    to_field: str = "to_node",
    keep_fields: list[str] | None = None,
    allow_non_metric: bool = False,
) -> Path:
    """Run two dangle passes around loop removal, then clean fields."""
    from prune_dangling_lines import linepruner_node_topology_v2
    from prune_line_loops import looppurger_node_topology
    from prune_shapefile_fields import fieldpruner

    validate_input(input_shp, from_field, to_field, allow_non_metric)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_shp.stem

    first_kept = output_dir / f"{stem}_01_dangle_kept.shp"
    first_removed = output_dir / f"{stem}_01_dangle_removed.shp"
    loop_kept = output_dir / f"{stem}_02_loop_kept.shp"
    loop_removed = output_dir / f"{stem}_02_loop_removed.shp"
    second_kept = output_dir / f"{stem}_03_dangle_kept.shp"
    second_removed = output_dir / f"{stem}_03_dangle_removed.shp"
    final_output = output_dir / f"{stem}_cleaned.shp"

    print("\n[1/4] 删除初始短悬挂线和短孤立线")
    linepruner_node_topology_v2(
        input_shp=str(input_shp),
        kept_shp=str(first_kept),
        removed_shp=str(first_removed),
        short_len=short_length,
        isolated_len=isolated_length,
        from_field=from_field,
        to_field=to_field,
    )

    print("\n[2/4] 删除自环和所有属于环的边")
    looppurger_node_topology(
        input_shp=str(first_kept),
        kept_shp=str(loop_kept),
        removed_shp=str(loop_removed),
        from_field=from_field,
        to_field=to_field,
    )

    print("\n[3/4] 再次删除删环后暴露出的短悬挂线")
    linepruner_node_topology_v2(
        input_shp=str(loop_kept),
        kept_shp=str(second_kept),
        removed_shp=str(second_removed),
        short_len=short_length,
        isolated_len=isolated_length,
        from_field=from_field,
        to_field=to_field,
    )

    print("\n[4/4] 删除中间诊断字段")
    fieldpruner(
        input_shp=str(second_kept),
        output_shp=str(final_output),
        keep_fields=keep_fields,
    )

    print(f"\n最终结果: {final_output}")
    return final_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_shp", type=Path, help="待清理的山脊/山谷线 Shapefile")
    parser.add_argument("output_dir", type=Path, help="各阶段结果的输出目录")
    parser.add_argument("--short-length", type=float, default=300.0, help="短悬挂线阈值，默认 300 米")
    parser.add_argument("--isolated-length", type=float, default=300.0, help="短孤立线阈值，默认 300 米")
    parser.add_argument("--from-field", default="from_node", help="起点节点字段名")
    parser.add_argument("--to-field", default="to_node", help="终点节点字段名")
    parser.add_argument(
        "--keep-fields",
        nargs="+",
        default=None,
        help="最终仅保留这些属性字段；省略则保留所有非诊断字段",
    )
    parser.add_argument(
        "--allow-non-metric",
        action="store_true",
        help="允许非米制坐标；此时长度阈值按图层坐标单位解释",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cleanup(
        input_shp=args.input_shp,
        output_dir=args.output_dir,
        short_length=args.short_length,
        isolated_length=args.isolated_length,
        from_field=args.from_field,
        to_field=args.to_field,
        keep_fields=args.keep_fields,
        allow_non_metric=args.allow_non_metric,
    )
