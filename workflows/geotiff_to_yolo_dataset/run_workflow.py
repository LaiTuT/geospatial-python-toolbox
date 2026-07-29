"""Build a tiled YOLO segmentation dataset from GeoTIFF and SHP/mask labels."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any


RASTER_EXTENSIONS = {".tif", ".tiff"}


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    base_dir = path.resolve().parent
    for key in ("raster_dir", "annotation_dir", "output_dir"):
        value = Path(config[key]).expanduser()
        config[key] = value if value.is_absolute() else (base_dir / value).resolve()
    return config


def list_files(folder: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def strip_suffixes(stem: str, suffixes: list[str]) -> str:
    result = stem
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if suffix and result.lower().endswith(suffix.lower()):
                result = result[: -len(suffix)]
                changed = True
                break
    return result


def unique_index(paths: list[Path], suffixes: list[str] | None = None) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in paths:
        key = strip_suffixes(path.stem, suffixes or [])
        if key in index:
            raise ValueError(f"名称归一化后发生冲突: {index[key]} 和 {path}")
        index[key] = path
    return index


def compare_names(left: dict[str, Path], right: dict[str, Path]) -> dict[str, Any]:
    left_names = set(left)
    right_names = set(right)
    return {
        "left_count": len(left_names),
        "right_count": len(right_names),
        "matched_count": len(left_names & right_names),
        "missing_on_right": sorted(left_names - right_names),
        "missing_on_left": sorted(right_names - left_names),
    }


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"输出目录非空: {output_dir}。确认可覆盖后在配置中设置 overwrite=true。"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def validate_paths(raster_dir: Path, annotation_dir: Path, output_dir: Path) -> None:
    if not raster_dir.is_dir():
        raise NotADirectoryError(f"源 TIF 目录不存在: {raster_dir}")
    if not annotation_dir.is_dir():
        raise NotADirectoryError(f"标注目录不存在: {annotation_dir}")

    output_resolved = output_dir.resolve()
    for source in (raster_dir.resolve(), annotation_dir.resolve()):
        if (
            output_resolved == source
            or source in output_resolved.parents
            or output_resolved in source.parents
        ):
            raise ValueError("输出目录与源数据目录不能互相包含。")


def tile_rasters(
    raster_index: dict[str, Path],
    tiles_dir: Path,
    config: dict[str, Any],
) -> int:
    from tile_geotiff import tile_single_tif

    total = 0
    for stem, raster_path in raster_index.items():
        total += tile_single_tif(
            str(raster_path),
            str(tiles_dir / stem),
            tile_size=int(config.get("tile_size", 1024)),
            overlap=int(config.get("overlap", 0)),
            nodata=config.get("nodata"),
            compress=str(config.get("compression", "LZW")),
            bigtiff="IF_SAFER",
            only_valid=bool(config.get("only_valid_tiles", False)),
        )
    return total


def create_labels_from_shapefiles(
    raster_index: dict[str, Path],
    annotation_index: dict[str, Path],
    tiles_dir: Path,
    labels_dir: Path,
    debug_masks_dir: Path | None,
    previews_dir: Path | None,
    config: dict[str, Any],
) -> None:
    from shapefile_to_yolo import batch_convert_from_tif_folder, batch_visualize_yolo_txt_as_mask

    shp_config = config.get("shapefile", {})
    class_values = {
        int(class_id): int(mask_value)
        for class_id, mask_value in shp_config.get(
            "class_mask_values", {"0": 255, "1": 128}
        ).items()
    }

    for stem in raster_index:
        tile_subdir = tiles_dir / stem
        label_subdir = labels_dir / stem
        mask_subdir = debug_masks_dir / stem if debug_masks_dir else None
        batch_convert_from_tif_folder(
            shp_path=str(annotation_index[stem]),
            field=str(shp_config.get("class_field", "grid_code")),
            tif_dir=str(tile_subdir),
            labels_dir=str(label_subdir),
            recursive=True,
            buffer_pixels=float(shp_config.get("buffer_pixels", 1.25)),
            simplify_tol=float(shp_config.get("simplify_tolerance", 0.0)),
            keep_subdirs=False,
            enable_buffer=True,
            masks_dir=str(mask_subdir) if mask_subdir else None,
            mask_ext=".png",
            min_area=float(config.get("minimum_area", 0.0)),
            mask_simplify_factor=float(config.get("simplify_factor", 0.0)),
            mask_class_values=class_values,
        )
        if previews_dir:
            batch_visualize_yolo_txt_as_mask(
                tif_dir=str(tile_subdir),
                labels_dir=str(label_subdir),
                out_vis_dir=str(previews_dir / stem),
                recursive=True,
                keep_subdirs=False,
            )


def create_labels_from_masks(
    raster_index: dict[str, Path],
    annotation_index: dict[str, Path],
    tiles_dir: Path,
    labels_dir: Path,
    debug_masks_dir: Path | None,
    config: dict[str, Any],
) -> None:
    from raster_mask_to_yolo import batch_geo_mask_to_yolo_single_source

    mask_config = config.get("mask", {})
    class_mapping = {
        int(pixel_value): int(class_id)
        for pixel_value, class_id in mask_config.get(
            "class_mapping", {"255": 0, "128": 1}
        ).items()
    }

    for stem in raster_index:
        batch_geo_mask_to_yolo_single_source(
            tif_dir=str(tiles_dir / stem),
            large_mask_path=str(annotation_index[stem]),
            out_dir=str(labels_dir / stem),
            class_mapping=class_mapping,
            debug_mask_dir=str(debug_masks_dir / stem) if debug_masks_dir else None,
            recursive=True,
            min_area=float(config.get("minimum_area", 100.0)),
            simplify_factor=float(config.get("simplify_factor", 0.001)),
            single_pixel_width=float(mask_config.get("single_pixel_width", 2.5)),
            debug_mode=bool(config.get("verbose", False)),
        )


def create_rgb_images(tiles_dir: Path, images_dir: Path) -> int:
    from create_yolo_terrain_composite import method_topo, read_raster, save_image

    completed = 0
    for tile_path in sorted(tiles_dir.rglob("*.tif")):
        relative = tile_path.relative_to(tiles_dir)
        output_path = images_dir / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raster = read_raster(str(tile_path))
        composite = method_topo({"dsm": raster})
        save_image(composite, str(output_path), raster["ds"])
        raster["ds"] = None
        completed += 1
    return completed


def recursive_stem_index(
    folder: Path, extensions: set[str] | None = None
) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or (extensions and path.suffix.lower() not in extensions):
            continue
        if path.stem in index:
            raise ValueError(f"瓦片主名重复，无法汇总数据集: {index[path.stem]} 和 {path}")
        index[path.stem] = path
    return index


def validate_yolo_labels(labels: dict[str, Path]) -> list[str]:
    issues: list[str] = []
    for stem, path in labels.items():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 7 or (len(parts) - 1) % 2:
                issues.append(f"{stem}.txt:{line_number}: 分割点数量无效")
                continue
            try:
                class_value = float(parts[0])
                coordinates = [float(value) for value in parts[1:]]
            except ValueError:
                issues.append(f"{stem}.txt:{line_number}: 包含非数值内容")
                continue
            if not class_value.is_integer():
                issues.append(f"{stem}.txt:{line_number}: 类别 ID 不是整数")
            if any(value < 0.0 or value > 1.0 for value in coordinates):
                issues.append(f"{stem}.txt:{line_number}: 坐标超出 [0, 1]")
    return issues


def assemble_dataset(
    images: dict[str, Path],
    labels: dict[str, Path],
    dataset_dir: Path,
    val_ratio: float,
    seed: int,
) -> tuple[int, int]:
    matched = sorted(set(images) & set(labels))
    if not matched:
        raise RuntimeError("没有可汇总的 RGB/标签配对。")
    if val_ratio < 0 or val_ratio >= 1:
        raise ValueError("validation_ratio 必须在 [0, 1) 范围内。")

    shuffled = matched[:]
    random.Random(seed).shuffle(shuffled)
    val_count = round(len(shuffled) * val_ratio)
    val_names = set(shuffled[:val_count])

    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for stem in matched:
        split = "val" if stem in val_names else "train"
        image_target = dataset_dir / "images" / split / images[stem].name
        label_target = dataset_dir / "labels" / split / labels[stem].name
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(images[stem], image_target)
        shutil.copy2(labels[stem], label_target)

    return len(matched) - val_count, val_count


def write_dataset_yaml(dataset_dir: Path, class_names: list[str]) -> None:
    # JSON is valid YAML 1.2 and avoids introducing another runtime dependency.
    content = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    (dataset_dir / "dataset.yaml").write_text(
        json.dumps(content, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    raster_dir: Path = config["raster_dir"]
    annotation_dir: Path = config["annotation_dir"]
    output_dir: Path = config["output_dir"]
    annotation_type = str(config.get("annotation_type", "shp")).lower()
    if annotation_type not in {"shp", "mask"}:
        raise ValueError("annotation_type 只能是 'shp' 或 'mask'。")

    validate_paths(raster_dir, annotation_dir, output_dir)
    raster_files = list_files(raster_dir, RASTER_EXTENSIONS)
    annotation_extensions = {".shp"} if annotation_type == "shp" else RASTER_EXTENSIONS
    annotation_files = list_files(annotation_dir, annotation_extensions)
    suffixes = list(
        config.get("annotation_name_suffixes", ["_buffer", "_cleaned", "_mask"])
    )
    raster_index = unique_index(raster_files)
    annotation_index = unique_index(annotation_files, suffixes)
    source_pairing = compare_names(raster_index, annotation_index)

    if not raster_index:
        raise FileNotFoundError(f"源目录中没有 GeoTIFF: {raster_dir}")
    if source_pairing["missing_on_right"]:
        raise ValueError(
            "以下源 TIF 没有同名标注: " + ", ".join(source_pairing["missing_on_right"])
        )

    prepare_output(output_dir, bool(config.get("overwrite", False)))
    tiles_dir = output_dir / "01_tiles"
    debug_masks_dir = (
        output_dir / "02_tile_masks" if config.get("write_debug_masks", True) else None
    )
    labels_dir = output_dir / "03_labels"
    previews_dir = (
        output_dir / "04_label_previews"
        if config.get("write_label_previews", True) and annotation_type == "shp"
        else None
    )
    images_dir = output_dir / "05_rgb_images"
    dataset_dir = output_dir / "dataset"
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] 切分源 GeoTIFF")
    tile_count = tile_rasters(raster_index, tiles_dir, config)

    print(f"\n[2/6] 按瓦片裁切 {annotation_type.upper()} 标注并生成 YOLO TXT")
    if annotation_type == "shp":
        create_labels_from_shapefiles(
            raster_index,
            annotation_index,
            tiles_dir,
            labels_dir,
            debug_masks_dir,
            previews_dir,
            config,
        )
    else:
        create_labels_from_masks(
            raster_index,
            annotation_index,
            tiles_dir,
            labels_dir,
            debug_masks_dir,
            config,
        )

    print("\n[3/6] 由瓦片生成三通道地形 RGB")
    image_count = create_rgb_images(tiles_dir, images_dir)

    print("\n[4/6] 对照 RGB 与 YOLO 标签名称和数量")
    image_index = recursive_stem_index(images_dir, RASTER_EXTENSIONS)
    label_index = recursive_stem_index(labels_dir, {".txt"})
    tile_pairing = compare_names(image_index, label_index)
    label_issues = validate_yolo_labels(label_index)
    if tile_pairing["missing_on_right"] or tile_pairing["missing_on_left"] or label_issues:
        report = {
            "source_pairing": source_pairing,
            "tile_pairing": tile_pairing,
            "label_issues": label_issues,
        }
        report_path = reports_dir / "workflow_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"RGB/标签检查失败，详见: {report_path}")

    print("\n[5/6] 汇总并拆分 train/val")
    train_count, val_count = assemble_dataset(
        image_index,
        label_index,
        dataset_dir,
        float(config.get("validation_ratio", 0.2)),
        int(config.get("random_seed", 42)),
    )
    write_dataset_yaml(dataset_dir, list(config.get("class_names", ["ridge", "valley"])))

    report = {
        "annotation_type": annotation_type,
        "source_pairing": source_pairing,
        "tile_pairing": tile_pairing,
        "label_issues": label_issues,
        "counts": {
            "tiles": tile_count,
            "rgb_images": image_count,
            "labels": len(label_index),
            "train": train_count,
            "val": val_count,
        },
        "dataset_yaml": str(dataset_dir / "dataset.yaml"),
    }
    report_path = reports_dir / "workflow_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[6/6] 完成。训练配置: {dataset_dir / 'dataset.yaml'}")
    print(f"审计报告: {report_path}")
    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        help="工作流 JSON 配置文件；相对路径以配置文件所在目录为基准",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.config)
