import argparse
from pathlib import Path


# 在这里修改你要追加的后缀
SUFFIX = ""

# 在这里设置要删除的内容(可选)
REMOVE_TEXT = "_topo"

# 模式: "append" 追加后缀, "remove" 只删除
MODE = "remove"

# 在这里修改要处理的目标文件夹路径
TARGET_FOLDER = r"D:\YOLOV8\Extract_RV\YOLO-RV\ultralytics\data\images\pscolor"


def build_new_name(
    file_path: Path,
    suffix: str,
    remove_text: str | None,
    only_remove: bool,
) -> str:
    stem = file_path.stem
    if remove_text:
        stem = stem.replace(remove_text, "")
    if only_remove:
        return f"{stem}{file_path.suffix}"
    return f"{stem}{suffix}{file_path.suffix}"


def rename_png_files(
    folder: Path,
    suffix: str,
    remove_text: str | None,
    only_remove: bool,
    recursive: bool,
    dry_run: bool,
) -> None:
    pattern = "**/*.tif" if recursive else "*.tif"
    png_files = sorted(folder.glob(pattern))

    if not png_files:
        print(f"未找到 TIF 文件: {folder}")
        return

    print(f"扫描到 {len(png_files)} 个 TIF 文件")
    renamed = 0
    skipped = 0

    for file_path in png_files:
        new_name = build_new_name(file_path, suffix, remove_text, only_remove)
        new_path = file_path.with_name(new_name)

        if new_path == file_path:
            skipped += 1
            print(f"跳过(名称未变化): {file_path.name}")
            continue

        if new_path.exists():
            skipped += 1
            print(f"跳过(目标已存在): {new_path}")
            continue

        if dry_run:
            print(f"预览: {file_path.name} -> {new_path.name}")
            renamed += 1
            continue

        file_path.rename(new_path)
        print(f"已重命名: {file_path.name} -> {new_path.name}")
        renamed += 1

    mode = "预览完成" if dry_run else "处理完成"
    print(f"{mode}: 成功 {renamed}，跳过 {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="给文件夹中的 PNG 文件名追加后缀或删除指定内容")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归处理子文件夹中的 PNG",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览重命名结果，不实际修改",
    )
    parser.add_argument(
        "--remove",
        default="",
        help="删除文件名中指定内容(可选)",
    )
    parser.add_argument(
        "--only-remove",
        action="store_true",
        help="只删除，不追加后缀",
    )

    args = parser.parse_args()
    folder = Path(TARGET_FOLDER)

    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"目标文件夹不存在或不是目录: {folder}")

    use_only_remove = MODE.lower() == "remove"
    use_remove_text = REMOVE_TEXT.strip() or None

    if args.remove.strip():
        use_remove_text = args.remove.strip()
    if args.only_remove:
        use_only_remove = True

    rename_png_files(
        folder=folder,
        suffix=SUFFIX,
        remove_text=use_remove_text,
        only_remove=use_only_remove,
        recursive=args.recursive,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
