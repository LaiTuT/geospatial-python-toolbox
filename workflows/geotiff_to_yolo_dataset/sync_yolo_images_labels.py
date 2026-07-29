from pathlib import Path
import os

# ====== 你需要改的路径 ======
TRAIN_IMAGES_DIR = r"D:\YOLOV8\Extract_RV\YOLO-RV\ultralytics\data\images\pscolor"
TRAIN_LABELS_DIR = r"D:\YOLOV8\Extract_RV\YOLO-RV\ultralytics\data\labels\train"

# 图片后缀（按你的实际补充/删减）
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]

# 是否输出报告文件
WRITE_REPORT = True
REPORT_PATH = r"D:\YOLOV8\Extract_RV\ultralytics\data\sync_report.txt"

# ====== 删除选项 ======
# 设置为 True 启用删除功能（谨慎使用！）
DELETE_ORPHAN_FILES = True

# 删除选项：可选择删除哪一侧
# "both" - 删除两侧不对应的文件
# "images" - 只删除没有对应标签的图片
# "labels" - 只删除没有对应图片的标签
DELETE_MODE = "labels"  # "both", "images", "labels"


def get_image_stems(images_dir: Path):
    """获取所有图片的文件名（不含扩展名）和完整路径的映射"""
    stem_to_paths = {}
    for ext in IMAGE_EXTS:
        for p in images_dir.glob(f"*{ext}"):
            stem_to_paths[p.stem] = p
    return stem_to_paths


def get_label_paths(labels_dir: Path):
    """获取所有标签文件的完整路径"""
    return {p.stem: p for p in labels_dir.glob("*.txt")}


def delete_file(file_path: Path, file_type: str):
    """安全删除文件"""
    try:
        os.remove(file_path)
        print(f"[Deleted] {file_type}: {file_path.name}")
        return True
    except Exception as e:
        print(f"[Error] Failed to delete {file_path}: {e}")
        return False


def main():
    images_dir = Path(TRAIN_IMAGES_DIR)
    labels_dir = Path(TRAIN_LABELS_DIR)

    if not images_dir.exists():
        raise FileNotFoundError(f"TRAIN_IMAGES_DIR not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"TRAIN_LABELS_DIR not found: {labels_dir}")

    img_stem_to_path = get_image_stems(images_dir)
    lbl_stem_to_path = get_label_paths(labels_dir)

    img_stems = set(img_stem_to_path.keys())
    lbl_stems = set(lbl_stem_to_path.keys())

    labels_no_image = sorted(lbl_stems - img_stems)   # label存在但找不到同名图片
    images_no_label = sorted(img_stems - lbl_stems)   # 图片存在但找不到同名label

    print(f"Images count: {len(img_stems)}")
    print(f"Labels count: {len(lbl_stems)}")
    print(f"Labels w/o images: {len(labels_no_image)}")
    print(f"Images w/o labels: {len(images_no_label)}")

    if labels_no_image:
        print("\n[Labels without matching images] (first 50)")
        for s in labels_no_image[:50]:
            print(s + ".txt")

    if images_no_label:
        print("\n[Images without matching labels] (first 50)")
        for s in images_no_label[:50]:
            print(s)

    # ====== 删除功能 ======
    if DELETE_ORPHAN_FILES and (labels_no_image or images_no_label):
        print("\n" + "=" * 50)
        print("开始删除不对应的文件...")
        print("=" * 50)

        deleted_images = 0
        deleted_labels = 0

        # 删除没有对应图片的标签
        if DELETE_MODE in ("both", "labels") and labels_no_image:
            print(f"\n正在删除 {len(labels_no_image)} 个孤立标签文件...")
            for stem in labels_no_image:
                if delete_file(lbl_stem_to_path[stem], "Label"):
                    deleted_labels += 1

        # 删除没有对应标签的图片
        if DELETE_MODE in ("both", "images") and images_no_label:
            print(f"\n正在删除 {len(images_no_label)} 个孤立图片文件...")
            for stem in images_no_label:
                if delete_file(img_stem_to_path[stem], "Image"):
                    deleted_images += 1

        print(f"\n删除完成：标签 {deleted_labels} 个，图片 {deleted_images} 个")

    elif DELETE_ORPHAN_FILES:
        print("\n所有文件都已配对，无需删除。")

    # 写报告
    if WRITE_REPORT:
        rp = Path(REPORT_PATH)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with rp.open("w", encoding="utf-8") as f:
            f.write(f"TRAIN_IMAGES_DIR={images_dir}\n")
            f.write(f"TRAIN_LABELS_DIR={labels_dir}\n")
            f.write(f"Images count: {len(img_stems)}\n")
            f.write(f"Labels count: {len(lbl_stems)}\n")
            f.write(f"Labels w/o images: {len(labels_no_image)}\n")
            f.write(f"Images w/o labels: {len(images_no_label)}\n\n")

            f.write("[Labels without matching images]\n")
            for s in labels_no_image:
                f.write(s + ".txt\n")
            f.write("\n[Images without matching labels]\n")
            for s in images_no_label:
                f.write(s + "\n")

        print(f"\nReport written to: {rp}")


if __name__ == "__main__":
    main()
