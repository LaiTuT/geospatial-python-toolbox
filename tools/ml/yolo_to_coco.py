import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
from collections import defaultdict


# ============================================================================
# 配置变量 - 在此处修改文件路径
# ============================================================================

# 图像文件夹路径
IMG_DIR = r"D:\YOLOV8\Extract_RV\ultralytics\data\images\azf45al30"

# YOLO 标签文件夹路径
LABEL_DIR = r"D:\YOLOV8\Extract_RV\ultralytics\data\labels\train"

# 输出目录 (生成 coco_train.json 和 coco_val.json)
OUTPUT_DIR = r"D:\Yolact\yolact\data\coco"

# 训练集比例 (0.0 - 1.0) 例如: 0.8 表示 80% 训练, 20% 验证
TRAIN_VAL_SPLIT = 0.8

# 类别名称列表 (可为空，则自动从标签推断)
# 注意：COCO 格式中，类别 ID 从 1 开始
CLASSES = ["ridge", "valley"]  # 示例: ["person", "car", "dog"]

# ============================================================================


class YOLO2COCOConverter:
    """Convert YOLO instance segmentation format to COCO object detection format with train/val split."""

    def __init__(self, img_dir: str, label_dir: str, output_dir: str, classes: List[str] = None,
                 train_val_split: float = 0.8):
        """
        Initialize converter.

        Args:
            img_dir: Directory containing image files
            label_dir: Directory containing YOLO txt label files
            output_dir: Output directory for COCO JSON files (coco_train.json, coco_val.json)
            classes: List of class names (if None, will infer from txt files)
            train_val_split: Fraction of data for training (0.0-1.0), rest for validation
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir)
        self.train_val_split = train_val_split
        self.classes = classes or []

        self.train_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.val_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.train_annotation_id = 1
        self.val_annotation_id = 1

    def _get_image_size(self, img_path: Path) -> Tuple[int, int]:
        """Get image dimensions using PIL or opencv."""
        try:
            from PIL import Image
            img = Image.open(img_path)
            return img.width, img.height
        except ImportError:
            try:
                import cv2
                img = cv2.imread(str(img_path))
                if img is not None:
                    return img.shape[1], img.shape[0]
            except ImportError:
                pass

        # Fallback: return default size
        return 640, 640

    def _yolo_to_coco_bbox(self, x_center: float, y_center: float, width: float, height: float,
                           img_width: int, img_height: int) -> List[float]:
        """Convert YOLO bbox format to COCO format.

        YOLO format: normalized center coordinates (x_center, y_center, width, height)
        COCO format: pixel coordinates [x, y, width, height] where (x,y) is top-left
        """
        # Convert normalized to pixel coordinates
        x_pixel = x_center * img_width
        y_pixel = y_center * img_height
        w_pixel = width * img_width
        h_pixel = height * img_height

        # Convert center to top-left corner
        x_top_left = x_pixel - w_pixel / 2
        y_top_left = y_pixel - h_pixel / 2

        return [x_top_left, y_top_left, w_pixel, h_pixel]

    def _yolo_polygon_to_coco(self, polygon_coords: List[float], img_width: int, img_height: int) -> List[float]:
        """Convert YOLO polygon (normalized) to COCO polygon (pixel coordinates)."""
        coco_polygon = []
        for i in range(0, len(polygon_coords), 2):
            if i + 1 < len(polygon_coords):
                x_norm = polygon_coords[i]
                y_norm = polygon_coords[i + 1]

                # Convert to pixel coordinates
                x_pixel = x_norm * img_width
                y_pixel = y_norm * img_height

                coco_polygon.extend([x_pixel, y_pixel])

        return coco_polygon

    def _calculate_bbox_from_polygon(self, polygon: List[float]) -> List[float]:
        """Calculate bounding box from polygon coordinates."""
        if not polygon or len(polygon) < 4:
            return [0, 0, 0, 0]

        xs = [polygon[i] for i in range(0, len(polygon), 2)]
        ys = [polygon[i] for i in range(1, len(polygon), 2)]

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)

        width = x_max - x_min
        height = y_max - y_min

        return [x_min, y_min, width, height]

    def _calculate_area(self, bbox: List[float]) -> float:
        """Calculate area from bounding box."""
        return bbox[2] * bbox[3]

    def process_image(self, img_path: Path, label_path: Path, image_id: int, is_train: bool) -> int:
        """
        Process a single image and its corresponding label file.

        Args:
            img_path: Path to image file
            label_path: Path to YOLO label file
            image_id: Unique image ID
            is_train: Whether this is a training or validation image

        Returns:
            Updated annotation_id
        """
        img_name = img_path.name

        # Get image dimensions
        img_width, img_height = self._get_image_size(img_path)

        # Select appropriate dataset
        target_data = self.train_data if is_train else self.val_data
        annotation_id = self.train_annotation_id if is_train else self.val_annotation_id

        # Add to images list
        target_data["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": img_width,
            "height": img_height
        })

        # Process annotations if label file exists
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:  # At minimum: class_id + 2 points (4 coords)
                        continue

                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]

                    # Update class info
                    if class_id >= len(self.classes):
                        for i in range(len(self.classes), class_id + 1):
                            self.classes.append(f"class_{i}")

                    # Check if coords are for bbox (4 values) or polygon (even number > 4)
                    if len(coords) == 4:
                        # YOLO bbox format: x_center, y_center, width, height
                        bbox = self._yolo_to_coco_bbox(
                            coords[0], coords[1], coords[2], coords[3],
                            img_width, img_height
                        )
                        segmentation = []
                    else:
                        # YOLO polygon format: x1, y1, x2, y2, ...
                        polygon = self._yolo_polygon_to_coco(coords, img_width, img_height)
                        segmentation = [polygon]
                        bbox = self._calculate_bbox_from_polygon(polygon)

                    # Create annotation (类别 ID 从 1 开始)
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id + 1,  # COCO 格式类别 ID 从 1 开始
                        "bbox": bbox,
                        "area": self._calculate_area(bbox),
                        "iscrowd": 0
                    }

                    if segmentation:
                        annotation["segmentation"] = segmentation

                    target_data["annotations"].append(annotation)
                    annotation_id += 1

        # Update annotation IDs
        if is_train:
            self.train_annotation_id = annotation_id
        else:
            self.val_annotation_id = annotation_id

        return annotation_id

    def build_categories(self):
        """Build categories from class list (ID starts from 1)."""
        categories = [
            {"id": i + 1, "name": class_name} for i, class_name in enumerate(self.classes)
        ]
        self.train_data["categories"] = categories
        self.val_data["categories"] = categories

    def convert(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Convert all YOLO format files with train/val split.

        Returns:
            Tuple of (train_data, val_data) COCO format dictionaries
        """
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        img_files = [f for f in self.img_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in image_extensions]
        img_files.sort()

        # Shuffle and split into train/val
        random.seed(42)  # For reproducibility
        random.shuffle(img_files)
        split_idx = int(len(img_files) * self.train_val_split)
        train_files = img_files[:split_idx]
        val_files = img_files[split_idx:]

        # Process training images
        train_image_id = 1
        for img_path in train_files:
            label_name = img_path.stem + '.txt'
            label_path = self.label_dir / label_name
            self.process_image(img_path, label_path, train_image_id, is_train=True)
            train_image_id += 1

        # Process validation images
        val_image_id = 1
        for img_path in val_files:
            label_name = img_path.stem + '.txt'
            label_path = self.label_dir / label_name
            self.process_image(img_path, label_path, val_image_id, is_train=False)
            val_image_id += 1

        # Build categories
        self.build_categories()

        return self.train_data, self.val_data

    def save(self) -> None:
        """Save COCO format data to separate train and validation JSON files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_file = self.output_dir / 'coco_train.json'
        val_file = self.output_dir / 'coco_val.json'

        with open(train_file, 'w') as f:
            json.dump(self.train_data, f, indent=2)

        with open(val_file, 'w') as f:
            json.dump(self.val_data, f, indent=2)

        print(f"\n{'='*60}")
        print(f"训练集: {train_file}")
        print(f"  图像数: {len(self.train_data['images'])}")
        print(f"  注释数: {len(self.train_data['annotations'])}")
        print(f"\n验证集: {val_file}")
        print(f"  图像数: {len(self.val_data['images'])}")
        print(f"  注释数: {len(self.val_data['annotations'])}")
        print(f"\n类别数: {len(self.train_data['categories'])}")
        print(f"{'='*60}\n")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Convert YOLO instance segmentation format to COCO JSON format with train/val split'
    )
    parser.add_argument('img_dir', nargs='?', default=IMG_DIR,
                        help=f'Directory containing image files (default: {IMG_DIR})')
    parser.add_argument('label_dir', nargs='?', default=LABEL_DIR,
                        help=f'Directory containing YOLO txt label files (default: {LABEL_DIR})')
    parser.add_argument('-d', '--output_dir', default=OUTPUT_DIR,
                        help=f'Output directory for COCO JSON files (default: {OUTPUT_DIR})')
    parser.add_argument('-s', '--split', type=float, default=TRAIN_VAL_SPLIT,
                        help=f'Train/val split ratio (default: {TRAIN_VAL_SPLIT})')
    parser.add_argument('-c', '--classes', nargs='+', default=CLASSES or None,
                        help='Class names in order (e.g., person car dog)')

    args = parser.parse_args()

    converter = YOLO2COCOConverter(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        classes=args.classes or [],
        train_val_split=args.split
    )

    converter.convert()
    converter.save()


if __name__ == '__main__':
    # =========================================================================
    # 方式1: 直接使用配置变量 (推荐 - 直接修改上面的配置变量即可)
    # =========================================================================
    converter = YOLO2COCOConverter(
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        output_dir=OUTPUT_DIR,
        classes=CLASSES or None,
        train_val_split=TRAIN_VAL_SPLIT
    )
    converter.convert()
    converter.save()

    # =========================================================================
    # 方式2: 命令行模式 (取消注释下面的行来使用)
    # 用法: python yolo_to_coco.py /path/to/images /path/to/labels -d /output/dir -s 0.8
    # =========================================================================
    # main()
