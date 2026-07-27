import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def check_yolo_dataset(dataset_path):
    # 路径设置
    img_dir = Path(dataset_path) / "images/azf45al30"
    lbl_dir = Path(dataset_path) / "labels/train"

    if not img_dir.exists():
        print(f"错误: 找不到路径 {img_dir}")
        return

    img_files = list(img_dir.glob("*.*"))
    print(f"开始检查，共发现 {len(img_files)} 张图片...")

    issues = []

    for img_path in tqdm(img_files):
        # --- 1. 检查图片有效性 ---
        img = cv2.imread(str(img_path))
        if img is None:
            issues.append(f"【坏图】无法读取图片: {img_path}")
            continue

        # 检查是否为纯色图 (全黑或全白)
        std_dev = np.std(img)
        if std_dev < 1.0: # 标准差极小说明几乎是纯色
            issues.append(f"【异常图】图像疑似纯色(全黑/全白): {img_path}")

        # --- 2. 检查对应的标签文件 ---
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            issues.append(f"【缺失】找不到对应的标签文件: {lbl_path}")
            continue

        # --- 3. 检查标签内容 ---
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                # YOLO允许空标签，但如果是NaN报错，最好检查是否有意外的空行
                continue

            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5: # 分割任务至少需要1个类ID + 2个坐标点(x,y)，通常更多
                    issues.append(f"【格式】标签行点数太少 (行 {i+1}): {lbl_path}")
                    continue

                try:
                    # 检查是否所有数值都是合法的浮点数
                    coords = [float(x) for x in parts]

                    # 检查类别ID (通常是整数)
                    class_id = coords[0]

                    # 检查坐标是否归一化 (0.0 - 1.0)
                    for val in coords[1:]:
                        if val < 0 or val > 1:
                            issues.append(f"【越界】坐标未归一化 (值={val}, 行 {i+1}): {lbl_path}")
                            break

                except ValueError:
                    issues.append(f"【损坏】标签包含非数字内容: {lbl_path}")

    # --- 报告结果 ---
    print("\n" + "="*50)
    if not issues:
        print("恭喜！未发现明显的脏数据。")
    else:
        print(f"检查完成，共发现 {len(issues)} 个潜在问题：")
        for issue in issues[:20]: # 只打印前20个
            print(issue)
        if len(issues) > 20:
            print(f"... 以及另外 {len(issues) - 20} 个错误。")
    print("="*50)

if __name__ == "__main__":
    # 修改为你数据集的根目录
    MY_PATH = r"D:\YOLOV8\Extract_RV\ultralytics\data"
    check_yolo_dataset(MY_PATH)
