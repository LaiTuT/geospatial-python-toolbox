import cv2
import numpy as np
import onnxruntime as ort

onnx_path = r"D:\YOLOV8\Extract_RV\YOLO-RV\ultralytics\runs\segment\yolopm_segment5\weights\best.onnx"
img_path = r"D:\YOLOV8\Extract_RV\YOLO-RV\ultralytics\data\images2\USGS_1m_x62y380_AL_25Co_B2_2017.png"

conf_thres = 0.25
iou_thres = 0.5
mask_thres = 0.50
min_instance_pixels = 100
window_stride = 100
save_path = r"D:\YOLOV8\Extract_RV\YOLO-RV\ultralytics\data\test_out\onnx_seg_result.jpg"
# BGR colors for class 0 and class 1 (OpenCV uses BGR).
CLASS_COLORS = [
    (0, 255, 0),   # class 0 -> green
    (0, 0, 255),   # class 1 -> red
]
CLASS_NAMES = ["class0", "class1"]
# Class priority for overlap merge. Larger value has higher priority.
CLASS_PRIORITY = {
    0: 1,
    1: 2,
}


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.zeros_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def class_aware_nms(xyxy: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray, iou: float) -> np.ndarray:
    keep = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        b = xyxy[idx]
        s = scores[idx]
        xywh = np.column_stack((b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])).tolist()
        ids = cv2.dnn.NMSBoxes(xywh, s.tolist(), score_threshold=0.0, nms_threshold=iou)
        if len(ids) > 0:
            ids = np.array(ids).reshape(-1)
            keep.extend(idx[ids].tolist())
    if not keep:
        return np.array([], dtype=np.int64)
    return np.array(keep, dtype=np.int64)


def clip_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes


def print_class_stats(title: str, cls_ids: np.ndarray, cls_conf: np.ndarray) -> None:
    if len(cls_ids) == 0:
        print(f"{title}: empty")
        return

    uniq, cnt = np.unique(cls_ids.astype(np.int64), return_counts=True)
    parts = []
    for u, c in zip(uniq.tolist(), cnt.tolist()):
        name = CLASS_NAMES[u] if 0 <= u < len(CLASS_NAMES) else f"class{u}"
        parts.append(f"{name}(id={u}): {c}")
    print(f"{title}: " + ", ".join(parts) + f" | conf_mean={float(np.mean(cls_conf)):.3f}")


def get_window_starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def preprocess_tile(tile_bgr: np.ndarray, in_w: int, in_h: int) -> np.ndarray:
    h, w = tile_bgr.shape[:2]
    padded = np.zeros((in_h, in_w, 3), dtype=np.uint8)
    padded[:h, :w] = tile_bgr
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    return x


def main() -> None:
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    inp = session.get_inputs()[0]
    print("input name:", inp.name, "shape:", inp.shape, "type:", inp.type)
    print("all outputs:")
    for i, o in enumerate(session.get_outputs()):
        print(f"  [{i}] name={o.name}, shape={o.shape}, type={o.type}")

    if inp.shape[2] is None or inp.shape[3] is None:
        raise ValueError("Model input size is dynamic. Please set fixed input shape before running this script.")

    h, w = int(inp.shape[2]), int(inp.shape[3])
    stride = int(window_stride) if int(window_stride) > 0 else min(h, w)

    img0 = cv2.imread(img_path)
    if img0 is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    h0, w0 = img0.shape[:2]
    x_starts = get_window_starts(w0, w, stride)
    y_starts = get_window_starts(h0, h, stride)

    print(f"Sliding window config: tile=({w}, {h}), stride={stride}, windows={len(x_starts) * len(y_starts)}")

    merged_cls_map = np.full((h0, w0), fill_value=-1, dtype=np.int16)
    merged_priority_map = np.full((h0, w0), fill_value=-1, dtype=np.int16)
    merged_conf_map = np.full((h0, w0), fill_value=-1.0, dtype=np.float32)
    pixel_filter_thres = max(0, int(min_instance_pixels))
    all_boxes = []
    all_conf = []
    all_cls = []
    filtered_small_instances = 0

    for y0 in y_starts:
        for x0 in x_starts:
            tile = img0[y0 : y0 + h, x0 : x0 + w]
            valid_h, valid_w = tile.shape[:2]
            x = preprocess_tile(tile, w, h)

            outputs = session.run(None, {inp.name: x})
            pred = outputs[0]
            proto = outputs[1] if len(outputs) > 1 else None

            if pred.ndim != 3 or pred.shape[0] != 1:
                raise ValueError(f"Unexpected pred shape: {pred.shape}")
            if proto is None:
                raise RuntimeError("No mask prototype output found. This does not look like a segmentation ONNX export.")

            pred = pred[0].T
            proto = proto[0]
            if proto.ndim != 3:
                raise ValueError(f"Unexpected proto shape: {proto.shape}")

            num_masks = int(proto.shape[0])
            nc = pred.shape[1] - 4 - num_masks
            if nc <= 0:
                raise ValueError(
                    f"Invalid channels: pred_dim={pred.shape[1]}, num_masks={num_masks}, parsed nc={nc}."
                )

            boxes = pred[:, :4]
            cls_scores = pred[:, 4 : 4 + nc]
            mask_coeff = pred[:, 4 + nc : 4 + nc + num_masks]

            conf = cls_scores.max(axis=1)
            cls_id = cls_scores.argmax(axis=1)
            keep = conf > conf_thres

            boxes = boxes[keep]
            conf = conf[keep]
            cls_id = cls_id[keep]
            mask_coeff = mask_coeff[keep]

            if len(boxes) == 0:
                continue

            xyxy = xywh_to_xyxy(boxes)
            keep_nms = class_aware_nms(xyxy, conf, cls_id, iou_thres)
            if len(keep_nms) == 0:
                continue

            xyxy = xyxy[keep_nms]
            conf = conf[keep_nms]
            cls_id = cls_id[keep_nms]
            mask_coeff = mask_coeff[keep_nms]

            xyxy = clip_boxes(xyxy, valid_w, valid_h)

            # Reconstruct masks in tile space and overwrite to global map.
            c, mh, mw = proto.shape
            proto_flat = proto.reshape(c, -1)
            masks = 1.0 / (1.0 + np.exp(-(mask_coeff @ proto_flat)))
            masks = masks.reshape(-1, mh, mw)

            for i in range(len(xyxy)):
                cls_i = int(cls_id[i])
                conf_i = float(conf[i])
                pri_i = int(CLASS_PRIORITY.get(cls_i, 0))

                gx1 = int(round(xyxy[i, 0])) + x0
                gy1 = int(round(xyxy[i, 1])) + y0
                gx2 = int(round(xyxy[i, 2])) + x0
                gy2 = int(round(xyxy[i, 3])) + y0

                gx1 = max(0, min(w0 - 1, gx1))
                gy1 = max(0, min(h0 - 1, gy1))
                gx2 = max(0, min(w0 - 1, gx2))
                gy2 = max(0, min(h0 - 1, gy2))

                m = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_LINEAR)
                m = (m[:valid_h, :valid_w] > mask_thres)

                # Constrain mask by its own box in local tile coordinates.
                lx1 = int(round(xyxy[i, 0]))
                ly1 = int(round(xyxy[i, 1]))
                lx2 = int(round(xyxy[i, 2]))
                ly2 = int(round(xyxy[i, 3]))
                lx1 = max(0, min(valid_w - 1, lx1))
                ly1 = max(0, min(valid_h - 1, ly1))
                lx2 = max(0, min(valid_w - 1, lx2))
                ly2 = max(0, min(valid_h - 1, ly2))
                if lx2 < lx1 or ly2 < ly1:
                    filtered_small_instances += 1
                    continue

                box_mask = np.zeros((valid_h, valid_w), dtype=bool)
                box_mask[ly1 : ly2 + 1, lx1 : lx2 + 1] = True
                m = m & box_mask

                if int(np.count_nonzero(m)) < pixel_filter_thres:
                    filtered_small_instances += 1
                    continue

                all_boxes.append([gx1, gy1, gx2, gy2])
                all_conf.append(conf_i)
                all_cls.append(cls_i)

                region_cls = merged_cls_map[y0 : y0 + valid_h, x0 : x0 + valid_w]
                region_pri = merged_priority_map[y0 : y0 + valid_h, x0 : x0 + valid_w]
                region_conf = merged_conf_map[y0 : y0 + valid_h, x0 : x0 + valid_w]

                better = (pri_i > region_pri) | ((pri_i == region_pri) & (conf_i >= region_conf))
                update_mask = m & better
                region_cls[update_mask] = cls_i
                region_pri[update_mask] = pri_i
                region_conf[update_mask] = conf_i

    if len(all_boxes) == 0:
        print("No detection in all sliding windows.")
        return

    print(f"Filtered small instances (<{pixel_filter_thres} pixels): {filtered_small_instances}")

    all_boxes = np.asarray(all_boxes, dtype=np.float32)
    all_conf = np.asarray(all_conf, dtype=np.float32)
    all_cls = np.asarray(all_cls, dtype=np.int64)

    print_class_stats("All windows merged", all_cls, all_conf)

    keep_global = class_aware_nms(all_boxes, all_conf, all_cls, iou_thres)
    if len(keep_global) == 0:
        print("No detection after global NMS.")
        return

    final_boxes = all_boxes[keep_global]
    final_conf = all_conf[keep_global]
    final_cls = all_cls[keep_global]
    print_class_stats("After global NMS", final_cls, final_conf)

    canvas = img0.copy()

    valid_cls = np.unique(merged_cls_map[merged_cls_map >= 0])
    for cid in valid_cls:
        mask = merged_cls_map == int(cid)
        if not np.any(mask):
            continue
        color = CLASS_COLORS[int(cid) % len(CLASS_COLORS)]
        color_layer = np.zeros_like(canvas, dtype=np.uint8)
        color_layer[:, :] = color
        canvas = np.where(mask[..., None], (0.65 * canvas + 0.35 * color_layer).astype(np.uint8), canvas)

    cv2.imwrite(save_path, canvas)
    print(f"Done. Final instances: {len(final_boxes)}")
    if len(np.unique(final_cls)) == 1:
        only_cls = int(np.unique(final_cls)[0])
        only_name = CLASS_NAMES[only_cls] if 0 <= only_cls < len(CLASS_NAMES) else f"class{only_cls}"
        print(f"[INFO] Only one class is detected in this image: {only_name}(id={only_cls}).")
    print(f"Saved result to: {save_path}")


if __name__ == "__main__":
    main()
