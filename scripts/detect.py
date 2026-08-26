"""
ATMS-Net Vehicle Detector — Real-World Inference Script.

Runs object detection on images or folders using trained ATMS-Net weights.
Draws color-coded bounding boxes, confidence scores, and outputs a detection summary.

Usage:
    # Single image
    python scripts/detect.py --weights checkpoints/best.pt --source test.jpg

    # Directory of images
    python scripts/detect.py --weights checkpoints/best.pt --source data/coco/val2017/ --conf-threshold 0.25

    # Specific device
    python scripts/detect.py --weights checkpoints/best.pt --source test.jpg --device cuda
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import torch
import numpy as np

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.detector.yolo_detector import ATMSDetector
from utils.nms import batch_nms
from utils.boxes import rescale_boxes
from utils.augmentations import letterbox


# Distinct high-visibility BGR colors for vehicle classes
CLASS_COLORS = {
    'car': (0, 230, 60),         # Vibrant Green
    'motorcycle': (0, 215, 255),  # Yellow
    'bus': (255, 140, 0),         # Deep Blue / Cyan
    'truck': (30, 70, 255),       # Bright Red-Orange
}

CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']


def get_device(preferred='auto'):
    """Auto-detect best available compute device."""
    if preferred != 'auto':
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(weights_path, device, num_classes=4):
    """
    Load ATMS-Net model from checkpoint.
    Prefers EMA weights if available for smoother inference.
    """
    print(f"\n[Model] Loading weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)

    # Instantiate model
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        m_cfg = checkpoint['config']['model']
        model = ATMSDetector(
            num_classes=m_cfg.get('num_classes', num_classes),
            depth_mul=m_cfg.get('depth_mul', 0.33),
            width_mul=m_cfg.get('width_mul', 0.5),
            in_channels=m_cfg.get('in_channels', 3),
        )
    else:
        model = ATMSDetector(num_classes=num_classes)

    # Extract state dict (prioritize EMA weights if present)
    if 'ema_state_dict' in checkpoint:
        state_dict = checkpoint['ema_state_dict']
        print("  → Loaded EMA weights")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("  → Loaded model weights")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    best_map = checkpoint.get('best_map', None)
    if best_map is not None:
        print(f"  → Checkpoint trained mAP@0.5: {best_map:.4f}")

    return model


def draw_detections(img, detections, conf_threshold=0.25):
    """
    Draw professional bounding box annotations with filled header tags.

    Args:
        img: Original BGR image numpy array
        detections: Tensor of shape (N, 7) [x1, y1, x2, y2, conf, cls_conf, cls_id]
        conf_threshold: Minimum confidence to draw

    Returns:
        annotated_img: Image with rendered bounding boxes
        summary_counts: Dict mapping class_name -> count
    """
    annotated = img.copy()
    h, w = img.shape[:2]
    counts = {name: 0 for name in CLASS_NAMES}

    if detections is None or len(detections) == 0:
        return annotated, counts

    for det in detections:
        x1, y1, x2, y2, conf, cls_conf, cls_id = det.tolist()

        if conf < conf_threshold:
            continue

        cls_id = int(cls_id)
        class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}'
        counts[class_name] += 1

        color = CLASS_COLORS.get(class_name, (200, 200, 200))
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

        # Draw modern label tag
        label_text = f"{class_name} {conf:.1%}"
        font_scale = 0.55
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX

        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Tag background rectangle
        tag_top = max(0, y1 - text_h - baseline - 4)
        tag_bot = y1
        cv2.rectangle(annotated, (x1, tag_top), (x1 + text_w + 6, tag_bot), color, -1)

        # Label text in high contrast dark color
        cv2.putText(
            annotated, label_text, (x1 + 3, tag_bot - baseline - 1),
            font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
        )

    return annotated, counts


@torch.no_grad()
def detect_single_image(model, img_path, device, img_size=416, conf_threshold=0.25, iou_threshold=0.45):
    """
    Run detection on a single image file.

    Returns:
        annotated_img: Annotated image with bounding boxes
        counts: Dict of detected vehicle counts
        infer_time_ms: Inference runtime in milliseconds
        det_list: List of detection details
    """
    orig_img = cv2.imread(str(img_path))
    if orig_img is None:
        print(f"  ⚠ Could not read image: {img_path}")
        return None, {}, 0.0, []

    orig_h, orig_w = orig_img.shape[:2]

    # Preprocessing: Letterbox resize to square (416x416)
    padded_img, ratio, (dw, dh) = letterbox(orig_img, target_size=img_size)

    # BGR -> RGB -> Tensor (1, 3, H, W) in [0, 1]
    input_tensor = padded_img[:, :, ::-1].transpose(2, 0, 1)  # HWC to CHW
    input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32) / 255.0
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device)

    # Forward pass
    t0 = time.perf_counter()
    raw_predictions = model(input_tensor)
    t1 = time.perf_counter()
    infer_time_ms = (t1 - t0) * 1000.0

    # NMS filtering
    detections = batch_nms(
        raw_predictions,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_detections=100,
    )[0]

    det_list = []
    if detections is not None and len(detections) > 0:
        # Rescale boxes back to original input image dimensions
        rescaled_boxes = rescale_boxes(detections[:, :4], (orig_h, orig_w), (img_size, img_size))
        detections[:, :4] = rescaled_boxes

        for d in detections:
            c = int(d[6].item())
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'class_{c}'
            score = float(d[4].item())
            det_list.append({'class': name, 'confidence': score, 'box': d[:4].tolist()})

    # Render boxes
    annotated_img, counts = draw_detections(orig_img, detections, conf_threshold=conf_threshold)

    return annotated_img, counts, infer_time_ms, det_list


def main():
    parser = argparse.ArgumentParser(description="ATMS-Net Vehicle Detection Inference")
    parser.add_argument('--weights', type=str, required=True, help="Path to checkpoint weights (.pt)")
    parser.add_argument('--source', type=str, required=True, help="Path to image file or directory")
    parser.add_argument('--conf-threshold', type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument('--iou-threshold', type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument('--img-size', type=int, default=416, help="Input resolution (default: 416)")
    parser.add_argument('--device', type=str, default='auto', help="Compute device: auto, cuda, mps, cpu")
    parser.add_argument('--save-dir', type=str, default='runs/detect', help="Directory to save output images")
    args = parser.parse_args()

    device = get_device(args.device)
    print("=" * 60)
    print("ATMS-Net Vehicle Detector — Inference Engine")
    print("=" * 60)
    print(f"  Device:         {device}")
    print(f"  Conf threshold: {args.conf_threshold}")
    print(f"  IoU threshold:  {args.iou_threshold}")
    print(f"  Image size:     {args.img_size}x{args.img_size}")

    model = load_model(args.weights, device)

    # Collect source images
    source_path = Path(args.source)
    if source_path.is_file():
        image_files = [source_path]
    elif source_path.is_dir():
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [p for p in source_path.iterdir() if p.suffix.lower() in valid_extensions]
    else:
        print(f"Error: Source not found: {args.source}")
        sys.exit(1)

    print(f"\nFound {len(image_files)} image(s) to process.")
    os.makedirs(args.save_dir, exist_ok=True)

    total_time = 0.0
    total_vehicles = 0

    print("\n" + "-" * 60)
    for idx, img_file in enumerate(image_files, 1):
        annotated_img, counts, infer_ms, det_list = detect_single_image(
            model=model,
            img_path=img_file,
            device=device,
            img_size=args.img_size,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
        )

        if annotated_img is None:
            continue

        total_time += infer_ms
        img_vehicles = sum(counts.values())
        total_vehicles += img_vehicles

        # Save annotated image
        out_filename = f"detected_{img_file.name}"
        out_path = os.path.join(args.save_dir, out_filename)
        cv2.imwrite(out_path, annotated_img)

        # Print detection line
        count_strs = [f"{v} {k}{'s' if v > 1 else ''}" for k, v in counts.items() if v > 0]
        summary_str = ", ".join(count_strs) if count_strs else "No vehicles detected"
        print(f"[{idx}/{len(image_files)}] {img_file.name} ({infer_ms:.1f}ms) → {summary_str}")

        for det in det_list:
            print(f"    • {det['class'].capitalize()}: {det['confidence'] * 100:.1f}% confidence")

    print("-" * 60)
    avg_fps = len(image_files) / (total_time / 1000.0) if total_time > 0 else 0
    print(f"\n✓ Completed! Results saved to: {args.save_dir}")
    print(f"  Total vehicles detected: {total_vehicles}")
    print(f"  Average speed:          {avg_fps:.1f} FPS ({total_time / max(len(image_files), 1):.1f} ms/image)")
    print("=" * 60)


if __name__ == '__main__':
    main()
