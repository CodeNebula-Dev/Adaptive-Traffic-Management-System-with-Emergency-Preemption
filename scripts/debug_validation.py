"""
Phase 1 Validation Pipeline Diagnostic Script.
Run this on Kaggle to trace exactly where mAP drops to 0.

Usage:
    python scripts/debug_validation.py --config configs/detector.yaml
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import yaml

from models.detector.yolo_detector import ATMSDetector
from data.coco.coco_dataset import COCOVehicleDataset, detection_collate_fn
from utils.nms import batch_nms
from utils.metrics import DetectionMetrics
from torch.utils.data import DataLoader


def main():
    # Load config
    config_path = os.path.join(project_root, 'configs', 'detector.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    model_cfg = config['model']
    eval_cfg = config['evaluation']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # === CHECK 1: Dataset label loading ===
    print("\n" + "=" * 60)
    print("CHECK 1: Dataset Label Loading")
    print("=" * 60)

    val_list = data_cfg['val_list']
    val_label_dir = data_cfg.get('val_label_dir', data_cfg.get('label_dir'))
    print(f"  val_list:      {val_list}")
    print(f"  val_label_dir: {val_label_dir}")
    print(f"  val_list exists: {os.path.exists(val_list)}")
    print(f"  val_label_dir exists: {os.path.exists(val_label_dir)}")

    if os.path.exists(val_list):
        with open(val_list) as f:
            val_paths = [l.strip() for l in f if l.strip()]
        print(f"  Number of val images: {len(val_paths)}")
        print(f"  First 3 image paths:")
        for p in val_paths[:3]:
            print(f"    {p}  (exists: {os.path.exists(p)})")
    else:
        print("  ERROR: val_list does not exist!")
        return

    # Check label file matching
    print(f"\n  Checking label files for first 10 val images:")
    labels_found = 0
    labels_missing = 0
    total_objects = 0
    for i, img_path in enumerate(val_paths[:10]):
        img_filename = os.path.basename(img_path)
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(val_label_dir, label_filename)
        exists = os.path.exists(label_path)
        n_obj = 0
        if exists:
            labels_found += 1
            data = np.loadtxt(label_path)
            if data.ndim == 0:
                n_obj = 0
            elif data.ndim == 1:
                n_obj = 1
            else:
                n_obj = len(data)
            total_objects += n_obj
        else:
            labels_missing += 1
        print(f"    [{i}] {label_filename}: exists={exists}, n_objects={n_obj}")
    print(f"  Summary: {labels_found} found, {labels_missing} missing, {total_objects} total objects")

    if labels_missing > 0:
        # Try alternative path
        alt_dir = val_label_dir.replace('val2017', 'train2017')
        print(f"\n  Trying alternative label dir: {alt_dir}")
        alt_found = 0
        for img_path in val_paths[:10]:
            img_filename = os.path.basename(img_path)
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            alt_path = os.path.join(alt_dir, label_filename)
            if os.path.exists(alt_path):
                alt_found += 1
        print(f"  Found {alt_found}/10 in alternative dir")

    # === CHECK 2: Dataset __getitem__ output ===
    print("\n" + "=" * 60)
    print("CHECK 2: Dataset __getitem__ Output")
    print("=" * 60)

    val_dataset = COCOVehicleDataset(
        img_list=val_list,
        label_dir=val_label_dir,
        img_size=model_cfg['img_size'],
        augment=False,
    )

    non_empty = 0
    for i in range(min(20, len(val_dataset))):
        img, targets = val_dataset[i]
        if targets.shape[0] > 0:
            non_empty += 1
            if non_empty <= 3:
                print(f"  [{i}] img shape={img.shape}, targets shape={targets.shape}")
                print(f"       First target: {targets[0]}")
                print(f"       Target value ranges: cls={targets[:, 1].unique().tolist()}, "
                      f"cx=[{targets[:, 2].min():.1f},{targets[:, 2].max():.1f}], "
                      f"cy=[{targets[:, 3].min():.1f},{targets[:, 3].max():.1f}], "
                      f"w=[{targets[:, 4].min():.1f},{targets[:, 4].max():.1f}], "
                      f"h=[{targets[:, 5].min():.1f},{targets[:, 5].max():.1f}]")

    print(f"\n  Images with targets: {non_empty}/20")
    if non_empty == 0:
        print("  CRITICAL: No validation images have targets!")
        print("  This is the ROOT CAUSE of mAP=0!")
        return

    # === CHECK 3: Collate function ===
    print("\n" + "=" * 60)
    print("CHECK 3: DataLoader Collate")
    print("=" * 60)

    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        collate_fn=detection_collate_fn, num_workers=0,
    )

    for images, targets in val_loader:
        print(f"  Batch images: {images.shape}")
        print(f"  Batch targets: {targets.shape}")
        if targets.shape[0] > 0:
            print(f"  First target: {targets[0]}")
            print(f"  Unique batch indices: {targets[:, 0].unique().tolist()}")
            print(f"  Target classes: {targets[:, 1].unique().tolist()}")
        else:
            print(f"  WARNING: Empty targets in first batch!")
        break

    # === CHECK 4: Model eval-mode predictions ===
    print("\n" + "=" * 60)
    print("CHECK 4: Model Eval-Mode Predictions")
    print("=" * 60)

    model = ATMSDetector(
        num_classes=model_cfg['num_classes'],
        depth_mul=model_cfg.get('depth_mul', 0.33),
        width_mul=model_cfg.get('width_mul', 0.5),
    ).to(device)

    # Load checkpoint if available
    ckpt_path = os.path.join(config['checkpoint']['save_dir'], 'last.pt')
    if os.path.exists(ckpt_path):
        print(f"  Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded epoch {ckpt.get('epoch', '?')}")
    else:
        print(f"  No checkpoint found at {ckpt_path}, using random weights")

    model.eval()

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)
            print(f"  Predictions shape: {predictions.shape}")
            print(f"  Predictions dtype: {predictions.dtype}")

            # Check prediction value ranges
            print(f"  cx range: [{predictions[..., 0].min():.1f}, {predictions[..., 0].max():.1f}]")
            print(f"  cy range: [{predictions[..., 1].min():.1f}, {predictions[..., 1].max():.1f}]")
            print(f"  w range:  [{predictions[..., 2].min():.1f}, {predictions[..., 2].max():.1f}]")
            print(f"  h range:  [{predictions[..., 3].min():.1f}, {predictions[..., 3].max():.1f}]")
            print(f"  obj_conf range: [{predictions[..., 4].min():.4f}, {predictions[..., 4].max():.4f}]")
            print(f"  cls_conf max per class: {predictions[..., 5:].max(dim=-2).values.max(dim=0).values.tolist()}")

            # Count how many predictions pass various thresholds
            obj_conf = predictions[..., 4]
            for thresh in [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]:
                n_pass = (obj_conf > thresh).sum().item()
                print(f"    obj_conf > {thresh}: {n_pass} predictions")

            break

    # === CHECK 5: NMS output ===
    print("\n" + "=" * 60)
    print("CHECK 5: NMS Output")
    print("=" * 60)

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)

            conf_thresh = eval_cfg['conf_threshold']
            iou_thresh = eval_cfg['iou_threshold']
            print(f"  Using conf_threshold={conf_thresh}, iou_threshold={iou_thresh}")

            detections = batch_nms(
                predictions,
                conf_threshold=conf_thresh,
                iou_threshold=iou_thresh,
                max_detections=eval_cfg['max_detections'],
            )

            total_dets = 0
            for b, det in enumerate(detections):
                if det is not None:
                    n = det.shape[0]
                    total_dets += n
                    if n > 0:
                        print(f"  Batch {b}: {n} detections")
                        print(f"    conf range: [{det[:, 4].min():.4f}, {det[:, 4].max():.4f}]")
                        print(f"    classes: {det[:, 6].unique().tolist()}")
                        print(f"    box[0]: {det[0, :4].tolist()}")
                else:
                    print(f"  Batch {b}: None (no detections)")

            print(f"  Total detections in batch: {total_dets}")

            if total_dets == 0:
                print("\n  ROOT CAUSE FOUND: NMS returns 0 detections!")
                print("  Trying with lower threshold...")
                for low_thresh in [0.001, 0.0001]:
                    dets_low = batch_nms(predictions, conf_threshold=low_thresh, iou_threshold=iou_thresh)
                    n_low = sum(d.shape[0] for d in dets_low if d is not None)
                    print(f"    conf_threshold={low_thresh}: {n_low} detections")

            break

    # === CHECK 6: Metrics accumulation ===
    print("\n" + "=" * 60)
    print("CHECK 6: Metrics Accumulation")
    print("=" * 60)

    metrics = DetectionMetrics(num_classes=model_cfg['num_classes'])

    with torch.no_grad():
        n_batches = 0
        total_gt = 0
        total_det = 0
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)
            detections = batch_nms(
                predictions,
                conf_threshold=conf_thresh,
                iou_threshold=iou_thresh,
                max_detections=eval_cfg['max_detections'],
            )

            metrics.update(detections, targets.cpu())

            n_batches += 1
            total_gt += targets.shape[0]
            total_det += sum(d.shape[0] for d in detections if d is not None)

            if n_batches >= 5:
                break

    print(f"  Processed {n_batches} batches")
    print(f"  Total GT targets: {total_gt}")
    print(f"  Total detections: {total_det}")
    print(f"  Images accumulated: {len(metrics.all_detections)}")

    results = metrics.compute()
    print(f"\n  mAP@0.5:     {results['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {results['mAP50_95']:.4f}")
    print(f"  Per-class AP: {results['per_class_ap50']}")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
