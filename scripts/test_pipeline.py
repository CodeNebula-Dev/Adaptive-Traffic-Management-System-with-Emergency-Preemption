"""
ATMS-Net Pipeline & Loss Verification Test.
"""

import sys
from pathlib import Path
import torch
import torch.optim as optim

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.detector.yolo_detector import ATMSDetector
from utils.losses import YOLOLoss
from utils.metrics import DetectionMetrics
from utils.nms import batch_nms

def test_overfit_single_batch():
    print("=" * 60)
    print("Running Pipeline Test: Overfitting Single Synthetic Batch")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = ATMSDetector(num_classes=4, depth_mul=0.33, width_mul=0.5).to(device)
    model.train()

    # Create loss criterion with box_weight = 5.0, cls_weight = 1.0, obj_weight = 1.0
    criterion = YOLOLoss(num_classes=4, box_weight=5.0, obj_weight=1.0, cls_weight=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.003)

    # Synthetic batch of 2 images with known boxes in pixel coords (0-416)
    dummy_images = torch.randn(2, 3, 416, 416, device=device)
    dummy_targets = torch.tensor([
        [0, 0, 208.0, 208.0, 100.0, 100.0],  # batch 0, class 0 (car) at center
        [1, 3, 100.0, 100.0, 80.0, 60.0],     # batch 1, class 3 (truck)
    ], device=device)

    print("\nStarting 100 overfitting iterations...")
    for i in range(1, 101):
        optimizer.zero_grad()
        preds = model(dummy_images)
        loss_dict = criterion(preds, dummy_targets)
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()

        if i % 20 == 0 or i == 1:
            print(f"  Step {i:3d} | Total Loss: {loss.item():.4f} | "
                  f"Box: {loss_dict['box_loss'].item():.4f} | "
                  f"Obj: {loss_dict['obj_loss'].item():.4f} | "
                  f"Cls: {loss_dict['cls_loss'].item():.4f}")

    # Test Validation / Evaluation
    model.eval()
    with torch.no_grad():
        eval_preds = model(dummy_images)
        detections = batch_nms(eval_preds, conf_threshold=0.01, iou_threshold=0.45)

    metrics = DetectionMetrics(num_classes=4)
    metrics.update(detections, dummy_targets.cpu())
    results = metrics.compute()

    print("\nValidation Results on Synthetic Batch:")
    print(f"  mAP@0.5: {results['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {results['mAP50_95']:.4f}")
    print(f"  Per-class AP: {results['per_class_ap50']}")

if __name__ == '__main__':
    test_overfit_single_batch()
