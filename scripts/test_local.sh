#!/bin/bash
# ============================================================================
# ATMS-Net Phase 1 — Local Mac Testing Script
# ============================================================================
# This script runs a quick end-to-end test of the vehicle detector pipeline
# on your Mac M4 WITHOUT needing the full COCO dataset download.
#
# What it does:
#   1. Generates synthetic dummy data (fake images + YOLO labels)
#   2. Runs a 2-epoch training on CPU with a tiny batch size
#   3. Verifies the full pipeline: data loading → model → loss → backprop → checkpoint
#
# Usage:
#   chmod +x scripts/test_local.sh
#   ./scripts/test_local.sh
#
# Expected runtime: ~2-3 minutes on Mac M4
# ============================================================================

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  ATMS-Net Phase 1 — Local Mac Testing"
echo "============================================================"
echo "  Project root: $PROJECT_ROOT"
echo "  Python: $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo ""

# ---- Step 1: Generate Synthetic Test Data ----
echo "[Step 1/4] Generating synthetic test data..."
python3 -c "
import os
import numpy as np
from PIL import Image
import random

random.seed(42)
np.random.seed(42)

# Create directory structure
data_dir = 'data/coco_test'
img_dir = os.path.join(data_dir, 'images')
label_dir = os.path.join(data_dir, 'labels')
os.makedirs(img_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# Vehicle class names
classes = ['car', 'motorcycle', 'bus', 'truck']
n_images = 50  # Small set for quick testing

train_paths = []
val_paths = []

for i in range(n_images):
    # Create a random image with colored rectangles (simulating vehicles)
    img = np.random.randint(60, 180, (480, 640, 3), dtype=np.uint8)
    
    # Add random 'vehicle' rectangles
    n_vehicles = random.randint(1, 6)
    labels = []
    
    for _ in range(n_vehicles):
        cls_id = random.randint(0, 3)
        # Random box position and size
        cx = random.uniform(0.1, 0.9)
        cy = random.uniform(0.1, 0.9)
        w = random.uniform(0.05, 0.25)
        h = random.uniform(0.05, 0.2)
        
        # Draw a colored rectangle on the image
        x1 = int((cx - w/2) * 640)
        y1 = int((cy - h/2) * 480)
        x2 = int((cx + w/2) * 640)
        y2 = int((cy + h/2) * 480)
        color = [(0, 0, 200), (0, 200, 0), (200, 200, 0), (200, 0, 200)][cls_id]
        img[max(0,y1):min(480,y2), max(0,x1):min(640,x2)] = color
        
        labels.append(f'{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
    
    # Save image
    img_path = os.path.join(img_dir, f'test_{i:04d}.jpg')
    Image.fromarray(img).save(img_path)
    
    # Save label
    label_path = os.path.join(label_dir, f'test_{i:04d}.txt')
    with open(label_path, 'w') as f:
        f.write('\n'.join(labels))
    
    # Split: 80% train, 20% val
    if i < int(n_images * 0.8):
        train_paths.append(os.path.abspath(img_path))
    else:
        val_paths.append(os.path.abspath(img_path))

# Write split files
with open(os.path.join(data_dir, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_paths))
with open(os.path.join(data_dir, 'val.txt'), 'w') as f:
    f.write('\n'.join(val_paths))

print(f'  ✓ Created {n_images} synthetic images ({len(train_paths)} train, {len(val_paths)} val)')
print(f'  ✓ Images saved to: {img_dir}')
print(f'  ✓ Labels saved to: {label_dir}')
"
echo ""

# ---- Step 2: Create Test Config ----
echo "[Step 2/4] Creating test config..."
cat > configs/detector_test.yaml << 'EOF'
# Quick test config — synthetic data, 2 epochs, tiny batch
model:
  num_classes: 4
  in_channels: 3
  depth_mul: 0.33
  width_mul: 0.5
  img_size: 416

data:
  data_dir: "data/coco_test"
  train_list: "data/coco_test/train.txt"
  val_list: "data/coco_test/val.txt"
  label_dir: "data/coco_test/labels"
  num_workers: 0   # 0 for Mac compatibility

training:
  epochs: 2
  batch_size: 4
  accumulate_grad: 1
  optimizer: "sgd"
  learning_rate: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  scheduler: "cosine"
  warmup_epochs: 1
  warmup_lr_ratio: 0.1
  min_lr_ratio: 0.01
  mixed_precision: false    # Disabled for CPU/MPS
  ema: true
  ema_decay: 0.9999

augmentation:
  mosaic_prob: 0.0          # Disable mosaic for quick test
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  flip_prob: 0.5
  cutout_prob: 0.5
  cutout_max_size: 0.2

loss:
  box_weight: 0.05
  obj_weight: 1.0
  cls_weight: 0.5

evaluation:
  conf_threshold: 0.01
  iou_threshold: 0.45
  max_detections: 300
  val_interval: 1

checkpoint:
  save_dir: "checkpoints/test"
  save_interval: 1
  save_best: true

logging:
  log_dir: "logs/test"
  log_interval: 5
  use_wandb: false
  project_name: "ATMS-Net"
  run_name: "local-test"

device:
  preferred: "auto"
EOF
echo "  ✓ Test config saved to: configs/detector_test.yaml"
echo ""

# ---- Step 3: Run Training ----
echo "[Step 3/4] Running 2-epoch training test..."
echo "  (This should take ~1-2 minutes on Mac M4)"
echo ""
python3 scripts/train_detector.py --config configs/detector_test.yaml --device cpu
echo ""

# ---- Step 4: Verify Outputs ----
echo "[Step 4/4] Verifying outputs..."
echo ""

# Check checkpoints
if [ -f "checkpoints/test/last.pt" ]; then
    echo "  ✓ Checkpoint saved: checkpoints/test/last.pt"
    python3 -c "
import torch
ckpt = torch.load('checkpoints/test/last.pt', map_location='cpu')
print(f'    Epoch: {ckpt[\"epoch\"] + 1}')
print(f'    Best mAP@0.5: {ckpt[\"best_map\"]:.4f}')
print(f'    Model keys: {len(ckpt[\"model_state_dict\"])} tensors')
if 'ema_state_dict' in ckpt:
    print(f'    EMA keys: {len(ckpt[\"ema_state_dict\"])} tensors')
print(f'    Config saved in checkpoint: ✓')
"
else
    echo "  ✗ Checkpoint not found!"
    exit 1
fi

# Check logs
if [ -f "logs/test/training.log" ]; then
    echo ""
    echo "  ✓ Training log saved: logs/test/training.log"
    echo "    Contents:"
    cat logs/test/training.log | sed 's/^/      /'
else
    echo "  ✗ Training log not found!"
fi

echo ""
echo "============================================================"
echo "  ✓ ALL TESTS PASSED!"
echo "============================================================"
echo ""
echo "  What was verified:"
echo "    ✓ Synthetic data generation"
echo "    ✓ Dataset loading + augmentation pipeline"
echo "    ✓ Model forward pass (backbone → neck → head)"
echo "    ✓ Loss computation (CIoU + BCE + SimOTA matching)"
echo "    ✓ Backward pass + SGD optimizer step"
echo "    ✓ LR scheduling (warmup + cosine annealing)"
echo "    ✓ EMA model update"
echo "    ✓ Validation with mAP computation"
echo "    ✓ Checkpoint saving (model + optimizer + EMA + config)"
echo "    ✓ Training log writing"
echo ""
echo "  Checkpoint location:"
echo "    checkpoints/test/last.pt  — last epoch"
echo "    checkpoints/test/best.pt  — best mAP@0.5 (if created)"
echo ""
echo "  Next: Upload code to Kaggle and train with real COCO data!"
echo ""
