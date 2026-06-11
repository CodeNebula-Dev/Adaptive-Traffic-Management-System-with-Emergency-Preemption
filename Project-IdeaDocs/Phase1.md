# Phase 1 — Custom Vehicle Detector: Implementation Plan

## Project Context — What We're Building

**ATMS-Net** (Adaptive Traffic Management System Network) is a unified deep learning pipeline that takes raw video from four intersection cameras and outputs **dynamic per-lane signal timing** while guaranteeing **emergency vehicle passage**. The project solves two problems simultaneously:

1. **Adaptive Signal Control** — Replace fixed-time traffic cycles with a learned policy that observes real-time vehicle density and outputs optimal green/red phases per lane.
2. **Emergency Vehicle Preemption** — Detect ambulances, fire trucks, and police cars from the camera feed and immediately override signal control to clear a green corridor.

### The Four Modules

| Module | What It Does | Type | Phase |
|--------|-------------|------|-------|
| **Module 1 — Vehicle Detector** | Multi-class vehicle detection + per-lane density estimation | Custom YOLO-style CNN (from scratch) | **Phase 1** (this plan) |
| **Module 2 — EV Detector** | High-recall emergency vehicle classification | Shared backbone + EV classification head | Phase 3 |
| **Module 3 — RL Signal Controller** | Optimal phase + duration selection | Deep Q-Network (trained in SUMO) | Phase 4 |
| **Module 4 — Emergency Override** | Safety-critical green corridor enforcement | Deterministic rule-based | Phase 5 |

### Training Pipeline Overview

```
Phase 1: Backbone + Neck + Head → train on MS COCO vehicle classes → mAP@0.5 > 75%
    ↓ checkpoint
Phase 2: Fine-tune on UA-DETRAC + CARLA intersection footage + add density head
    ↓ checkpoint
Phase 3: Freeze backbone → train EV classification head → recall > 95%
    ↓ checkpoint
Phase 4: Deploy detector in SUMO → train DQN RL controller → 500k steps
    ↓ checkpoint
Phase 5: Full integration test — EV events injected → evaluate end-to-end
```

> [!IMPORTANT]
> **Phase 1 is the foundation of the entire system.** Every subsequent module depends on the detector backbone being well-trained. We are training from scratch (random initialisation) — not using pretrained weights — which is a deliberate design choice to demonstrate full architectural understanding.

---

## Phase 1 Scope: Module 1 — Custom Vehicle Detector

### What We're Building

A YOLO-style single-stage object detector trained from random initialisation on MS COCO vehicle classes. By the end of Phase 1, we will have:

- A working **CSP backbone** that extracts multi-scale features at strides 8, 16, 32
- An **FPN + PANet neck** for bidirectional feature fusion
- An **anchor-free detection head** that predicts bounding boxes, objectness, and class labels
- A complete **training pipeline** with loss computation, data augmentation, and evaluation
- A trained checkpoint achieving **mAP@0.5 > 75%** on the COCO vehicle subset

### Architecture Diagram

```
Input Image (416×416×3)
       │
       ▼
┌──────────────────────────────────────┐
│           CSP BACKBONE               │
│                                      │
│  Stem (Conv 6×6/2) → 208×208×32     │
│       │                              │
│  Stage 1: CSP Block → 104×104×64    │
│       │                              │
│  Stage 2: CSP Block → 52×52×128  ────── P3 (stride 8, large objects)
│       │                              │
│  Stage 3: CSP Block → 26×26×256  ────── P4 (stride 16, medium objects)
│       │                              │
│  Stage 4: CSP Block + SPP → 13×13×512 ── P5 (stride 32, small objects)
│                                      │
└──────────────────────────────────────┘
       │ P3, P4, P5
       ▼
┌──────────────────────────────────────┐
│        FPN + PANet NECK              │
│                                      │
│  Top-down path (semantic info):      │
│    P5 → upsample + concat P4 → N4   │
│    N4 → upsample + concat P3 → N3   │
│                                      │
│  Bottom-up path (localisation info): │
│    N3 → downsample + concat N4 → F4 │
│    F4 → downsample + concat P5 → F5 │
│                                      │
│  Output: N3, F4, F5                  │
└──────────────────────────────────────┘
       │ 3 fused feature maps
       ▼
┌──────────────────────────────────────┐
│     ANCHOR-FREE DETECTION HEAD       │
│                                      │
│  Per feature map, per spatial cell:  │
│    → 4D bbox offset (x, y, w, h)    │
│    → 1 objectness score             │
│    → C class probabilities          │
│                                      │
│  Decoupled head design:             │
│    Shared stem → cls branch (BCE)    │
│                → reg branch (CIoU)   │
│                → obj branch (BCE)    │
└──────────────────────────────────────┘
       │
       ▼
  Predictions: [{x, y, w, h, conf, class}]
```

---

## User Review Required

> [!IMPORTANT]
> **Input Resolution**: The plan uses **416×416** as the default training resolution (standard YOLO starting point, memory-friendly for single GPU). We can also support 640×640 for higher accuracy at the cost of ~2.4× more VRAM. Which do you prefer?

> [!IMPORTANT]
> **Vehicle Classes**: We will filter MS COCO to 4 vehicle classes: `car`, `truck`, `bus`, `motorcycle`. Should we also include `bicycle` as a 5th class, since cyclists are common at intersections?

---

## Open Questions

> [!WARNING]
> **GPU Hardware**: What GPU do you have available? The doc mentions "minimum RTX 3060 12GB". Training from scratch for ~50 epochs on 80k images will take approximately:
> - RTX 3060 12GB: ~8-10 hours per training run
> - RTX 3090 24GB / A100: ~3-4 hours
> - CPU-only: Not feasible (days)
>
> This affects batch size selection and whether we enable FP16 mixed precision.

> [!NOTE]
> **Dataset Download**: MS COCO 2017 training set is ~18GB. We'll create a download script, but make sure you have sufficient disk space (~30GB with extracted images + filtered annotations).

> [!NOTE]
> **Experiment Tracking**: The project docs specify Weights & Biases. Shall I configure W&B integration from the start, or use local TensorBoard-style logging for Phase 1 and add W&B later?

---

## Proposed Changes

This is a greenfield implementation — all files below are **new**. The project structure follows the layout defined in the README.

### Repository Structure for Phase 1

```
ATMS-Net/
├── configs/
│   └── detector.yaml              # [NEW] Training configuration
├── data/
│   └── coco/
│       ├── download_coco.py        # [NEW] COCO download + vehicle-class filter script
│       └── coco_dataset.py         # [NEW] PyTorch Dataset for COCO vehicle subset
├── models/
│   ├── __init__.py                 # [NEW]
│   ├── backbone/
│   │   ├── __init__.py             # [NEW]
│   │   └── csp_darknet.py          # [NEW] CSP backbone with residual blocks
│   ├── neck/
│   │   ├── __init__.py             # [NEW]
│   │   └── fpn_panet.py            # [NEW] FPN + PANet neck
│   └── detector/
│       ├── __init__.py             # [NEW]
│       ├── detection_head.py       # [NEW] Anchor-free decoupled detection head
│       └── yolo_detector.py        # [NEW] Full detector (backbone + neck + head)
├── utils/
│   ├── __init__.py                 # [NEW]
│   ├── losses.py                   # [NEW] CIoU loss, BCE loss, combined YOLO loss
│   ├── metrics.py                  # [NEW] mAP computation, precision, recall
│   ├── nms.py                      # [NEW] Non-maximum suppression
│   ├── boxes.py                    # [NEW] Box format conversions, IoU computation
│   └── augmentations.py            # [NEW] Mosaic, HSV jitter, cutout, random flip
├── scripts/
│   └── train_detector.py           # [NEW] Phase 1 training entry point
├── requirements.txt                # [NEW] Python dependencies
└── configs/
    └── detector.yaml               # [NEW] All hyperparameters
```

---

### Configuration

#### [NEW] [detector.yaml](file:///Users/devanshkhosla/Projects/ATMS-Net/configs/detector.yaml)

Central YAML config file containing all training hyperparameters:

- **Model**: input resolution (416), backbone depth/width multipliers, number of classes (4), anchor-free head channels
- **Training**: epochs (50), batch size (16), learning rate (0.01), SGD momentum (0.937), weight decay (5e-4), cosine annealing schedule with warm restarts, warmup epochs (3)
- **Data**: COCO data root, train/val split, number of workers
- **Augmentation**: mosaic probability (1.0), HSV jitter ranges (h=0.015, s=0.7, v=0.4), random flip probability (0.5), cutout params
- **Loss weights**: box loss weight (0.05), objectness loss weight (1.0), classification loss weight (0.5)
- **Evaluation**: NMS IoU threshold (0.45), confidence threshold (0.25), mAP IoU thresholds

---

### Data Pipeline

#### [NEW] [download_coco.py](file:///Users/devanshkhosla/Projects/ATMS-Net/data/coco/download_coco.py)

Script to download and prepare the MS COCO 2017 vehicle subset:

1. Download COCO 2017 train images + annotations (or symlink if already present)
2. Filter annotations to vehicle classes only: `car` (id=3), `truck` (id=8), `bus` (id=6), `motorcycle` (id=4)
3. Remap class IDs to 0-3 contiguous indices
4. Convert COCO JSON format to per-image YOLO `.txt` label files (`class cx cy w h` normalised)
5. Create `train.txt` and `val.txt` listing image paths (80/20 split of vehicle-containing images)
6. Print dataset statistics: total images, per-class instance counts, images-per-class distribution

#### [NEW] [coco_dataset.py](file:///Users/devanshkhosla/Projects/ATMS-Net/data/coco/coco_dataset.py)

PyTorch `Dataset` class for the COCO vehicle subset:

- Loads image + corresponding YOLO-format label file
- Applies augmentation pipeline (mosaic, HSV jitter, random flip, cutout)
- Resizes and pads to target resolution (416×416) with letterbox
- Returns `(image_tensor, targets)` where targets is a tensor of `[batch_idx, class, cx, cy, w, h]`
- Includes a collate function for variable-length target tensors
- Mosaic augmentation: combines 4 random images into one training sample for scale/context diversity

---

### Model Architecture

#### [NEW] [csp_darknet.py](file:///Users/devanshkhosla/Projects/ATMS-Net/models/backbone/csp_darknet.py)

The CSP (Cross-Stage Partial) backbone — the feature extractor:

- **ConvBnAct**: Basic building block — `Conv2d → BatchNorm2d → SiLU`
- **ResidualBlock**: Two ConvBnAct layers with a skip connection
- **CSPBlock**: Splits input channels into two paths — one goes through N residual blocks, other passes through directly — then concatenates. This reduces computation while preserving gradient flow
- **SPPBlock** (Spatial Pyramid Pooling): Applies max-pooling at kernel sizes 5, 9, 13 and concatenates — expands receptive field without increasing parameters
- **CSPDarknet**: Full backbone composed of:
  - Stem: 6×6 stride-2 conv (aggressive downsampling)
  - 4 stages of CSP blocks with progressively wider channels (64→128→256→512)
  - SPP at the end of stage 4
  - Returns feature maps at 3 scales: P3 (stride 8), P4 (stride 16), P5 (stride 32)

#### [NEW] [fpn_panet.py](file:///Users/devanshkhosla/Projects/ATMS-Net/models/neck/fpn_panet.py)

The FPN + PANet neck — bidirectional feature fusion:

- **Top-down pathway (FPN)**: P5 → upsample → concat with P4 → CSP block → N4; N4 → upsample → concat with P3 → CSP block → N3
- **Bottom-up pathway (PANet)**: N3 → downsample conv → concat with N4 → CSP block → F4; F4 → downsample conv → concat with P5 → CSP block → F5
- Output: Three fused feature maps (N3, F4, F5) at strides 8, 16, 32
- PANet's bottom-up pass adds localisation-strong features back to deep layers, critical for detecting vehicles at varying distances from the camera

#### [NEW] [detection_head.py](file:///Users/devanshkhosla/Projects/ATMS-Net/models/detector/detection_head.py)

Anchor-free decoupled detection head:

- **Decoupled design**: Separate branches for classification and regression (empirically shown to improve convergence vs. coupled heads)
- For each of the 3 feature maps, applies:
  - Shared 3×3 conv stem (256 channels)
  - Classification branch: 3×3 conv → 1×1 conv → C outputs (sigmoid)
  - Regression branch: 3×3 conv → 1×1 conv → 4 outputs (bbox offsets)
  - Objectness branch: 3×3 conv → 1×1 conv → 1 output (sigmoid)
- Grid-based decoding: predicted offsets are relative to the grid cell center, decoded to absolute coordinates at inference
- No anchor boxes — the model directly predicts center offset + width/height

#### [NEW] [yolo_detector.py](file:///Users/devanshkhosla/Projects/ATMS-Net/models/detector/yolo_detector.py)

Assembles backbone + neck + head into the full detector:

- `ATMSDetector(nn.Module)`: composes `CSPDarknet`, `FPNPANet`, and `DetectionHead`
- `forward()` returns raw predictions during training, decoded boxes during inference
- Includes a `from_config(yaml_path)` class method for instantiation from config
- Model summary: prints parameter count, FLOPs estimate, per-layer shapes

---

### Training Utilities

#### [NEW] [losses.py](file:///Users/devanshkhosla/Projects/ATMS-Net/utils/losses.py)

Loss functions for YOLO-style detection:

- **CIoU Loss** (Complete IoU): Combines IoU with center distance penalty and aspect ratio consistency. Better convergence than standard IoU/GIoU for bounding box regression
- **BCE with Logits**: For objectness and classification outputs
- **YOLOLoss**: Combined loss that:
  1. Assigns ground-truth boxes to grid cells using SimOTA (simplified optimal transport assignment)
  2. Computes CIoU loss on matched box predictions
  3. Computes BCE loss on objectness scores (positive + negative samples)
  4. Computes BCE loss on class predictions (positive samples only)
  5. Returns weighted sum: `λ_box × L_box + λ_obj × L_obj + λ_cls × L_cls`

#### [NEW] [metrics.py](file:///Users/devanshkhosla/Projects/ATMS-Net/utils/metrics.py)

Evaluation metrics:

- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5 (primary Phase 1 metric)
- **mAP@0.5:0.95**: COCO-style mAP averaged over IoU thresholds 0.5 to 0.95 in steps of 0.05
- **Per-class AP**: Breakdown by car, truck, bus, motorcycle
- **Precision-Recall curves**: For visualization
- **Confusion matrix**: For error analysis

#### [NEW] [nms.py](file:///Users/devanshkhosla/Projects/ATMS-Net/utils/nms.py)

Non-maximum suppression:

- Standard NMS with configurable IoU threshold
- Class-aware NMS (apply NMS per class independently)
- Batch NMS for efficient inference

#### [NEW] [boxes.py](file:///Users/devanshkhosla/Projects/ATMS-Net/utils/boxes.py)

Bounding box utilities:

- Format conversions: `xywh ↔ xyxy ↔ cxcywh`
- IoU computation (standard, GIoU, DIoU, CIoU)
- Box clipping to image boundaries
- Letterbox coordinate rescaling

#### [NEW] [augmentations.py](file:///Users/devanshkhosla/Projects/ATMS-Net/utils/augmentations.py)

Data augmentation pipeline:

- **Mosaic**: Stitch 4 random images into a 2×2 grid — forces the model to see partial objects and diverse context in every sample. This is the single most impactful YOLO augmentation
- **HSV Jitter**: Random hue, saturation, value shifts — robustness to lighting/weather
- **Random Horizontal Flip**: Standard geometric augmentation
- **Cutout / Random Erase**: Randomly mask small rectangular patches — forces redundant feature learning
- **Letterbox Resize**: Resize maintaining aspect ratio with gray padding

---

### Training Script

#### [NEW] [train_detector.py](file:///Users/devanshkhosla/Projects/ATMS-Net/scripts/train_detector.py)

Main training entry point:

```
Usage: python scripts/train_detector.py --config configs/detector.yaml
```

Training loop:

1. **Setup**: Parse config, create model, optimizer (SGD), scheduler (cosine annealing), loss function
2. **Data loading**: Create train/val datasets and dataloaders with mosaic collate
3. **Warmup**: Linear LR warmup for first 3 epochs (momentum and LR ramp from 0)
4. **Training loop** (50 epochs):
   - Forward pass → compute YOLOLoss → backward → SGD step
   - Log loss components per batch (box, obj, cls) to console and logfile
   - End of epoch: run validation (mAP@0.5 on val set)
   - Save checkpoint if best mAP so far
   - Save checkpoint every 10 epochs
5. **Mixed precision**: Use `torch.cuda.amp` for FP16 training (2× memory savings)
6. **EMA** (Exponential Moving Average): Maintain an EMA copy of model weights — smoother evaluation performance
7. **Logging**: Per-epoch: train loss, val mAP@0.5, mAP@0.5:0.95, per-class AP, learning rate, GPU memory

---

### Dependencies

#### [NEW] [requirements.txt](file:///Users/devanshkhosla/Projects/ATMS-Net/requirements.txt)

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=9.5.0
PyYAML>=6.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pycocotools>=2.0.7
```

---

## Implementation Order

The work will proceed in this sequence to ensure each component can be tested before building on it:

| Step | Component | Dependencies | Validation |
|------|-----------|-------------|------------|
| 1 | `requirements.txt` + project skeleton (`__init__.py` files) | None | `pip install -r requirements.txt` succeeds |
| 2 | `utils/boxes.py` | NumPy, Torch | Unit tests: format conversions, IoU values match expected |
| 3 | `models/backbone/csp_darknet.py` | PyTorch | Smoke test: random tensor → 3 feature maps of correct shapes |
| 4 | `models/neck/fpn_panet.py` | Backbone | Smoke test: 3 backbone outputs → 3 fused outputs, correct shapes |
| 5 | `models/detector/detection_head.py` | PyTorch | Smoke test: feature map → predictions with correct dimensions |
| 6 | `models/detector/yolo_detector.py` | Backbone + Neck + Head | Full forward pass: `(B, 3, 416, 416)` → list of predictions |
| 7 | `utils/losses.py` | `boxes.py` | CIoU loss on known box pairs matches expected |
| 8 | `utils/nms.py` | `boxes.py` | NMS correctly filters overlapping boxes |
| 9 | `utils/metrics.py` | `nms.py`, `boxes.py` | mAP on synthetic predictions matches manual calculation |
| 10 | `data/coco/download_coco.py` | pycocotools | Script runs, produces filtered labels + stats |
| 11 | `data/coco/coco_dataset.py` + `utils/augmentations.py` | OpenCV, PIL | Dataloader yields correct shapes, visualise augmented samples |
| 12 | `configs/detector.yaml` | All above | Config loads without error |
| 13 | `scripts/train_detector.py` | Everything | 1-epoch training run completes, loss decreases |
| 14 | Full 50-epoch training run | GPU hardware | mAP@0.5 > 75% on val set |

---

## Verification Plan

### Automated Tests

```bash
# Smoke test: model forward pass
python -c "from models.detector.yolo_detector import ATMSDetector; import torch; m = ATMSDetector(); out = m(torch.randn(1,3,416,416)); print('OK', [o.shape for o in out])"

# Unit tests for box utilities
python -m pytest utils/test_boxes.py -v

# 1-epoch training sanity check
python scripts/train_detector.py --config configs/detector.yaml --epochs 1 --batch-size 4
```

### Manual Verification

- **Visualise augmented training samples**: Render 8-10 mosaic-augmented images with ground truth boxes overlaid — verify boxes are correct after transforms
- **Loss curve inspection**: After 50 epochs, plot training loss (should decrease steadily after warmup) and validation mAP (should plateau around target)
- **Inference visualisation**: Run trained model on 10 held-out images, draw predicted boxes with confidence scores — qualitative check
- **Per-class AP breakdown**: Ensure no single class is drastically underperforming (indicates data or loss imbalance)

### Success Criteria

| Metric | Target | Hard Minimum |
|--------|--------|-------------|
| mAP@0.5 on COCO vehicle val set | > 75% | > 65% |
| mAP@0.5:0.95 | > 50% | > 40% |
| Inference speed (416×416) | > 30 FPS | > 20 FPS |
| Training completes without OOM | ✓ | ✓ |
