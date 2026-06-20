# ATMS-Net — Phase 1 Progress Report

### Everything That Was Built, How It Works, and What Comes Next

*Last updated: June 20, 2026*

---

## Table of Contents

1. [The Big Picture — What Is ATMS-Net?](#1-the-big-picture)
2. [What Phase 1 Is About](#2-what-phase-1-is-about)
3. [Project Folder Structure — What Got Created](#3-project-folder-structure)
4. [The Model Architecture — How It Works](#4-the-model-architecture)
5. [Every File Explained](#5-every-file-explained)
6. [The Training Pipeline — How the Model Learns](#6-the-training-pipeline)
7. [The Local Test — What We Verified](#7-the-local-test)
8. [Where Things Get Stored](#8-where-things-get-stored)
9. [Kaggle Training Notebook — GPU Training Setup](#9-kaggle-training-notebook)
10. [What Comes Next](#10-what-comes-next)

---

## 1. The Big Picture

### What Are We Building?

**ATMS-Net** (Adaptive Traffic Management System Network) is a deep learning system that watches live traffic camera feeds at an intersection and does two things:

1. **Smart Signal Control** — Instead of dumb fixed-timer traffic lights (30 seconds green, 30 seconds red, repeat forever), our system actually *looks* at how many cars are on each road and gives more green time to the busier lanes.

2. **Emergency Vehicle Priority** — When an ambulance or fire truck approaches, the system instantly detects it and clears a green path for it. No more ambulances stuck at red lights.

### The Four Modules

The system is built from four separate pieces that work together:

```
┌─────────────────────────────────────────────────────────────┐
│  Module 1: Vehicle Detector  ←── THIS IS WHAT WE JUST BUILT │
│  "I can see cars, trucks, buses, and motorcycles"           │
│  "I know how many vehicles are on each lane"                │
├─────────────────────────────────────────────────────────────┤
│  Module 2: Emergency Vehicle Detector  (Phase 3 — later)    │
│  "I can spot ambulances, fire trucks, and police cars"      │
├─────────────────────────────────────────────────────────────┤
│  Module 3: RL Signal Controller  (Phase 4 — later)          │
│  "I decide which lane gets green and for how long"          │
├─────────────────────────────────────────────────────────────┤
│  Module 4: Emergency Override  (Phase 5 — later)            │
│  "When an ambulance is coming, I override everything"       │
└─────────────────────────────────────────────────────────────┘
```

### Why Phase 1 Matters

Phase 1 (the Vehicle Detector) is the **foundation** of the entire system. Every other module depends on it:

- Module 2 (EV Detector) reuses the backbone we train here
- Module 3 (RL Controller) needs the vehicle count from our detector
- Module 4 (Override) needs Module 2 which needs our backbone

If the detector doesn't work, nothing else works. That's why we built it first.

---

## 2. What Phase 1 Is About

### The Goal

Build a **YOLO-style object detector** that can:
- Look at a single frame from a traffic camera
- Draw boxes around every vehicle it sees
- Label each vehicle: car, motorcycle, bus, or truck
- Count how many vehicles are on each lane

### What "YOLO-style" Means

YOLO stands for **"You Only Look Once"**. It's a type of neural network architecture designed specifically for object detection. The key idea:

> Instead of scanning the image multiple times looking for objects (slow), YOLO processes the entire image in a single forward pass through the network (fast).

This is critical for us because we need to process live video at 30+ frames per second. We can't afford a slow detector.

### Why "From Scratch"?

Most research papers cheat — they download a pretrained YOLO model that someone else trained on millions of images and just fine-tune it. We're training **every single weight from random noise**. This is:
- Harder (takes longer to converge)
- More educational (we understand every layer)
- More impressive for a research contribution

### The Target

After training on real data (COCO dataset, on Kaggle), the detector should achieve:
- **mAP@0.5 > 75%** (mAP = Mean Average Precision — the standard metric for detectors)
- **> 30 FPS** inference speed on a GPU

---

## 3. Project Folder Structure

Here's every folder and file that was created, with explanations:

```
ATMS-Net/
│
├── Project-IdeaDocs/                    ←  Project planning documents
│   ├── ATMS-Net.md                      ← Original project idea (existed before)
│   ├── Phase1.md                        ← Phase 1 implementation plan
│   └── Traffic Management ... Model.md  ← Full technical roadmap
│
├── configs/                             ←   Configuration files
│   ├── detector.yaml                    ← Main training config (all hyperparameters)
│   └── detector_test.yaml               ← Quick local test config (auto-generated)
│
├── data/                                ←  Dataset handling
│   ├── __init__.py                      ← Makes this a Python package
│   └── coco/                            ← COCO-specific data code
│       ├── __init__.py
│       ├── download_coco.py             ← Downloads + filters COCO to vehicles only
│       └── coco_dataset.py              ← PyTorch Dataset class + data loading
│
├── models/                              ←  Neural network architecture
│   ├── __init__.py
│   ├── backbone/                        ← Feature extractor (the "eyes")
│   │   ├── __init__.py
│   │   └── csp_darknet.py               ← CSP backbone with residual blocks
│   ├── neck/                            ← Feature fusion (the "brain bridge")
│   │   ├── __init__.py
│   │   └── fpn_panet.py                 ← FPN + PANet bidirectional fusion
│   └── detector/                        ← Detection logic (the "decision maker")
│       ├── __init__.py
│       ├── detection_head.py            ← Anchor-free prediction head
│       └── yolo_detector.py             ← Full model assembly + EMA
│
├── utils/                               ←  Utility functions
│   ├── __init__.py
│   ├── augmentations.py                 ← Image augmentation (mosaic, flip, etc.)
│   ├── boxes.py                         ← Bounding box math (IoU, conversions)
│   ├── losses.py                        ← Training loss functions
│   ├── metrics.py                       ← Evaluation (mAP computation)
│   └── nms.py                           ← Non-maximum suppression
│
├── kaggle/                              ←  Kaggle GPU training
│   ├── README.md                        ← Step-by-step Kaggle guide
│   └── atms_net_phase1_training.ipynb   ← Complete training notebook
│
├── scripts/                             ←  Executable scripts
│   ├── train_detector.py                ← Main training entry point
│   └── test_local.sh                    ← Mac local testing shell script
│
├── checkpoints/                         ←  Saved model weights (git-ignored)
│   └── test/                            ← Test run checkpoints
│       ├── best.pt                      ← Best model by mAP
│       ├── last.pt                      ← Most recent checkpoint
│       ├── epoch_1.pt                   ← Epoch 1 checkpoint
│       └── epoch_2.pt                   ← Epoch 2 checkpoint
│
├── logs/                                ←  Training logs (git-ignored)
│   └── test/
│       └── training.log                 ← Per-epoch loss + metrics
│
├── data/coco_test/                      ←  Synthetic test data (git-ignored)
│   ├── images/                          ← 50 fake images with colored rectangles
│   ├── labels/                          ← YOLO-format label files
│   ├── train.txt                        ← List of 40 training image paths
│   └── val.txt                          ← List of 10 validation image paths
│
├── requirements.txt                     ←  Python dependencies
├── README.md                            ←  Project README
└── .gitignore                           ←  Files excluded from git
```

### What `__init__.py` Files Are

You'll notice many folders have an `__init__.py` file. In Python, this turns a regular folder into a **package** — meaning other code can import from it:

```python
# Without __init__.py → this would fail
from models.backbone.csp_darknet import CSPDarknet

# The __init__.py files make Python recognize the folder hierarchy
```

---

## 4. The Model Architecture

### How the Detector Processes a Video Frame

A video is just a sequence of images (frames). Our detector processes one frame at a time:

```
Live Camera Feed (30 FPS)
    │
    │  OpenCV grabs one frame
    ▼
Single Frame (e.g. 1920×1080 pixels)
    │
    │  Resize to 416×416 (with letterbox padding to keep proportions)
    ▼
Input Tensor (416 × 416 × 3)     ← 3 channels = Red, Green, Blue
    │
    │  BACKBONE: Extract features at 3 scales
    ▼
P3 (52×52×128)  ← Large objects (cars close to camera)
P4 (26×26×256)  ← Medium objects
P5 (13×13×512)  ← Small objects (cars far from camera)
    │
    │  NECK: Fuse features bidirectionally
    ▼
N3, F4, F5       ← Features now have both detail AND context
    │
    │  HEAD: Make predictions at each grid cell
    ▼
3,549 predictions ← (52² + 26² + 13² = 3,549 grid cells)
Each prediction: [cx, cy, width, height, confidence, car?, motorcycle?, bus?, truck?]
    │
    │  NMS: Remove duplicate overlapping boxes
    ▼
Final Detections: [{box, class, confidence}, ...]
```

### Part 1: The Backbone (`csp_darknet.py`)

Think of the backbone as the **eyes** of the model. It looks at the raw pixels and extracts meaningful features.

#### What's a "feature"?

Raw pixels are just numbers (0-255 for brightness). Features are *patterns* the network learns to recognize — edges, corners, shapes, textures, and eventually "this looks like a car hood" or "this looks like a wheel."

#### How the CSP Backbone works:

```
Input Image (416×416×3)
      │
      ▼
[Stem] 6×6 convolution, stride 2
      → Aggressively reduces size: 416→208
      → Learns basic edge/color features
      │
      ▼
[Stage 1] CSP Block
      → 208→104 (stride 2 downsample + cross-stage partial processing)
      │
      ▼
[Stage 2] CSP Block → outputs P3 (52×52×128)
      → 104→52
      → P3 captures fine details — good for large vehicles close to camera
      │
      ▼
[Stage 3] CSP Block → outputs P4 (26×26×256)
      → 52→26
      → P4 captures medium-level features
      │
      ▼
[Stage 4] CSP Block + SPP → outputs P5 (13×13×512)
      → 26→13
      → P5 captures high-level semantic meaning — "there's a vehicle-shaped blob here"
      → SPP (Spatial Pyramid Pooling) expands the field of view
```

#### What's a CSP Block?

CSP = Cross-Stage Partial. The trick: **split the input into two paths**, only process half through the expensive bottleneck layers, then recombine. This cuts computation in half while keeping accuracy nearly the same.

```
Input Channels
    ├──── Path 1 (50% channels): Goes through multiple residual blocks → learns complex features
    └──── Path 2 (50% channels): Direct 1×1 conv → preserves gradient flow cheaply
              │
              ▼
        Concatenate both paths → Fuse with 1×1 conv → Output
```

#### What's SPP?

SPP = Spatial Pyramid Pooling. It applies max-pooling at three different kernel sizes (5×5, 9×9, 13×13) and concatenates the results. This lets the network see the same features at different scales without adding learnable parameters.

### Part 2: The Neck (`fpn_panet.py`)

Think of the neck as a **bridge** that lets information flow between the three feature scales.

**Problem:** P3 has great fine-grained detail (it knows exactly where edges are) but poor semantic understanding (it doesn't know what a "car" is). P5 knows what a "car" is but has blurry spatial precision (13×13 is coarse).

**Solution:** Two-way information flow:

```
Top-Down (FPN — Feature Pyramid Network):
P5 (knows "car") → upsample → combine with P4 → combine with P3
Result: P3 now has semantic knowledge from P5

Bottom-Up (PANet — Path Aggregation Network):
P3 (knows exact positions) → downsample → combine with P4 → combine with P5
Result: P5 now has fine spatial precision from P3
```

After this bidirectional fusion, all three scales have both detailed positions AND semantic understanding.

### Part 3: The Detection Head (`detection_head.py`)

The head takes the fused features and makes actual predictions.

#### "Anchor-free" — what does that mean?

Older YOLO versions used "anchor boxes" — predefined box shapes (e.g., "wide and short for buses", "tall and narrow for trucks") placed at each grid cell. The network then adjusted these anchors to fit the actual objects.

Our detector is **anchor-free**: it directly predicts the center position and dimensions of each box. No predefined shapes needed. This is simpler and works better for traffic scenes where vehicle sizes vary a lot with camera distance.

#### "Decoupled" head design

Instead of one branch predicting everything (box + class + confidence), we split into three separate branches:

```
Feature Map
     │
     ├── Classification Branch → "What class is this?" (car/motorcycle/bus/truck)
     ├── Regression Branch → "Where exactly is the box?" (x, y, width, height)  
     └── Objectness Branch → "Is there anything here at all?" (yes/no confidence)
```

This separation improves training convergence because each branch can specialize.

### Part 4: Full Assembly (`yolo_detector.py`)

This file just connects the three parts together:

```python
class ATMSDetector(nn.Module):
    def __init__(self):
        self.backbone = CSPDarknet()    # Extract features
        self.neck = FPNPANet()          # Fuse features
        self.head = DetectionHead()     # Make predictions

    def forward(self, image):
        features = self.backbone(image)  # → (P3, P4, P5)
        fused = self.neck(features)      # → (N3, F4, F5)
        output = self.head(fused)        # → predictions
        return output
```

It also includes:
- **`from_config()`** — create a model from a YAML config file
- **`summary()`** — print model stats (13.2M params, 50.3 MB)
- **`ModelEMA`** — Exponential Moving Average for stable evaluation

### Model Stats

| Property | Value |
|----------|-------|
| Total parameters | 13,173,691 |
| Model size (FP32) | 50.3 MB |
| Detection classes | 4 (car, motorcycle, bus, truck) |
| Input size | 416 × 416 × 3 |
| Feature strides | 8, 16, 32 |
| Grid cells per image | 3,549 (52² + 26² + 13²) |
| Predictions per image | 3,549 × 9 = 31,941 values |

---

## 5. Every File Explained

### `configs/detector.yaml` — The Control Panel

This YAML file controls **every** aspect of training without changing code. Key settings:

```yaml
model:
  num_classes: 4        # car, motorcycle, bus, truck
  img_size: 416         # Input resolution
  depth_mul: 0.33       # Makes network shallower (fewer layers)
  width_mul: 0.5        # Makes network narrower (fewer channels)

training:
  epochs: 50            # How many times to go through all training data
  batch_size: 16        # How many images to process at once
  learning_rate: 0.01   # How fast the model learns (too high = unstable, too low = slow)
  mixed_precision: true # Use FP16 for 2× memory savings on GPU

loss:
  box_weight: 0.05      # How much to penalize wrong box positions
  obj_weight: 1.0       # How much to penalize wrong object/no-object predictions
  cls_weight: 0.5       # How much to penalize wrong class predictions
```

### `utils/boxes.py` — Bounding Box Math

All the math for working with rectangular boxes:

- **Format conversions**: `xyxy` (corners) ↔ `xywh` (top-left + size) ↔ `cxcywh` (center + size)
- **IoU** (Intersection over Union): Measures how much two boxes overlap (0 = no overlap, 1 = perfect match)
- **CIoU** (Complete IoU): A better version that also penalizes center distance and aspect ratio difference — makes training converge faster
- **Box clipping**: Ensures boxes don't go outside the image
- **Letterbox rescaling**: Converts box coordinates back to original image size after prediction

### `utils/losses.py` — How the Model Learns from Mistakes

The loss function tells the model **how wrong** its predictions are. Lower loss = better predictions. Three components:

1. **Box Loss (CIoU)**: "Your predicted box is 3 pixels off from the real car" → penalize
2. **Objectness Loss (BCE)**: "You said there's a car here but there isn't" → penalize
3. **Classification Loss (BCE)**: "You said this is a truck but it's actually a bus" → penalize

**SimOTA Matching** (the most complex part): Before computing loss, we need to decide which predictions match which ground-truth objects. SimOTA does this dynamically:
- For each real object, it finds the best-matching predictions based on a cost combining classification accuracy + box overlap
- Objects with high IoU get more positive matches (busier areas get more attention)
- This is much better than the old approach of "just match to the nearest grid cell"

### `utils/augmentations.py` — Making Training Data More Diverse

If you only show the model the same images over and over, it memorizes them instead of learning general features. Augmentations create diversity:

- **Mosaic**: Stitches 4 random images into a 2×2 grid → the model sees partial objects, diverse backgrounds, and varied scales in every sample. **This is the single most important YOLO augmentation.**
- **HSV Jitter**: Randomly shifts hue, saturation, and brightness → robustness to different lighting (sunny, cloudy, night)
- **Random Flip**: Mirrors the image horizontally → doubles the effective dataset
- **Cutout**: Randomly erases small patches → forces the model to not rely on any single feature (e.g., if you erase the wheels, it still needs to recognize the car)
- **Letterbox Resize**: Resizes while keeping aspect ratio, pads with gray → no distortion

### `utils/nms.py` — Cleaning Up Predictions

The network makes 3,549 predictions per image. Most of them are garbage (low confidence). And for a single car, multiple nearby grid cells might all predict the same car.

**NMS** (Non-Maximum Suppression) cleans this up:
1. Sort all predictions by confidence (highest first)
2. Take the highest-confidence prediction → keep it
3. Remove all other predictions that overlap with it by more than 45% (IoU threshold)
4. Repeat until no predictions left

"Class-aware NMS" does this separately per class — so a car prediction won't suppress a motorcycle prediction even if they overlap (different classes can overlap).

### `utils/metrics.py` — Measuring How Good the Model Is

**mAP@0.5** (Mean Average Precision at IoU threshold 0.5):
- For each predicted box, check if it overlaps with a ground-truth box by ≥ 50%
- If yes → True Positive. If no → False Positive.
- Compute Precision-Recall curve for each class
- AP = area under the Precision-Recall curve
- mAP = average AP across all 4 classes

**mAP@0.5:0.95** (COCO-style): Same thing but averaged across IoU thresholds from 0.5 to 0.95 in steps of 0.05 — much stricter, requires very precise boxes.

### `data/coco/download_coco.py` — Getting the Training Data

MS COCO (Common Objects in Context) is a massive dataset with 80 object categories. We only need vehicles, so this script:

1. Downloads COCO 2017 (~18GB of images + annotations)
2. Filters to 4 vehicle classes: car (id=3), motorcycle (id=4), bus (id=6), truck (id=8)
3. Converts COCO's JSON annotation format to YOLO's `.txt` format:
   ```
   # Each line: class_id  center_x  center_y  width  height  (all normalized 0-1)
   0 0.453125 0.634722 0.187500 0.122222
   3 0.721875 0.451389 0.093750 0.076389
   ```
4. Splits into 80% train / 20% validation
5. Prints statistics (how many images, how many cars/trucks/buses/motorcycles)

### `data/coco/coco_dataset.py` — Feeding Data to PyTorch

A PyTorch `Dataset` class that:
- Reads the image paths from `train.txt` / `val.txt`
- Loads each image + its YOLO label file
- Applies augmentations (mosaic, HSV, flip, cutout)
- Resizes to 416×416 with letterbox
- Converts to tensors: image `(3, 416, 416)` and targets `(N, 6)`
- Includes a custom `collate_fn` because each image has a different number of objects (standard PyTorch can't handle this)

### `scripts/train_detector.py` — The Main Training Loop

The big script that orchestrates everything:

```
1. Load config (detector.yaml)
2. Auto-detect device (CUDA > MPS > CPU)
3. Create model (ATMSDetector)
4. Create dataset + dataloader
5. Create optimizer (SGD) + scheduler (cosine annealing)
6. For each epoch:
   a. Train: forward pass → compute loss → backward pass → update weights
   b. Update EMA model
   c. Validate: run model on val set → compute mAP
   d. Save checkpoint if best mAP improved
   e. Log everything
7. Print final results
```

Key features:
- **Warmup**: First 3 epochs use a very low learning rate that gradually increases → prevents the randomly-initialized model from making huge destructive weight updates
- **Cosine annealing**: Learning rate follows a cosine curve, starting high and smoothly decreasing → fine-tuning in later epochs
- **EMA**: Maintains a "smoothed" copy of the model weights (exponential moving average) → more stable validation performance
- **Mixed precision**: Uses FP16 (half precision) on GPU → 2× less memory, ~1.5× faster training
- **Gradient accumulation**: Can simulate larger batch sizes on limited memory
- **Checkpointing**: Saves `best.pt` (highest mAP), `last.pt` (most recent), and periodic saves

### `scripts/test_local.sh` — Quick Mac Testing

A shell script that tests the entire pipeline without needing real data:

1. **Generates 50 synthetic images**: Random noise with colored rectangles as fake vehicles
2. **Creates a test config**: 2 epochs, batch size 4, no mosaic, CPU mode
3. **Runs training**: Full pipeline — data loading → model → loss → backprop → checkpoint
4. **Verifies outputs**: Checks that checkpoints and logs were created correctly

Takes ~11 seconds on Mac M4. Run it anytime to verify nothing is broken:
```bash
./scripts/test_local.sh
```

---

## 6. The Training Pipeline

### How Training Actually Works

Here's what happens on every single training step:

```
Step 1: Load a batch of 16 images + their ground truth labels
         images: (16, 3, 416, 416) — 16 images, 3 colors, 416×416 pixels
         targets: (N, 6) — [batch_idx, class, cx, cy, w, h] for every object

Step 2: Forward pass — run images through the model
         → The model outputs raw predictions at 3 scales

Step 3: SimOTA matching — figure out which predictions should match which objects
         → For each real car/bus/truck, find the best-matching prediction cells

Step 4: Compute loss — measure how wrong the predictions are
         Loss = 0.05 × CIoU_loss + 1.0 × objectness_loss + 0.5 × classification_loss

Step 5: Backward pass — compute gradients (how to adjust each weight to reduce loss)
         → PyTorch's autograd does this automatically

Step 6: Optimizer step — SGD updates all 13.2M weights by a tiny amount
         weight_new = weight_old - learning_rate × gradient

Step 7: Update EMA — update the smoothed model copy
         ema_weight = 0.9999 × ema_weight + 0.0001 × current_weight

Step 8: Repeat for next batch
```

### What Happens During Validation

Every epoch, we check how well the model is doing:

```
1. Switch model to eval mode (no dropout, fixed batch norm)
2. Run all validation images through the model
3. Apply NMS to clean up predictions
4. Compare predictions to ground truth using mAP@0.5
5. If mAP improved → save as best.pt
```

### Learning Rate Schedule

```
Epoch:  1    2    3    4    5    ...    25    ...    45    50
LR:   0.001 0.005 0.01 0.0098 0.0095  ...  0.005  ...  0.001 0.0001
      ├─── Warmup ───┤├─── Cosine Annealing (smooth decrease) ─────┤
```

- **Warmup (epochs 1-3)**: Start with very low LR (0.001) and linearly increase to full LR (0.01). This prevents the randomly-initialized model from exploding.
- **Cosine Annealing (epochs 4-50)**: Smoothly decrease LR following a cosine curve. Early epochs = big changes. Later epochs = fine-tuning.

---

## 7. The Local Test

### What We Ran

```bash
./scripts/test_local.sh
```

### What Happened

| Step | What | Result |
|------|------|--------|
| 1 | Generated 50 synthetic images with colored rectangles | ✅ 40 train + 10 val |
| 2 | Created test config (2 epochs, batch 4, CPU) | ✅ |
| 3a | **Epoch 1**: Loss = 4.49, mAP@0.5 = 0.0003 | ✅ Model learning from random init |
| 3b | **Epoch 2**: Loss = 1.87, mAP@0.5 = 0.0157 | ✅ Loss dropped 58%! |
| 4a | Checkpoint saved (414 tensors + EMA + config) | ✅ `checkpoints/test/best.pt` |
| 4b | Training log written | ✅ `logs/test/training.log` |

**Key observation:** The loss went from 4.49 → 1.87 (58% drop) in just 2 epochs. This proves:
- ✅ Gradients are flowing correctly (backward pass works)
- ✅ Weights are being updated (optimizer works)
- ✅ The model is learning something (loss decreasing)
- ✅ The mAP is near-zero (expected — these are random colored rectangles, not real vehicles)

### What the mAP Numbers Mean

- **mAP@0.5 = 0.0157** on synthetic data is *expected to be near zero* — the model can't learn real vehicle detection from random colored rectangles
- On real COCO data with 50 epochs of training, we expect mAP@0.5 > 75%

---

## 8. Where Things Get Stored

### Model Checkpoints (`.pt` files)

```
checkpoints/
├── test/
│   ├── best.pt       ← Best model by mAP@0.5
│   ├── last.pt       ← Most recent epoch
│   ├── epoch_1.pt    ← Checkpoint at epoch 1
│   └── epoch_2.pt    ← Checkpoint at epoch 2
```

Each `.pt` file contains:

| Content | What It Is | Why |
|---------|-----------|-----|
| `model_state_dict` | All 13.2M model weights | The actual trained model |
| `ema_state_dict` | EMA model weights | Smoother version for inference |
| `optimizer_state_dict` | SGD internal state | To resume training exactly |
| `scheduler_state_dict` | LR schedule position | To resume training exactly |
| `epoch` | Which epoch this was saved at | For tracking |
| `best_map` | Best mAP@0.5 seen so far | To know when to save "best" |
| `config` | Full YAML config snapshot | For reproducibility |

> **Note:** `.pt` files and `checkpoints/` are in `.gitignore` — they're large binary files that shouldn't go in git.

### Training Logs

```
logs/
└── test/
    └── training.log     ← One line per epoch with all metrics
```

Example content:
```
epoch=1 loss=4.4949 box=3.5586 obj=0.0145 cls=8.6050 lr=0.010000
epoch=2 loss=1.8693 box=2.8213 obj=0.0211 cls=3.4142 lr=0.000100
```

### Weights & Biases (W&B)

This is an **experiment tracking cloud platform** (like a fancy dashboard for loss curves). It's currently **disabled** (`use_wandb: false` in the config). We can enable it later when training on Kaggle for nice visualizations.

---

## 9. Kaggle Training Notebook

### Why Kaggle?

Our model has 13.2 million parameters. Training it from scratch on a CPU (even a fast M4 Mac) would take **days**. Kaggle provides a free NVIDIA Tesla T4 GPU (16 GB VRAM) with ~30 hours/week of compute, which can finish 50 epochs of training in **2–3 hours**.

We created a dedicated `kaggle/` folder with a ready-to-run Jupyter notebook.

### What's Inside `kaggle/`

| File | Purpose |
|------|---------|
| `README.md` | Step-by-step guide: how to upload, enable GPU, run, and download results |
| `atms_net_phase1_training.ipynb` | Complete training notebook with 6 sections |

### The 6 Sections of the Notebook

Here is exactly what each section of the Kaggle notebook does:

#### Section 1 — Environment Setup
- Verifies the GPU is available (prints GPU name + VRAM)
- Clones the ATMS-Net repository from GitHub
- Installs Python dependencies (`pycocotools`, `opencv`, `tqdm`, etc.)
- Sanity-checks by importing the model and printing its summary

#### Section 2 — Dataset Preparation
- **Option A**: Downloads the full COCO 2017 dataset (~20 GB) using our `download_coco.py` script
- **Option B** (faster): Links Kaggle's pre-loaded COCO dataset via symlinks (avoids re-downloading)
- Filters annotations to only 4 vehicle classes (car, motorcycle, bus, truck)
- Converts labels from COCO JSON format to YOLO `.txt` format
- Splits into 80% train / 20% validation
- Visualizes 6 random training images with their ground-truth bounding boxes

#### Section 3 — Training Configuration
- Loads the default `configs/detector.yaml`
- Applies Kaggle-specific overrides:
  - `batch_size: 32` (T4 has 16 GB VRAM — fits larger batches)
  - `mixed_precision: true` (FP16 halves memory, speeds up 1.5×)
  - `num_workers: 4` (Kaggle has 4 CPU cores for data loading)
  - Checkpoint and log paths set to `/kaggle/working/` (persists after kernel restart)
- Saves as `configs/kaggle_detector.yaml`

#### Section 4 — Training
- Runs the full training command:
  ```bash
  python scripts/train_detector.py --config configs/kaggle_detector.yaml --device cuda
  ```
- 50 epochs with warmup + cosine annealing LR schedule
- Validates every epoch and saves `best.pt` when mAP@0.5 improves
- Expected time: ~2–3 hours on Tesla T4

#### Section 5 — Results & Evaluation
- **Loss curves**: Plots total loss, box/obj/cls components, and LR schedule over all epochs
- **Checkpoint inspection**: Loads `best.pt` and prints epoch, mAP, tensor count, file size
- **Inference demo**: Loads the trained model, runs it on 6 random validation images, draws predicted bounding boxes with class labels and confidence scores

#### Section 6 — Export & Download
- Copies `best.pt`, `last.pt`, training log, and plots into a clean output folder
- Zips everything into `atms_net_phase1_trained.zip` for easy download
- Prints instructions: download from Kaggle's Output tab → place `best.pt` in local `checkpoints/`

### How to Use It (Quick Version)

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. **File → Import Notebook** → upload `atms_net_phase1_training.ipynb`
3. **Settings**: GPU T4 x2 + Internet ON
4. Click **Run All** → wait ~2–3 hours
5. Download `best.pt` from the Output tab

### Expected Results After Training

| Metric | Expected Value |
|--------|---------------|
| mAP@0.5 | 70–80% (target: >75%) |
| car AP | ~85% |
| truck AP | ~70% |
| bus AP | ~75% |
| motorcycle AP | ~65% |
| Training time | ~2–3 hours on T4 |

---

## 10. What Comes Next

### Immediate Next Step: Run the Kaggle Notebook

The code and the Kaggle notebook are both complete. To train the model:

1. ✅ ~~Set up Kaggle notebook~~ — **Done!** (see `kaggle/` folder)
2. **Train 50 epochs on Tesla T4** — Upload the notebook and run it
3. **Evaluate** — Check mAP@0.5 > 75%
4. **Download `best.pt`** — Place it in your local `checkpoints/` folder

### After Phase 1: The Remaining Phases

| Phase | What | Builds On |
|-------|------|-----------|
| **Phase 2** | Fine-tune detector on intersection footage (UA-DETRAC) + add density estimation | Phase 1 checkpoint |
| **Phase 3** | Train emergency vehicle detection head (ambulance, fire truck, police) | Phase 1 backbone (frozen) |
| **Phase 4** | Train DQN reinforcement learning signal controller in SUMO simulator | Phase 2 density output |
| **Phase 5** | Full system integration test — all 4 modules working together | Everything |

### Git History

| Commit | Files | Description |
|--------|-------|-------------|
| `c1780ce` | 24 files, 3,808 lines | Phase 1 code — full detector implementation |
| `e1a4a8e` | 1 file, 668 lines | Progress documentation (this file) |
| `bfb8153` | 2 files, 764 lines | Kaggle notebook + README for GPU training |

- Branch: `main`
- Repository: `CodeNebula-Dev/Adaptive-Traffic-Management-System-with-Emergency-Preemption`

---

*Phase 1 code complete. Kaggle training notebook ready. Next: run training on GPU!* 🚀
