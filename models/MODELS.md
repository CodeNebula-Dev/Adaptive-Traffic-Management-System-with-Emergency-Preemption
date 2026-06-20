# ATMS-Net — Models Folder Documentation

### Every Important Function, How Files Communicate, and How the Full Model Works

*Last updated: June 20, 2026*

---

## Table of Contents

1. [Folder Layout](#1-folder-layout)
2. [How the Files Communicate — The Data Flow](#2-how-the-files-communicate)
3. [File 1: `backbone/csp_darknet.py` — The Eyes](#3-backbone)
4. [File 2: `neck/fpn_panet.py` — The Bridge](#4-neck)
5. [File 3: `detector/detection_head.py` — The Decision Maker](#5-detection-head)
6. [File 4: `detector/yolo_detector.py` — The Full Assembly](#6-full-assembly)
7. [How the Model Connects to the Rest of the Project](#7-external-connections)
8. [Quick Reference: All Important Functions](#8-quick-reference)

---

## 1. Folder Layout

```
models/
├── __init__.py                    ← Makes models/ a Python package
├── MODELS.md                      ← This documentation file
│
├── backbone/                      ← Feature extractor (the "eyes")
│   ├── __init__.py                ← Exports: CSPDarknet
│   └── csp_darknet.py             ← 5 classes, 261 lines
│
├── neck/                          ← Feature fusion (the "bridge")
│   ├── __init__.py                ← Exports: FPNPANet
│   └── fpn_panet.py               ← 1 class, 119 lines
│
└── detector/                      ← Detection logic (the "decision maker")
    ├── __init__.py                ← Exports: (empty)
    ├── detection_head.py          ← 2 classes, 261 lines
    └── yolo_detector.py           ← 2 classes, 197 lines (main entry point)
```

---

## 2. How the Files Communicate

### The Big Picture — What Calls What

When a single image passes through the model, the files interact like this:

```
yolo_detector.py  (the orchestrator — called by train_detector.py)
    │
    │ 1. self.backbone(image)
    │    calls CSPDarknet.forward()
    ▼
csp_darknet.py
    │  Returns: (P3, P4, P5) — three feature maps at different scales
    │
    │ 2. self.neck(features)
    │    calls FPNPANet.forward()
    ▼
fpn_panet.py
    │  Returns: (N3, F4, F5) — three fused feature maps
    │
    │ 3. self.head(fused_features)
    │    calls DetectionHead.forward()
    ▼
detection_head.py
    │  Returns:
    │    Training → dict of raw logits per scale (for loss computation)
    │    Inference → (B, 3549, 9) decoded bounding boxes
    ▼
Output: Vehicle detections [cx, cy, w, h, confidence, car?, motorcycle?, bus?, truck?]
```

### The Import Chain

Here is exactly which file imports from which and what it uses:

```
csp_darknet.py          ← imports nothing from our code (root dependency)
     │
     │ Exports: ConvBnAct, CSPBlock, CSPDarknet
     │
     ├──────────────── fpn_panet.py
     │                    imports: ConvBnAct, CSPBlock
     │                    (reuses the same building blocks for the neck)
     │
     └──────────────── detection_head.py
                          imports: ConvBnAct
                          (reuses the Conv→BN→SiLU block for head branches)

yolo_detector.py         ← the assembly file
     imports:
       ├── CSPDarknet      from models.backbone.csp_darknet
       ├── FPNPANet        from models.neck.fpn_panet
       └── DetectionHead   from models.detector.detection_head
```

### What This Means

- `csp_darknet.py` is the **foundation**. It defines the basic building blocks (`ConvBnAct`, `Bottleneck`, `CSPBlock`) that every other file reuses. If you change something here, it affects the backbone, the neck, and the head.

- `fpn_panet.py` does **not** define its own conv block — it imports `ConvBnAct` and `CSPBlock` from the backbone. This keeps the architecture consistent (same batch normalization, same SiLU activation everywhere).

- `detection_head.py` also imports `ConvBnAct` from the backbone for its classification and regression branches.

- `yolo_detector.py` imports one class from each sub-module and wires them into a single `nn.Module`. This is the only file the rest of the project needs to import.

### Channel Count Communication

The three components must agree on tensor shapes. Here's how they negotiate:

```python
# In yolo_detector.py:

self.backbone = CSPDarknet(...)
# backbone.out_channels = [128, 256, 512]  ← backbone tells the neck its output sizes

self.neck = FPNPANet(in_channels=self.backbone.out_channels, ...)
# neck.out_channels = [128, 256, 512]      ← neck tells the head its output sizes

self.head = DetectionHead(in_channels_list=self.neck.out_channels, ...)
# head creates one DecoupledHead per channel count
```

The `out_channels` attribute is the **contract** between components. The backbone sets it, the neck reads it and passes its own, and the head reads the neck's.

---

## 3. `backbone/csp_darknet.py` — The Eyes

This file extracts raw visual features from the input image. It's the deepest dependency — everything else builds on top of it.

### Classes Defined

| Class | Lines | Purpose |
|-------|-------|---------|
| `ConvBnAct` | 14–42 | Fundamental building block: Conv2d → BatchNorm2d → SiLU |
| `Bottleneck` | 45–73 | Two-conv residual block with optional skip connection |
| `CSPBlock` | 76–118 | Cross-Stage Partial block — splits gradient flow for efficiency |
| `SPPBlock` | 121–156 | Spatial Pyramid Pooling — multi-scale receptive field expansion |
| `CSPDarknet` | 159–261 | Full backbone — assembles stem + 4 stages + SPP |

### Important Functions

#### `ConvBnAct.forward(x)` — The Most Called Function in the Model

```python
def forward(self, x):
    return self.act(self.bn(self.conv(x)))
```

This is a `Conv2d` → `BatchNorm2d` → `SiLU` sequence. It's used **everywhere** — in the backbone, neck, and head. Every learnable layer in the network passes through this pattern.

- **Why BatchNorm?** Normalizes activations to zero mean and unit variance. This stabilizes training and allows higher learning rates.
- **Why SiLU (Swish)?** `SiLU(x) = x * sigmoid(x)`. Smoother than ReLU, avoids the "dying neuron" problem where ReLU outputs zero for negative inputs and never recovers.
- **Why `bias=False`?** When followed by BatchNorm, the bias in Conv2d is redundant (BatchNorm has its own learnable bias). Removing it saves memory.

#### `Bottleneck.forward(x)` — The Residual Block

```python
def forward(self, x):
    out = self.conv2(self.conv1(x))    # 1×1 reduce → 3×3 process
    if self.use_shortcut:
        out = out + x                   # skip connection
    return out
```

The skip connection (`out + x`) is critical for training from scratch. Without it, gradients vanish in deep networks. The residual learning idea is: instead of learning the full transformation `F(x)`, the block learns the **residual** `F(x) - x`, which is easier to optimize.

#### `CSPBlock.forward(x)` — The Efficiency Trick

```python
def forward(self, x):
    path1 = self.bottlenecks(self.conv1(x))  # expensive path: N bottleneck blocks
    path2 = self.conv2(x)                     # cheap path: single 1×1 conv
    return self.conv3(torch.cat([path1, path2], dim=1))  # fuse both
```

This is the **Cross-Stage Partial** idea. Instead of sending all channels through the expensive bottleneck chain, it:
1. Splits input into two halves
2. Sends half through the bottleneck chain (learns complex features)
3. Sends the other half through a cheap 1×1 conv (preserves gradient flow)
4. Concatenates both and fuses with a final 1×1 conv

Result: ~50% less computation for nearly the same accuracy.

#### `SPPBlock.forward(x)` — Multi-Scale Vision

```python
def forward(self, x):
    x = self.conv1(x)
    pool_outputs = [x] + [pool(x) for pool in self.pools]  # 5×5, 9×9, 13×13
    return self.conv2(torch.cat(pool_outputs, dim=1))
```

Max-pooling at 3 different kernel sizes (5×5, 9×9, 13×13) lets the network "see" at different scales without adding any learnable parameters. This is placed at the very end of the backbone (after stage 4) so the model captures both fine local details and broad global context.

#### `CSPDarknet.forward(x)` — The Full Backbone Pass

```python
def forward(self, x):
    x = self.stem(x)       # (B, 3, 416, 416) → (B, 32, 208, 208)
    x = self.stage1(x)     # → (B, 64, 104, 104)
    p3 = self.stage2(x)    # → (B, 128, 52, 52)   ← stride 8
    p4 = self.stage3(p3)   # → (B, 256, 26, 26)   ← stride 16
    p5 = self.stage4(p4)   # → (B, 512, 13, 13)   ← stride 32
    return p3, p4, p5
```

Returns 3 feature maps at 3 different resolutions. **This is the only function the neck calls.**

#### `CSPDarknet._init_weights()` — Weight Initialization

```python
def _init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
```

Since we train from scratch (no pretrained weights), initialization matters a lot:
- **Kaiming normal** for Conv layers: Initializes weights so the variance of activations stays constant across layers. Without this, deep networks either explode (values → ∞) or vanish (values → 0).
- **BatchNorm**: weight=1, bias=0 means "start by doing nothing" — the identity transform.

---

## 4. `neck/fpn_panet.py` — The Bridge

The neck takes the 3 feature maps from the backbone and fuses information between them bidirectionally.

### Why Is This Needed?

| Feature Map | Spatial Detail | Semantic Understanding |
|-------------|---------------|----------------------|
| P3 (52×52) | High — knows exactly where edges are | Low — doesn't know what a "car" is |
| P5 (13×13) | Low — coarse spatial resolution | High — understands "car-shaped blob" |

The neck solves this mismatch by letting information flow both ways.

### Classes Defined

| Class | Lines | Purpose |
|-------|-------|---------|
| `FPNPANet` | 24–119 | Full bidirectional fusion neck |

### Important Functions

#### `FPNPANet.__init__()` — How It Reuses Backbone Building Blocks

```python
from models.backbone.csp_darknet import ConvBnAct, CSPBlock
```

The neck imports `ConvBnAct` and `CSPBlock` directly from the backbone file. This means the neck uses the **same** Conv→BN→SiLU pattern and the **same** cross-stage partial fusion blocks as the backbone. The only difference: neck CSP blocks use fewer bottlenecks (typically 1) since the neck's job is fusion, not heavy feature extraction.

#### `FPNPANet.forward(features)` — The Bidirectional Fusion

```python
def forward(self, features):
    p3, p4, p5 = features

    # ---- Top-down pathway (FPN) ----
    # P5 → reduce channels → upsample 2× → concat with P4 → CSP fuse
    p5_lateral = self.lateral_p5(p5)            # (B, 512→256, 13, 13)
    p5_up = self.upsample(p5_lateral)           # (B, 256, 26, 26)
    n4 = self.fpn_csp_p4(cat([p5_up, p4]))      # (B, 512→256, 26, 26)

    # N4 → reduce channels → upsample 2× → concat with P3 → CSP fuse
    n4_lateral = self.lateral_n4(n4)            # (B, 256→128, 26, 26)
    n4_up = self.upsample(n4_lateral)           # (B, 128, 52, 52)
    n3 = self.fpn_csp_p3(cat([n4_up, p3]))      # (B, 256→128, 52, 52)

    # ---- Bottom-up pathway (PANet) ----
    # N3 → downsample 2× → concat with N4 → CSP fuse
    n3_down = self.down_n3(n3)                  # (B, 128, 26, 26)
    f4 = self.pan_csp_p4(cat([n3_down, n4]))    # (B, 384→256, 26, 26)

    # F4 → downsample 2× → concat with P5 → CSP fuse
    f4_down = self.down_f4(f4)                  # (B, 256, 13, 13)
    f5 = self.pan_csp_p5(cat([f4_down, p5]))    # (B, 768→512, 13, 13)

    return n3, f4, f5
```

**Top-down (FPN)**: Semantic information flows from P5 (deep, semantic-rich) to P3 (shallow, detail-rich). After this, P3 knows "what" it's looking at, not just "where" edges are.

**Bottom-up (PANet)**: Spatial precision flows from N3 (high-res) back to F5 (low-res). After this, F5 has fine-grained position information, not just coarse "car-shaped blob."

**How upsample/downsample work**:
- **Upsample**: `nn.Upsample(scale_factor=2, mode='nearest')` — doubles spatial dimensions using nearest-neighbor interpolation (no learnable params)
- **Downsample**: `ConvBnAct(kernel_size=3, stride=2)` — halves spatial dimensions using a learned stride-2 convolution

**Why `torch.cat` instead of addition?** Concatenation preserves all channels from both sources. Addition forces the two tensors to have the same shape and merges their information destructively. Concatenation + CSP fusion gives the network freedom to decide how to combine them.

---

## 5. `detector/detection_head.py` — The Decision Maker

The head takes the 3 fused feature maps and produces actual predictions: "there's a car at position (x, y) with dimensions (w, h)."

### Classes Defined

| Class | Lines | Purpose |
|-------|-------|---------|
| `DecoupledHead` | 23–107 | Single-scale prediction head with separate branches |
| `DetectionHead` | 110–261 | Multi-scale wrapper + grid decoding |

### Important Functions

#### `DecoupledHead.forward(x)` — The Three-Branch Architecture

```python
def forward(self, x):
    x = self.stem(x)                    # shared 1×1 conv

    cls_feat = self.cls_conv(x)         # two 3×3 convs
    cls_out = self.cls_pred(cls_feat)   # 1×1 conv → (B, 4, H, W) class logits

    reg_feat = self.reg_conv(x)         # two 3×3 convs
    reg_out = self.reg_pred(reg_feat)   # 1×1 conv → (B, 4, H, W) box offsets
    obj_out = self.obj_pred(reg_feat)   # 1×1 conv → (B, 1, H, W) objectness

    return cls_out, reg_out, obj_out
```

**Why "decoupled"?** Classification ("is this a car or a truck?") and regression ("where exactly is this box?") are fundamentally different tasks. A single shared branch tries to optimize both simultaneously, causing conflicts. Decoupled branches let each task specialize independently. This was shown by the YOLOX paper to improve convergence speed.

**Why does objectness share the regression branch?** Objectness ("is there anything here at all?") is strongly correlated with spatial location — if the regression branch thinks there's a box here, the objectness should also be high. Sharing features between them saves computation.

#### `DecoupledHead._init_biases()` — Preventing Early False Positives

```python
def _init_biases(self):
    prior_prob = 0.01
    bias_value = -math.log((1 - prior_prob) / prior_prob)  # ≈ -4.595

    nn.init.constant_(self.cls_pred.bias, bias_value)
    nn.init.constant_(self.obj_pred.bias, bias_value)
```

At the start of training (random weights), every grid cell would predict ~50% probability of containing an object. With 3,549 grid cells, that's ~1,775 false positives per image. By setting the bias to `log(0.01 / 0.99) ≈ -4.6`, the initial sigmoid output is ~1%, so the model starts conservative and learns to "activate" cells only when it's confident.

#### `DetectionHead._make_grid(h, w, stride)` — The Coordinate System

```python
def _make_grid(self, h, w, stride, device, dtype):
    yv, xv = torch.meshgrid(
        torch.arange(h), torch.arange(w), indexing='ij'
    )
    grid = torch.stack([xv, yv], dim=-1)  # (H, W, 2)
    return grid.unsqueeze(0).unsqueeze(0)
```

Creates a grid of (x, y) coordinates for each cell in the feature map. For a 52×52 feature map at stride 8, cell (3, 7) corresponds to pixel position (3×8, 7×8) = (24, 56) in the original image.

This grid is **precomputed once** and reused for every image. It's lazily initialized on the first forward pass (`self.grids[i] = None` initially).

#### `DetectionHead.decode_predictions()` — From Network Output to Real Boxes

```python
def decode_predictions(self, cls_out, reg_out, obj_out, stride, grid):
    # Center: sigmoid constrains offset to [-0.5, 1.5], then add grid position
    xy = (reg_out[..., :2].sigmoid() * 2 - 0.5 + grid) * stride

    # Size: sigmoid constrains to [0, 2], square it, then scale by stride
    wh = (reg_out[..., 2:4].sigmoid() * 2) ** 2 * stride

    # Confidence: sigmoid to [0, 1]
    obj_conf = obj_out.sigmoid()
    cls_conf = cls_out.sigmoid()

    return torch.cat([xy, wh, obj_conf, cls_conf], dim=-1)
```

The network doesn't predict absolute pixel coordinates — it predicts **offsets** relative to each grid cell. This function converts them:

| Network Output | Transformation | Final Value |
|---------------|---------------|-------------|
| x_offset (raw) | `(sigmoid(x) * 2 - 0.5 + grid_x) * stride` | Pixel x-coordinate of box center |
| y_offset (raw) | `(sigmoid(y) * 2 - 0.5 + grid_y) * stride` | Pixel y-coordinate of box center |
| w_pred (raw) | `(sigmoid(w) * 2)² * stride` | Box width in pixels |
| h_pred (raw) | `(sigmoid(h) * 2)² * stride` | Box height in pixels |
| obj (raw) | `sigmoid(obj)` | Confidence [0, 1] |
| cls (raw) | `sigmoid(cls)` | Per-class probability [0, 1] |

#### `DetectionHead.forward(features)` — Training vs. Inference

```python
def forward(self, features, targets=None):
    for i, (feat, head) in enumerate(zip(features, self.heads)):
        cls_out, reg_out, obj_out = head(feat)
        # ...

    if self.training:
        # Return raw logits — the loss function needs unprocessed outputs
        return {
            'cls': outputs_cls,    # 3 tensors: [(B,4,52,52), (B,4,26,26), (B,4,13,13)]
            'reg': outputs_reg,    # 3 tensors: [(B,4,52,52), (B,4,26,26), (B,4,13,13)]
            'obj': outputs_obj,    # 3 tensors: [(B,1,52,52), (B,1,26,26), (B,1,13,13)]
            'strides': [8, 16, 32],
        }
    else:
        # Decode to absolute coordinates and concatenate all scales
        return torch.cat(decoded_outputs, dim=1)  # (B, 3549, 9)
```

**Why two modes?**
- **Training**: The loss function (`utils/losses.py`) needs the raw logits (before sigmoid) to use `BCEWithLogitsLoss`, which is numerically more stable than applying sigmoid first and then computing cross-entropy.
- **Inference**: The user needs actual bounding boxes in pixel coordinates. Decoding + sigmoid are applied, and all 3 scales are concatenated into a single tensor of 3,549 predictions.

---

## 6. `detector/yolo_detector.py` — The Full Assembly

This file is the **only file** that external code needs to import. It wires backbone + neck + head into one `nn.Module`.

### Classes Defined

| Class | Lines | Purpose |
|-------|-------|---------|
| `ATMSDetector` | 31–144 | Full model: image → detections |
| `ModelEMA` | 147–197 | Exponential Moving Average of model weights |

### Important Functions

#### `ATMSDetector.__init__()` — Wiring the Components

```python
def __init__(self, num_classes=4, depth_mul=0.33, width_mul=0.5, in_channels=3):
    self.backbone = CSPDarknet(in_channels, depth_mul, width_mul)
    self.neck = FPNPANet(self.backbone.out_channels, depth_mul)
    self.head = DetectionHead(self.neck.out_channels, num_classes, strides=(8,16,32))
```

Three lines of code wire the entire architecture together. The key contract: each component exposes `.out_channels` so the next one knows the expected input shape.

#### `ATMSDetector.forward(x)` — The One-Line-Per-Component Design

```python
def forward(self, x):
    features = self.backbone(x)     # → (P3, P4, P5)
    fused = self.neck(features)     # → (N3, F4, F5)
    output = self.head(fused)       # → predictions
    return output
```

This is the entire inference pipeline in 3 lines. Each line calls exactly one component.

#### `ATMSDetector.from_config(config_path)` — YAML-Driven Construction

```python
@classmethod
def from_config(cls, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model_cfg = config.get('model', {})
    return cls(
        num_classes=model_cfg.get('num_classes', 4),
        depth_mul=model_cfg.get('depth_mul', 0.33),
        width_mul=model_cfg.get('width_mul', 0.5),
    )
```

Creates a model from a YAML config file. This is how `train_detector.py` builds the model — it reads `configs/detector.yaml` and constructs the architecture accordingly.

#### `ModelEMA.update(model)` — Smoothing Weights During Training

```python
def update(self, model):
    self.updates += 1
    d = self.decay * (1 - math.exp(-self.updates / 2000))  # ramp up decay

    for name, ema_param in self.ema.named_parameters():
        if name in model_params:
            ema_param.data.mul_(d).add_(model_params[name].data, alpha=1 - d)
```

EMA maintains a "shadow" copy of the model where each weight is a running exponential average:

```
ema_weight = 0.9999 × ema_weight + 0.0001 × current_weight
```

This produces a smoother model that's less sensitive to noisy gradients in individual batches. The EMA model is used for **validation and inference** while the raw model is used for training.

The decay ramps up from 0 to 0.9999 over the first ~2000 updates. Early on (updates near 0), the EMA is essentially a copy of the current model. Later, it becomes a very smooth average.

---

## 7. How the Model Connects to the Rest of the Project

### Who Imports from `models/`

```
scripts/train_detector.py
    │
    ├── from models.detector.yolo_detector import ATMSDetector, ModelEMA
    │   └── Used to: create the model, create the EMA copy
    │
    └── (Everything else is accessed through ATMSDetector internally)

utils/losses.py
    │
    └── Does NOT import from models/
        The loss function receives the output dict from DetectionHead.forward()
        It only needs the raw tensors — no model knowledge required

kaggle/atms_net_phase1_training.ipynb
    │
    └── from models.detector.yolo_detector import ATMSDetector
        Used for: loading the trained checkpoint and running inference
```

### What the Model Receives and Returns

**Input** (from `data/coco/coco_dataset.py`):
```python
images: Tensor of shape (B, 3, 416, 416)  # B images, RGB, 416×416 pixels
        values in [0, 1]                    # normalized by dividing by 255
```

**Output during training** (consumed by `utils/losses.py`):
```python
{
    'cls': [                               # class logits per scale
        Tensor(B, 4, 52, 52),              # scale 0 — stride 8
        Tensor(B, 4, 26, 26),              # scale 1 — stride 16
        Tensor(B, 4, 13, 13),              # scale 2 — stride 32
    ],
    'reg': [                               # box offset predictions per scale
        Tensor(B, 4, 52, 52),
        Tensor(B, 4, 26, 26),
        Tensor(B, 4, 13, 13),
    ],
    'obj': [                               # objectness logits per scale
        Tensor(B, 1, 52, 52),
        Tensor(B, 1, 26, 26),
        Tensor(B, 1, 13, 13),
    ],
    'strides': [8, 16, 32],
}
```

**Output during inference** (consumed by `utils/nms.py`):
```python
Tensor(B, 3549, 9)
# 3549 predictions = 52² + 26² + 13²
# 9 values per prediction = [cx, cy, w, h, obj_conf, car, motorcycle, bus, truck]
# cx, cy, w, h are in absolute pixel coordinates (0–416)
# obj_conf and class scores are in [0, 1] after sigmoid
```

---

## 8. Quick Reference — All Important Functions

### `backbone/csp_darknet.py`

| Function | What It Does | Called By |
|----------|-------------|-----------|
| `ConvBnAct.forward(x)` | Conv2d → BatchNorm → SiLU | Everything (backbone, neck, head) |
| `Bottleneck.forward(x)` | 1×1 conv → 3×3 conv + skip connection | CSPBlock |
| `CSPBlock.forward(x)` | Split channels → bottleneck path + direct path → concat → fuse | CSPDarknet stages, FPNPANet CSP blocks |
| `SPPBlock.forward(x)` | Multi-scale max pooling (5×5, 9×9, 13×13) → concat | CSPDarknet stage 4 |
| `CSPDarknet.forward(x)` | Full backbone: image → (P3, P4, P5) | ATMSDetector.forward() |
| `CSPDarknet._init_weights()` | Kaiming normal for convs, identity for BN | CSPDarknet.__init__() |

### `neck/fpn_panet.py`

| Function | What It Does | Called By |
|----------|-------------|-----------|
| `FPNPANet.forward(features)` | Top-down FPN + Bottom-up PANet fusion → (N3, F4, F5) | ATMSDetector.forward() |

### `detector/detection_head.py`

| Function | What It Does | Called By |
|----------|-------------|-----------|
| `DecoupledHead.forward(x)` | Feature → (cls_logits, box_offsets, objectness) | DetectionHead.forward() |
| `DecoupledHead._init_biases()` | Sets initial sigmoid output to ~1% (prevents false positives) | DecoupledHead.__init__() |
| `DetectionHead.forward(features)` | Training: returns raw dict. Inference: returns decoded boxes | ATMSDetector.forward() |
| `DetectionHead._make_grid(h, w, stride)` | Creates (x, y) coordinate grid for decoding | DetectionHead.forward() (lazy) |
| `DetectionHead.decode_predictions(...)` | Raw offsets → absolute pixel coordinates | DetectionHead.forward() (eval mode) |

### `detector/yolo_detector.py`

| Function | What It Does | Called By |
|----------|-------------|-----------|
| `ATMSDetector.forward(x)` | Full pipeline: image → backbone → neck → head → output | train_detector.py |
| `ATMSDetector.from_config(path)` | Build model from YAML config file | train_detector.py |
| `ATMSDetector.summary()` | Print param count, size, architecture info | train_detector.py, Kaggle notebook |
| `ATMSDetector.get_param_count()` | Returns (total_params, trainable_params) | ATMSDetector.summary() |
| `ModelEMA.update(model)` | Update EMA weights: shadow = 0.9999 × shadow + 0.0001 × current | train_detector.py (after each batch) |

---

*This document covers the `models/` folder as of Phase 1. Future phases will add modules here (e.g., emergency vehicle detection head in Phase 3).*
