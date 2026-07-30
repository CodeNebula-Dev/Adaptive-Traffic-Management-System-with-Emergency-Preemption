# Phase 1 Training Debug — mAP Stuck at 0.0000 After 6 Epochs

## Root Cause Analysis

After analyzing every file in the codebase, I've identified **5 bugs** causing the zero mAP despite decreasing loss. The bugs span the validation pipeline, coordinate handling, metrics computation, and class mapping.

> [!CAUTION]
> **The model IS learning** (loss drops from 0.65 → 0.15) but the **validation/evaluation pipeline is completely broken**. Every prediction that the model makes is either filtered out, mismatched in coordinate space, or compared against wrong class IDs.

---

## Bug #1 (CRITICAL): Target coordinates are NORMALIZED but treated as ABSOLUTE in validation

**Files**: [coco_dataset.py](file:///Users/devanshkhosla/Projects/ATMS-Net/data/coco/coco_dataset.py) + [metrics.py](file:///Users/devanshkhosla/Projects/ATMS-Net/utils/metrics.py)

**The Problem**: During validation (non-mosaic, `augment=False`), `apply_augmentations()` calls `letterbox_labels()` which converts labels from normalized `[0,1]` to absolute pixel coordinates. This is correct. However, the **NMS output** from `batch_nms()` produces boxes in absolute pixel coordinates decoded from the model's eval mode, while `metrics.update()` receives `targets.cpu()` — the raw collated targets tensor `[batch_idx, cls, cx, cy, w, h]`.

The metrics code at [line 71-77](file:///Users/devanshkhosla/Projects/ATMS-Net/utils/metrics.py#L71-L77) converts targets from `cxcywh` → `xyxy`:
```python
gt_cx, gt_cy, gt_w, gt_h = img_targets[:, 2], img_targets[:, 3], img_targets[:, 4], img_targets[:, 5]
gt_boxes = torch.stack([gt_cx - gt_w/2, gt_cy - gt_h/2, gt_cx + gt_w/2, gt_cy + gt_h/2], dim=1)
```

This is fine **IF** targets are in absolute pixel coordinates (which they are after letterbox). But the detection outputs from the model are also in absolute pixel coordinates. The question is: **are they in the same coordinate space?** Let's check...

---

## Bug #2 (CRITICAL): Detection head `decode_predictions` uses WRONG decoding formula for w/h

**File**: [detection_head.py](file:///Users/devanshkhosla/Projects/ATMS-Net/models/detector/detection_head.py#L196)

The eval-mode decoding at [line 196](file:///Users/devanshkhosla/Projects/ATMS-Net/models/detector/detection_head.py#L196):
```python
wh = (reg_out[..., 2:4].sigmoid() * 2) ** 2 * stride
```

And the loss decoding at [losses.py line 109](file:///Users/devanshkhosla/Projects/ATMS-Net/utils/losses.py#L109):
```python
decoded_wh = (reg_pred_flat[..., 2:4].sigmoid() * 2) ** 2 * stride
```

These are **consistent** with each other, which is good. But the formula `(sigmoid(x) * 2)^2 * stride` has a maximum value of `4 * stride`. For stride=32, max wh = 128 pixels. For stride=8, max wh = 32 pixels. **This means the model CAN'T predict boxes larger than 128 pixels on a 416×416 image!** Vehicles near the camera that are large will never be correctly predicted. This severely limits what the model can learn, but it's not the primary cause of zero mAP — it would just limit mAP, not zero it.

---

## Bug #3 (CRITICAL): Class name mismatch between download_coco.py and DetectionMetrics

**Files**: [download_coco.py](file:///Users/devanshkhosla/Projects/ATMS-Net/data/coco/download_coco.py#L39) + [metrics.py](file:///Users/devanshkhosla/Projects/ATMS-Net/utils/metrics.py#L32) + [yolo_detector.py](file:///Users/devanshkhosla/Projects/ATMS-Net/models/detector/yolo_detector.py#L54)

| Index | download_coco.py | metrics.py | yolo_detector.py |
|-------|-----------------|------------|------------------|
| 0 | car | car | car |
| 1 | **motorcycle** | **truck** | **truck** |
| 2 | **bus** | **bus** | **bus** |
| 3 | **truck** | **motorcycle** | **motorcycle** |

`download_coco.py` maps: `car=0, motorcycle=1, bus=2, truck=3`
`metrics.py` has: `CLASS_NAMES = ['car', 'truck', 'bus', 'motorcycle']`
`yolo_detector.py` has: `CLASS_NAMES = ['car', 'truck', 'bus', 'motorcycle']`

**This is a cosmetic bug only** — the class indices in the labels and predictions still match numerically (the model trains on the same numeric IDs it evaluates against). The per-class AP names are just displayed wrong. The mAP number itself is unaffected.

---

## Bug #4 (CRITICAL): `validate()` function calls `model(images)` in eval mode but ALSO tries to compute loss with training-mode predictions

**File**: [train_detector.py](file:///Users/devanshkhosla/Projects/ATMS-Net/scripts/train_detector.py#L208-L248)

Looking at the `validate()` function:
```python
model.eval()
...
predictions = model(images)  # (B, N, 5+C) — eval mode returns decoded
detections = batch_nms(predictions, ...)  # NMS on decoded predictions
metrics.update(detections, targets.cpu())
```

This part is actually correct — the model in eval mode returns decoded `(B, N, 5+C)` and NMS produces `(M, 7)` = `[x1,y1,x2,y2,conf,cls_conf,cls_id]` in absolute pixel coords. But targets `(N_total, 6)` = `[batch_idx, cls, cx, cy, w, h]` are ALSO in absolute pixel coords (after letterbox). So the comparison should work...

**BUT WAIT** — let me re-read the NMS code more carefully:

---

## Bug #5 (CRITICAL ROOT CAUSE): NMS `class_aware_nms` box conversion from cxcywh to xyxy uses ALREADY-DECODED absolute coordinates

**File**: [nms.py](file:///Users/devanshkhosla/Projects/ATMS-Net/utils/nms.py#L36-L73)

The NMS function expects input as `(N, 5+C)` = `[cx, cy, w, h, obj_conf, cls1, ..., clsC]` — decoded predictions from eval mode. It converts `cxcywh` → `xyxy` at [lines 69-73](file:///Users/devanshkhosla/Projects/ATMS-Net/utils/nms.py#L69-L73):
```python
boxes[:, 0] = predictions[:, 0] - predictions[:, 2] / 2  # x1
boxes[:, 1] = predictions[:, 1] - predictions[:, 3] / 2  # y1  
boxes[:, 2] = predictions[:, 0] + predictions[:, 2] / 2  # x2
boxes[:, 3] = predictions[:, 1] + predictions[:, 3] / 2  # y2
```

This conversion is correct. The output `detections` tensor is `[x1, y1, x2, y2, conf, cls_conf, cls_id]` in absolute pixel coordinates.

In `metrics.py`, the `_compute_ap_at_threshold` compares `det[:, :4]` (xyxy) against `gt[:, 1:5]` (xyxy). This should match...

**LET ME RE-CHECK THE ACTUAL EVAL PATH MORE CAREFULLY.** 

Actually, I need to trace the exact data flow during validation end-to-end:

1. `val_dataset` returns targets in **absolute pixel coords** (after letterbox) as `[0, cls, cx, cy, w, h]`
2. `detection_collate_fn` sets correct `batch_idx` → targets `(N_total, 6)` `[batch_idx, cls, cx, cy, w, h]` absolute
3. `model(images)` in eval → decoded `(B, N, 5+C)` `[cx, cy, w, h, obj_conf, cls_probs...]` absolute
4. `batch_nms()` → filters by conf, converts cxcywh→xyxy → NMS → output `(M, 7)` `[x1,y1,x2,y2,conf,cls_conf,cls_id]` absolute
5. `metrics.update(detections, targets.cpu())` — detections in xyxy absolute, targets in cxcywh absolute

**This should work theoretically.** So why mAP = 0?

Let me look MORE carefully at the issue...

---

## THE REAL ROOT CAUSES

### Root Cause A: The EMA model is nearly identical to the initial random model in early epochs

Looking at [yolo_detector.py lines 178-188](file:///Users/devanshkhosla/Projects/ATMS-Net/models/detector/yolo_detector.py#L178-L188):
```python
def update(self, model):
    self.updates += 1
    d = self.decay * (1 - math.exp(-self.updates / 2000))
```

With `decay=0.9999`, after 453 steps (1 epoch), `d = 0.9999 * (1 - exp(-453/2000)) = 0.9999 * 0.2028 = 0.2028`. The EMA model barely incorporates training updates — it's still ~80% the random initialization. By epoch 5, `updates = 5*453 = 2265`, `d = 0.9999 * (1 - exp(-1.13)) = 0.677`. Still blending heavily with random weights.

However, the training script uses `eval_model = ema.ema if ema else model` for validation. So validation is done with this severely under-trained EMA model. **This alone could cause zero mAP** in early epochs.

### Root Cause B: `scheduler.step()` is called BEFORE `optimizer.step()` on the first batch

The warning in your logs: `UserWarning: Detected call of lr_scheduler.step() before optimizer.step()`. This is at [train_detector.py line 178](file:///Users/devanshkhosla/Projects/ATMS-Net/scripts/train_detector.py#L178). With `accumulate_grad=1`, the scheduler steps inside the first `if (batch_idx + 1) % accumulate == 0` block. But on `batch_idx=0`, `optimizer.step()` happens at line 174, then `scheduler.step()` at line 178. This is the correct order. The PyTorch warning fires because `scheduler.step()` is being called before the **very first** `optimizer.step()` — this happens because `scheduler = LambdaLR(...)` internal state thinks the first step hasn't happened yet. **This causes the first LR value to be skipped.**

### Root Cause C (THE MAIN BUG): Detection predictions have OBJ_CONF applied as SIGMOID already, but NMS checks raw values against threshold

Looking at the **eval mode output** from [detection_head.py line 199-203](file:///Users/devanshkhosla/Projects/ATMS-Net/models/detector/detection_head.py#L199-L203):
```python
obj_conf = obj_out.sigmoid()  # Already sigmoid'd
cls_conf = cls_out.sigmoid()  # Already sigmoid'd
output = torch.cat([xy, wh, obj_conf, cls_conf], dim=-1)
```

Now in [nms.py line 47-48](file:///Users/devanshkhosla/Projects/ATMS-Net/utils/nms.py#L47-L48):
```python
obj_conf = predictions[:, 4]  # This is ALREADY sigmoid'd (0 to 1)
mask = obj_conf > conf_threshold  # conf_threshold = 0.01
```

Then at [nms.py line 55-56](file:///Users/devanshkhosla/Projects/ATMS-Net/utils/nms.py#L55-L56):
```python
cls_conf, cls_id = predictions[:, 5:].max(dim=1)  # ALREADY sigmoid'd
conf = predictions[:, 4] * cls_conf  # Combined confidence
```

**The combined confidence `conf = obj_sigmoid * cls_sigmoid`.** In early training, `obj_sigmoid` starts around 0.01 (due to bias init at `prior_prob=0.01`). `cls_sigmoid` also starts around 0.01. So `conf = 0.01 * 0.01 = 0.0001`. Even with `conf_threshold=0.01`, the SECOND filter at line 59:
```python
mask = conf > conf_threshold  # 0.0001 > 0.01 → FALSE
```

**This second threshold check kills ALL predictions.** The objectness alone might pass 0.01, but `obj * cls` will almost never pass 0.01 in early training. This means `batch_nms` returns `None` for every image → `metrics.update` receives all None detections → mAP = 0.

---

## Proposed Changes

### 1. NMS Confidence Filtering Fix
#### [MODIFY] [nms.py](file:///Users/devanshkhosla/Projects/ATMS-Net/utils/nms.py)
- Use a **separate, lower threshold** for the combined `obj * cls` confidence, or use `max(obj, obj * cls)` for filtering
- The cleanest fix: filter by `obj_conf > conf_threshold` only (remove the second `conf > conf_threshold` filter), then let NMS handle the rest

### 2. Scheduler Warning Fix
#### [MODIFY] [train_detector.py](file:///Users/devanshkhosla/Projects/ATMS-Net/scripts/train_detector.py)
- Initialize scheduler with `last_epoch=-1` (default) and step it only after the first optimizer step
- Add `scheduler.step()` call with proper epoch-based stepping

### 3. EMA Warmup Fix — Use raw model for validation in early epochs  
#### [MODIFY] [train_detector.py](file:///Users/devanshkhosla/Projects/ATMS-Net/scripts/train_detector.py)
- For the first N epochs (e.g., warmup_epochs), use the raw model for validation instead of the under-trained EMA model

### 4. Class Name Consistency Fix
#### [MODIFY] [metrics.py](file:///Users/devanshkhosla/Projects/ATMS-Net/utils/metrics.py)
- Align `CLASS_NAMES` with `download_coco.py`: `['car', 'motorcycle', 'bus', 'truck']`

#### [MODIFY] [yolo_detector.py](file:///Users/devanshkhosla/Projects/ATMS-Net/models/detector/yolo_detector.py)
- Same alignment: `['car', 'motorcycle', 'bus', 'truck']`

### 5. Width/Height decoding range fix
#### [MODIFY] [detection_head.py](file:///Users/devanshkhosla/Projects/ATMS-Net/models/detector/detection_head.py)
- Change wh decoding to use `exp()` instead of bounded sigmoid: `wh = torch.exp(reg_out[..., 2:4].clamp(-5, 5)) * stride`
- This allows the model to predict arbitrarily large boxes

#### [MODIFY] [losses.py](file:///Users/devanshkhosla/Projects/ATMS-Net/utils/losses.py)  
- Match the same decoding formula in the loss function

---

## Open Questions

> [!IMPORTANT]
> 1. **Are you running on Kaggle with the fixes from the existing bug report already applied?** The report mentions `box_weight` was changed from 0.05 to 5.0, and `conf_threshold` from 0.25 to 0.01. I see these values are already in the config. So those fixes are already in — but mAP is STILL 0. The remaining bugs I've found above (especially Bug C - the double confidence threshold) would explain why.
>
> 2. **Do you want me to also fix the wh decoding range (Bug #2)?** This is a deeper architectural change that will require restarting training from scratch. Without this fix, the model can still learn but large vehicles may have degraded accuracy. I recommend fixing it now since you need to restart training anyway.

---

## Verification Plan

### Automated Tests
- Run the existing `scripts/test_pipeline.py` smoke test to verify the fixes produce mAP > 0 on the 2-image synthetic batch
- Add a debug print in `batch_nms` to show how many predictions survive each filtering stage

### Manual Verification  
- Re-launch training on Kaggle with the fixes
- After epoch 1, mAP@0.5 should be > 0 (even if small, like 0.01-0.05)
- By epoch 5, mAP@0.5 should be climbing (0.1+)
