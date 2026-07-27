# ATMS-Net — Dataset Analysis & Kaggle Training Guide

## Part 1: Is MS COCO 2017 The Best Option?

### Short Answer

**Yes, MS COCO 2017 is the right choice for Phase 1 — but it's not the only good option.** Here's a breakdown:

---

### Your Current Setup

Your [download_coco.py](file:///Users/devanshkhosla/Projects/ATMS-Net/data/coco/download_coco.py) filters COCO 2017 to **4 vehicle classes**:

| COCO ID | COCO Label | Your Class ID |
|---------|-----------|---------------|
| 3 | car | 0 |
| 4 | motorcycle | 1 |
| 6 | bus | 2 |
| 8 | truck | 3 |

This gives you **~40,000 images** with **~120,000 vehicle annotations** (after filtering).

---

### Dataset Comparison for Phase 1 (Vehicle Detection Backbone Training)

| Dataset | Images | Vehicle Annotations | Scene Type | Viewpoint | Free on Kaggle? | Verdict |
|---------|--------|-------------------|------------|-----------|-----------------|---------|
| **MS COCO 2017** (current) | ~40K (filtered) | ~120K | Mixed (streets, parking, highways) | Mixed angles | ✅ Yes | ✅ **Best for Phase 1** |
| **KITTI** | 7,481 | ~40K | Highway/urban | Dashboard cam | ✅ Yes | ⚠️ Too small, single viewpoint |
| **BDD100K** | 100K | ~1.8M | Driving scenes | Dashboard cam | ✅ Yes | ⚠️ Great size, but wrong viewpoint for intersections |
| **UA-DETRAC** | 140K frames | ~1.2M | Intersections | Overhead/elevated | ✅ Yes | 🔥 **Best for Phase 2 fine-tuning** (already in your plan) |
| **VisDrone** | 10K | ~54K | Aerial drone | Top-down | ✅ Yes | ⚠️ Too different from intersection cameras |
| **Cityscapes** | 5K | ~30K | Urban streets | Dashboard cam | ❌ No (needs registration) | ❌ Small, segmentation-focused |
| **nuScenes** | 40K | ~1.4M | Driving scenes | Multi-camera | ❌ No | ⚠️ 3D-focused, overkill for 2D detection |

---

### Why COCO 2017 Is Right for Phase 1

> [!TIP]
> **COCO is the correct foundation because Phase 1 is about teaching the backbone *what* vehicles look like, not *where* they appear in intersections. That's Phase 2's job (UA-DETRAC fine-tuning).**

1. **Diversity of scenes and angles** — COCO has vehicles in parking lots, highways, city streets, varied weather. This forces the backbone to learn robust features, not overfit to one camera angle.

2. **High-quality annotations** — COCO annotations are human-verified with bounding box quality checks. Many traffic-specific datasets have noisier labels.

3. **4 of your exact target classes exist** — car, motorcycle, bus, truck are native COCO categories. No label remapping gymnastics needed.

4. **Available on Kaggle natively** — `coco-2017-dataset` is a first-party Kaggle dataset. You can symlink it instead of downloading 20GB.

5. **Standard benchmark** — If your model gets mAP@0.5 > 75% on COCO vehicles, that's a meaningful, publishable number.

---

### What Would Be *Better*?

If you wanted to squeeze more performance, you could **combine COCO with one of these during Phase 1**:

| Strategy | Dataset Addition | Benefit | Effort |
|----------|-----------------|---------|--------|
| **COCO only** (current plan) | None | Simple, proven | ✅ Zero |
| **COCO + Open Images V7** | +200K vehicle images | More data → better features | 🟡 Medium — need to write OID label converter |
| **COCO + BDD100K** | +100K driving images | Dashboard cam exposure early | 🟡 Medium — BDD uses different label format |

> [!IMPORTANT]
> **My recommendation: Stick with COCO for Phase 1.** Your training pipeline ([train_detector.py](file:///Users/devanshkhosla/Projects/ATMS-Net/scripts/train_detector.py)), dataset class ([coco_dataset.py](file:///Users/devanshkhosla/Projects/ATMS-Net/data/coco/coco_dataset.py)), and loss function ([losses.py](file:///Users/devanshkhosla/Projects/ATMS-Net/utils/losses.py)) are already built and tested for this exact setup. Train first, evaluate, **then** decide if you need more data in Phase 2.

---

## Part 2: Kaggle Training Guide — Step by Step

You already have a working notebook at [atms_net_phase1_training.ipynb](file:///Users/devanshkhosla/Projects/ATMS-Net/kaggle/atms_net_phase1_training.ipynb). Here's the complete walkthrough to get it running:

---

### Prerequisites

Before you start:
- A **Kaggle account** (free at [kaggle.com](https://www.kaggle.com))
- Your ATMS-Net repo **pushed to GitHub** (it's already at `CodeNebula-Dev/Adaptive-Traffic-Management-System-with-Emergency-Preemption`)
- You get **30 hours/week** of free GPU time on Kaggle

---

### Step 1: Upload the Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"+ New Notebook"**
3. Click **File → Import Notebook**
4. Upload `kaggle/atms_net_phase1_training.ipynb` from your local project

![Screenshot placeholder — Kaggle new notebook](https://www.kaggle.com/code)

---

### Step 2: Configure GPU & Internet

This is the most critical step. Without GPU, training takes **days** instead of **hours**.

1. Click the **⚙ Settings** panel (right sidebar)
2. Under **Accelerator** → select **GPU T4 x2**
   - If T4 is unavailable, **GPU P100** also works
3. Under **Internet** → toggle **On**
   - Required to clone your repo and download COCO
4. Under **Persistence** → select **Files only** (saves your checkpoints even if kernel restarts)

> [!WARNING]
> If you forget to enable Internet, the `git clone` and `pip install` cells will fail silently or timeout.

---

### Step 3: Add COCO as a Kaggle Dataset (Optional but Recommended)

This **saves 15-20 minutes** by skipping the 20GB COCO download:

1. In your notebook sidebar, click **"+ Add Data"**
2. Search for **"COCO 2017"**
3. Add the dataset by `awsaf49` or the official COCO 2017 dataset
4. It will mount at `/kaggle/input/coco-2017-dataset/`
5. **Uncomment** the alternative cell **2.1b** in the notebook and **comment out** cell 2.1

If you skip this, cell 2.1 will download COCO from scratch (works fine, just slower).

---

### Step 4: Run the Notebook

Click **Run All** (or run cells sequentially). Here's what each section does:

| Section | What It Does | Time |
|---------|-------------|------|
| **1. Environment Setup** | Verifies GPU, clones repo, installs dependencies | ~2 min |
| **2. Dataset Preparation** | Downloads/links COCO, filters to vehicles, creates YOLO labels | ~5–15 min |
| **3. Training Configuration** | Creates Kaggle-optimized config (batch_size=32, FP16) | Instant |
| **4. Training** | Runs 50 epochs of training | **~2–3 hours** |
| **5. Results & Evaluation** | Plots loss curves, runs inference on sample images | ~5 min |
| **6. Export & Download** | Packages `best.pt` for download | ~1 min |

---

### Step 5: Monitor Training

While training runs (Section 4), you'll see a progress bar like:

```
Epoch 1: 100%|██████████| 625/625 [03:45<00:00, loss: 0.2341, box: 0.0412, obj: 0.1523, cls: 0.0406, lr: 0.001000]

  Epoch 1/50 (225.1s) — loss: 0.2341 [box: 0.0412, obj: 0.1523, cls: 0.0406]
  Val mAP@0.5: 0.0523  |  mAP@0.5:0.95: 0.0212
  Per-class AP@0.5: car=0.068  motorcycle=0.032  bus=0.051  truck=0.039
```

> [!NOTE]
> **Don't panic if early epoch mAP is very low (~5-10%).** The model is training from scratch (random weights). Meaningful detection usually kicks in around epoch 10-15. By epoch 30-40, you should see mAP@0.5 > 50%.

**Expected progression:**

| Epoch | Approximate mAP@0.5 | What's Happening |
|-------|---------------------|-----------------|
| 1–5 | 2–10% | Warmup phase, model learning basic features |
| 5–15 | 10–30% | Starting to detect large vehicles (buses, trucks) |
| 15–30 | 30–55% | Cars and motorcycles start being detected |
| 30–50 | 55–80% | Refinement, small object detection improves |

---

### Step 6: Download Your Trained Model

After training completes:

1. Go to the **Output** tab (right sidebar)
2. Download `atms_net_phase1_trained.zip`
3. Extract and place `best.pt` in your local project:

```bash
# On your Mac
cp ~/Downloads/atms_net_phase1_trained/best.pt /Users/devanshkhosla/Projects/ATMS-Net/checkpoints/best.pt
```

---

### Troubleshooting Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `No GPU detected` | Accelerator not set | Settings → Accelerator → GPU T4 x2 |
| `git clone` fails | Internet not enabled | Settings → Internet → On |
| `CUDA out of memory` | Batch size too large | Reduce `batch_size` from 32 to 16 |
| Notebook kernel restarts mid-training | Session timeout (6-hour limit on GPU) | Resume from `last.pt` — add `--resume /kaggle/working/checkpoints/last.pt` |
| Training very slow (~20+ min/epoch) | Running on CPU | Verify GPU is enabled; check first cell output |
| `ModuleNotFoundError` for utils | sys.path not set | Make sure cell 1.4 ran successfully |
| mAP stays at 0% after 10 epochs | Bug in data pipeline | Check cell 2.2 — verify image count is ~40K |

---

### Kaggle Session Limits

> [!CAUTION]
> Kaggle has a **6-hour session limit** for GPU notebooks and a **30-hour weekly GPU quota**. If training gets interrupted:
> 1. Your `last.pt` checkpoint is saved after every epoch
> 2. Re-run the notebook and modify cell 4.1 to add `--resume /kaggle/working/checkpoints/last.pt`

---

### After Training: What's Next?

Once you have `best.pt` with mAP@0.5 > 70%:

| Phase | What to Do | Dataset |
|-------|-----------|---------|
| **Phase 2** | Fine-tune on intersection footage | UA-DETRAC + CARLA |
| **Phase 3** | Train emergency vehicle head | HERO dataset |
| **Phase 4** | Train RL signal controller | SUMO simulator |
| **Phase 5** | Full system integration test | SUMO + synthetic EV events |

Your complete training pipeline from [detector.yaml](file:///Users/devanshkhosla/Projects/ATMS-Net/configs/detector.yaml) is already configured for this. The Kaggle notebook handles the GPU-intensive Phase 1 — everything after that can likely run on your Mac or a smaller GPU.
