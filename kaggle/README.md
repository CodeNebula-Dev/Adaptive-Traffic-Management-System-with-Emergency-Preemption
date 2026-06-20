# ATMS-Net — Kaggle Training Guide

## Overview

This folder contains the Kaggle notebook for training the ATMS-Net custom YOLO-style
vehicle detector on a **free Tesla T4 GPU** using the **MS COCO 2017** dataset.

## Why Kaggle?

Training a deep neural network with 13.2M parameters on a CPU (even an M4 Mac) would
take days. Kaggle provides **30 hours/week of free GPU** (NVIDIA Tesla T4, 16GB VRAM)
which can finish Phase 1 training in a few hours.

## Files

| File | Description |
|------|-------------|
| `atms_net_phase1_training.ipynb` | Complete Jupyter notebook for Kaggle GPU training |
| `README.md` | This guide |

## How to Use

### Step 1: Upload to Kaggle
1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"+ New Notebook"**
3. Click **File → Import Notebook** and upload `atms_net_phase1_training.ipynb`

### Step 2: Enable GPU
1. In the notebook, click **Settings** (gear icon on the right)
2. Under **Accelerator**, select **GPU T4 x2** (or **GPU P100**)
3. Set **Internet** to **On** (needed to download COCO dataset)

### Step 3: Run All Cells
Click **Run All** or run cells one by one. The notebook will:
1. Clone the ATMS-Net repo from GitHub
2. Install dependencies
3. Download and prepare the COCO vehicle subset
4. Train the detector for 50 epochs on GPU
5. Save the best model checkpoint

### Step 4: Download Trained Model
After training, download `checkpoints/best.pt` from the notebook output
and place it in your local project's `checkpoints/` folder.

## Expected Training Time

| Setting | Time |
|---------|------|
| 50 epochs, batch_size=16, T4 GPU | ~2-3 hours |
| 50 epochs, batch_size=32, T4 GPU | ~1.5-2 hours |

## Expected Results

After 50 epochs on the COCO vehicle subset:
- **mAP@0.5**: 70-80% (target: >75%)
- **Per-class AP**: car ~85%, truck ~70%, bus ~75%, motorcycle ~65%
