"""
ATMS-Net Phase 1 — Vehicle Detector Training Script.

Trains the custom YOLO-style detector from scratch on the MS COCO
vehicle subset (car, motorcycle, bus, truck).

This script handles:
    - Config loading from YAML
    - Device auto-detection (CUDA → MPS → CPU)
    - Dataset creation with augmentation pipeline
    - Training loop with warmup + cosine annealing LR
    - Mixed precision training (FP16) for GPU
    - EMA model for stable validation
    - Periodic validation with mAP@0.5
    - Checkpointing (best + periodic)
    - Comprehensive logging

Usage:
    # Full training
    python scripts/train_detector.py --config configs/detector.yaml

    # Quick sanity check (1 epoch, small batch)
    python scripts/train_detector.py --config configs/detector.yaml --epochs 1 --batch-size 4

    # Override device
    python scripts/train_detector.py --config configs/detector.yaml --device cpu

    # Resume from checkpoint
    python scripts/train_detector.py --config configs/detector.yaml --resume checkpoints/last.pt
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.detector.yolo_detector import ATMSDetector, ModelEMA
from data.coco.coco_dataset import COCOVehicleDataset, detection_collate_fn
from utils.losses import YOLOLoss
from utils.nms import batch_nms
from utils.metrics import DetectionMetrics


def get_device(preferred='auto'):
    """Auto-detect the best available compute device."""
    if preferred != 'auto':
        return torch.device(preferred)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  → Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"  → Using Apple MPS (Metal)")
    else:
        device = torch.device('cpu')
        print(f"  → Using CPU")

    return device


def build_optimizer(model, config):
    """Create optimizer with per-parameter weight decay handling."""
    training_cfg = config['training']

    # Separate parameters: no weight decay for bias and BatchNorm
    pg0, pg1, pg2 = [], [], []  # BN weights, Conv weights, biases
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if '.bias' in name:
            pg2.append(param)
        elif '.bn.' in name or '.weight' in name and param.ndim == 1:
            pg0.append(param)
        else:
            pg1.append(param)

    optimizer = optim.SGD(
        pg0, lr=training_cfg['learning_rate'],
        momentum=training_cfg['momentum'], nesterov=True,
    )
    optimizer.add_param_group({'params': pg1, 'weight_decay': training_cfg['weight_decay']})
    optimizer.add_param_group({'params': pg2})  # biases, no weight decay

    print(f"  → Optimizer: SGD (lr={training_cfg['learning_rate']}, "
          f"momentum={training_cfg['momentum']}, wd={training_cfg['weight_decay']})")
    print(f"  → Parameter groups: BN={len(pg0)}, Conv={len(pg1)}, Bias={len(pg2)}")

    return optimizer


def build_scheduler(optimizer, config, steps_per_epoch):
    """Create cosine annealing LR scheduler with warmup."""
    training_cfg = config['training']
    total_steps = training_cfg['epochs'] * steps_per_epoch
    warmup_steps = training_cfg['warmup_epochs'] * steps_per_epoch
    min_lr_ratio = training_cfg.get('min_lr_ratio', 0.01)

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return training_cfg.get('warmup_lr_ratio', 0.1) + \
                   (1 - training_cfg.get('warmup_lr_ratio', 0.1)) * step / warmup_steps
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler,
                    device, epoch, config, ema=None):
    """
    Train for one epoch.

    Returns:
        dict with average loss components
    """
    model.train()
    training_cfg = config['training']
    log_interval = config['logging'].get('log_interval', 50)
    accumulate = training_cfg.get('accumulate_grad', 1)
    use_amp = training_cfg.get('mixed_precision', False) and device.type == 'cuda'

    total_box = 0.0
    total_obj = 0.0
    total_cls = 0.0
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass (with optional mixed precision)
        if use_amp:
            with autocast():
                predictions = model(images)
                loss_dict = criterion(predictions, targets)
                loss = loss_dict['loss'] / accumulate
            scaler.scale(loss).backward()
        else:
            predictions = model(images)
            loss_dict = criterion(predictions, targets)
            loss = loss_dict['loss'] / accumulate
            loss.backward()

        # Optimizer step (with gradient accumulation)
        if (batch_idx + 1) % accumulate == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            # Update EMA
            if ema is not None:
                ema.update(model)

        # Step scheduler per batch
        scheduler.step()

        # Accumulate losses
        total_box += loss_dict['box_loss'].item()
        total_obj += loss_dict['obj_loss'].item()
        total_cls += loss_dict['cls_loss'].item()
        total_loss += loss_dict['loss'].item()
        n_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['loss'].item():.4f}",
            'box': f"{loss_dict['box_loss'].item():.4f}",
            'obj': f"{loss_dict['obj_loss'].item():.4f}",
            'cls': f"{loss_dict['cls_loss'].item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
        })

    return {
        'box_loss': total_box / max(n_batches, 1),
        'obj_loss': total_obj / max(n_batches, 1),
        'cls_loss': total_cls / max(n_batches, 1),
        'total_loss': total_loss / max(n_batches, 1),
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device, config):
    """
    Run validation and compute mAP.

    Returns:
        dict with mAP@0.5, mAP@0.5:0.95, per-class AP, and loss
    """
    model.eval()
    eval_cfg = config['evaluation']

    metrics = DetectionMetrics(num_classes=config['model']['num_classes'])
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="  Validating", leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass in eval mode → decoded predictions
        predictions = model(images)  # (B, N, 5+C)

        # Apply NMS
        detections = batch_nms(
            predictions,
            conf_threshold=eval_cfg['conf_threshold'],
            iou_threshold=eval_cfg['iou_threshold'],
            max_detections=eval_cfg['max_detections'],
        )

        # Update metrics
        metrics.update(detections, targets.cpu())

        n_batches += 1

    # Compute mAP
    results = metrics.compute()

    return results


def save_checkpoint(model, ema, optimizer, scheduler, epoch, best_map, config, filename):
    """Save a training checkpoint."""
    save_dir = config['checkpoint']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_map': best_map,
        'config': config,
    }

    if ema is not None:
        checkpoint['ema_state_dict'] = ema.ema.state_dict()

    torch.save(checkpoint, filepath)
    print(f"  → Saved checkpoint: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='ATMS-Net Phase 1: Train Vehicle Detector')
    parser.add_argument('--config', type=str, default='configs/detector.yaml',
                        help='Path to training config YAML')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, mps, cpu, or auto')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # ---- Load Config ----
    print("=" * 60)
    print("ATMS-Net Phase 1 — Vehicle Detector Training")
    print("=" * 60)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size

    # ---- Device ----
    device = get_device(args.device)

    # Disable mixed precision on non-CUDA devices
    if device.type != 'cuda':
        config['training']['mixed_precision'] = False

    # ---- Model ----
    print("\n[Model]")
    model = ATMSDetector.from_config(args.config)
    model.summary()
    model = model.to(device)

    # ---- EMA ----
    ema = None
    if config['training'].get('ema', False):
        ema = ModelEMA(model, decay=config['training'].get('ema_decay', 0.9999))
        print("  → EMA enabled")

    # ---- Dataset ----
    print("\n[Dataset]")
    data_cfg = config['data']

    # Check if data files exist
    if not os.path.exists(data_cfg['train_list']):
        print(f"\n  ⚠ Training data not found: {data_cfg['train_list']}")
        print(f"  Run this first: python data/coco/download_coco.py --data-dir {data_cfg['data_dir']}")
        print(f"\n  For a quick smoke test without data, use: --epochs 0")
        sys.exit(1)

    train_dataset = COCOVehicleDataset(
        img_list=data_cfg['train_list'],
        label_dir=data_cfg['label_dir'],
        img_size=config['model']['img_size'],
        augment=True,
        mosaic_prob=config['augmentation'].get('mosaic_prob', 0.5),
    )

    val_dataset = COCOVehicleDataset(
        img_list=data_cfg['val_list'],
        label_dir=data_cfg['label_dir'],
        img_size=config['model']['img_size'],
        augment=False,  # No augmentation for validation
        mosaic_prob=0.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 4),
        collate_fn=detection_collate_fn,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 4),
        collate_fn=detection_collate_fn,
        pin_memory=(device.type == 'cuda'),
    )

    print(f"  → Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  → Val:   {len(val_dataset)} images, {len(val_loader)} batches")

    # ---- Loss ----
    loss_cfg = config['loss']
    criterion = YOLOLoss(
        num_classes=config['model']['num_classes'],
        strides=(8, 16, 32),
        box_weight=loss_cfg['box_weight'],
        obj_weight=loss_cfg['obj_weight'],
        cls_weight=loss_cfg['cls_weight'],
    )

    # ---- Optimizer & Scheduler ----
    print("\n[Optimizer]")
    optimizer = build_optimizer(model, config)

    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch)
    print(f"  → Scheduler: Cosine annealing (warmup={config['training']['warmup_epochs']} epochs)")

    # ---- Mixed Precision ----
    scaler = GradScaler(enabled=config['training']['mixed_precision'])
    if config['training']['mixed_precision']:
        print("  → Mixed precision (FP16) enabled")

    # ---- Resume ----
    start_epoch = 0
    best_map = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"\n[Resume] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_map = ckpt.get('best_map', 0.0)
        if ema and 'ema_state_dict' in ckpt:
            ema.ema.load_state_dict(ckpt['ema_state_dict'])
        print(f"  → Resuming from epoch {start_epoch}, best mAP@0.5: {best_map:.4f}")

    # ---- Logging Setup ----
    log_dir = config['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')

    # ---- Training Loop ----
    print("\n" + "=" * 60)
    print(f"Starting training: {config['training']['epochs']} epochs")
    print("=" * 60 + "\n")

    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_start = time.time()

        # Train
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch + 1, config, ema=ema,
        )

        epoch_time = time.time() - epoch_start

        # Log training results
        print(f"\n  Epoch {epoch + 1}/{config['training']['epochs']} "
              f"({epoch_time:.1f}s) — "
              f"loss: {train_losses['total_loss']:.4f} "
              f"[box: {train_losses['box_loss']:.4f}, "
              f"obj: {train_losses['obj_loss']:.4f}, "
              f"cls: {train_losses['cls_loss']:.4f}]")

        # Validate
        val_interval = config['evaluation'].get('val_interval', 1)
        if (epoch + 1) % val_interval == 0:
            eval_model = ema.ema if ema else model
            val_results = validate(eval_model, val_loader, criterion, device, config)

            map50 = val_results['mAP50']
            map50_95 = val_results['mAP50_95']

            print(f"  Val mAP@0.5: {map50:.4f}  |  mAP@0.5:0.95: {map50_95:.4f}")
            print(f"  Per-class AP@0.5: ", end='')
            for cls_name, ap in val_results['per_class_ap50'].items():
                print(f"{cls_name}={ap:.3f}  ", end='')
            print()

            # Save best model
            if map50 > best_map:
                best_map = map50
                save_checkpoint(model, ema, optimizer, scheduler, epoch, best_map, config, 'best.pt')
                print(f"  ★ New best mAP@0.5: {best_map:.4f}")

        # Save periodic checkpoint
        save_interval = config['checkpoint'].get('save_interval', 10)
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(model, ema, optimizer, scheduler, epoch, best_map, config, f'epoch_{epoch+1}.pt')

        # Save last checkpoint (always)
        save_checkpoint(model, ema, optimizer, scheduler, epoch, best_map, config, 'last.pt')

        # Write to log file
        with open(log_file, 'a') as f:
            f.write(f"epoch={epoch+1} "
                    f"loss={train_losses['total_loss']:.4f} "
                    f"box={train_losses['box_loss']:.4f} "
                    f"obj={train_losses['obj_loss']:.4f} "
                    f"cls={train_losses['cls_loss']:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}\n")

        print()

    print("=" * 60)
    print(f"Training complete! Best mAP@0.5: {best_map:.4f}")
    print(f"Best model saved to: {os.path.join(config['checkpoint']['save_dir'], 'best.pt')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
