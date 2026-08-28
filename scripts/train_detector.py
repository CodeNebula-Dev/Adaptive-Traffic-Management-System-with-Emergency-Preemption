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
from torch.amp import GradScaler, autocast

from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.detector.yolo_detector import ATMSDetector, ModelEMA
from data.coco.coco_dataset import COCOVehicleDataset, detection_collate_fn
try:
    from data.ua_detrac.uadetrac_dataset import UADetracDataset
except ImportError:
    UADetracDataset = None
from utils.losses import YOLOLoss
from utils.nms import batch_nms
from utils.metrics import DetectionMetrics


def get_device(preferred='auto'):
    """Auto-detect the best available compute device."""
    if preferred != 'auto':
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_optimizer(model, config):
    """
    Build optimizer with parameter group separation.

    Separates parameters into 3 groups:
        1. BatchNorm weights and biases: NO weight decay
        2. Conv weights: WITH weight decay
        3. Conv biases: NO weight decay
    """
    train_cfg = config['training']
    base_lr = train_cfg['learning_rate']
    weight_decay = train_cfg.get('weight_decay', 0.0005)

    bn_params = []
    weight_params = []
    bias_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bn' in name or 'norm' in name:
            bn_params.append(param)
        elif 'bias' in name:
            bias_params.append(param)
        else:
            weight_params.append(param)

    param_groups = [
        {'params': bn_params, 'weight_decay': 0.0, 'lr': base_lr},
        {'params': weight_params, 'weight_decay': weight_decay, 'lr': base_lr},
        {'params': bias_params, 'weight_decay': 0.0, 'lr': base_lr},
    ]

    opt_name = train_cfg.get('optimizer', 'sgd').lower()
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            lr=base_lr,
            momentum=train_cfg.get('momentum', 0.937),
            nesterov=True,
        )
    elif opt_name == 'adam':
        optimizer = optim.Adam(param_groups, lr=base_lr)
    elif opt_name == 'adamw':
        optimizer = optim.AdamW(param_groups, lr=base_lr)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    print(f"  → Optimizer: {opt_name.upper()} (lr={base_lr}, momentum={train_cfg.get('momentum', 0.937)}, wd={weight_decay})")
    print(f"  → Parameter groups: BN={len(bn_params)}, Conv={len(weight_params)}, Bias={len(bias_params)}")

    return optimizer


def build_scheduler(optimizer, config, steps_per_epoch):
    """Build a learning rate scheduler with warmup and cosine annealing."""
    train_cfg = config['training']
    total_epochs = train_cfg['epochs']
    warmup_epochs = train_cfg.get('warmup_epochs', 3)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    min_lr_ratio = train_cfg.get('min_lr_ratio', 0.01)
    warmup_lr_ratio = train_cfg.get('warmup_lr_ratio', 0.1)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            alpha = current_step / max(1, warmup_steps)
            return warmup_lr_ratio + (1.0 - warmup_lr_ratio) * alpha
        else:
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler,
                    device, epoch, config, ema=None):
    """Train the model for one epoch."""
    model.train()
    total_loss_meter = AverageMeter('Total Loss')
    box_loss_meter = AverageMeter('Box Loss')
    obj_loss_meter = AverageMeter('Obj Loss')
    cls_loss_meter = AverageMeter('Cls Loss')

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
    accumulate_grad = config['training'].get('accumulate_grad', 1)

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast('cuda', enabled=config['training']['mixed_precision']):
            predictions = model(images)
            loss_dict = criterion(predictions, targets)
            loss_val = loss_dict.get('loss', loss_dict.get('total_loss'))
            loss = loss_val / accumulate_grad

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulate_grad == 0 or (batch_idx + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if ema is not None:
                ema.update(model)

        scheduler.step()

        total_loss_meter.update(loss_val.item(), images.size(0))
        box_loss_meter.update(loss_dict['box_loss'].item(), images.size(0))
        obj_loss_meter.update(loss_dict['obj_loss'].item(), images.size(0))
        cls_loss_meter.update(loss_dict['cls_loss'].item(), images.size(0))

        pbar.set_postfix({
            'loss': f"{total_loss_meter.avg:.4f}",
            'box': f"{box_loss_meter.avg:.4f}",
            'obj': f"{obj_loss_meter.avg:.4f}",
            'cls': f"{cls_loss_meter.avg:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
        })

    return {
        'total_loss': total_loss_meter.avg,
        'box_loss': box_loss_meter.avg,
        'obj_loss': obj_loss_meter.avg,
        'cls_loss': cls_loss_meter.avg,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device, config, epoch=0):
    """Run validation and compute vectorized mAP@0.5 and mAP@0.5:0.95."""
    model.eval()
    num_classes = config['model']['num_classes']
    class_names = getattr(model, 'CLASS_NAMES', ['car', 'motorcycle', 'bus', 'truck', 'unknown_vehicle'])[:num_classes]

    eval_cfg = config.get('evaluation', {})
    conf_thresh = eval_cfg.get('conf_threshold', 0.01)
    iou_thresh = eval_cfg.get('iou_threshold', 0.45)
    max_dets = eval_cfg.get('max_detections', 300)

    metrics = DetectionMetrics(
        num_classes=num_classes,
        class_names=class_names,
        iou_thresholds=None,
        conf_threshold=conf_thresh,
    )

    total_detections = 0
    total_ground_truths = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        img_h, img_w = images.shape[2:]

        predictions = model(images)
        batch_detections = batch_nms(
            predictions,
            conf_threshold=conf_thresh,
            iou_threshold=iou_thresh,
            max_detections=max_dets,
        )

        for b_idx in range(images.size(0)):
            dets = batch_detections[b_idx]
            gt_mask = targets[:, 0] == b_idx
            img_targets = targets[gt_mask]

            if dets is not None and len(dets) > 0:
                pred_boxes = dets[:, :4].cpu()
                pred_scores = dets[:, 4].cpu()
                pred_labels = dets[:, 6].long().cpu()
                total_detections += len(dets)
            else:
                pred_boxes = torch.zeros((0, 4))
                pred_scores = torch.zeros((0,))
                pred_labels = torch.zeros((0,), dtype=torch.long)

            if len(img_targets) > 0:
                gt_cx = img_targets[:, 2] * img_w
                gt_cy = img_targets[:, 3] * img_h
                gt_w = img_targets[:, 4] * img_w
                gt_h = img_targets[:, 5] * img_h

                gt_x1 = gt_cx - gt_w / 2
                gt_y1 = gt_cy - gt_h / 2
                gt_x2 = gt_cx + gt_w / 2
                gt_y2 = gt_cy + gt_h / 2

                gt_boxes = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=1).cpu()
                gt_labels = img_targets[:, 1].long().cpu()
                total_ground_truths += len(img_targets)
            else:
                gt_boxes = torch.zeros((0, 4))
                gt_labels = torch.zeros((0,), dtype=torch.long)

            metrics.update(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)

    results = metrics.compute()
    print(f"  [Val diagnostics] conf_thresh={conf_thresh:.4f}, total_dets={total_detections}, total_gt={total_ground_truths}, batches={len(dataloader)}")
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


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    parser = argparse.ArgumentParser(description='ATMS-Net Phase 1 & 2.5: Train Vehicle Detector')
    parser.add_argument('--config', type=str, default='configs/detector.yaml',
                        help='Path to training config YAML')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, mps, cpu, or auto')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (resumes epoch & optimizer)')
    parser.add_argument('--finetune', type=str, default=None,
                        help='Path to pretrained checkpoint to fine-tune from (loads weights, fresh optimizer/schedule)')
    args = parser.parse_args()

    # ---- Load Config ----
    print("=" * 60)
    print("ATMS-Net — Vehicle Detector Training & Fine-Tuning")
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
    model.to(device)

    # ---- Fine-tune Weight Loading (with class expansion support) ----
    best_map = 0.0
    start_epoch = 0

    if args.finetune and os.path.exists(args.finetune):
        model.load_pretrained_with_class_expansion(args.finetune, device=device)
        print("  → Initialized fresh optimizer & learning rate schedule for fine-tuning")

    model.summary()

    # ---- EMA ----
    ema = None
    if config['training'].get('ema', False):
        ema = ModelEMA(model, decay=config['training'].get('ema_decay', 0.9999))
        print("  → EMA enabled")

    # ---- Dataset ----
    print("\n[Dataset]")
    data_cfg = config['data']
    dataset_type = data_cfg.get('dataset_type', 'coco').lower()

    if not os.path.exists(data_cfg['train_list']):
        print(f"\n  ⚠ Training data not found: {data_cfg['train_list']}")
        if dataset_type == 'uadetrac':
            print(f"  Run this first: python data/ua_detrac/download_uadetrac.py --data-dir {data_cfg['data_dir']}")
        else:
            print(f"  Run this first: python data/coco/download_coco.py --data-dir {data_cfg['data_dir']}")
        print(f"\n  For a quick smoke test without data, use: --epochs 0")
        sys.exit(1)

    if dataset_type == 'uadetrac' and UADetracDataset is not None:
        print("  → Ingesting UA-DETRAC Traffic Surveillance Dataset (5 classes)")
        train_dataset = UADetracDataset(
            img_list=data_cfg['train_list'],
            label_dir=data_cfg.get('label_dir', 'data/ua_detrac/labels'),
            img_size=config['model']['img_size'],
            augment=True,
            mosaic_prob=config['augmentation'].get('mosaic_prob', 0.4),
            num_classes=config['model']['num_classes'],
        )
        val_dataset = UADetracDataset(
            img_list=data_cfg['val_list'],
            label_dir=data_cfg.get('label_dir', 'data/ua_detrac/labels'),
            img_size=config['model']['img_size'],
            augment=False,
            mosaic_prob=0.0,
            num_classes=config['model']['num_classes'],
        )
    else:
        train_label_dir = data_cfg.get('train_label_dir', data_cfg.get('label_dir', 'data/coco/labels/train2017'))
        val_label_dir = data_cfg.get('val_label_dir', data_cfg.get('label_dir', 'data/coco/labels/train2017').replace('train2017', 'val2017'))

        train_dataset = COCOVehicleDataset(
            img_list=data_cfg['train_list'],
            label_dir=train_label_dir,
            img_size=config['model']['img_size'],
            augment=True,
            mosaic_prob=config['augmentation'].get('mosaic_prob', 0.5),
        )
        val_dataset = COCOVehicleDataset(
            img_list=data_cfg['val_list'],
            label_dir=val_label_dir,
            img_size=config['model']['img_size'],
            augment=False,
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
    print(f"  → Scheduler: Cosine annealing (warmup={config['training']['warmup_epochs']} epochs, lr={config['training']['learning_rate']})")

    # ---- Mixed Precision ----
    scaler = GradScaler('cuda', enabled=config['training']['mixed_precision'])
    if config['training']['mixed_precision']:
        print("  → Mixed precision (FP16) enabled")

    # ---- Resume Mode ----
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
        warmup_epochs = config['training'].get('warmup_epochs', 3)
        if (epoch + 1) % val_interval == 0:
            # Use raw model during early warmup while EMA is still initializing
            eval_model = ema.ema if (ema and (epoch + 1) > warmup_epochs) else model
            val_results = validate(eval_model, val_loader, criterion, device, config, epoch=epoch + 1)

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
