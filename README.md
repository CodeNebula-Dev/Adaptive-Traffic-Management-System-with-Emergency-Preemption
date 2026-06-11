# ATMS-Net: Adaptive Traffic Management System with Emergency Preemption

A deep learning pipeline for real-time adaptive traffic signal control with integrated emergency vehicle preemption. ATMS-Net takes raw intersection camera feeds and outputs dynamic per-lane signal timing — while guaranteeing unobstructed passage for emergency vehicles.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Proposed Solution](#proposed-solution)
- [System Architecture](#system-architecture)
- [Module Breakdown](#module-breakdown)
- [Dataset Strategy](#dataset-strategy)
- [Technology Stack](#technology-stack)
- [Training Pipeline](#training-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Milestones](#project-milestones)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [References](#references)
- [License](#license)

---

## Problem Statement

Urban traffic congestion costs cities billions of productive hours annually. The majority of traffic light systems still operate on **fixed-time cycles** — pre-programmed intervals that are completely blind to actual vehicle presence, queue length, or real-time density. A lane with 40 waiting vehicles gets the same green window as an empty one.

Beyond everyday congestion, fixed-cycle systems present a **life-safety problem**. Emergency vehicles — ambulances, fire engines, police cars — are routinely delayed at red lights during critical interventions.

ATMS-Net addresses a two-fold problem:

> **How can a traffic intersection autonomously decide the optimal green/red window for each approach lane in real time, based purely on visual data, and simultaneously guarantee unobstructed passage for emergency vehicles?**

This requires solving several hard sub-problems simultaneously:

- Real-time multi-class vehicle detection from camera footage, robust to occlusion and lighting variation
- Per-lane vehicle counting and density estimation at inference speed
- Emergency vehicle classification — distinguishing ambulances, fire trucks, and police cars from ordinary traffic
- Sequential decision-making under uncertainty for signal phase and duration
- Safe integration of a hard-interrupt preemption system that overrides learned control without destabilising it

---

## Proposed Solution

ATMS-Net is a full deep learning pipeline that takes raw video from four cameras positioned at the four approach lanes of a standard four-way intersection and outputs:

- **Dynamic red/green signal windows** for each lane, updated every decision cycle
- **Emergency preemption signals** that immediately clear a lane for detected emergency vehicles

The system is composed of four co-designed modules:

| Module | Role | Type |
|--------|------|------|
| Vehicle Detector | Multi-class detection + per-lane density estimation | Custom YOLO-style CNN (trained from scratch) |
| Emergency Vehicle Detector | High-recall EV classification | Shared backbone + dedicated EV head |
| RL Signal Controller | Optimal phase and duration selection | Deep Q-Network (SUMO-trained) |
| Emergency Override | Safety-critical green corridor enforcement | Deterministic rule-based |

**Design philosophy:** Learned where learning is appropriate, deterministic where safety is critical.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                             │
│  [Camera N]   [Camera S]   [Camera E]   [Camera W]              │
│       ↓            ↓            ↓            ↓                  │
└─────────────────────────────────────────────────────────────────┘
                         ↓ video frames
┌─────────────────────────────────────────────────────────────────┐
│                      INTELLIGENCE LAYER                         │
│                                                                 │
│  ┌───────────────────────┐    ┌───────────────────────────────┐ │
│  │   Vehicle Detector    │───→│     RL Signal Controller      │ │
│  │   (Custom YOLO CNN)   │    │     (DQN — SUMO trained)      │ │
│  │  → BBoxes + classes   │    │  State:  density × 4 lanes    │ │
│  │  → Lane density score │    │  Action: phase + duration     │ │
│  └───────────────────────┘    │  Reward: −waiting time        │ │
│            │                  └───────────────────────────────┘ │
│            ↓                               │                    │
│  ┌───────────────────────┐    ┌────────────↓──────────────────┐ │
│  │  Emergency Detector   │───→│    Emergency Override         │ │
│  │  (Shared backbone +   │    │    (Rule-based preemption)    │ │
│  │   EV head)            │    │  → EV lane → GREEN            │ │
│  │  → EV flag + lane ID  │    │  → All others → RED           │ │
│  └───────────────────────┘    └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                         ↓ phase + duration command
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT LAYER                             │
│  ┌───────────────────────────┐   ┌───────────────────────────┐  │
│  │    Signal Actuator        │   │   Monitoring Dashboard    │  │
│  │  Dynamic per-lane R/G/Y   │   │   Live density · EV alert │  │
│  │  Millisecond-response     │   │   Phase logs · metrics    │  │
│  └───────────────────────────┘   └───────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Breakdown

### Module 1 — Custom Vehicle Detector

A from-scratch YOLO-style single-stage detector with:

- **Backbone:** CSP (Cross-Stage Partial) convolutional network with residual blocks and cross-stage partial connections. Outputs feature maps at strides 8, 16, and 32.
- **Neck:** Feature Pyramid Network (FPN) with Path Aggregation Network (PANet) for bidirectional feature fusion.
- **Head:** Anchor-free detection head predicting 4D bounding box offset, objectness score, and class vector per spatial cell.

**Output per frame per camera:**
- Bounding boxes `{x, y, w, h, confidence, class}`
- Per-lane vehicle count (via spatial lane mask)
- Normalised lane density score `d ∈ [0, 1]`

**Training:** CIoU regression loss + BCE classification loss + objectness loss. Mosaic augmentation, HSV jitter, cutout. SGD with cosine annealing LR schedule.

### Module 2 — Emergency Vehicle Detector

Shared backbone from Module 1 (frozen/partially frozen) with a dedicated EV classification head:

- Global average pool → two-layer MLP → sigmoid output per EV class (ambulance, fire truck, police car)
- Zero additional computation for backbone — EV head runs as a parallel branch
- **High-recall operating point:** threshold set conservatively (missed EV = safety failure; false positive = brief unnecessary preemption)
- Trained with class-weighted cross-entropy and focal loss for class imbalance

**Output:** `{ev_detected: bool, ev_class: str, lane_id: int, confidence: float}`

### Module 3 — RL Signal Controller

Deep Q-Network trained in SUMO (Simulation of Urban MObility) via the TraCI API.

**MDP Formulation:**

| Component | Definition |
|-----------|-----------|
| State `s` | `[d_N, d_S, d_E, d_W, current_phase, elapsed_time_normalised]` |
| Action `a` | `(phase_id, duration_bin)` — duration bins ∈ {15s, 30s, 45s, 60s} |
| Reward `r` | `−Σ(queue_length_per_lane) − α × max(queue_length)` |
| Discount `γ` | 0.95 |

The reward function penalises both total queue length (efficiency) and maximum queue on any single lane (fairness), preventing degenerate policies that starve low-density lanes.

**Network:** Input (6D state) → 256 → 128 (ReLU) → 16 outputs (4 phases × 4 durations). Experience replay buffer of 50,000 transitions. Target network updated every 500 steps. ε-greedy exploration with linear decay.

### Module 4 — Emergency Override

A deterministic rule-based module that sits above the RL controller:

```python
def get_phase_command(rl_output, ev_flag):
    if ev_flag.detected:
        return PhaseCommand(
            green_lanes=[ev_flag.lane_id],
            red_lanes=all_lanes - {ev_flag.lane_id},
            duration=EV_CLEARANCE_TIME,
            source="EV_OVERRIDE"
        )
    else:
        return rl_output
```

After preemption, control returns to the RL controller with a **state refresh** — the density vector is re-read to prevent stale-state value estimation.

---

## Dataset Strategy

| Purpose | Dataset | Notes |
|---------|---------|-------|
| Vehicle detection backbone training | MS COCO (vehicle classes) | 80-class COCO filtered to car, truck, bus, motorcycle |
| Intersection-specific fine-tuning | UA-DETRAC | 140,000 frames from 24 locations |
| Adverse condition robustness | DAWN Dataset | Rain, fog, sand, snow |
| Emergency vehicle classification | HERO Dataset + manual curation | Augmented with affine transforms and lighting jitter |
| Synthetic intersection rendering | CARLA Simulator | Custom intersection map, controllable EV injection |
| RL training environment | SUMO + OpenStreetMap | Real intersection geometry imported from OSM |
| Evaluation benchmark | Held-out SUMO scenarios | 5 traffic demand profiles × 3 time-of-day conditions |

All datasets are publicly available for academic use. CARLA and SUMO synthetic pipelines allow unlimited data generation for edge cases (nighttime EV detection, multi-vehicle occlusion).

---

## Technology Stack

| Layer | Tool / Library | Role |
|-------|---------------|------|
| Model development | PyTorch 2.x | All neural network implementation and training |
| Traffic simulation | SUMO 1.19 + TraCI | RL environment, episode generation |
| RL framework | Custom Gym wrapper + Stable-Baselines3 | DQN training and evaluation |
| Computer vision | OpenCV 4.x | Frame capture, lane masking, pre-processing |
| Experiment tracking | Weights & Biases | Loss curves, reward plots, model versioning |
| Synthetic data | CARLA 0.9.15 | Photorealistic intersection video generation |
| Data handling | NumPy, Pandas, Albumentations | Preprocessing, augmentation pipelines |
| Visualisation | Matplotlib, Seaborn | Evaluation plots and result figures |
| Version control | Git + GitHub | Repository, README, releases |

**Hardware requirement:** Single GPU (minimum RTX 3060 12GB or equivalent). Mixed precision training (FP16) used throughout.

---

## Training Pipeline

The training is structured in five sequential phases:

**Phase 1 — Detector Pre-training**
Train the backbone + neck + detection head on MS COCO vehicle classes from random initialisation. Target: mAP@0.5 > 75% on the COCO vehicle subset (~50 epochs on 80k images).

**Phase 2 — Intersection Fine-tuning**
Fine-tune the full detector on UA-DETRAC and CARLA intersection footage. Add and train the lane density estimation head. Density labels computed from SUMO ground-truth vehicle positions projected into camera space.

**Phase 3 — EV Head Training**
Freeze the backbone. Train the EV classification head on HERO dataset + CARLA EV sequences using focal loss. Target: recall > 0.95 on the EV test set.

**Phase 4 — RL Controller Training**
Deploy the trained detector inside the SUMO Gym environment. Train DQN for 500,000 environment steps. Log reward curves, queue length distributions, and throughput metrics via W&B.

**Phase 5 — Integration Testing**
Run the full four-module system end-to-end inside SUMO. Inject EV events at random timesteps. Evaluate EV clearance time, false preemption rate, and impact on overall network throughput.

---

## Evaluation Metrics

### Detection (Modules 1 & 2)

| Metric | Description | Target |
|--------|-------------|--------|
| mAP@0.5 | Mean average precision at IoU 0.5 | > 85% (vehicle detector) |
| mAP@0.5:0.95 | COCO-style mAP | > 60% |
| EV Recall | True positive rate for EV class | > 95% |
| EV Precision | Positive predictive value for EV | > 85% |
| Inference FPS | Frames processed per second | > 30 FPS on target GPU |

### RL Controller (Module 3)

| Metric | Description |
|--------|-------------|
| Average waiting time | Mean seconds per vehicle at intersection |
| Max lane queue | Maximum queue length across all lanes (fairness) |
| Throughput | Vehicles cleared per minute |
| Phase efficiency | Fraction of green time with non-zero flow |

All RL metrics reported relative to the fixed-time baseline as percentage improvement.

### Full System Integration

| Metric | Description |
|--------|-------------|
| EV clearance time | Seconds from EV detection to intersection clearing |
| Preemption overhead | Extra waiting time imposed on non-EV traffic per event |
| False preemption rate | EV override events triggered without a true EV present |

---

## Project Milestones

| Phase | Deliverable | Success Criterion |
|-------|------------|-------------------|
| Phase 1 | Custom vehicle detector trained from scratch | mAP@0.5 > 85% on held-out test |
| Phase 2 | Intersection fine-tuning + density estimation | Per-lane density MAE < 0.08 |
| Phase 3 | EV classification head | EV recall > 95% on test set |
| Phase 4 | Trained DQN signal controller | ≥ 20% reduction in avg. waiting time vs fixed-time baseline |
| Phase 5 | Full integrated system in SUMO | EV clearance < 8s, false preemption rate < 2% |
| Final | Complete repository with documentation | Reproducible README, W&B run logs, demo video |

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (minimum 12GB VRAM recommended)
- SUMO 1.19+
- CARLA 0.9.15 (for synthetic data generation)

### Installation

```bash
# Clone the repository
git clone https://github.com/CodeNebula-Dev/Adaptive-Traffic-Management-System-with-Emergency-Preemption.git
cd Adaptive-Traffic-Management-System-with-Emergency-Preemption

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Phase 1: Train the vehicle detector
python train_detector.py --config configs/detector.yaml

# Phase 3: Train the EV classification head
python train_ev_head.py --config configs/ev_head.yaml

# Phase 4: Train the RL signal controller
python train_rl.py --config configs/rl_controller.yaml

# Phase 5: Run the full integrated system
python run_system.py --config configs/integration.yaml
```

> **Note:** Detailed instructions for each phase will be added as the project progresses. The commands above represent the planned CLI interface.

---

## Repository Structure

```
ATMS-Net/
├── Project-IdeaDocs/           # Project ideation and technical roadmap
├── configs/                    # Training and system configuration files
├── data/                       # Dataset scripts and data loaders
│   ├── coco/                   # MS COCO vehicle subset processing
│   ├── ua_detrac/              # UA-DETRAC preprocessing
│   ├── dawn/                   # DAWN dataset (adverse conditions)
│   └── hero/                   # HERO emergency vehicle dataset
├── models/                     # Neural network architectures
│   ├── backbone/               # CSP backbone implementation
│   ├── neck/                   # FPN + PANet neck
│   ├── detector/               # Full vehicle detector
│   ├── ev_head/                # Emergency vehicle classification head
│   └── rl_agent/               # DQN signal controller
├── envs/                       # SUMO Gym environment wrappers
├── modules/                    # System integration modules
│   └── override/               # Emergency override logic
├── utils/                      # Utility functions (lane masks, metrics, etc.)
├── scripts/                    # Training and evaluation scripts
├── checkpoints/                # Saved model weights
├── logs/                       # Training logs and W&B exports
├── results/                    # Evaluation results and figures
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

> **Note:** This structure will be populated as development progresses. Some directories are planned and will be created during the respective development phases.

---

## Key Innovations

1. **End-to-end trained detection backbone** — Unlike existing work that uses pretrained YOLO weights as black-box extractors, ATMS-Net trains the full detection backbone from random initialisation on intersection-specific data.

2. **Fairness-weighted RL reward function** — The reward explicitly penalises the maximum queue across lanes (`α × max(queue)` term), enforcing fairness as a hard constraint and preventing degenerate policies that starve sparse lanes.

3. **Safe handoff between RL and deterministic override** — A clean state-refresh protocol ensures the RL controller resumes from fresh state after emergency preemption ends, preventing stale-state value estimation.

---

## References

1. Abbas, S. et al. (2024). *Vision based intelligent traffic light management system using Faster R-CNN.* CAAI Transactions on Intelligence Technology.
2. Charoenpong, T. et al. (2024). *Adaptive traffic light control using vision-based deep learning for vehicle density estimation.* Proceedings of APIT 2024, ACM.
3. Johny, C. & Sharma, A. (2024). *Deep Learning for Emergency Vehicle Identification: A YOLOv8-Based Approach for Smart City Solutions.* Journal of Electrical Systems, 20(3).
4. Scribano, C. & Muzzini, F. (2025). *Real-time traffic signal adjustment using YOLOv8 for improved integration of emergency vehicles in smart traffic systems.* Signal, Image and Video Processing.
5. Hu, Y. et al. (2024). *A multi-agent deep reinforcement learning approach for traffic signal coordination.* IET Intelligent Transport Systems.
6. Yang, G. et al. (2025). *Multi-Agent Deep Reinforcement Learning with Graph Attention Network for Traffic Signal Control.* Proceedings of COSITE 2025.
7. Wu, Q. et al. (2025). *Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control with Spatio-Temporal Attention Mechanism.* Applied Sciences, 15(15).
8. Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518.
9. Redmon, J. & Farhadi, A. (2018). *YOLOv3: An Incremental Improvement.* arXiv:1804.02767.
10. Lopez, P.A. et al. (2018). *Microscopic Traffic Simulation using SUMO.* Proceedings of IEEE ITSC 2018.

---

## License

This project is intended for academic and research purposes. See [LICENSE](LICENSE) for details.

---

**ATMS-Net** — Learned where learning is appropriate, deterministic where safety is critical.
