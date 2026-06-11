# ATMS-Net: Adaptive Traffic Management System with Emergency Preemption

### A Deep Learning Project — Cool lets get started ig i this is the overall idea and a bit of techinal roadmap so lets see 

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Research Background](#2-research-background)
3. [Proposed Solution](#3-proposed-solution)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [Module Breakdown](#5-module-breakdown)
6. [Dataset Strategy](#6-dataset-strategy)
7. [Technology Stack](#7-technology-stack)
8. [Training Pipeline](#8-training-pipeline)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Innovation Over Prior Work](#10-innovation-over-prior-work)
11. [Project Milestones](#11-project-milestones)
12. [References](#12-references)

---

## 1. Problem Statement

Urban traffic congestion is one of the defining infrastructure challenges of the 21st century. According to transportation research, cities across the world lose billions of productive hours annually to traffic delays, with a disproportionate share caused not by road capacity limits but by **inefficient signal timing**.

The overwhelming majority of traffic light systems deployed today operate on **fixed-time cycles** — pre-programmed intervals that do not respond to actual vehicle presence, queue length, or real-time density. A north-bound lane with 40 vehicles waiting receives the same green window as an empty east-bound lane. This static design is provably suboptimal and becomes increasingly damaging as urban density grows.

Beyond everyday congestion, fixed-cycle systems present a **life-safety problem** for emergency response. Ambulances, fire engines, and police vehicles are routinely delayed at red lights during critical interventions. Every additional minute an ambulance spends at an intersection can directly affect patient survival outcomes.

The problem this project addresses is therefore two-fold:

> **How can a traffic intersection autonomously decide the optimal green/red window for each approach lane in real time, based purely on visual data, and simultaneously guarantee unobstructed passage for emergency vehicles?**

Solving this requires addressing several hard technical sub-problems simultaneously:

- Real-time multi-class vehicle detection from camera footage, robust to occlusion and lighting variation
- Per-lane vehicle counting and density estimation at inference speed
- Emergency vehicle classification — distinguishing ambulances, fire trucks, and police cars from ordinary traffic
- Sequential decision-making under uncertainty: deciding which lane gets green, and for how long, given continuously changing traffic state
- Safe integration of a hard-interrupt preemption system that overrides learned control without destabilising it

No single deployed system today solves all of these from a unified, trained-from-scratch deep learning architecture.

---

## 2. Research Background

### 2.1 Classical Traffic Signal Control

Traditional traffic signal control falls into three broad categories:

**Fixed-time control** uses pre-computed cycle plans derived from historical traffic surveys. It is simple and predictable but completely blind to real-time conditions. Webster's formula (1958) remains the mathematical foundation for most deployed fixed-time systems.

**Actuated control** uses loop detectors or radar sensors embedded in the road to detect vehicle presence and extend green phases dynamically. While more responsive than fixed-time, it is limited to binary presence detection and cannot estimate queue length, vehicle class, or overall network state.

**Adaptive control systems** such as SCOOT (Split Cycle Offset Optimisation Technique) and SCATS (Sydney Coordinated Adaptive Traffic System) use sensor networks to adjust cycle plans in near real time. These are the current gold standard in smart cities but require expensive dedicated hardware infrastructure and operate on aggregate flow models rather than direct visual perception.

### 2.2 Deep Learning Approaches to Vehicle Detection

The emergence of CNN-based object detectors transformed the field of intelligent transportation. The YOLO (You Only Look Once) family of architectures demonstrated that single-pass detection could achieve real-time throughput without sacrificing meaningful accuracy. YOLOv3 established the multi-scale anchor-based detection paradigm, while subsequent versions (v5 through v9) progressively improved backbone efficiency, neck design, and training strategies.

Parallel work on two-stage detectors — particularly Faster R-CNN — showed that region proposal networks could achieve higher accuracy at the cost of latency. For intersection control, where per-frame latency must stay under 100ms, single-stage detectors are the natural choice.

Vehicle counting for signal timing specifically was explored by Abbas et al. (2024), who used Faster R-CNN with a per-lane counting head and achieved detection accuracies of 95.7% across day and night conditions. Work presented at APIT 2024 demonstrated a complete pipeline using YOLOv3 + DeepSORT for counting, with green-light duration calculated as a linear function of vehicle count — a key early result, but one limited to a single approach lane and a hand-crafted timing rule.

### 2.3 Emergency Vehicle Detection and Signal Preemption

Emergency vehicle preemption (EVP) has been an active research area since the 1990s. Traditional EVP systems use GPS transponders or acoustic sirens — the Opticom system, for example, uses infrared emitters mounted on emergency vehicles that trigger receivers at intersections. These require dedicated hardware on both the vehicle and the intersection.

Vision-based EVP using deep learning is comparatively recent. A 2024 study in the Journal of Electrical Systems applied YOLOv8 to detect ambulances and fire trucks from live CCTV feeds, demonstrating precision and recall competitive with transponder-based systems. A 2025 study in Signal, Image and Video Processing extended this to a multi-camera system with real-time signal adjustment, noting residual challenges in adverse weather and heavy occlusion.

A key finding across this literature is that YOLOv8's anchor-free detection head and improved feature pyramid neck significantly outperform earlier YOLO versions on small, partially occluded emergency vehicle instances — the dominant failure mode in dense urban traffic.

### 2.4 Reinforcement Learning for Adaptive Signal Control

Formulating traffic signal control as a Markov Decision Process (MDP) and solving it with reinforcement learning is an established paradigm. The state space encodes traffic density across lanes; the action space defines which phase (lane group) to activate and for how long; the reward function typically penalises cumulative waiting time or queue length.

Early RL approaches used tabular Q-learning on discretised state spaces, limiting scalability. Deep Q-Networks (DQN), introduced by Mnih et al. (2015) in the context of Atari games, provided the key insight that a neural network approximator could generalise across states — enabling RL to scale to realistic intersection configurations.

More recent work has moved toward multi-agent RL for coordinated control across multiple intersections. A 2024 multi-agent deep RL approach using Double DQN demonstrated a measurable reduction in average vehicle waiting time compared to fixed-time and actuated baselines in SUMO simulation. A 2025 paper proposed spatio-temporal attention networks within a multi-agent DRL framework, achieving further improvements on large-scale urban road networks by explicitly modelling inter-intersection dependencies.

### 2.5 Gap in Existing Work

Despite this rich body of literature, the following gap remains unaddressed:

> **No existing open-source system trains a unified end-to-end architecture — from raw pixel input to signal output — that jointly optimises normal traffic throughput via RL and guarantees emergency vehicle passage via hard preemption, with all component models trained from scratch rather than fine-tuned from pretrained weights.**

Most academic prototypes either use a pretrained backbone (making no contribution to detection architecture), employ a fixed timing rule instead of learned control, handle emergency vehicles as a post-processing flag rather than an integrated module, or address only a single intersection rather than a generalisable system. This project is designed to close all four gaps simultaneously.

---

## 3. Proposed Solution

ATMS-Net (Adaptive Traffic Management System Network) is a full deep learning pipeline that takes raw video from four cameras positioned at the four approach lanes of a standard four-way intersection and outputs:

- Dynamic red/green signal windows for each lane, updated every decision cycle
- An emergency preemption signal that immediately clears a lane for an emergency vehicle when one is detected

The system is composed of four co-designed modules that share computation where beneficial and are integrated through a clean interface protocol:

**Module 1 — Vehicle Detector:** A custom YOLO-style convolutional neural network trained from scratch for multi-class vehicle detection and per-lane density estimation.

**Module 2 — Emergency Vehicle Detector:** A fine-tuned classification head on top of Module 1's shared backbone, specialised for high-recall detection of ambulances, fire trucks, and police vehicles.

**Module 3 — RL Signal Controller:** A Deep Q-Network that takes the per-lane density vector as state and outputs the optimal phase and duration for the next green window, trained in a SUMO simulation environment.

**Module 4 — Emergency Override Module:** A deterministic rule-based preemption module that intercepts Module 3's output when an EV flag is raised and enforces the safety-critical green corridor.

The key design philosophy is: **learned where learning is appropriate, deterministic where safety is critical.**

---

## 4. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                             │
│  [Camera N]   [Camera S]   [Camera E]   [Camera W]              │
│   ↓               ↓           ↓              ↓                  │
└─────────────────────────────────────────────────────────────────┘
                         ↓ video frames
┌─────────────────────────────────────────────────────────────────┐
│                      INTELLIGENCE LAYER                         │
│                                                                 │
│  ┌───────────────────────┐    ┌───────────────────────────────┐ │
│  │   Vehicle Detector    │───▶│     RL Signal Controller      │ │
│  │   (Custom YOLO CNN)   │    │     (DQN — SUMO trained)      │ │
│  │  → BBoxes + classes   │    │  State:  density × 4 lanes    │ │
│  │  → Lane density score │    │  Action: phase + duration     │ │
│  └───────────────────────┘    │  Reward: −waiting time        │ │
│            │                  └───────────────────────────────┘ │
│            ▼                               │                    │
│  ┌───────────────────────┐    ┌────────────▼──────────────────┐ │
│  │  Emergency Detector   │───▶│    Emergency Override         | │
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

## 5. Module Breakdown

### 5.1 Module 1 — Custom Vehicle Detector

**Architecture:** A from-scratch YOLO-style single-stage detector.

The backbone is a CSP (Cross-Stage Partial) convolutional network: a series of residual blocks with cross-stage partial connections that split the gradient flow, reducing redundant computation while preserving feature richness. The backbone outputs three feature maps at strides 8, 16, and 32 — corresponding to large, medium, and small object scales.

The neck is a Feature Pyramid Network (FPN) with a Path Aggregation Network (PANet) extension, which enables bidirectional feature fusion: top-down semantic information is combined with bottom-up localisation information to produce three fused multi-scale feature maps.

The head is an anchor-free detection head: for each spatial cell in each feature map, it predicts a 4D bounding box offset, an objectness score, and a C-dimensional class vector. This eliminates the hyperparameter sensitivity of anchor boxes — important for intersection scenes where vehicle sizes vary drastically with distance.

**Output per frame per camera:**

- Set of bounding boxes `{x, y, w, h, confidence, class}`
- Per-lane vehicle count (via spatial lane mask applied to detections)
- Normalised lane density score `d ∈ [0, 1]`

**Training details:**

- Loss: CIoU regression loss + Binary Cross-Entropy classification loss + objectness loss
- Augmentation: Mosaic augmentation, random horizontal flip, HSV jitter, cutout
- Optimiser: SGD with cosine annealing LR schedule, warm restarts

### 5.2 Module 2 — Emergency Vehicle Detector

**Architecture:** Shared backbone from Module 1 (frozen or partially frozen) + a dedicated EV classification head.

The classification head takes the highest-resolution fused feature map from the neck, applies a global average pool, and feeds into a two-layer MLP with a sigmoid output per EV class (ambulance, fire truck, police car). At inference, a positive detection in any class with confidence > threshold triggers the EV flag.

**Key design choices:**

- Shared backbone means zero additional computation for the first forward pass — the EV head runs as a parallel branch
- High-recall operating point: the threshold is set conservatively. A missed EV (false negative) is a safety failure; a false positive only costs a brief unnecessary preemption
- Trained with class-weighted cross-entropy to handle the naturally long-tail distribution of EV instances

**Output:** `{ev_detected: bool, ev_class: str, lane_id: int, confidence: float}`

### 5.3 Module 3 — RL Signal Controller

**Environment:** SUMO (Simulation of Urban MObility) wrapped as an OpenAI Gym-compatible environment using the TraCI API. SUMO provides a physically realistic microscopic traffic model — each vehicle has individual acceleration, deceleration, and reaction-time parameters.

**MDP Formulation:**

|Component|Definition|
|---|---|
|State `s`|`[d_N, d_S, d_E, d_W, current_phase, elapsed_time_normalised]`|
|Action `a`|`(phase_id, duration_bin)` where duration bins ∈ {15s, 30s, 45s, 60s}|
|Reward `r`|`−Σ(queue_length_per_lane) − α × max(queue_length)`|
|Episode|One simulated hour of traffic at a synthetic intersection|
|Discount `γ`|0.95|

The reward function penalises both total queue length (efficiency objective) and the maximum queue on any single lane (fairness objective, weighted by `α`). This prevents the RL agent from discovering degenerate policies that starve low-density lanes.

**Network:** Deep Q-Network (DQN) with:

- Input layer: 6-dimensional state vector
- Two hidden layers: 256 → 128 neurons, ReLU activations
- Output layer: one Q-value per action (4 phases × 4 duration bins = 16 outputs)
- Experience replay buffer: 50,000 transitions
- Target network updated every 500 steps
- ε-greedy exploration with linear decay from 1.0 to 0.05 over 200,000 steps

**Baseline comparison:** The trained DQN will be evaluated against (a) a fixed 30-second cycle baseline and (b) a Webster-formula actuated baseline.

### 5.4 Module 4 — Emergency Override Module

This module is explicitly rule-based and sits above the RL controller in the call stack. Its logic:

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

After the preemption window ends, control is returned to the RL controller with a state refresh — the density vector is re-read from the detector so the agent does not attempt to continue from a stale state.

---

## 6. Dataset Strategy

|Purpose|Dataset|Notes|
|---|---|---|
|Vehicle detection backbone training|MS COCO (vehicle classes)|80-class COCO filtered to car, truck, bus, motorcycle|
|Intersection-specific fine-tuning|UA-DETRAC|140,000 frames from 24 locations|
|Adverse condition robustness|DAWN Dataset|Rain, fog, sand, snow|
|Emergency vehicle classification|HERO Dataset + manual curation|Augmented with affine transforms and lighting jitter|
|Synthetic intersection rendering|CARLA Simulator|Custom intersection map, controllable EV injection|
|RL training environment|SUMO + OpenStreetMap|Real intersection geometry imported from OSM|
|Evaluation benchmark|Held-out SUMO scenarios|5 traffic demand profiles × 3 time-of-day conditions|

All datasets are publicly available and free for academic use. The CARLA and SUMO synthetic pipelines allow unlimited data generation for long-tail edge cases such as nighttime EV detection and multi-vehicle occlusion.

---

## 7. Technology Stack

|Layer|Tool / Library|Role|
|---|---|---|
|Model development|PyTorch 2.x|All neural network implementation and training|
|Traffic simulation|SUMO 1.19 + TraCI|RL environment, episode generation|
|RL framework|Custom Gym wrapper + Stable-Baselines3|DQN training and evaluation|
|Computer vision|OpenCV 4.x|Frame capture, lane masking, pre-processing|
|Experiment tracking|Weights & Biases|Loss curves, reward plots, model versioning|
|Synthetic data|CARLA 0.9.15|Photorealistic intersection video generation|
|Data handling|NumPy · Pandas · Albumentations|Preprocessing, augmentation pipelines|
|Visualisation|Matplotlib · Seaborn|Evaluation plots and result figures|
|Version control|Git + GitHub|Repository, README, releases|

All tools are open source. The full project is designed to run on a single GPU (minimum: RTX 3060 12GB or equivalent). Mixed precision training (FP16) is used throughout to stay within memory budget.

---

## 8. Training Pipeline

The training is structured in four sequential phases, each of which produces a checkpoint that feeds the next phase.

**Phase 1 — Detector Pre-training**

Train the backbone + neck + detection head on MS COCO vehicle classes from random initialisation. Objective: achieve stable convergence and a mAP@0.5 > 75% on the COCO vehicle subset. This is the longest training phase (~50 epochs on a 80k-image subset).

**Phase 2 — Intersection Fine-tuning**

Fine-tune the full detector on UA-DETRAC and CARLA-generated intersection footage. The lane density estimation head is added and trained here. Density labels are computed programmatically from SUMO ground-truth vehicle positions projected into camera space.

**Phase 3 — EV Head Training**

Freeze the backbone. Train the EV classification head on the HERO dataset + CARLA EV sequences. Use focal loss to handle class imbalance. Target: recall > 0.95 on the EV test set (precision is secondary).

**Phase 4 — RL Controller Training**

Load the trained detector. Deploy it inside the SUMO Gym environment. The detector runs inference on SUMO's rendered camera frames, producing the density state vector every decision step. Train DQN for 500,000 environment steps. Log reward curves, queue length distributions, and throughput metrics via W&B.

**Phase 5 — Integration Testing**

Run the full four-module system end-to-end inside SUMO. Inject EV events at random timesteps. Evaluate EV clearance time, false preemption rate, and impact of preemption on overall network throughput.

---

## 9. Evaluation Metrics

### Detection Metrics (Modules 1 & 2)

|Metric|Description|Target|
|---|---|---|
|mAP@0.5|Mean average precision at IoU 0.5|> 85% (vehicle detector)|
|mAP@0.5:0.95|COCO-style mAP|> 60%|
|EV Recall|True positive rate for EV class|> 95%|
|EV Precision|Positive predictive value for EV|> 85%|
|Inference FPS|Frames processed per second|> 30 FPS on target GPU|

### RL Metrics (Module 3)

|Metric|Description|
|---|---|
|Average waiting time|Mean seconds per vehicle at intersection|
|Max lane queue|Maximum queue length across all lanes (fairness)|
|Throughput|Vehicles cleared per minute|
|Phase efficiency|Fraction of green time with non-zero flow|

All RL metrics are reported relative to the fixed-time baseline as a percentage improvement.

### Integration Metrics (Full System)

|Metric|Description|
|---|---|
|EV clearance time|Seconds from EV detection to intersection clearing|
|Preemption overhead|Extra waiting time imposed on non-EV traffic per event|
|False preemption rate|EV override events triggered without a true EV present|

---

## 10. Innovation Over Prior Work

This project makes three technical contributions that collectively distinguish it from existing literature:

**1. End-to-end trained detection backbone**

Existing papers (Abbas et al. 2024, APIT 2024, Johny & Sharma 2024) use pretrained YOLOv5/v8 weights from the Ultralytics model zoo and apply them as black-box feature extractors. This project trains the full detection backbone from random initialisation on intersection-specific data. This is a stronger and more educational contribution — it demonstrates understanding of the architecture rather than just its API.

**2. Fairness-weighted RL reward function**

Standard RL formulations for traffic control minimise average waiting time across all lanes. This is vulnerable to a degenerate policy that grants excessive green time to always-busy lanes and starves sparse lanes. The reward function here explicitly penalises the maximum queue across lanes (the `α × max(queue)` term), enforcing fairness as a hard constraint in the learned policy. This is a novel reward shaping contribution.

**3. Safe handoff between RL and deterministic override**

The interaction between a learned continuous controller and a hard-interrupt safety system is a generally challenging problem in applied RL. This project implements a clean state-refresh protocol: when the EV override ends, the density state vector is re-read from the detector before the RL controller resumes, preventing value estimation from a stale state. The mechanism is logged and evaluated explicitly, making the safety property measurable rather than assumed.

---

## 11. Project Milestones

|Phase|Deliverable|Success Criterion|
|---|---|---|
|Phase 1|Custom vehicle detector trained from scratch|mAP@0.5 > 85% on held-out test|
|Phase 2|Intersection-specific fine-tuning + density estimation|Per-lane density MAE < 0.08|
|Phase 3|EV classification head|EV recall > 95% on test set|
|Phase 4|Trained DQN signal controller|≥ 20% reduction in avg. waiting time vs fixed-time baseline|
|Phase 5|Full integrated system in SUMO|EV clearance < 8 seconds, false preemption rate < 2%|
|Final|GitHub repository with full documentation|Reproducible README, W&B run logs, demo video|

---

## 12. References

1. Abbas, S. et al. (2024). Vision based intelligent traffic light management system using Faster R-CNN. _CAAI Transactions on Intelligence Technology._ https://doi.org/10.1049/cit2.12309
    
2. Charoenpong, T. et al. (2024). Adaptive traffic light control using vision-based deep learning for vehicle density estimation. _Proceedings of APIT 2024, ACM._ https://doi.org/10.1145/3651623.3651629
    
3. Johny, C. & Sharma, A. (2024). Deep Learning for Emergency Vehicle Identification: A YOLOv8-Based Approach for Smart City Solutions. _Journal of Electrical Systems, 20_(3), 6952–6960.
    
4. Scribano, C. & Muzzini, F. (2025). Real-time traffic signal adjustment using YOLOv8 for improved integration of emergency vehicles in smart traffic systems. _Signal, Image and Video Processing._ https://doi.org/10.1007/s11760-025-04210-8
    
5. Hu, Y. et al. (2024). A multi-agent deep reinforcement learning approach for traffic signal coordination. _IET Intelligent Transport Systems._ https://doi.org/10.1049/itr2.12521
    
6. Yang, G. et al. (2025). Multi-Agent Deep Reinforcement Learning with Graph Attention Network for Traffic Signal Control. _Proceedings of COSITE 2025._
    
7. Wu, Q. et al. (2025). Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control with Spatio-Temporal Attention Mechanism. _Applied Sciences, 15_(15), 8605.
    
8. Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. _Nature, 518_, 529–533.
    
9. Redmon, J. & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. _arXiv:1804.02767._
    
10. Lopez, P.A. et al. (2018). Microscopic Traffic Simulation using SUMO. _Proceedings of IEEE ITSC 2018._
    

---
