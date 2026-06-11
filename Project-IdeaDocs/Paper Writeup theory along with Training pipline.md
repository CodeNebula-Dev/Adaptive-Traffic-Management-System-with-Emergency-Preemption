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
    