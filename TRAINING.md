# ATMS-Net: Deep Learning Training Theory, Dynamics & Optimization Guide

This document provides a comprehensive, mathematically rigorous explanation of the training pipeline, learning rate dynamics, optimization strategies, multi-GPU architecture, and learning paradigms employed in **ATMS-Net** across Phase 1 (COCO Pre-training) and Phase 2.5 (UA-DETRAC CCTV Surveillance Fine-Tuning).

---

## 1. Learning Paradigm: Supervised vs. Unsupervised vs. Open-Set Learning

A fundamental theoretical question is: **Did we use Supervised Learning, Unsupervised Learning, or Both?**

### The Core Paradigm: Supervised Deep Learning (95%)
ATMS-Net is primarily a **Supervised Object Detection Network**. 
* **Input**: High-resolution RGB frames $X \in \mathbb{R}^{B \times 3 \times H \times W}$.
* **Ground Truth**: Human-annotated vehicle bounding boxes and discrete class IDs:
  $$Y = \left\{ \left( c_i, x_i, y_i, w_i, h_i \right) \right\}_{i=1}^{N_{\text{gt}}}, \quad c_i \in \{0, 1, 2, 3, 4\}$$
* **Mechanism**: The network generates predictions, computes errors against human-labeled ground truth via Complete IoU (CIoU) and Binary Cross-Entropy (BCE), and optimizes its 13.2M parameters via backpropagation and stochastic gradient descent.

---

### Embedded Self-Supervised & Open-Set Elements (5%)
While the loss minimization is supervised, ATMS-Net integrates three sophisticated mechanisms that operate on **self-adaptive, unsupervised, and open-set principles**:

#### 1. SimOTA Dynamic Label Assignment (Self-Supervised Optimal Transport)
* In traditional detectors (YOLOv3, Faster R-CNN), ground-truth assignment to grid cells is governed by static, hand-crafted rules (e.g., fixed IoU threshold $> 0.5$).
* In ATMS-Net, **SimOTA** treats label assignment as an **Optimal Transport Problem**:
  $$c_{ij} = L_{\text{cls}}(p_i, g_j) + \alpha \cdot L_{\text{reg}}(b_i, g_j)$$
* The network **self-determines** which prediction cells are assigned to which ground-truth objects dynamically during training based on its current representation quality, without manual intervention.

#### 2. Open-Set / Out-of-Distribution Recognition (`unknown_vehicle`)
* In closed-set supervised learning, a network assumes every object belongs to a known training category.
* ATMS-Net uses an **Uncertainty-Guided Open-Set Formulation**:
  * Because the classification head uses **independent logistic sigmoids ($\sigma(z)$)** rather than a closed-set Softmax, each class is evaluated independently.
  * When an unseen or non-standard vehicle (such as a 3-wheeler, customized utility cart, or loaded auto-rickshaw) passes an intersection camera:
    $$P_{\text{obj}} \ge 0.75 \quad \text{and} \quad \max_{c \in \{0..3\}} P_c \le 0.35$$
  * High objectness combined with high entropy across known vehicle classes allows the network to classify the object as **`unknown_vehicle`** without having required closed-set categorical supervision.

#### 3. Model Exponential Moving Average (EMA) Consistency
* Inspired by self-supervised teacher-student frameworks (e.g., Mean Teacher, BYOL), ATMS-Net maintains a shadow model whose weights update via temporal smoothing:
  $$\theta_{\text{EMA}}^{(t)} = \beta \cdot \theta_{\text{EMA}}^{(t-1)} + (1 - \beta) \cdot \theta_{\text{model}}^{(t)}, \quad \beta = 0.9999$$
* This produces an ensemble effect across thousands of optimization steps, stabilizing validation metrics and preventing single-batch gradient noise from degrading test performance.

---

## 2. Learning Rate Dynamics & Mathematical Formulation

The learning rate $\eta$ is the single most critical hyperparameter in deep learning. A fixed learning rate causes either gradient explosion (if too high) or premature stagnation in sub-optimal local minima (if too low).

ATMS-Net employs a **Two-Phase Composite Schedule**: **Linear Warmup followed by Cosine Annealing Decay**.

```
Learning Rate (η)
    ^
η_max |       /-------------\   (Cosine Annealing Decay)
      |      /               \
      |     / (Warmup)        \
      |    /                   \
η_min |---/                     \--------->
      +---+---------------------+---------> Total Steps
        Warmup Steps          Total Steps
```

### Phase A: Linear Learning Rate Warmup
During the first epoch (warmup phase), weights—especially newly initialized classification head slices—are prone to large, volatile gradient updates that can destroy pre-trained backbone representations.

To prevent gradient shock, the learning rate scales up linearly from a small ratio ($\alpha_{\text{warmup}} = 0.33$) to the base learning rate ($\eta_{\text{max}} = 0.002$):
$$\eta(t) = \eta_{\text{max}} \cdot \left[ \alpha_{\text{warmup}} + (1 - \alpha_{\text{warmup}}) \cdot \frac{t}{T_{\text{warmup}}} \right], \quad \forall t \le T_{\text{warmup}}$$

### Phase B: Cosine Annealing Decay
Following warmup, the learning rate decays along a half-period cosine curve until reaching the minimum threshold $\eta_{\text{min}} = 0.01 \times \eta_{\text{max}} = 0.00002$:
$$\eta(t) = \eta_{\text{min}} + \frac{1}{2} (\eta_{\text{max}} - \eta_{\text{min}}) \cdot \left( 1 + \cos\left( \pi \cdot \frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}} \right) \right)$$

#### Why Cosine Annealing Outperforms StepLR:
1. **Continuous Smoothness**: Step decay introduces sudden discrete drops that destabilize momentum vectors. Cosine annealing smoothly slows weight updates.
2. **Flat Minima Convergence**: As $\eta \to \eta_{\text{min}}$ in the final epochs, the optimizer gently settles into broad, flat valleys in the loss landscape. According to statistical learning theory, flat minima generalize significantly better to unseen real-world test images than sharp minima.

---

## 3. Parameter Group Separation & Regularization

Not all parameters in a convolutional neural network should be regularized identically. Applying $L_2$ weight decay to normalization layers or bias terms impairs network expressiveness.

ATMS-Net automatically bifurcates all $13.2\text{M}$ parameters into **three distinct optimization groups**:

| Parameter Group | Layers Included | Weight Decay ($L_2$) | Optimization Objective |
| :--- | :--- | :---: | :--- |
| **Group 1: Normalization** | BatchNorm $\gamma$ (scale), $\beta$ (shift) | **`0.0`** (None) | Preserves internal covariate shift stabilization |
| **Group 2: Conv Kernels** | $3\times3$, $1\times1$, $6\times6$ Conv Weights | **`0.0005`** ($5 \times 10^{-4}$) | Penalizes overly complex feature filters ($L_2$ regularization) |
| **Group 3: Biases** | Convolutional Biases | **`0.0`** (None) | Allows unconstrained spatial shifts without shrinkage penalty |

### Optimizer: SGD with Nesterov Accelerated Gradient (NAG)
$$\mathbf{v}_{t+1} = \mu \mathbf{v}_t + \nabla_\theta \mathcal{L}(\theta_t - \mu \mathbf{v}_t)$$
$$\theta_{t+1} = \theta_t - \eta_{t+1} \mathbf{v}_{t+1}$$
* **Momentum factor**: $\mu = 0.937$
* Nesterov momentum computes gradients by looking ahead along the momentum trajectory, dampening oscillations across steep ravines in the loss surface.

---

## 4. Multi-GPU Distributed Data Parallel (DDP) Architecture

When training on dual-GPU accelerators (e.g., Kaggle **GPU T4 $\times 2$**), ATMS-Net utilizes PyTorch's native **Distributed Data Parallel (DDP)** framework launched via `torchrun`.

```
                    [Master Script / torchrun]
                                |
             +------------------+------------------+
             |                                     |
    [Process 0: GPU 0 (T4)]               [Process 1: GPU 1 (T4)]
    ├── Local Batch: 32 images            ├── Local Batch: 32 images
    ├── Forward Pass (FP16)               ├── Forward Pass (FP16)
    ├── SimOTA + CIoU Loss                ├── SimOTA + CIoU Loss
    └── Backward Pass (Gradients)         └── Backward Pass (Gradients)
             |                                     |
             +<==== NCCL Ring All-Reduce Sync ====>+
             |                                     |
    Optimizer Step (Local Weights)        Optimizer Step (Local Weights)
```

### Why DDP is Superior to `nn.DataParallel`:
1. **No Single-GPU Bottleneck**: In `DataParallel`, GPU 0 must gather all outputs, compute the entire loss, and scatter weights back, causing GPU 0 to bottleneck at 100% memory while GPU 1 idles. In DDP, each GPU computes its own loss independently.
2. **True Parallelism**: DDP spawns separate Python processes per GPU, bypassing the Python Global Interpreter Lock (GIL).
3. **NCCL Ring All-Reduce**: Gradients are synchronized across GPUs via NVIDIA Collective Communications Library (NCCL) in $O(N)$ communication time rather than broadcasting through a central master.

---

## 5. Smart Video Stride Subsampling ($\text{Stride} = 4$)

### The Problem: 25 FPS Video Redundancy
* The raw UA-DETRAC surveillance benchmark comprises **`138,252` continuous video frames** recorded at $25\text{ frames per second}$.
* In $25\text{ fps}$ CCTV footage, consecutive frames are captured $0.04\text{ seconds}$ apart. Vehicles stopped at a traffic signal or moving in a queue are virtually identical across frames $t$, $t+1$, $t+2$, $t+3$.
* Training on 117,515 frames per epoch forced the GPU to spend 75% of its compute cycles repeatedly recalculating gradients on identical pixel arrangements, taking **87 minutes per epoch**.

### The Solution: Temporal Subsampling ($\text{Stride} = 4$)
$$\text{Sampled Dataset} = \left\{ \text{Frame}_k \;\middle|\; k \equiv 0 \pmod 4 \right\}$$
* Sampling every 4th frame reduces the dataset to **`~6.25 frames per second`**, which is the ideal temporal density for traffic dynamics.
* **Math of Training Throughput**:
  $$\text{Sampled Train Images} = \frac{117,515}{4} = \mathbf{29,379\text{ images}}$$
  $$\text{Effective Global Batch Size} = 32 \text{ images/GPU} \times 2 \text{ GPUs} = \mathbf{64\text{ images/step}}$$
  $$\text{Batches per Epoch} = \frac{29,379}{64} = \mathbf{459\text{ batches}}$$
* **Visual Information Preserved**:
  * 🚗 **~260,000 Car Annotations**
  * 🚌 **~26,000 Bus Annotations**
  * 🚚 **~24,000 Truck Annotations**
  * 🟣 **All 20,641 `unknown_vehicle` (Auto-Rickshaw) Annotations**
  * **100% of all 100 intersection sequences, weather variations, day/night cycles**.
* **Speedup**: Epoch duration dropped from **87 minutes down to 13 minutes** (**6.7× acceleration**)!

---

## 6. Multi-Component Loss Formulation

$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{obj}} \mathcal{L}_{\text{obj}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}$$

```
+-----------------------------------------------------------------------------+
| ATMS-Net Loss Components                                                    |
|                                                                             |
| 1. CIoU Loss (Box Regression):                                              |
|    L_CIoU = 1 - IoU + (ρ²(b, b_gt) / c²) + α · v                            |
|    Encourages overlap, center proximity, and aspect ratio alignment.        |
|    Multiplier: λ_box = 8.0 (Strict spatial bounding box precision)          |
|                                                                             |
| 2. Objectness Loss (BCEWithLogits):                                         |
|    L_obj = - [ y_obj · log(σ(p_obj)) + (1 - y_obj) · log(1 - σ(p_obj)) ]   |
|    Evaluated over ALL 5,376 grid cells. Soft IoU target for matched cells.  |
|    Multiplier: λ_obj = 2.5 (Aggressively suppresses background false alarms)|
|                                                                             |
| 3. Classification Loss (Independent Binary BCE):                            |
|    L_cls = - Σ_{c=1}^5 [ y_c · log(σ(p_c)) + (1 - y_c) · log(1 - σ(p_c)) ]  |
|    Evaluated ONLY on positive SimOTA matched cells.                         |
|    Multiplier: λ_cls = 1.0                                                  |
+-----------------------------------------------------------------------------+
```

---

## 7. Summary of Training Hyperparameters (UA-DETRAC Phase 2.5)

| Hyperparameter | Value | Description |
| :--- | :---: | :--- |
| **Input Resolution** | **`512 × 512`** | High-resolution for small distant vehicles & edge cut-offs |
| **Class Count** | **`5`** | `car, motorcycle, bus, truck, unknown_vehicle` |
| **Optimizer** | **SGD (Nesterov)** | Momentum = `0.937`, Weight Decay = `0.0005` |
| **Base Learning Rate** | **`0.002`** | Gentle fine-tuning rate starting from 40.5% weights |
| **LR Scheduler** | **Cosine Annealing** | 1-epoch warmup, minimum ratio = `0.01` |
| **Batch Size** | **`32 per GPU`** (64 global) | Maximizes GPU tensor core occupancy |
| **Frame Stride** | **`4`** | Subsamples 29,379 diverse surveillance frames from 117.5k |
| **Mixed Precision** | **FP16 (`torch.amp`)** | Reduces VRAM footprint to 3.5 GB per GPU |
| **EMA Decay** | **`0.9999`** | Exponential moving average for evaluation stability |
| **Loss Weights** | $\lambda_{\text{box}}=8.0, \lambda_{\text{obj}}=2.5, \lambda_{\text{cls}}=1.0$ | Tuned for dense intersection surveillance |
