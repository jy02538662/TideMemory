# TideMemory

**Topological vortex memory using Ginzburg-Landau dynamics — noise-robust, topology-protected information storage.**

---

## What is TideMemory?

TideMemory stores binary information as **topological winding numbers** (±1) of complex-field vortex lines threading a 3D grid. Because winding numbers are topological invariants, they survive substantial noise — the Ginzburg-Landau (GL) evolution layer acts as a physics-based self-repair process that restores corrupted fields back to their attractor.

---

## Architecture

```
Input x ∈ R^32  +  label n ∈ {-1, +1}
          │
          ▼
  ┌───────────────────┐
  │  Conditional      │   Linear(33→256) → ReLU → Linear(256→2·G³)
  │  Encoder          │   output: amp_residual, phi_residual
  └───────────────────┘
          │
          ▼
  ┌───────────────────┐
  │  Vortex Template  │   Analytic vortex: ψ_tpl = V_BG·tanh(r/ξ)·exp(i·n·θ)
  │  × Residual Init  │   ψ₀ = ψ_tpl · exp(i·φ_res) · a_res
  └───────────────────┘
          │
          ▼
  ┌───────────────────┐
  │  GL Evolution     │   ∂ψ/∂t = ∇²ψ + α(V²_BG − |ψ|²)ψ
  │  (self-repair)    │   iterated DT × EVO_STEPS
  └───────────────────┘
          │
          ▼
  ┌───────────────────┐
  │  Winding-Number   │   Sample phase on ring of radius R around vortex core
  │  Readout          │   n̂ = (1/2π) ∮ dθ  →  sign(n̂) = stored bit
  └───────────────────┘
```

**Multi-memory extension** (`demo_unified.py`):  
Partition the z-axis of a `GRID³` field into N equal segments. Each segment independently hosts one vortex, giving N parallel memory channels with zero spatial overlap.

```
z-axis layout (N=4, GRID=64, k=16):
  ┌──────────┬──────────┬──────────┬──────────┐
  │ Memory 0 │ Memory 1 │ Memory 2 │ Memory 3 │
  │  z:0-15  │  z:16-31 │  z:32-47 │  z:48-63 │
  └──────────┴──────────┴──────────┴──────────┘
```

---

## Core Algorithm

| Component | Description |
|---|---|
| **Vortex template** | Analytic GL ground state: amplitude `tanh(r/ξ)`, phase `n·atan2(y,x)` |
| **Residual init** | Neural network predicts small amplitude/phase corrections on top of the template |
| **GL evolution** | Discrete Euler step of the time-dependent GL equation — drives ψ toward nearest topological attractor |
| **Winding readout** | Discrete sum of phase increments along a ring contour; gives continuous estimate of integer winding number |
| **Spatial segmentation** | z-axis partitioned into N bands; each band independent — enables multi-memory without cross-talk |

**Loss function** (training):

```
L = W_wind · ‖n̂ − n‖²       # winding accuracy
  + W_cons · Var_z(n̂)        # z-slice consistency
  + W_smooth · ‖∇²ψ‖²        # field smoothness
  + W_res · ‖residual‖²      # regularize network output
  + W_bg · bg amplitude loss  # background amplitude
  + W_core · core amplitude   # vortex core sharpness
```

---

## File Structure

```
TideMemory/
├── main.py            # Industrial-grade single-memory benchmark
│                      #   Adam + StepLR + grad clip
│                      #   5-run statistics with 95% CI
│                      #   SNR sweep, TPR, attractor-basin, evo-steps, RING_R
│
├── plot_results.py    # Visualization suite
│                      #   Trains network, sweeps all conditions
│                      #   Outputs: fig1_snr_acc.png
│                      #            fig2_evo_steps.png
│                      #            fig3_ring_r.png
│
├── demo_unified.py    # Multi-memory experiment (spatial z-segmentation)
│                      #   Capacity curve: N = 1,2,4,8,16,32
│                      #   AI retrieval: TideMemory vs RAG (cosine)
│                      #   Outputs: fig8_segment_capacity.png
│                      #            fig9_segment_vs_rag.png
│
└── figures/           # All output PNG figures
```

---

## Requirements

```
torch >= 1.12
numpy
matplotlib
```

No GPU required. All experiments run on CPU.

---

## Quick Start

```bash
# 1. Single-memory industrial benchmark (training + full robustness test)
python main.py

# 2. Generate visualization figures
python plot_results.py

# 3. Multi-memory capacity + AI retrieval comparison
python demo_unified.py
```

---

## Key Results

### Single Memory (main.py)

| Condition | Acc (before Evo) | Acc (after Evo) | TPR |
|---|---|---|---|
| Clean | 1.0000 | 1.0000 | 1.0000 |
| Phase noise σ=1.0 | ~0.53 | **0.9994** | ~0.999 |
| Additive noise σ=1.0 | ~0.50 | **0.964** | ~0.96 |

GL evolution recovers **+46% accuracy** under SNR = 0 dB additive noise.

### Multi-Memory Capacity (demo_unified.py)

| N memories | Clean Acc | Noisy Acc (σ=0.5) |
|---|---|---|
| 1 | 1.000 | 1.000 |
| 2 | 1.000 | 1.000 |
| 4 | 1.000 | 1.000 |
| 8 | 1.000 | ~0.98 |
| 16 | ~0.72 | ~0.65 |

### TideMemory vs RAG (AI Retrieval, N=8 items)

| Storage Noise σ | TideMemory Top-1 | RAG (Cosine) Top-1 | Delta |
|---|---|---|---|
| 0.0 | 1.000 | 1.000 | 0% |
| 0.3 | 1.000 | 0.875 | **+12.5%** |
| 0.5 | 1.000 | 0.575 | **+42.5%** |
| 0.8 | 0.975 | 0.625 | **+35.0%** |
| 1.2 | 0.925 | 0.450 | **+47.5%** |

TideMemory significantly outperforms cosine-similarity RAG under high storage noise.

---

## Why Topology?

Winding numbers are **discrete topological invariants** — they cannot be changed by continuous deformation of the field. This means:

- Small perturbations (noise) cannot flip the stored bit
- The GL evolution has discrete attractors at n = ±1, 0
- Recovery is deterministic, not probabilistic

This is fundamentally different from floating-point vector memories (e.g., RAG embeddings), which degrade continuously under noise.

---

## Potential Applications

- **Noise-robust AI memory systems** — long-term memory for LLMs under noisy retrieval/storage conditions
- **Topological quantum computing simulation** — classical emulator of topological qubit encoding
- **Error-correcting memory** — physics-inspired alternative to LDPC / surface codes for soft-decision scenarios
- **Condensed matter / superconductor modeling** — GL field dynamics on discrete lattices

---

## Author Note

This is an independent research prototype demonstrating that topological field dynamics can serve as a practical noise-robust memory primitive. Core algorithm, architecture, and experimental design by a single author.
