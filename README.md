# TideMemory

**TideMemory** is an experimental topological field-memory prototype. It stores memory states as vortex winding numbers in a complex-valued field and studies whether field dynamics can provide robust, self-repairing memory under strong storage noise.

Status: **research prototype / experimental**.

This project is not a production RAG replacement. It explores a complementary memory substrate that may be useful for robust long-term AI memory, agent memory, and physics-inspired retrieval systems.

---

## Latest Validation Results

See [RESULTS.md](RESULTS.md) for the latest v3 validation tables.

Highlights from the current prototype:

- Unified clean sanity, N=8: **1.000 +/- 0.000**
- Unified multi-vortex E_full at sigma=3.0: **1.000**
- Segmented E_full at sigma=3.0: **0.625**
- TideMemory at sigma=3.0: **0.550** vs RAG noise-aware **0.338**
- Multi-symbol probe `{-2, -1, +1, +2}` at sigma=1.2: unified **0.833**

Important: unified multi-vortex results are in an addressable-memory setting where vortex readout centers are known.

---

## Core Idea

TideMemory represents memory using a complex field:

```text
psi(x, y, z) in C
```

A memory item is encoded as a vortex winding state. The readout estimates the winding number:

```text
n_hat = (1 / 2pi) integral d arg(psi)
```

The current prototypes study two storage modes:

1. **Segmented storage**: memory channels occupy separate z-axis segments.
2. **Unified multi-vortex storage**: multiple vortices coexist in one background field using a phase-superposition ansatz.

The main repair mechanism combines:

- GL-style field relaxation
- unified background-field projection
- noise-adaptive resonance frequency
- robust winding-number readout

---

## What This Repository Contains

```text
TideMemory/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── main.py                    # older single-memory benchmark
├── plot_results.py            # older plotting utilities
├── demo_unified.py            # original multi-memory demo
├── demo_unified_v2.py         # optimized segmented-memory demo + RAG comparison
├── demo_unified_v3.py         # validation suite: ablation, fairer RAG, unified vortices, multi-symbol probe
├── figures/                   # generated figures
└── results/
    └── v3_validation_results.md
```

Recommended entry points:

```text
demo_unified_v2.py  -> segmented optimized TideMemory vs RAG
demo_unified_v3.py  -> validation suite and unified multi-vortex experiments
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements:

```text
numpy >= 1.24
matplotlib >= 3.7
torch >= 2.0
```

The current demos run on CPU. GPU is optional but not required.

---

## Quick Start

Run the optimized v2 demo:

```bash
python demo_unified_v2.py
```

Run the v3 validation suite:

```bash
python demo_unified_v3.py
```

The v3 suite can take several minutes on CPU because it runs 3D complex-field dynamics, FFT-based colored resonance noise, ablations, and multiple repeated trials.

---

## Key Mechanisms

### 1. Vortex Memory

Each memory is represented by a vortex winding state:

```text
psi = A(r) exp(i n theta)
```

where `n` is the topological winding number.

### 2. GL-style Relaxation

The field evolves with a Ginzburg-Landau-inspired update:

```text
psi <- psi + dt * (laplacian(psi) + alpha * (V_BG^2 - |psi|^2) * psi)
```

This acts as an attractor-like repair process.

### 3. Unified Background Projection

A background amplitude field pulls corrupted states back toward a stable shell while preserving phase structure.

### 4. Noise-Adaptive Resonance

The resonance frequency is scheduled from a log noise-ratio estimate, with a default log-ratio limit of `2.5`.

### 5. Robust Winding Readout

Readout estimates winding along rings around vortex cores and uses robust statistics to reduce the effect of damaged slices.

---

## Representative Results

The following results are from `demo_unified_v3.py`. Full output is saved in:

```text
results/v3_validation_results.md
```

### A-E Ablation, Segmented Storage, N=8

| Mode | sigma=1.5 | sigma=2.0 | sigma=2.5 | sigma=3.0 |
|---|---:|---:|---:|---:|
| A_base | 0.891 | 0.438 | 0.172 | 0.000 |
| B_bg | 0.938 | 0.656 | 0.578 | 0.375 |
| C_adapt | 0.844 | 0.500 | 0.125 | 0.047 |
| D_bg_adapt | 0.844 | 0.703 | 0.344 | 0.516 |
| E_full | 0.938 | 0.797 | 0.750 | 0.625 |

### A-E Ablation, Unified Multi-Vortex Storage, N=8

Unified clean sanity:

```text
1.000 +/- 0.000
```

| Mode | sigma=1.5 | sigma=2.0 | sigma=2.5 | sigma=3.0 |
|---|---:|---:|---:|---:|
| A_base | 1.000 | 0.958 | 0.125 | 0.000 |
| B_bg | 1.000 | 1.000 | 1.000 | 0.875 |
| C_adapt | 1.000 | 0.979 | 0.104 | 0.000 |
| D_bg_adapt | 1.000 | 1.000 | 0.958 | 0.896 |
| E_full | 1.000 | 1.000 | 1.000 | 1.000 |

Important: the unified multi-vortex result is an **addressable-memory** experiment. The vortex readout centers are known during readout.

### Stable AI Task + Fairer RAG Baseline

| sigma | TideMemory | RAG cosine | RAG noise-aware |
|---:|---:|---:|---:|
| 0.00 | 1.000 | 1.000 | 1.000 |
| 0.50 | 1.000 | 0.787 | 1.000 |
| 1.20 | 1.000 | 0.412 | 0.800 |
| 1.50 | 0.863 | 0.312 | 0.762 |
| 1.80 | 0.875 | 0.212 | 0.625 |
| 2.00 | 0.812 | 0.325 | 0.700 |
| 2.50 | 0.575 | 0.212 | 0.512 |
| 3.00 | 0.550 | 0.175 | 0.338 |

Note: this comparison uses a controlled cosine/noise-aware vector retrieval baseline, not a production RAG system.

### Multi-Symbol Topological Probe

Alphabet:

```text
{-2, -1, +1, +2}
```

| sigma | Segmented | Unified |
|---:|---:|---:|
| 0.50 | 1.000 | 1.000 |
| 1.20 | 0.656 | 0.833 |
| 2.00 | 0.500 | 0.500 |

This suggests a path beyond binary winding memory, but multi-symbol encoding is still preliminary.

---

## Relationship to RAG

TideMemory is not intended as a drop-in replacement for RAG.

A more realistic role is:

```text
encoder -> TideMemory robust memory field -> candidate recall -> RAG/reranker/LLM
```

The current RAG baselines are controlled vector retrieval baselines used to test robustness under storage noise.

---

## Limitations

Please read these limitations before interpreting the results:

1. Current experiments are controlled simulations, not production retrieval benchmarks.
2. Unified multi-vortex experiments are addressable-memory experiments; readout uses known vortex centers.
3. The semantic task is simplified compared with real-world RAG.
4. Current code is CPU-oriented and not optimized.
5. Multi-symbol encoding is preliminary.
6. Blind retrieval, address discovery, address noise, real embedding datasets, and learned semantic-to-vortex encoders are future work.
7. The term "quantum-inspired" should be understood as a computational metaphor involving phase, waves, resonance, and topology. This repository does not claim a new theory of physical quantum mechanics.

---

## Roadmap

Planned next steps:

- Add command-line options for fast/full benchmark modes.
- Add unified-storage AI fair benchmark.
- Add address-noise tests for vortex readout centers.
- Add blind retrieval / address discovery experiments.
- Add learned encoder from semantic embeddings to vortex parameters.
- Optimize field dynamics with GPU/CUDA/Triton or cached FFT filters.
- Add a dual-wave embodied toy-world demo.

---

## Citation / Attribution

If you use this code or build on the idea, please cite or link this repository.

Suggested informal citation:

```text
TideMemory: A topological field-memory prototype with unified vortex encoding and noise-adaptive resonance repair.
```

---

## License

MIT License. See `LICENSE`.
