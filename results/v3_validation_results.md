# TideMemory v3 validation output

Run command:

```bash
python demo_unified_v3.py
```

Environment:

- Device: CPU
- Grid: 64^3
- Runs: segmented ablation 8, unified ablation 6, fair RAG 10, multi-symbol 8/6

## [0] Unified clean sanity, N=8

| Setting | Accuracy |
|---|---:|
| Unified clean sanity | 1.000 +/- 0.000 |

## [1] A-E ablation, segmented storage, N=8

| Mode | sigma=1.5 | sigma=2.0 | sigma=2.5 | sigma=3.0 |
|---|---:|---:|---:|---:|
| A_base | 0.891 | 0.438 | 0.172 | 0.000 |
| B_bg | 0.938 | 0.656 | 0.578 | 0.375 |
| C_adapt | 0.844 | 0.500 | 0.125 | 0.047 |
| D_bg_adapt | 0.844 | 0.703 | 0.344 | 0.516 |
| E_full | 0.938 | 0.797 | 0.750 | 0.625 |

## [4] A-E ablation, unified multi-vortex storage, N=8

| Mode | sigma=1.5 | sigma=2.0 | sigma=2.5 | sigma=3.0 |
|---|---:|---:|---:|---:|
| A_base | 1.000 | 0.958 | 0.125 | 0.000 |
| B_bg | 1.000 | 1.000 | 1.000 | 0.875 |
| C_adapt | 1.000 | 0.979 | 0.104 | 0.000 |
| D_bg_adapt | 1.000 | 1.000 | 0.958 | 0.896 |
| E_full | 1.000 | 1.000 | 1.000 | 1.000 |

## [2-3] Stable AI task + fairer RAG baseline

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

Note: this fair-RAG comparison currently uses the segmented TideMemory storage path.

## [5] Multi-symbol topological probe, alphabet={-2,-1,+1,+2}, N=4

| sigma | Segmented | Unified |
|---:|---:|---:|
| 0.50 | 1.000 | 1.000 |
| 1.20 | 0.656 | 0.833 |
| 2.00 | 0.500 | 0.500 |

## Interpretation

- The segmented path shows clear gains from background-field projection, adaptive resonance, and robust readout.
- The unified multi-vortex ansatz is viable in the addressable-memory setting.
- The current RAG baseline is a controlled cosine/noise-aware vector retrieval baseline, not a production RAG stack.
- The multi-symbol result suggests a path beyond binary winding memory, but it is preliminary.
