# -*- coding: utf-8 -*-
"""
TideMemory — Spatial-Segment Multi-Memory Experiment
======================================================
Extends single-vortex TideMemory to store N independent memories
by partitioning the z-axis of a GRID^3 Ginzburg-Landau field into N segments.

Key Design:
  - Unified background field (V_BG = 1.0) across the entire 3D volume
  - Each memory i occupies z ∈ [i·k, (i+1)·k), k = GRID // N
  - Independent topological vortex per segment; no spatial overlap
  - Standard GL evolution for self-repair after noise injection

Experiments:
  1. Capacity curve  — Acc vs N (N=1,2,4,8,16,32) under noise σ=0~1.2
  2. AI retrieval    — TideMemory vs RAG (cosine) under increasing storage noise

Run:
    python demo_unified.py
"""

import os, math, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cpu")
torch.manual_seed(42)
np.random.seed(42)

# ─── Parameters (aligned with main.py) ───
GRID = 64          # 4 memories → 16 z-layers/seg; 8 memories → 8 z-layers/seg
M    = 16
EPS  = 1e-6
XI   = 1.8
V_BG = 1.0
DT   = 0.05
ALPHA = 1.0
AMP_MAX = 3.0
EVO_STEPS = 8
RING_R = 2.0

Xg = torch.arange(GRID, device=device).float().view(1, GRID, 1)
Yg = torch.arange(GRID, device=device).float().view(1, 1, GRID)

# ─── Centerline generator ───
def gen_centerline(z_count, ring_r, seed=1):
    center = (GRID-1)/2.0
    rng = np.random.default_rng(seed)
    z = np.arange(z_count)
    cx = center + 2.8*(0.6*np.sin(2*np.pi*z/z_count)+0.4*np.sin(4*np.pi*z/z_count+0.7))
    cy = center + 2.8*(0.6*np.cos(2*np.pi*z/z_count+0.2)+0.4*np.cos(4*np.pi*z/z_count+1.1))
    cx += rng.normal(0,0.15,size=z_count)
    cy += rng.normal(0,0.15,size=z_count)
    margin = max(3.0, ring_r+1.5)
    return (torch.tensor(np.clip(cx,margin,GRID-1-margin),dtype=torch.float32,device=device),
            torch.tensor(np.clip(cy,margin,GRID-1-margin),dtype=torch.float32,device=device))

# ─── Single vortex template ───
def make_single_vortex(n_sign, cx_z, cy_z):
    """Build a single vortex field (GRID, GRID, GRID) in complex dtype."""
    psi_re = torch.zeros(GRID,GRID,GRID,device=device)
    psi_im = torch.zeros(GRID,GRID,GRID,device=device)
    for zi in range(GRID):
        dx = (Xg - cx_z[zi]).expand(1,GRID,GRID).squeeze(0)   # (GRID,GRID)
        dy = (Yg - cy_z[zi]).expand(1,GRID,GRID).squeeze(0)
        r  = torch.sqrt(dx*dx + dy*dy + 1e-12)
        A  = torch.tanh(r / XI)
        theta = n_sign * torch.atan2(dy, dx)
        psi_re[:,:,zi] = V_BG * A * torch.cos(theta)
        psi_im[:,:,zi] = V_BG * A * torch.sin(theta)
    return torch.complex(psi_re, psi_im)

# ─── Laplacian ───
def laplacian_roll(psi):
    return (torch.roll(psi,1,0)+torch.roll(psi,-1,0)
           +torch.roll(psi,1,1)+torch.roll(psi,-1,1)
           +torch.roll(psi,1,2)+torch.roll(psi,-1,2) - 6.0*psi)

# ─── GL Evolution ───
def evolve(psi, steps=EVO_STEPS):
    for _ in range(steps):
        lap = laplacian_roll(psi)
        nonlin = ALPHA*(V_BG*V_BG - torch.abs(psi)**2)*psi
        psi = psi + DT*(lap + nonlin)
        amp = torch.abs(psi)
        psi = psi * torch.clamp(AMP_MAX/(amp+1e-12), max=1.0)
    return psi

# ─── Winding-number estimator (identical to main.py) ───
def bilinear_sample_2d(img_xy, x, y):
    B, H, W = img_xy.shape
    x_norm = (y/(W-1))*2-1
    y_norm = (x/(H-1))*2-1
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1)
    inp  = img_xy.unsqueeze(1)
    out  = F.grid_sample(inp, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out[:,0,0,:]

def winding_estimate(psi, cx, cy, ring_r):
    """psi: (1,X,Y,Z), cx/cy: (1,Z) → n_hat: (1,Z)"""
    B = psi.shape[0]
    thetas = torch.linspace(0, 2*math.pi, M+1, device=device)[:-1]
    cos_t = torch.cos(thetas).view(1,1,M).expand(B,GRID,M)
    sin_t = torch.sin(thetas).view(1,1,M).expand(B,GRID,M)
    x_ring = cx.unsqueeze(-1) + ring_r * cos_t
    y_ring = cy.unsqueeze(-1) + ring_r * sin_t
    amp = torch.abs(psi) + EPS
    u   = psi / amp
    n_hat_z = []
    for zi in range(GRID):
        u_re = bilinear_sample_2d(u.real[:,:,:,zi], x_ring[:,zi,:], y_ring[:,zi,:])
        u_im = bilinear_sample_2d(u.imag[:,:,:,zi], x_ring[:,zi,:], y_ring[:,zi,:])
        norm = torch.sqrt(u_re*u_re + u_im*u_im + 1e-12)
        u_re, u_im = u_re/norm, u_im/norm
        u_re_n = torch.roll(u_re, -1, 1)
        u_im_n = torch.roll(u_im, -1, 1)
        re = u_re_n*u_re + u_im_n*u_im
        im = u_im_n*u_re - u_re_n*u_im
        dtheta = torch.atan2(im, re)
        n_hat_z.append(dtheta.sum(dim=1) / (2*math.pi))
    return torch.stack(n_hat_z, dim=1)  # (1, Z)

# ─── Sanity check: single vortex write → evolve → read ───
def test_single_vortex():
    print("  [Sanity Check] Single vortex write → evolve → read ... ", end="")
    cx, cy = gen_centerline(GRID, RING_R, seed=1)
    psi = make_single_vortex(1.0, cx, cy)
    psi_evolved = evolve(psi.clone(), EVO_STEPS)

    psi_b = psi_evolved.unsqueeze(0)         # (1,X,Y,Z)
    cx_b  = cx.view(1, GRID)
    cy_b  = cy.view(1, GRID)
    n_hat = winding_estimate(psi_b, cx_b, cy_b, RING_R)
    mean_n = n_hat.mean().item()
    ok = abs(mean_n - 1.0) < 0.2
    print(f"n_hat_mean={mean_n:.4f}  {'OK' if ok else 'FAIL'}")
    return ok

# ─── Spatial-segment multi-memory: each memory occupies an independent z-band ───
def write_segmented_field(N, seeds, n_signs):
    """
    Divide GRID z-layers equally among N memories.
    Memory i occupies z ∈ [i*k, (i+1)*k), k = GRID // N.
    Each segment gets an independent vortex; non-segment layers hold V_BG background.
    """
    k = GRID // N
    channels = []
    # Initialize with uniform V_BG background
    field = torch.complex(
        torch.full((GRID,GRID,GRID), V_BG, device=device),
        torch.zeros(GRID,GRID,GRID, device=device))

    for i in range(N):
        z_s = i * k
        z_e = (i + 1) * k
        cx_z, cy_z = gen_centerline(k, RING_R, seed=int(seeds[i]))
        # Write vortex into z ∈ [z_s, z_e)
        for zi_local in range(k):
            zi = z_s + zi_local
            if zi_local < len(cx_z):
                dx = (Xg.squeeze() - cx_z[zi_local])
                dy = (Yg.squeeze() - cy_z[zi_local])
                # 需要正确 meshgrid
                dx_2d = dx.unsqueeze(1).expand(GRID, GRID)
                dy_2d = dy.unsqueeze(0).expand(GRID, GRID)
                r = torch.sqrt(dx_2d*dx_2d + dy_2d*dy_2d + 1e-12)
                A = torch.tanh(r / XI)
                theta = n_signs[i] * torch.atan2(dy_2d, dx_2d)
                field[:,:,zi] = V_BG * A * torch.exp(1j * theta)
        channels.append({
            'i': i, 'z_s': z_s, 'z_e': z_e, 'k': k,
            'cx_z': cx_z, 'cy_z': cy_z, 'n_sign': n_signs[i],
        })
    return field, channels

def read_segment(field, ch):
    """读取单个 z 段的绕数"""
    seg = field[:, :, ch['z_s']:ch['z_e']]  # (GRID, GRID, k)
    # 需要在 z 方向也做成完整维度给 winding_estimate
    # 方法：直接在段内逐 z 层测量
    nz = ch['k']
    thetas = torch.linspace(0, 2*math.pi, M+1, device=device)[:-1]
    amp = torch.abs(seg) + EPS
    u = seg / amp

    n_hats = []
    for zi in range(nz):
        if zi >= len(ch['cx_z']):
            break
        x_ring = ch['cx_z'][zi] + RING_R * torch.cos(thetas)  # (M,)
        y_ring = ch['cy_z'][zi] + RING_R * torch.sin(thetas)
        # 双线性采样
        H, W = GRID, GRID
        x_n = (y_ring/(W-1))*2-1
        y_n = (x_ring/(H-1))*2-1
        grid = torch.stack([x_n, y_n], -1).view(1,1,-1,2)
        u_re_slice = u.real[:,:,zi].unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        u_im_slice = u.imag[:,:,zi].unsqueeze(0).unsqueeze(0)
        ur = F.grid_sample(u_re_slice, grid, mode="bilinear",
                           padding_mode="border", align_corners=True)[0,0,0,:]
        ui = F.grid_sample(u_im_slice, grid, mode="bilinear",
                           padding_mode="border", align_corners=True)[0,0,0,:]
        nm = torch.sqrt(ur*ur+ui*ui+1e-12)
        ur, ui = ur/nm, ui/nm
        ur_n = torch.roll(ur,-1); ui_n = torch.roll(ui,-1)
        dth = torch.atan2(ui_n*ur-ur_n*ui, ur_n*ur+ui_n*ui)
        n_hats.append(dth.sum()/(2*math.pi))

    if len(n_hats) == 0:
        return 0.0
    return torch.stack(n_hats).mean().item()


# ─── Capacity experiment ───
def run_capacity(n_list, noise_sigmas, runs=5):
    results = {}
    for N in n_list:
        print(f"  N={N:2d} ...", end="", flush=True)
        accs = {s: [] for s in noise_sigmas}
        for run in range(runs):
            rng = np.random.default_rng(run*100+N*7)
            seeds = rng.integers(10,9990,size=N).tolist()
            n_signs = [1.0 if rng.random()>0.5 else -1.0 for _ in range(N)]
            field, channels = write_segmented_field(N, seeds, n_signs)
            # 写入后演化稳定
            field = evolve(field, EVO_STEPS)

            for sigma in noise_sigmas:
                if sigma > 0:
                    noise = sigma*(torch.randn_like(field.real)+1j*torch.randn_like(field.real))
                    fn = field + noise
                else:
                    fn = field.clone()
                # 演化修复
                fn = evolve(fn, EVO_STEPS)

                correct = 0
                for ch in channels:
                    w = read_segment(fn, ch)
                    if abs(w - ch['n_sign']) < 0.4:
                        correct += 1
                accs[sigma].append(correct / N)

        results[N] = {s: (np.mean(v), 1.96*np.std(v,ddof=1)/math.sqrt(runs) if runs>1 else 0)
                      for s,v in accs.items()}
        print(f"  clean={results[N][0.0][0]:.3f}  noisy(0.5)={results[N][0.5][0]:.3f}")
    return results


# ─── AI retrieval task: TideMemory vs RAG (cosine similarity) ───
def run_ai_task(N_ITEMS=4, RUNS=5, EMBED_DIM=64):
    import torch.nn as nn
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
    topo_top1 = {s: [] for s in noise_levels}
    rag_top1  = {s: [] for s in noise_levels}

    class MapNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(EMBED_DIM,64),nn.ReLU(),nn.Linear(64,1))
        def forward(self,x): return torch.sign(self.fc(x).squeeze(-1))

    mapper = MapNet().to(device)
    opt = torch.optim.Adam(mapper.parameters(),lr=1e-3)
    for _ in range(300):
        emb=torch.randn(32,EMBED_DIM,device=device)
        lbl=(2*(emb[:,0]>0).float()-1)
        loss=F.binary_cross_entropy_with_logits(mapper.fc(emb).squeeze(-1),(lbl+1)/2)
        opt.zero_grad();loss.backward();opt.step()

    rng=np.random.default_rng(0)
    for run in range(RUNS):
        seeds=rng.integers(10,9990,size=N_ITEMS).tolist()
        db_emb=F.normalize(torch.randn(N_ITEMS,EMBED_DIM,device=device),dim=-1)
        mapper.eval()
        with torch.no_grad(): n_signs=mapper(db_emb).tolist()
        field,channels=write_segmented_field(N_ITEMS,seeds,n_signs)
        field=evolve(field,EVO_STEPS)
        for sigma in noise_levels:
            if sigma>0:
                noise=sigma*(torch.randn_like(field.real)+1j*torch.randn_like(field.real))
                fn=field+noise
            else: fn=field.clone()
            fn=evolve(fn,EVO_STEPS)
            t1t=0;t1r=0
            for q in range(N_ITEMS):
                w=read_segment(fn,channels[q])
                if abs(w-n_signs[q])<0.4: t1t+=1
                db_n=db_emb.clone()
                if sigma>0: db_n=db_n+sigma*0.8*torch.randn_like(db_n)
                q_sim=F.cosine_similarity(db_emb[q].unsqueeze(0),db_n,dim=-1)
                if q_sim.argmax().item()==q: t1r+=1
            topo_top1[sigma].append(t1t/N_ITEMS)
            rag_top1[sigma].append(t1r/N_ITEMS)

    def agg(d):
        return{s:(np.mean(v),1.96*np.std(v,ddof=1)/math.sqrt(RUNS))for s,v in d.items()}
    return agg(topo_top1),agg(rag_top1),noise_levels


# ─── Plotting ───
def plot_capacity(results, n_list, save_path):
    fig,ax = plt.subplots(figsize=(9,5.5),dpi=150)
    fig.suptitle("TideMemory · Spatial-Segment Multi-Memory Capacity\n"
                 "z-Slice Multiplexing + Topological Protection",
                 fontsize=14,fontweight="bold",y=1.02)
    colors=["#4C72B0","#DD8452","#55A868","#C44E52"]
    labels={0.0:"No Noise",0.3:"σ=0.3",0.5:"σ=0.5",0.8:"σ=0.8"}
    for idx,sigma in enumerate([0.0,0.3,0.5,0.8]):
        mus=[results[N][sigma][0] for N in n_list]
        cis=[results[N][sigma][1] for N in n_list]
        ax.fill_between(n_list,[m-c for m,c in zip(mus,cis)],[m+c for m,c in zip(mus,cis)],alpha=0.12,color=colors[idx])
        ax.plot(n_list,mus,"o-",color=colors[idx],lw=2,ms=6,label=labels[sigma])
    ax.axhline(0.95,color="tomato",lw=1,ls=":",label="Threshold 0.95")
    ax.set_xlabel("Number of Memories N",fontsize=11);ax.set_ylabel("Recall Accuracy",fontsize=11)
    ax.legend(fontsize=9,framealpha=0.88);ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
    ax.grid(axis="y",color="lightgray",lw=0.5);ax.set_ylim(-0.02,1.12);ax.tick_params(which="both",direction="in")
    plt.tight_layout();fig.savefig(save_path,dpi=150,bbox_inches="tight");plt.close(fig)
    print(f"\n  [Capacity] Saved -> {save_path}")

def plot_ai(topo,rag,nl,save_path):
    fig,ax=plt.subplots(figsize=(8,5.5),dpi=150)
    fig.suptitle("TideMemory (Spatial-Segment) vs RAG\nTop-1 Recall Under Storage Noise",
                 fontsize=14,fontweight="bold",y=1.02)
    tm=[topo[s][0] for s in nl];tc=[topo[s][1] for s in nl]
    rm=[rag[s][0]  for s in nl];rc=[rag[s][1]  for s in nl]
    ax.fill_between(nl,[m-c for m,c in zip(tm,tc)],[m+c for m,c in zip(tm,tc)],alpha=0.12,color="#DD8452")
    ax.fill_between(nl,[m-c for m,c in zip(rm,rc)],[m+c for m,c in zip(rm,rc)],alpha=0.12,color="#4C72B0")
    ax.plot(nl,tm,"s-",color="#DD8452",lw=2.5,ms=7,label="TideMemory")
    ax.plot(nl,rm,"o--",color="#4C72B0",lw=2.5,ms=7,label="RAG (Cosine)")
    deltas=[t-r for t,r in zip(tm,rm)]
    mi=int(np.argmax(deltas))
    if deltas[mi]>0.02:
        ax.annotate(f"+{deltas[mi]*100:.0f}%",xy=(nl[mi],tm[mi]),xytext=(nl[mi]-0.2,tm[mi]+0.05),
                    fontsize=11,color="#DD8452",fontweight="bold",arrowprops=dict(arrowstyle="->",color="#DD8452",lw=1.2))
    ax.set_xlabel("Storage Noise σ",fontsize=11);ax.set_ylabel("Top-1 Recall",fontsize=11)
    ax.set_ylim(-0.02,1.12);ax.legend(fontsize=10,framealpha=0.88)
    ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
    ax.grid(axis="y",color="lightgray",lw=0.5);ax.tick_params(which="both",direction="in")
    plt.tight_layout();fig.savefig(save_path,dpi=150,bbox_inches="tight");plt.close(fig)
    print(f"  [AI Task] Saved -> {save_path}")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  TideMemory · Spatial-Segment Multi-Memory")
    print(f"  Grid={GRID}^3  V_BG={V_BG}  EvoSteps={EVO_STEPS}")
    print("=" * 60)

    # 基础验证
    ok = test_single_vortex()
    if not ok:
        print("  !! Sanity check failed, aborting.")
        exit(1)

    # 容量实验
    n_list = [1, 2, 4, 8, 16, 32]
    noise_sigmas = [0.0, 0.3, 0.5, 0.8, 1.2]
    print("\n[Experiment 1] Capacity")
    t0 = time.time()
    cap = run_capacity(n_list, noise_sigmas, runs=5)
    print(f"\n  Elapsed: {time.time()-t0:.1f}s")

    print("\n" + "=" * 60)
    for sigma in [0.0, 0.5]:
        lbl = "Clean" if sigma == 0 else f"Noisy(σ={sigma})"
        print(f"  [{lbl}]")
        for N in n_list:
            mu, ci = cap[N][sigma]
            print(f"    N={N:3d}  Acc={mu:.4f} ± {ci:.4f}")
    plot_capacity(cap, n_list, os.path.join(out_dir, "fig8_segment_capacity.png"))

    # AI 任务
    print("\n[Experiment 2] TideMemory vs RAG")
    t1 = time.time()
    topo, rag, nl = run_ai_task(N_ITEMS=8, RUNS=5)
    print(f"  Elapsed: {time.time()-t1:.1f}s")
    print(f"\n  {'σ':>6} | {'TideMemory':>10} | {'RAG':>10} | {'Delta':>8}")
    print("  " + "-" * 44)
    for s in nl:
        tm=topo[s][0]; rm=rag[s][0]
        print(f"  {s:6.2f} | {tm:>10.4f} | {rm:>10.4f} | {(tm-rm)*100:>+7.1f}%")
    plot_ai(topo, rag, nl, os.path.join(out_dir, "fig9_segment_vs_rag.png"))

    print("\n" + "=" * 60)
    print("  Done.")