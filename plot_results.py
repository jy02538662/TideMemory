# -*- coding: utf-8 -*-
"""
TideMemory — Visualization Suite
==================================
Trains TopoGenesisNet B then generates three industrial-grade analysis figures:
  Fig 1 — SNR vs Winding-Number Accuracy (phase noise vs additive noise)
  Fig 2 — Evolution Steps vs Accuracy (noise σ=1.0)
  Fig 3 — Measurement Ring Radius (RING_R) vs Accuracy

Output: figures/fig1_snr_acc.png / fig2_evo_steps.png / fig3_ring_r.png

Run:
    python plot_results.py
"""

import os, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")          # 非交互后端，无需 GUI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator

# ─────────────────────── 全局配置（与 main.py 保持一致）────────────────────
device = torch.device("cpu")
torch.manual_seed(42)
np.random.seed(42)

GRID = 16
M    = 16
EPS  = 1e-6
RING_R_DEFAULT = 2.0
XI   = 1.8
V_BG = 1.0
DT   = 0.05
ALPHA = 1.0
EVO_STEPS_TRAIN = 6
AMP_MAX = 3.0
PHI_SCALE = 0.25
AMP_SCALE = 0.25
W_WIND=1.0; W_CONS=0.2; W_SMOOTH=0.01; W_RES=0.02; W_BG=0.05; W_CORE=0.05
TRAIN_STEPS   = 400
BATCH_SIZE    = 16
LR            = 1e-3
LR_DECAY_STEP = 200
LR_DECAY_GAMMA = 0.5
GRAD_CLIP     = 1.0
EVAL_RUNS     = 6          # 多轮统计次数
EVAL_BATCH    = 64

# ─────────────────────── 核心算法（完全复用）────────────────────────────────
def gen_centerline_np(z_count, ring_r, drift=2.8, bend=1.0, seed=1):
    center = (GRID-1)/2.0
    rng = np.random.default_rng(seed)
    z = np.arange(z_count)
    cx = center + drift*(0.6*np.sin(2*np.pi*z/z_count)+0.4*np.sin(4*np.pi*z/z_count+0.7))
    cy = center + drift*(0.6*np.cos(2*np.pi*z/z_count+0.2)+0.4*np.cos(4*np.pi*z/z_count+1.1))
    cx += bend*rng.normal(0,0.15,size=z_count)
    cy += bend*rng.normal(0,0.15,size=z_count)
    margin = max(3.0, ring_r+1.5)
    return np.clip(cx, margin, GRID-1-margin), np.clip(cy, margin, GRID-1-margin)

def build_centerline_tensors(ring_r, seed=1):
    cx_np, cy_np = gen_centerline_np(GRID, ring_r, seed=seed)
    return torch.from_numpy(cx_np).float().to(device), torch.from_numpy(cy_np).float().to(device)

Xg = torch.arange(GRID, device=device).float().view(1,GRID,1)
Yg = torch.arange(GRID, device=device).float().view(1,1,GRID)

def vortex_template(n_target, cx_z, cy_z):
    B = n_target.shape[0]
    psi_re = torch.zeros(B,GRID,GRID,GRID,device=device)
    psi_im = torch.zeros(B,GRID,GRID,GRID,device=device)
    amp_tpl= torch.zeros(B,GRID,GRID,GRID,device=device)
    for zi in range(GRID):
        dx = (Xg - cx_z[zi]).expand(B,GRID,GRID)
        dy = (Yg - cy_z[zi]).expand(B,GRID,GRID)
        r  = torch.sqrt(dx*dx+dy*dy+1e-12)
        A  = torch.tanh(r/XI)
        theta = n_target.view(B,1,1)*torch.atan2(dy,dx)
        psi_re[:,:,:,zi] = V_BG*A*torch.cos(theta)
        psi_im[:,:,:,zi] = V_BG*A*torch.sin(theta)
        amp_tpl[:,:,:,zi]= V_BG*A
    return torch.complex(psi_re,psi_im), amp_tpl

def laplacian_roll(psi):
    return (torch.roll(psi,1,1)+torch.roll(psi,-1,1)
           +torch.roll(psi,1,2)+torch.roll(psi,-1,2)
           +torch.roll(psi,1,3)+torch.roll(psi,-1,3) - 6.0*psi)

class EvolutionLayer(nn.Module):
    def __init__(self, alpha=1.0, dt=0.05, steps=6, amp_max=3.0):
        super().__init__()
        self.alpha=float(alpha); self.dt=float(dt)
        self.steps=int(steps);   self.amp_max=float(amp_max)
    def forward(self, psi):
        for _ in range(self.steps):
            lap   = laplacian_roll(psi)
            nonlin= self.alpha*(V_BG*V_BG-torch.abs(psi)**2)*psi
            psi   = psi + self.dt*(lap+nonlin)
            amp   = torch.abs(psi)
            psi   = psi * torch.clamp(self.amp_max/(amp+1e-12), max=1.0)
        return psi

def bilinear_sample_2d(img_xy, x, y):
    B,H,W = img_xy.shape
    grid = torch.stack([(y/(W-1))*2-1, (x/(H-1))*2-1], dim=-1).unsqueeze(1)
    out  = F.grid_sample(img_xy.unsqueeze(1), grid,
                         mode="bilinear", padding_mode="border", align_corners=True)
    return out[:,0,0,:]

def winding_estimate_per_z(psi, cx, cy, ring_r):
    B      = psi.shape[0]
    thetas = torch.linspace(0,2*math.pi,M+1,device=psi.device)[:-1]
    cos_t  = torch.cos(thetas).view(1,1,M).expand(B,GRID,M)
    sin_t  = torch.sin(thetas).view(1,1,M).expand(B,GRID,M)
    x_ring = cx.unsqueeze(-1)+ring_r*cos_t
    y_ring = cy.unsqueeze(-1)+ring_r*sin_t
    u      = psi/(torch.abs(psi)+EPS)
    n_hat_z= []
    for zi in range(GRID):
        u_re = bilinear_sample_2d(u.real[:,:,:,zi], x_ring[:,zi,:], y_ring[:,zi,:])
        u_im = bilinear_sample_2d(u.imag[:,:,:,zi], x_ring[:,zi,:], y_ring[:,zi,:])
        norm = torch.sqrt(u_re*u_re+u_im*u_im+1e-12)
        u_re,u_im = u_re/norm, u_im/norm
        u_re_n = torch.roll(u_re,-1,1); u_im_n = torch.roll(u_im,-1,1)
        re = u_re_n*u_re+u_im_n*u_im; im = u_im_n*u_re-u_re_n*u_im
        n_hat_z.append(torch.atan2(im,re).sum(1)/(2*math.pi))
    return torch.stack(n_hat_z, dim=1)

class TopoGenesisNetB(nn.Module):
    def __init__(self, ring_r=RING_R_DEFAULT, evo_steps=EVO_STEPS_TRAIN):
        super().__init__()
        self.ring_r = float(ring_r)
        self.cx_z, self.cy_z = build_centerline_tensors(self.ring_r, seed=1)
        self.encoder = nn.Sequential(
            nn.Linear(33,256), nn.ReLU(),
            nn.Linear(256, 2*GRID*GRID*GRID))
        self.evo = EvolutionLayer(alpha=ALPHA, dt=DT, steps=evo_steps, amp_max=AMP_MAX)
    def forward(self, x, n_target):
        B = x.shape[0]
        out = self.encoder(torch.cat([x, n_target.view(B,1)],1)).view(B,2,GRID,GRID,GRID)
        amp_res_raw, phi_res_raw = out[:,0], out[:,1]
        a_res   = torch.exp(AMP_SCALE*torch.tanh(amp_res_raw))
        phi_res = PHI_SCALE*torch.tanh(phi_res_raw)
        psi_tpl, amp_tpl = vortex_template(n_target, self.cx_z, self.cy_z)
        cos_r,sin_r = torch.cos(phi_res), torch.sin(phi_res)
        psi0 = torch.complex(
            (psi_tpl.real*cos_r - psi_tpl.imag*sin_r)*a_res,
            (psi_tpl.real*sin_r + psi_tpl.imag*cos_r)*a_res)
        psi  = self.evo(psi0)
        cx = self.cx_z.view(1,GRID).expand(B,-1)
        cy = self.cy_z.view(1,GRID).expand(B,-1)
        return psi, cx, cy, {"psi0":psi0,"amp_tpl":amp_tpl,
                              "amp_res_raw":amp_res_raw,"phi_res_raw":phi_res_raw}

def acc_from_nhat(n_hat, n_target, tol=0.2):
    return torch.mean((torch.abs(n_hat-n_target.unsqueeze(1))<tol).float()).item()

def topo_protection_rate(n_hat, n_target, tol=0.45):
    return ((n_hat*n_target.unsqueeze(1)>0) &
            (torch.abs(n_hat-n_target.unsqueeze(1))<tol)).float().mean().item()

def set_residual_scales(phi,amp):
    global PHI_SCALE, AMP_SCALE
    PHI_SCALE=float(phi); AMP_SCALE=float(amp)

# ─────────────────────── 评估函数 ────────────────────────────────────────────
def eval_once(net, x, n_target, ring_r, evo_steps, phase_sigma=0.0, add_sigma=0.0):
    net.eval()
    with torch.no_grad():
        psi, cx, cy, aux = net(x, n_target)
        psi0 = aux["psi0"]
        psi0n = psi0
        if phase_sigma>0:
            psi0n = psi0n * torch.exp(1j*phase_sigma*torch.randn_like(psi0.real))
        if add_sigma>0:
            psi0n = psi0n + add_sigma*(torch.randn_like(psi0.real)+1j*torch.randn_like(psi0.real))
        n0 = winding_estimate_per_z(psi0n, cx, cy, ring_r)
        a0 = acc_from_nhat(n0, n_target); t0 = topo_protection_rate(n0, n_target)
        psi1 = EvolutionLayer(alpha=ALPHA,dt=DT,steps=evo_steps,amp_max=AMP_MAX).to(device)(psi0n)
        n1 = winding_estimate_per_z(psi1, cx, cy, ring_r)
        a1 = acc_from_nhat(n1, n_target); t1 = topo_protection_rate(n1, n_target)
        return a0, a1, t0, t1

def multi_run(net, ring_r, evo_steps, phase_sigma=0.0, add_sigma=0.0,
              runs=EVAL_RUNS, batch=EVAL_BATCH):
    res = defaultdict(list)
    for _ in range(runs):
        x  = torch.randn(batch,32).to(device)
        nt = (2*torch.randint(0,2,(batch,),device=device)-1).float()
        a0,a1,t0,t1 = eval_once(net,x,nt,ring_r,evo_steps,phase_sigma,add_sigma)
        for k,v in zip(["a0","a1","t0","t1"],[a0,a1,t0,t1]):
            res[k].append(v)
    out={}
    for k,vals in res.items():
        mu = np.mean(vals)
        ci = 1.96*np.std(vals,ddof=1)/math.sqrt(runs) if runs>1 else 0.0
        out[k]=(mu,ci)
    return out

# ─────────────────────── 训练 ────────────────────────────────────────────────
def train_network():
    print("=" * 60)
    print("  [1/4] Training TopoGenesisNet B  ...")
    print("=" * 60)
    net = TopoGenesisNetB(ring_r=RING_R_DEFAULT, evo_steps=EVO_STEPS_TRAIN).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)
    t0  = time.time()
    for step in range(TRAIN_STEPS+1):
        x  = torch.randn(BATCH_SIZE,32).to(device)
        nt = (2*torch.randint(0,2,(BATCH_SIZE,),device=device)-1).float()
        psi,cx,cy,aux = net(x,nt)
        n_hat = winding_estimate_per_z(psi,cx,cy,net.ring_r)
        amp   = torch.abs(psi)
        amp_tpl = aux["amp_tpl"].detach()
        loss  = (W_WIND*((n_hat-nt.unsqueeze(1))**2).mean()
                +W_CONS*n_hat.var(1).mean()
                +W_SMOOTH*(torch.abs(laplacian_roll(psi))**2).mean()
                +W_RES*(aux["amp_res_raw"]**2+aux["phi_res_raw"]**2).mean()
                +W_BG*(amp_tpl**2*(amp-V_BG)**2).mean()
                +W_CORE*((1-amp_tpl)**2*amp**2).mean())
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
        opt.step(); sch.step()
        if step % 100 == 0:
            acc = acc_from_nhat(n_hat, nt)
            print(f"  Step {step:4d} | loss {loss.item():.5f} | acc {acc:.4f} | "
                  f"lr {opt.param_groups[0]['lr']:.6f}")
    print(f"  Training done in {time.time()-t0:.1f}s\n")
    return net

# ─────────────────────── 数据采集 ────────────────────────────────────────────
def collect_snr_data(net):
    print("  [2/4] Collecting SNR vs Acc data ...")
    with torch.no_grad():
        _x  = torch.randn(EVAL_BATCH,32).to(device)
        _nt = (2*torch.randint(0,2,(EVAL_BATCH,),device=device)-1).float()
        _,_,_,_aux = net(_x,_nt)
        clean_pow = (_aux["psi0"].real**2+_aux["psi0"].imag**2).mean().item()

    phase_sigmas = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.10, 1.30, 1.60, 2.00]
    add_sigmas   = [0.0, 0.08, 0.15, 0.25, 0.40, 0.55, 0.70, 0.90, 1.10, 1.40, 1.80]

    def to_snr(s, mode):
        if s == 0: return float('inf')
        pwr = (s**2)*clean_pow if mode=="phase" else 2*s**2
        return 10*math.log10(clean_pow/(pwr+1e-30))

    ph_data = []
    for s in phase_sigmas:
        m = multi_run(net, RING_R_DEFAULT, EVO_STEPS_TRAIN, phase_sigma=s)
        ph_data.append((to_snr(s,"phase"), m["a0"], m["a1"]))

    add_data = []
    for s in add_sigmas:
        m = multi_run(net, RING_R_DEFAULT, EVO_STEPS_TRAIN, add_sigma=s)
        add_data.append((to_snr(s,"add"), m["a0"], m["a1"]))

    print("  Done.\n")
    return ph_data, add_data, clean_pow

def collect_evo_steps_data(net):
    print("  [3/4] Collecting Evolution Steps data ...")
    steps_list = [0, 1, 2, 3, 4, 6, 8, 10, 14, 20]
    ph_res, add_res = [], []
    for st in steps_list:
        m_ph  = multi_run(net, RING_R_DEFAULT, st, phase_sigma=1.0)
        m_add = multi_run(net, RING_R_DEFAULT, st, add_sigma=1.0)
        ph_res.append((st, m_ph["a0"], m_ph["a1"]))
        add_res.append((st, m_add["a0"], m_add["a1"]))
    print("  Done.\n")
    return steps_list, ph_res, add_res

def collect_ring_r_data(net):
    print("  [4/4] Collecting RING_R data ...")
    ring_rs = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0]
    ph_res, add_res = [], []
    for rr in ring_rs:
        net_tmp = TopoGenesisNetB(ring_r=rr, evo_steps=EVO_STEPS_TRAIN).to(device)
        net_tmp.load_state_dict(net.state_dict(), strict=False)
        m_ph  = multi_run(net_tmp, rr, EVO_STEPS_TRAIN, phase_sigma=1.0)
        m_add = multi_run(net_tmp, rr, EVO_STEPS_TRAIN, add_sigma=1.0)
        ph_res.append((rr, m_ph["a0"], m_ph["a1"]))
        add_res.append((rr, m_add["a0"], m_add["a1"]))
    print("  Done.\n")
    return ring_rs, ph_res, add_res

# ─────────────────────── 绘图风格 ─────────────────────────────────────────────
COLORS = {
    "phase_no" : "#4C72B0",   # 蓝
    "phase_evo": "#DD8452",   # 橙
    "add_no"   : "#55A868",   # 绿
    "add_evo"  : "#C44E52",   # 红
    "neutral"  : "#8172B2",   # 紫
    "fill_ph"  : "#4C72B0",
    "fill_add" : "#C44E52",
}

def apply_style(ax, xlabel="", ylabel="", title="", legend=True):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="both", which="both", direction="in", labelsize=10)
    ax.set_ylim(-0.02, 1.08)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    if legend:
        ax.legend(fontsize=9, framealpha=0.85, edgecolor="lightgray",
                  loc="lower right")
    ax.grid(axis="y", color="lightgray", linewidth=0.5, zorder=0)

# ─────────────────────── Fig 1: SNR vs Acc ───────────────────────────────────
def plot_fig1(ph_data, add_data, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    fig.suptitle("Fig 1 — SNR vs Winding-Number Accuracy\n"
                 "TideMemory · TopoGenesisNet B",
                 fontsize=14, fontweight="bold", y=1.02)

    # ── 左图：相位噪声 ──
    ax = axes[0]
    snr_ph  = [d[0] for d in ph_data if d[0] != float('inf')]
    a0_ph   = [d[1][0] for d in ph_data if d[0] != float('inf')]
    ci0_ph  = [d[1][1] for d in ph_data if d[0] != float('inf')]
    a1_ph   = [d[2][0] for d in ph_data if d[0] != float('inf')]
    ci1_ph  = [d[2][1] for d in ph_data if d[0] != float('inf')]

    ax.fill_between(snr_ph,
                    [a-c for a,c in zip(a0_ph,ci0_ph)],
                    [a+c for a,c in zip(a0_ph,ci0_ph)],
                    alpha=0.15, color=COLORS["phase_no"])
    ax.fill_between(snr_ph,
                    [a-c for a,c in zip(a1_ph,ci1_ph)],
                    [a+c for a,c in zip(a1_ph,ci1_ph)],
                    alpha=0.15, color=COLORS["phase_evo"])
    ax.plot(snr_ph, a0_ph, "o-", color=COLORS["phase_no"],
            lw=2, ms=6, label="Before Evo (Phase)")
    ax.plot(snr_ph, a1_ph, "s-", color=COLORS["phase_evo"],
            lw=2, ms=6, label="After Evo  (Phase)")

    # 标注 SNR=0dB
    for snr_val, a0, a1, c in [(0.0, a0_ph, a1_ph, "black")]:
        idx = min(range(len(snr_ph)), key=lambda i: abs(snr_ph[i]-snr_val))
        ax.annotate(f"SNR=0dB\n↑{(a1[idx]-a0[idx])*100:.1f}%",
                    xy=(snr_ph[idx], a1[idx]),
                    xytext=(snr_ph[idx]+1.2, a1[idx]-0.12),
                    fontsize=8, color=COLORS["phase_evo"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["phase_evo"], lw=1.2))

    apply_style(ax, xlabel="SNR (dB)", ylabel="Accuracy (tol=0.2)",
                title="(a) Phase Noise: psi × exp(jN(0,σ))")

    # ── 右图：加性噪声 ──
    ax2 = axes[1]
    snr_add = [d[0] for d in add_data if d[0] != float('inf')]
    a0_add  = [d[1][0] for d in add_data if d[0] != float('inf')]
    ci0_add = [d[1][1] for d in add_data if d[0] != float('inf')]
    a1_add  = [d[2][0] for d in add_data if d[0] != float('inf')]
    ci1_add = [d[2][1] for d in add_data if d[0] != float('inf')]

    ax2.fill_between(snr_add,
                     [a-c for a,c in zip(a0_add,ci0_add)],
                     [a+c for a,c in zip(a0_add,ci0_add)],
                     alpha=0.15, color=COLORS["add_no"])
    ax2.fill_between(snr_add,
                     [a-c for a,c in zip(a1_add,ci1_add)],
                     [a+c for a,c in zip(a1_add,ci1_add)],
                     alpha=0.15, color=COLORS["add_evo"])
    ax2.plot(snr_add, a0_add, "^-", color=COLORS["add_no"],
             lw=2, ms=6, label="Before Evo (Additive)")
    ax2.plot(snr_add, a1_add, "D-", color=COLORS["add_evo"],
             lw=2, ms=6, label="After Evo  (Additive)")

    idx2 = min(range(len(snr_add)), key=lambda i: abs(snr_add[i]-0.0))
    ax2.annotate(f"SNR=0dB\n↑{(a1_add[idx2]-a0_add[idx2])*100:.1f}%",
                 xy=(snr_add[idx2], a1_add[idx2]),
                 xytext=(snr_add[idx2]+1.2, a1_add[idx2]-0.15),
                 fontsize=8, color=COLORS["add_evo"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["add_evo"], lw=1.2))

    apply_style(ax2, xlabel="SNR (dB)", ylabel="Accuracy (tol=0.2)",
                title="(b) Additive Noise: psi + σ(N+jN)")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig 1] Saved -> {save_path}")


# ─────────────────────── Fig 2: Evolution Steps ──────────────────────────────
def plot_fig2(steps_list, ph_res, add_res, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    fig.suptitle("Fig 2 — Evolution Steps vs Winding-Number Accuracy\n"
                 "TideMemory · TopoGenesisNet B  (Noise σ = 1.0)",
                 fontsize=14, fontweight="bold", y=1.02)

    def _unpack(res):
        steps = [r[0] for r in res]
        a0    = [r[1][0] for r in res]; ci0 = [r[1][1] for r in res]
        a1    = [r[2][0] for r in res]; ci1 = [r[2][1] for r in res]
        return steps, a0, ci0, a1, ci1

    for ax, res, noise_lbl, c_no, c_evo, sub in [
        (axes[0], ph_res,  "Phase σ=1.0", COLORS["phase_no"], COLORS["phase_evo"], "(a)"),
        (axes[1], add_res, "Add   σ=1.0", COLORS["add_no"],   COLORS["add_evo"],   "(b)"),
    ]:
        steps, a0, ci0, a1, ci1 = _unpack(res)
        ax.fill_between(steps, [v-e for v,e in zip(a0,ci0)], [v+e for v,e in zip(a0,ci0)],
                        alpha=0.15, color=c_no)
        ax.fill_between(steps, [v-e for v,e in zip(a1,ci1)], [v+e for v,e in zip(a1,ci1)],
                        alpha=0.15, color=c_evo)
        ax.plot(steps, a0, "o--", color=c_no,  lw=2, ms=6, label=f"Before Evo ({noise_lbl})")
        ax.plot(steps, a1, "s-",  color=c_evo, lw=2, ms=6, label=f"After  Evo ({noise_lbl})")

        # 标注最大 Acc
        best_idx = int(np.argmax(a1))
        ax.annotate(f"Peak {a1[best_idx]:.4f}\n@ steps={steps[best_idx]}",
                    xy=(steps[best_idx], a1[best_idx]),
                    xytext=(steps[best_idx]+1.5, a1[best_idx]-0.10),
                    fontsize=8, color=c_evo,
                    arrowprops=dict(arrowstyle="->", color=c_evo, lw=1.2))

        # 收敛阈值线
        ax.axhline(0.95, color="tomato", lw=1.0, ls=":", alpha=0.8, label="Threshold 0.95")

        apply_style(ax, xlabel="Evolution Steps",
                    ylabel="Accuracy (tol=0.2)",
                    title=f"{sub} {noise_lbl}")
        ax.set_xlim(-0.5, max(steps)+1)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig 2] Saved -> {save_path}")


# ─────────────────────── Fig 3: RING_R ───────────────────────────────────────
def plot_fig3(ring_rs, ph_res, add_res, save_path):
    fig = plt.figure(figsize=(15, 10), dpi=150)
    fig.suptitle("Fig 3 — Measurement Ring Radius (RING_R) vs Accuracy\n"
                 "TideMemory · TopoGenesisNet B  (Noise σ = 1.0)",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    def _unpack(res):
        rrs = [r[0] for r in res]
        a0=[r[1][0] for r in res]; ci0=[r[1][1] for r in res]
        a1=[r[2][0] for r in res]; ci1=[r[2][1] for r in res]
        return rrs, a0, ci0, a1, ci1

    # ── 子图 (0,0) 相位噪声 Acc ──
    ax00 = fig.add_subplot(gs[0,0])
    rrs, a0, ci0, a1, ci1 = _unpack(ph_res)
    ax00.fill_between(rrs,[v-e for v,e in zip(a0,ci0)],[v+e for v,e in zip(a0,ci0)],
                      alpha=0.15,color=COLORS["phase_no"])
    ax00.fill_between(rrs,[v-e for v,e in zip(a1,ci1)],[v+e for v,e in zip(a1,ci1)],
                      alpha=0.15,color=COLORS["phase_evo"])
    ax00.plot(rrs, a0, "o--", color=COLORS["phase_no"],  lw=2,ms=6,label="Before Evo")
    ax00.plot(rrs, a1, "s-",  color=COLORS["phase_evo"], lw=2,ms=6,label="After Evo")
    ax00.axvline(RING_R_DEFAULT, color="gray", lw=1, ls=":", label=f"Default R={RING_R_DEFAULT}")
    apply_style(ax00, xlabel="RING_R", ylabel="Accuracy",
                title="(a) Phase Noise σ=1.0 — Accuracy")

    # ── 子图 (0,1) 加性噪声 Acc ──
    ax01 = fig.add_subplot(gs[0,1])
    rrs2, a0b, ci0b, a1b, ci1b = _unpack(add_res)
    ax01.fill_between(rrs2,[v-e for v,e in zip(a0b,ci0b)],[v+e for v,e in zip(a0b,ci0b)],
                      alpha=0.15,color=COLORS["add_no"])
    ax01.fill_between(rrs2,[v-e for v,e in zip(a1b,ci1b)],[v+e for v,e in zip(a1b,ci1b)],
                      alpha=0.15,color=COLORS["add_evo"])
    ax01.plot(rrs2, a0b, "^--", color=COLORS["add_no"],  lw=2,ms=6,label="Before Evo")
    ax01.plot(rrs2, a1b, "D-",  color=COLORS["add_evo"], lw=2,ms=6,label="After Evo")
    ax01.axvline(RING_R_DEFAULT, color="gray", lw=1, ls=":", label=f"Default R={RING_R_DEFAULT}")
    apply_style(ax01, xlabel="RING_R", ylabel="Accuracy",
                title="(b) Additive Noise σ=1.0 — Accuracy")

    # ── 子图 (1,0) Evo 提升量 ──
    ax10 = fig.add_subplot(gs[1,0])
    delta_ph  = [(a1[i]-a0[i])*100 for i in range(len(rrs))]
    delta_add = [(a1b[i]-a0b[i])*100 for i in range(len(rrs2))]
    ax10.bar([r-0.08 for r in rrs],  delta_ph,  width=0.14,
             color=COLORS["phase_evo"], alpha=0.85, label="Phase σ=1.0")
    ax10.bar([r+0.08 for r in rrs2], delta_add, width=0.14,
             color=COLORS["add_evo"],   alpha=0.85, label="Additive σ=1.0")
    ax10.axvline(RING_R_DEFAULT, color="gray", lw=1, ls=":", label=f"Default R={RING_R_DEFAULT}")
    ax10.set_xlabel("RING_R", fontsize=11)
    ax10.set_ylabel("Delta Acc (%)", fontsize=11)
    ax10.set_title("(c) Evo Improvement vs RING_R", fontsize=13, fontweight="bold")
    ax10.legend(fontsize=9, framealpha=0.85)
    ax10.spines["top"].set_visible(False); ax10.spines["right"].set_visible(False)
    ax10.grid(axis="y", color="lightgray", lw=0.5)
    ax10.tick_params(axis="both", which="both", direction="in", labelsize=10)

    # ── 子图 (1,1) Evo Acc 热力图式对比 ──
    ax11 = fig.add_subplot(gs[1,1])
    x = np.array(rrs); w = 0.15
    ax11.bar(x-w, a0,  width=w*1.8, color=COLORS["phase_no"],  alpha=0.7, label="Phase: Before Evo")
    ax11.bar(x+w, a1,  width=w*1.8, color=COLORS["phase_evo"], alpha=0.7, label="Phase: After Evo")
    for xi, ai in zip(x, a1):
        ax11.text(xi+w, ai+0.01, f"{ai:.2f}", ha="center", fontsize=7, color=COLORS["phase_evo"])
    ax11.axvline(RING_R_DEFAULT, color="gray", lw=1, ls=":", label=f"Default R={RING_R_DEFAULT}")
    ax11.set_ylim(-0.02, 1.12)
    ax11.set_xlabel("RING_R", fontsize=11)
    ax11.set_ylabel("Accuracy", fontsize=11)
    ax11.set_title("(d) Phase Noise: Before vs After Evo", fontsize=13, fontweight="bold")
    ax11.legend(fontsize=9, framealpha=0.85)
    ax11.spines["top"].set_visible(False); ax11.spines["right"].set_visible(False)
    ax11.tick_params(axis="both", which="both", direction="in", labelsize=10)
    ax11.grid(axis="y", color="lightgray", lw=0.5)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig 3] Saved -> {save_path}")


# ─────────────────────── 主流程 ──────────────────────────────────────────────
if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)

    t_all = time.time()

    # Step 1: 训练
    net = train_network()

    # Step 2: 采集数据
    ph_data, add_data, clean_pow = collect_snr_data(net)
    steps_list, ph_steps, add_steps = collect_evo_steps_data(net)
    ring_rs, ph_ring, add_ring = collect_ring_r_data(net)

    # Step 3: 绘图
    print("  Plotting figures ...")
    plot_fig1(ph_data, add_data,
              os.path.join(out_dir, "fig1_snr_acc.png"))
    plot_fig2(steps_list, ph_steps, add_steps,
              os.path.join(out_dir, "fig2_evo_steps.png"))
    plot_fig3(ring_rs, ph_ring, add_ring,
              os.path.join(out_dir, "fig3_ring_r.png"))

    # Step 4: 汇总分析
    elapsed = time.time() - t_all
    print()
    print("=" * 60)
    print("  Analysis Summary")
    print("=" * 60)

    # SNR 分析
    snr_finite_ph  = [(d[0],d[1][0],d[2][0]) for d in ph_data  if d[0]!=float('inf')]
    snr_finite_add = [(d[0],d[1][0],d[2][0]) for d in add_data if d[0]!=float('inf')]
    # 找 Evo 后 Acc >= 0.95 的最低 SNR
    ph_thresh  = next((s for s,_,a1 in snr_finite_ph  if a1>=0.95), None)
    add_thresh = next((s for s,_,a1 in snr_finite_add if a1>=0.95), None)

    print(f"\n  [Fig 1 - SNR vs Acc]")
    print(f"    Clean signal power    : {clean_pow:.4f}")
    print(f"    Phase noise  : Evo maintains Acc>=0.95 down to SNR~{ph_thresh:.1f}dB" if ph_thresh else
          f"    Phase noise  : no SNR threshold found for Acc>=0.95")
    print(f"    Additive noise: Evo maintains Acc>=0.95 down to SNR~{add_thresh:.1f}dB" if add_thresh else
          f"    Additive noise: no SNR threshold found for Acc>=0.95")

    best_ph_step  = max(ph_steps,  key=lambda r: r[2][0])
    best_add_step = max(add_steps, key=lambda r: r[2][0])
    print(f"\n  [Fig 2 - Evolution Steps]")
    print(f"    Phase noise  best Acc={best_ph_step[2][0]:.4f}  @ steps={best_ph_step[0]}")
    print(f"    Additive best Acc={best_add_step[2][0]:.4f} @ steps={best_add_step[0]}")

    best_ph_r   = max(ph_ring,  key=lambda r: r[2][0])
    best_add_r  = max(add_ring, key=lambda r: r[2][0])
    print(f"\n  [Fig 3 - RING_R]")
    print(f"    Phase noise  best Acc={best_ph_r[2][0]:.4f}   @ R={best_ph_r[0]:.1f}")
    print(f"    Additive best Acc={best_add_r[2][0]:.4f}  @ R={best_add_r[0]:.1f}")
    print(f"    Default R={RING_R_DEFAULT} is {'optimal' if best_ph_r[0]==RING_R_DEFAULT else 'suboptimal — consider R='+str(best_ph_r[0])}")

    print(f"\n  Total elapsed : {elapsed:.1f}s")
    print(f"  Output dir    : {out_dir}")
    print("=" * 60)
