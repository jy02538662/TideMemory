# -*- coding: utf-8 -*-
"""
TideMemory — Industrial-Grade Benchmark
========================================
Core Algorithm : TopoGenesisNet B
  - Conditional input encoding
  - Vortex template × residual initialization
  - Ginzburg-Landau evolution (topological self-repair)
  - Winding-number readout

Benchmark Suite:
  - Multi-run statistics with 95% confidence intervals
  - Phase noise + additive noise robustness sweep
  - Topological Protection Rate (TPR)
  - Attractor-basin analysis (residual perturbation)
  - Evolution-steps sweep
  - Measurement-ring radius sweep

Run:
    python main.py
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# ---------------------- 全局配置 ----------------------
device = torch.device("cpu")
torch.manual_seed(42)
np.random.seed(42)

GRID = 16
M    = 16
EPS  = 1e-6

RING_R_DEFAULT = 2.0
XI     = 1.8
V_BG   = 1.0

DT             = 0.05
ALPHA          = 1.0
EVO_STEPS_TRAIN = 6
AMP_MAX        = 3.0

PHI_SCALE = 0.25
AMP_SCALE = 0.25

W_WIND   = 1.0
W_CONS   = 0.2
W_SMOOTH = 0.01
W_RES    = 0.02
W_BG     = 0.05
W_CORE   = 0.05

# 训练超参数
TRAIN_STEPS   = 400        # 增加步数让收敛更充分
BATCH_SIZE    = 16         # 更大 batch 提升估计稳定性
LR            = 1e-3
LR_DECAY_STEP = 200        # 步数 milestone
LR_DECAY_GAMMA = 0.5       # LR × 0.5
GRAD_CLIP     = 1.0        # 梯度裁剪阈值
EVAL_RUNS     = 5          # 多轮评估次数（置信区间）
EVAL_BATCH    = 64         # 评估用更大 batch


# ─────────────────────────── 打印工具 ───────────────────────────
BAR = "=" * 72
BAR2 = "-" * 72
BAR3 = "·" * 72

def section(title: str):
    pad = max(0, 70 - len(title))
    left = pad // 2
    right = pad - left
    print(f"\n{'='*72}")
    print(f"{'='*left}  {title}  {'='*right}")
    print(f"{'='*72}")

def subsection(title: str):
    print(f"\n{'-'*72}")
    print(f"  >> {title}")
    print(f"{'-'*72}")

def kv(label: str, value, unit: str = ""):
    print(f"  {label:<40s}: {value}  {unit}")


# ─────────────────────────── 中心线模板 ─────────────────────────
def gen_centerline_np(z_count, ring_r, drift=2.8, bend=1.0, seed=1):
    center = (GRID - 1) / 2.0
    rng = np.random.default_rng(seed)
    z = np.arange(z_count)
    cx = center + drift * (0.6*np.sin(2*np.pi*z/z_count) + 0.4*np.sin(4*np.pi*z/z_count + 0.7))
    cy = center + drift * (0.6*np.cos(2*np.pi*z/z_count + 0.2) + 0.4*np.cos(4*np.pi*z/z_count + 1.1))
    cx += bend * rng.normal(0, 0.15, size=z_count)
    cy += bend * rng.normal(0, 0.15, size=z_count)
    margin = max(3.0, ring_r + 1.5)
    cx = np.clip(cx, margin, GRID - 1 - margin)
    cy = np.clip(cy, margin, GRID - 1 - margin)
    return cx, cy


def build_centerline_tensors(ring_r, seed=1):
    cx_np, cy_np = gen_centerline_np(GRID, ring_r, seed=seed)
    cx = torch.from_numpy(cx_np).float().to(device)
    cy = torch.from_numpy(cy_np).float().to(device)
    return cx, cy


Xg = torch.arange(GRID, device=device).float().view(1, GRID, 1)
Yg = torch.arange(GRID, device=device).float().view(1, 1, GRID)


def vortex_template(n_target: torch.Tensor, cx_z: torch.Tensor, cy_z: torch.Tensor):
    B = n_target.shape[0]
    psi_re  = torch.zeros(B, GRID, GRID, GRID, device=device)
    psi_im  = torch.zeros(B, GRID, GRID, GRID, device=device)
    amp_tpl = torch.zeros(B, GRID, GRID, GRID, device=device)
    for zi in range(GRID):
        dx = (Xg - cx_z[zi]).expand(B, GRID, GRID)
        dy = (Yg - cy_z[zi]).expand(B, GRID, GRID)
        r  = torch.sqrt(dx*dx + dy*dy + 1e-12)
        A  = torch.tanh(r / XI)
        theta = n_target.view(B, 1, 1) * torch.atan2(dy, dx)
        psi_re[:, :, :, zi]  = V_BG * A * torch.cos(theta)
        psi_im[:, :, :, zi]  = V_BG * A * torch.sin(theta)
        amp_tpl[:, :, :, zi] = V_BG * A
    return torch.complex(psi_re, psi_im), amp_tpl


# ─────────────────────────── 动力学 ─────────────────────────────
def laplacian_roll(psi: torch.Tensor):
    return (
        torch.roll(psi,  1, dims=1) + torch.roll(psi, -1, dims=1) +
        torch.roll(psi,  1, dims=2) + torch.roll(psi, -1, dims=2) +
        torch.roll(psi,  1, dims=3) + torch.roll(psi, -1, dims=3) -
        6.0 * psi
    )


class EvolutionLayer(nn.Module):
    def __init__(self, alpha=1.0, dt=0.05, steps=6, amp_max=3.0):
        super().__init__()
        self.alpha   = float(alpha)
        self.dt      = float(dt)
        self.steps   = int(steps)
        self.amp_max = float(amp_max)

    def forward(self, psi):
        for _ in range(self.steps):
            lap    = laplacian_roll(psi)
            nonlin = self.alpha * (V_BG*V_BG - torch.abs(psi)**2) * psi
            psi    = psi + self.dt * (lap + nonlin)
            amp    = torch.abs(psi)
            scale  = torch.clamp(self.amp_max / (amp + 1e-12), max=1.0)
            psi    = psi * scale
        return psi


# ─────────────────────────── 绕数检测 ───────────────────────────
def bilinear_sample_2d(img_xy, x, y):
    B, H, W = img_xy.shape
    x_norm = (y / (W - 1)) * 2 - 1
    y_norm = (x / (H - 1)) * 2 - 1
    grid   = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1)
    inp    = img_xy.unsqueeze(1)
    out    = F.grid_sample(inp, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out[:, 0, 0, :]


def winding_estimate_per_z(psi, cx, cy, ring_r):
    B      = psi.shape[0]
    thetas = torch.linspace(0, 2*math.pi, M+1, device=psi.device)[:-1]
    cos_t  = torch.cos(thetas).view(1, 1, M).expand(B, GRID, M)
    sin_t  = torch.sin(thetas).view(1, 1, M).expand(B, GRID, M)
    x_ring = cx.unsqueeze(-1) + ring_r * cos_t
    y_ring = cy.unsqueeze(-1) + ring_r * sin_t
    amp    = torch.abs(psi) + EPS
    u      = psi / amp
    n_hat_z = []
    for zi in range(GRID):
        u_re = bilinear_sample_2d(u.real[:, :, :, zi], x_ring[:, zi, :], y_ring[:, zi, :])
        u_im = bilinear_sample_2d(u.imag[:, :, :, zi], x_ring[:, zi, :], y_ring[:, zi, :])
        norm = torch.sqrt(u_re*u_re + u_im*u_im + 1e-12)
        u_re, u_im = u_re / norm, u_im / norm
        u_re_next = torch.roll(u_re, shifts=-1, dims=1)
        u_im_next = torch.roll(u_im, shifts=-1, dims=1)
        re = u_re_next * u_re + u_im_next * u_im
        im = u_im_next * u_re - u_re_next * u_im
        dtheta = torch.atan2(im, re)
        n_hat_z.append(dtheta.sum(dim=1) / (2*math.pi))
    return torch.stack(n_hat_z, dim=1)


# ─────────────────────────── 主网络 ─────────────────────────────
class TopoGenesisNetB(nn.Module):
    def __init__(self, ring_r=RING_R_DEFAULT, evo_steps=EVO_STEPS_TRAIN):
        super().__init__()
        self.ring_r = float(ring_r)
        self.cx_z, self.cy_z = build_centerline_tensors(self.ring_r, seed=1)
        self.encoder = nn.Sequential(
            nn.Linear(33, 256), nn.ReLU(),
            nn.Linear(256, 2 * GRID * GRID * GRID),
        )
        self.evo = EvolutionLayer(alpha=ALPHA, dt=DT, steps=evo_steps, amp_max=AMP_MAX)

    def forward(self, x, n_target):
        B     = x.shape[0]
        x_cond = torch.cat([x, n_target.view(B, 1)], dim=1)
        out   = self.encoder(x_cond).view(B, 2, GRID, GRID, GRID)
        amp_res_raw, phi_res_raw = out[:, 0], out[:, 1]
        a_res   = torch.exp(AMP_SCALE * torch.tanh(amp_res_raw))
        phi_res = PHI_SCALE * torch.tanh(phi_res_raw)
        psi_tpl, amp_tpl = vortex_template(n_target, self.cx_z, self.cy_z)
        cos_r  = torch.cos(phi_res)
        sin_r  = torch.sin(phi_res)
        psi0_re = (psi_tpl.real * cos_r - psi_tpl.imag * sin_r) * a_res
        psi0_im = (psi_tpl.real * sin_r + psi_tpl.imag * cos_r) * a_res
        psi0   = torch.complex(psi0_re, psi0_im)
        psi    = self.evo(psi0)
        cx     = self.cx_z.view(1, GRID).expand(B, -1)
        cy     = self.cy_z.view(1, GRID).expand(B, -1)
        aux    = {"psi0": psi0, "amp_tpl": amp_tpl,
                  "amp_res_raw": amp_res_raw, "phi_res_raw": phi_res_raw}
        return psi, cx, cy, aux


# ─────────────────────────── 评估工具 ────────────────────────────
def acc_from_nhat(n_hat, n_target, tol=0.2):
    return torch.mean((torch.abs(n_hat - n_target.unsqueeze(1)) < tol).float()).item()


def snr_db(signal: torch.Tensor, noise: torch.Tensor) -> float:
    """信号-噪声比（dB）= 10·log10(E[signal²] / E[noise²])"""
    sig_pwr   = (signal**2).mean().item()
    noise_pwr = (noise**2).mean().item()
    if noise_pwr < 1e-15:
        return float('inf')
    return 10.0 * math.log10(sig_pwr / (noise_pwr + 1e-15))


def topo_protection_rate(n_hat, n_target, tol=0.45):
    """拓扑保护率：|n̂ - n| < tol 且符号正确"""
    sign_ok  = (n_hat * n_target.unsqueeze(1)) > 0
    close_ok = torch.abs(n_hat - n_target.unsqueeze(1)) < tol
    return (sign_ok & close_ok).float().mean().item()


def eval_winding_once(net, x, n_target, ring_r, evo_steps,
                      phase_sigma=0.0, add_sigma=0.0):
    net.eval()
    with torch.no_grad():
        psi, cx, cy, aux = net(x, n_target)
        psi0 = aux["psi0"]
        psi0n = psi0
        if phase_sigma > 0:
            ph    = phase_sigma * torch.randn_like(psi0.real)
            psi0n = psi0n * torch.exp(1j * ph)
        if add_sigma > 0:
            noise = add_sigma * (torch.randn_like(psi0.real) + 1j*torch.randn_like(psi0.real))
            psi0n = psi0n + noise

        n_hat0 = winding_estimate_per_z(psi0n, cx, cy, ring_r)
        acc0   = acc_from_nhat(n_hat0, n_target, tol=0.2)
        err0   = torch.mean(torch.abs(n_hat0 - n_target.unsqueeze(1))).item()
        std0   = n_hat0.std(dim=1).mean().item()
        tpr0   = topo_protection_rate(n_hat0, n_target)

        evo  = EvolutionLayer(alpha=ALPHA, dt=DT, steps=evo_steps, amp_max=AMP_MAX).to(device)
        psi1 = evo(psi0n)
        n_hat1 = winding_estimate_per_z(psi1, cx, cy, ring_r)
        acc1   = acc_from_nhat(n_hat1, n_target, tol=0.2)
        err1   = torch.mean(torch.abs(n_hat1 - n_target.unsqueeze(1))).item()
        std1   = n_hat1.std(dim=1).mean().item()
        tpr1   = topo_protection_rate(n_hat1, n_target)

        return acc0, acc1, err0, err1, std0, std1, tpr0, tpr1


def multi_run_eval(net, ring_r, evo_steps, phase_sigma=0.0, add_sigma=0.0,
                   runs=EVAL_RUNS, batch=EVAL_BATCH):
    """多轮评估，返回均值和95%置信区间（±1.96σ/√n）"""
    metrics = defaultdict(list)
    for _ in range(runs):
        x        = torch.randn(batch, 32).to(device)
        n_target = (2 * torch.randint(0, 2, (batch,), device=device) - 1).float()
        r = eval_winding_once(net, x, n_target, ring_r, evo_steps,
                               phase_sigma, add_sigma)
        keys = ["acc0", "acc1", "err0", "err1", "std0", "std1", "tpr0", "tpr1"]
        for k, v in zip(keys, r):
            metrics[k].append(v)
    out = {}
    for k, vals in metrics.items():
        mu  = np.mean(vals)
        ci  = 1.96 * np.std(vals, ddof=1) / math.sqrt(runs) if runs > 1 else 0.0
        out[k] = (mu, ci)
    return out


def set_residual_scales(phi_scale, amp_scale):
    global PHI_SCALE, AMP_SCALE
    PHI_SCALE = float(phi_scale)
    AMP_SCALE = float(amp_scale)


# ─────────────────────────── MAIN ────────────────────────────────
if __name__ == "__main__":
    t_start_total = time.time()

    section("TideMemory · 工业级基准测试  (TopoGenesisNet B / A+B)")
    kv("设备",         device)
    kv("网格尺寸",     f"{GRID}^3 = {GRID**3} 点")
    kv("训练步数",     TRAIN_STEPS)
    kv("批大小",       BATCH_SIZE)
    kv("初始学习率",   LR)
    kv("演化步数",     EVO_STEPS_TRAIN)
    kv("DT / AMP_MAX", f"{DT} / {AMP_MAX}")
    kv("多轮评估次数", EVAL_RUNS)
    kv("评估批大小",   EVAL_BATCH)

    # ================================================================
    # 训练阶段
    # ================================================================
    section("训练阶段  (SGD + Adam + LR Schedule + Grad Clip)")

    net = TopoGenesisNetB(ring_r=RING_R_DEFAULT, evo_steps=EVO_STEPS_TRAIN).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)

    total_params = sum(p.numel() for p in net.parameters())
    kv("模型参数量", f"{total_params:,}")

    print()
    header = (f"{'Step':>6} | {'Loss':>8} | {'L_wind':>7} | {'L_cons':>7} | "
              f"{'L_smooth':>8} | {'Acc(tol=0.2)':>12} | {'LR':>9} | {'Δt(s)':>6}")
    print(header)
    print(BAR2)

    loss_history = []
    acc_history  = []
    t_step = time.time()
    best_acc  = 0.0
    best_step = 0
    converged_at = None

    for step in range(TRAIN_STEPS + 1):
        B_tr = BATCH_SIZE
        x        = torch.randn(B_tr, 32).to(device)
        n_target = (2 * torch.randint(0, 2, (B_tr,), device=device) - 1).float()

        psi, cx, cy, aux = net(x, n_target)
        n_hat = winding_estimate_per_z(psi, cx, cy, net.ring_r)

        L_wind   = ((n_hat - n_target.unsqueeze(1))**2).mean()
        L_cons   = n_hat.var(dim=1).mean()
        lap      = laplacian_roll(psi)
        L_smooth = (torch.abs(lap)**2).mean()
        L_res    = (aux["amp_res_raw"]**2).mean() + (aux["phi_res_raw"]**2).mean()

        amp      = torch.abs(psi)
        amp_tpl  = aux["amp_tpl"].detach()
        noncore  = amp_tpl**2
        core     = (1.0 - amp_tpl)**2
        L_bg     = (noncore * (amp - V_BG)**2).mean()
        L_core   = (core * (amp**2)).mean()

        loss = (W_WIND   * L_wind
              + W_CONS   * L_cons
              + W_SMOOTH * L_smooth
              + W_RES    * L_res
              + W_BG     * L_bg
              + W_CORE   * L_core)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
        opt.step()
        scheduler.step()

        loss_history.append(loss.item())
        acc = acc_from_nhat(n_hat, n_target, tol=0.2)
        acc_history.append(acc)

        if acc > best_acc:
            best_acc  = acc
            best_step = step

        # 收敛检测：最近 50 步 acc > 0.95
        if converged_at is None and step >= 50:
            recent = np.mean(acc_history[-50:])
            if recent >= 0.95:
                converged_at = step

        if step % 40 == 0:
            now_lr = opt.param_groups[0]['lr']
            dt_s   = time.time() - t_step
            t_step = time.time()
            print(f"{step:6d} | {loss.item():8.4f} | {L_wind.item():7.4f} | "
                  f"{L_cons.item():7.4f} | {L_smooth.item():8.5f} | "
                  f"{acc:12.3f} | {now_lr:9.6f} | {dt_s:6.2f}s")

    print(BAR2)
    print(f"  训练完成 · 最佳 Acc = {best_acc:.3f} @ Step {best_step}")
    if converged_at:
        print(f"  收敛检测 · 首次 50步均值 ≥ 0.95 在 Step {converged_at}")
    else:
        print(f"  收敛检测 · 训练期间未观测到 50步均值 ≥ 0.95（可增加步数）")

    # ================================================================
    # 最终准确率（多轮统计）
    # ================================================================
    section("最终准确率  (多轮统计 · 95% 置信区间)")
    m = multi_run_eval(net, net.ring_r, EVO_STEPS_TRAIN,
                       phase_sigma=0.0, add_sigma=0.0)
    print(f"\n  {'指标':<28} {'无演化':>14} {'有演化(Evo)':>14}")
    print(f"  {'-'*56}")
    print(f"  {'准确率 Acc(tol=0.2)':<28} {m['acc0'][0]:>8.4f}±{m['acc0'][1]:.4f}  {m['acc1'][0]:>8.4f}±{m['acc1'][1]:.4f}")
    print(f"  {'绝对误差 |n̂-n|':<28} {m['err0'][0]:>8.4f}±{m['err0'][1]:.4f}  {m['err1'][0]:>8.4f}±{m['err1'][1]:.4f}")
    print(f"  {'截面一致性 STD_z':<28} {m['std0'][0]:>8.4f}±{m['std0'][1]:.4f}  {m['std1'][0]:>8.4f}±{m['std1'][1]:.4f}")
    print(f"  {'拓扑保护率 TPR':<28} {m['tpr0'][0]:>8.4f}±{m['tpr0'][1]:.4f}  {m['tpr1'][0]:>8.4f}±{m['tpr1'][1]:.4f}")
    print(f"\n  [解读] Evo前/后 Acc 提升 = {(m['acc1'][0]-m['acc0'][0])*100:+.2f}%，"
          f"误差下降 = {(m['err0'][0]-m['err1'][0]):.4f}")

    # ================================================================
    # 验证 1：抗噪声 — 相位噪声 + 加性噪声
    # ================================================================
    section("验证 1 · 抗噪声鲁棒性  (Phase + Additive Noise)")

    phase_sigmas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    add_sigmas   = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]

    # SNR 基准（clean signal power）
    with torch.no_grad():
        _x = torch.randn(EVAL_BATCH, 32).to(device)
        _n = (2 * torch.randint(0, 2, (EVAL_BATCH,), device=device) - 1).float()
        _psi, _cx, _cy, _aux = net(_x, _n)
        clean_power = (_aux["psi0"].real**2 + _aux["psi0"].imag**2).mean().item()
    kv("洁净信号功率 (均值)", f"{clean_power:.4f}")

    subsection("相位噪声  psi0 × exp(j·N(0,σ))")
    hdr = (f"  {'σ':>5} | {'SNR(dB)':>8} | "
           f"{'Acc_no':>7} {'Acc_evo':>8} | "
           f"{'Err_no':>7} {'Err_evo':>8} | "
           f"{'TPR_no':>7} {'TPR_evo':>8} | {'Δ Acc':>7}")
    print(hdr)
    print("  " + "-" * 68)
    for s in phase_sigmas:
        # 近似 SNR：相位噪声功率 ≈ σ²·clean_power
        noise_pwr = (s**2) * clean_power if s > 0 else 1e-30
        snr_val   = 10 * math.log10(clean_power / noise_pwr) if s > 0 else float('inf')
        m2 = multi_run_eval(net, net.ring_r, EVO_STEPS_TRAIN,
                            phase_sigma=s, add_sigma=0.0)
        snr_str = f"{snr_val:8.2f}" if s > 0 else "    +inf"
        delta   = (m2['acc1'][0] - m2['acc0'][0]) * 100
        print(f"  {s:5.2f} | {snr_str} | "
              f"{m2['acc0'][0]:7.4f} {m2['acc1'][0]:8.4f} | "
              f"{m2['err0'][0]:7.4f} {m2['err1'][0]:8.4f} | "
              f"{m2['tpr0'][0]:7.4f} {m2['tpr1'][0]:8.4f} | {delta:+6.2f}%")

    subsection("加性复数噪声  psi0 + σ·(N+jN)")
    hdr2 = (f"  {'σ':>5} | {'SNR(dB)':>8} | "
            f"{'Acc_no':>7} {'Acc_evo':>8} | "
            f"{'Err_no':>7} {'Err_evo':>8} | "
            f"{'TPR_no':>7} {'TPR_evo':>8} | {'Δ Acc':>7}")
    print(hdr2)
    print("  " + "-" * 68)
    for s in add_sigmas:
        noise_pwr = 2 * s**2 if s > 0 else 1e-30   # real + imag
        snr_val   = 10 * math.log10(clean_power / noise_pwr) if s > 0 else float('inf')
        m3 = multi_run_eval(net, net.ring_r, EVO_STEPS_TRAIN,
                            phase_sigma=0.0, add_sigma=s)
        snr_str = f"{snr_val:8.2f}" if s > 0 else "    +inf"
        delta   = (m3['acc1'][0] - m3['acc0'][0]) * 100
        print(f"  {s:5.2f} | {snr_str} | "
              f"{m3['acc0'][0]:7.4f} {m3['acc1'][0]:8.4f} | "
              f"{m3['err0'][0]:7.4f} {m3['err1'][0]:8.4f} | "
              f"{m3['tpr0'][0]:7.4f} {m3['tpr1'][0]:8.4f} | {delta:+6.2f}%")

    # ================================================================
    # 验证 2：吸引域 — 残差强度扫描
    # ================================================================
    section("验证 2 · 吸引域分析  (残差扰动强度扫描)")
    phi0_saved, amp0_saved = PHI_SCALE, AMP_SCALE
    scales = [0.10, 0.25, 0.50, 0.80, 1.20, 1.80]

    print(f"\n  {'Scale':>6} | {'Acc_no':>8} {'±CI':>6} | {'Acc_evo':>8} {'±CI':>6} | "
          f"{'Err_no':>7} | {'Err_evo':>7} | {'TPR_no':>7} {'TPR_evo':>8}")
    print("  " + "-" * 72)
    for sc in scales:
        set_residual_scales(phi_scale=sc, amp_scale=sc)
        ms = multi_run_eval(net, net.ring_r, EVO_STEPS_TRAIN,
                            phase_sigma=0.0, add_sigma=0.0)
        print(f"  {sc:6.2f} | {ms['acc0'][0]:8.4f} {ms['acc0'][1]:6.4f} | "
              f"{ms['acc1'][0]:8.4f} {ms['acc1'][1]:6.4f} | "
              f"{ms['err0'][0]:7.4f} | {ms['err1'][0]:7.4f} | "
              f"{ms['tpr0'][0]:7.4f} {ms['tpr1'][0]:8.4f}")

    set_residual_scales(phi_scale=phi0_saved, amp_scale=amp0_saved)
    print(f"\n  [已恢复] PHI_SCALE={PHI_SCALE}  AMP_SCALE={AMP_SCALE}")

    # ================================================================
    # Sanity A：Evolution Steps 扫描
    # ================================================================
    section("Sanity A · Evolution Steps 扫描  (σ_phase=1.0 / σ_add=1.0)")
    evo_steps_list = [0, 1, 3, 6, 12, 20]

    subsection("相位噪声 σ=1.0")
    print(f"  {'Steps':>5} | {'Acc_no':>7} {'Acc_evo':>8} | {'Err_no':>7} {'Err_evo':>8} | {'TPR_no':>7} {'TPR_evo':>8}")
    print("  " + "-" * 60)
    for st in evo_steps_list:
        ms = multi_run_eval(net, net.ring_r, st, phase_sigma=1.0, add_sigma=0.0)
        print(f"  {st:5d} | {ms['acc0'][0]:7.4f} {ms['acc1'][0]:8.4f} | "
              f"{ms['err0'][0]:7.4f} {ms['err1'][0]:8.4f} | "
              f"{ms['tpr0'][0]:7.4f} {ms['tpr1'][0]:8.4f}")

    subsection("加性噪声 σ=1.0")
    print(f"  {'Steps':>5} | {'Acc_no':>7} {'Acc_evo':>8} | {'Err_no':>7} {'Err_evo':>8} | {'TPR_no':>7} {'TPR_evo':>8}")
    print("  " + "-" * 60)
    for st in evo_steps_list:
        ms = multi_run_eval(net, net.ring_r, st, phase_sigma=0.0, add_sigma=1.0)
        print(f"  {st:5d} | {ms['acc0'][0]:7.4f} {ms['acc1'][0]:8.4f} | "
              f"{ms['err0'][0]:7.4f} {ms['err1'][0]:8.4f} | "
              f"{ms['tpr0'][0]:7.4f} {ms['tpr1'][0]:8.4f}")

    # ================================================================
    # Sanity B：RING_R 测量几何对比
    # ================================================================
    section("Sanity B · 测量环半径 RING_R 对比  (σ_phase/add = 1.0)")
    ring_rs = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

    subsection("相位噪声 σ=1.0")
    print(f"  {'R':>4} | {'Acc_no':>7} {'Acc_evo':>8} | {'Err_no':>7} {'Err_evo':>8} | {'Std_no':>7} {'Std_evo':>8}")
    print("  " + "-" * 62)
    for rr in ring_rs:
        net_tmp = TopoGenesisNetB(ring_r=rr, evo_steps=EVO_STEPS_TRAIN).to(device)
        net_tmp.load_state_dict(net.state_dict(), strict=False)
        ms = multi_run_eval(net_tmp, rr, EVO_STEPS_TRAIN, phase_sigma=1.0, add_sigma=0.0)
        print(f"  {rr:4.1f} | {ms['acc0'][0]:7.4f} {ms['acc1'][0]:8.4f} | "
              f"{ms['err0'][0]:7.4f} {ms['err1'][0]:8.4f} | "
              f"{ms['std0'][0]:7.4f} {ms['std1'][0]:8.4f}")

    subsection("加性噪声 σ=1.0")
    print(f"  {'R':>4} | {'Acc_no':>7} {'Acc_evo':>8} | {'Err_no':>7} {'Err_evo':>8} | {'Std_no':>7} {'Std_evo':>8}")
    print("  " + "-" * 62)
    for rr in ring_rs:
        net_tmp = TopoGenesisNetB(ring_r=rr, evo_steps=EVO_STEPS_TRAIN).to(device)
        net_tmp.load_state_dict(net.state_dict(), strict=False)
        ms = multi_run_eval(net_tmp, rr, EVO_STEPS_TRAIN, phase_sigma=0.0, add_sigma=1.0)
        print(f"  {rr:4.1f} | {ms['acc0'][0]:7.4f} {ms['acc1'][0]:8.4f} | "
              f"{ms['err0'][0]:7.4f} {ms['err1'][0]:8.4f} | "
              f"{ms['std0'][0]:7.4f} {ms['std1'][0]:8.4f}")

    # ================================================================
    # 汇总报告
    # ================================================================
    t_total = time.time() - t_start_total
    m_clean = multi_run_eval(net, net.ring_r, EVO_STEPS_TRAIN)
    m_noisy = multi_run_eval(net, net.ring_r, EVO_STEPS_TRAIN, phase_sigma=1.0, add_sigma=0.5)

    section("最终汇总报告  (Industrial Benchmark Summary)")
    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │                TideMemory · 测试报告                         │
  ├─────────────────────────────────────────────────────────────┤
  │  网络架构         : TopoGenesisNet B (条件输入+模板×残差)    │
  │  总参数量         : {total_params:>10,} 个                        │
  │  训练步数         : {TRAIN_STEPS:>10} 步 (batch={BATCH_SIZE})           │
  │  总运行时间       : {t_total:>10.1f} 秒                         │
  ├─────────────────────────────────────────────────────────────┤
  │  [洁净条件] 绕数检测准确率                                   │
  │    无演化 Acc     : {m_clean['acc0'][0]:>8.4f} ± {m_clean['acc0'][1]:.4f}               │
  │    有演化 Acc     : {m_clean['acc1'][0]:>8.4f} ± {m_clean['acc1'][1]:.4f}               │
  │    拓扑保护率 TPR : {m_clean['tpr1'][0]:>8.4f} ± {m_clean['tpr1'][1]:.4f}               │
  │    截面一致性 STD : {m_clean['std1'][0]:>8.4f} ± {m_clean['std1'][1]:.4f}               │
  ├─────────────────────────────────────────────────────────────┤
  │  [噪声条件] σ_phase=1.0, σ_add=0.5                          │
  │    无演化 Acc     : {m_noisy['acc0'][0]:>8.4f} ± {m_noisy['acc0'][1]:.4f}               │
  │    有演化 Acc     : {m_noisy['acc1'][0]:>8.4f} ± {m_noisy['acc1'][1]:.4f}               │
  │    拓扑保护率 TPR : {m_noisy['tpr1'][0]:>8.4f} ± {m_noisy['tpr1'][1]:.4f}               │
  │    Evo 带来提升   : {(m_noisy['acc1'][0]-m_noisy['acc0'][0])*100:>+7.2f}%                       │
  ├─────────────────────────────────────────────────────────────┤
  │  训练最佳 Acc     : {best_acc:>8.4f} @ Step {best_step:<6d}                │
  │  收敛(≥0.95)      : {'Step ' + str(converged_at) if converged_at else '未触发（可增加步数）':<35s}│
  └─────────────────────────────────────────────────────────────┘
""")
    print(f"  {'='*60}")
    print(f"  全部基准测试完成。总耗时 {t_total:.1f}s")
    print(f"  {'='*60}\n")
