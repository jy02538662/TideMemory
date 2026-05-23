# -*- coding: utf-8 -*-
"""TideMemory v3 validation: ablation, stable runs, fair RAG, unified vortices, multibit."""
import os, math, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device=torch.device("cpu"); torch.manual_seed(42); np.random.seed(42)
GRID=64; M=16; EPS=1e-6; XI=1.8; V0=1.0; DT=0.05; ALPHA=1.0; AMP_MAX=3.0; RING_R=2.0
BASE=8; MED=10; HIGH=14; EXT=18; V_MED=1.5; V_HIGH=2.0
NOISE_CENTER=0.3; NOISE_WIDTH=0.1; LOG_LIMIT=2.5; RMIN=0.18; RMAX=0.55; BG_BLEND=0.18
MODES=["A_base","B_bg","C_adapt","D_bg_adapt","E_full"]
Xg=torch.arange(GRID,device=device).float().view(1,GRID,1); Yg=torch.arange(GRID,device=device).float().view(1,1,GRID)

def gen_centerline(n,seed=1,offset=(0,0),wobble=2.8):
    c=(GRID-1)/2; rng=np.random.default_rng(seed); z=np.arange(n); d=max(n,1)
    cx=c+offset[0]+wobble*(.6*np.sin(2*np.pi*z/d)+.4*np.sin(4*np.pi*z/d+.7))+rng.normal(0,.15,n)
    cy=c+offset[1]+wobble*(.6*np.cos(2*np.pi*z/d+.2)+.4*np.cos(4*np.pi*z/d+1.1))+rng.normal(0,.15,n)
    m=max(3.0,RING_R+1.5)
    return torch.tensor(np.clip(cx,m,GRID-1-m),dtype=torch.float32,device=device),torch.tensor(np.clip(cy,m,GRID-1-m),dtype=torch.float32,device=device)

def offsets(N,r=17):
    if N==1: return [(0,0)]
    rings=max(1,math.ceil(N/8)); out=[]
    for i in range(N):
        rr=r*(1+(i//8)%rings)/rings; a=2*math.pi*(i%8)/8+.35*(i//8); out.append((rr*math.cos(a),rr*math.sin(a)))
    return out

def lap(psi):
    return torch.roll(psi,1,0)+torch.roll(psi,-1,0)+torch.roll(psi,1,1)+torch.roll(psi,-1,1)+torch.roll(psi,1,2)+torch.roll(psi,-1,2)-6*psi

def colored(shape,center,width,amp,dev):
    w=torch.randn(shape,device=dev); f=torch.fft.fftn(w); freqs=[torch.fft.fftfreq(n).to(dev) for n in shape]
    km=torch.zeros(shape,device=dev)
    for g in torch.meshgrid(*freqs,indexing="ij"): km+=g*g
    y=torch.fft.ifftn(f*torch.exp(-((torch.sqrt(km)-center)**2)/(2*width**2))).real
    return y/(y.std()+1e-8)*amp

def gl(psi,steps=BASE,V=V0):
    for _ in range(steps):
        psi=psi+DT*(lap(psi)+ALPHA*(V*V-torch.abs(psi)**2)*psi); a=torch.abs(psi); psi=psi*torch.clamp(AMP_MAX/(a+1e-12),max=1.0)
    return psi.detach()

def project(psi,V=V0,blend=BG_BLEND):
    a=torch.abs(psi); ph=psi/(a+1e-12); ta=torch.clamp(a,min=.15*V,max=AMP_MAX); return (((1-blend)*ta+blend*V)*ph).detach()

def log_ratio(psi,V=V0):
    a=torch.abs(psi); ae=torch.mean((a-V)**2); rough=sum(torch.mean(torch.abs(psi-torch.roll(psi,1,d))**2) for d in range(3))/3
    return float(torch.log1p((ae+.25*rough)/(torch.mean(a*a)+1e-8)).detach().cpu())

def params(sigma,psi=None):
    lr=max(math.log1p(float(sigma)**2),log_ratio(psi) if psi is not None else 0); x=min(max(lr/LOG_LIMIT,0),1)
    c=RMIN+(RMAX-RMIN)*(x**.7); wid=.08+.08*x; inten=.012+.038*x
    if x<.25: return c,wid,inten,BASE,V0
    if x<.55: return c,wid,inten,MED,V_MED
    if x<.85: return c,wid,inten,HIGH,V_HIGH
    return c,wid,inten*.85,EXT+6,V_HIGH+.35

def reson(psi,steps,inten,V,c,wid,use_bg):
    if use_bg: psi=project(psi,V)
    for t in range(steps):
        d=DT*(lap(psi)+ALPHA*(V*V-torch.abs(psi)**2)*psi)
        nr=colored(psi.shape,c,wid,inten*(1-t/max(steps,1)),psi.device); ni=colored(psi.shape,c,wid,inten*(1-t/max(steps,1)),psi.device)
        psi=psi+d+torch.complex(nr,ni); a=torch.abs(psi); psi=psi*torch.clamp(AMP_MAX/(a+1e-12),max=1.0)
        if use_bg and t%3==2: psi=project(psi,V,BG_BLEND*.5)
    return psi.detach()

def repair(psi,sigma,mode):
    if mode=="A_base":
        if sigma<=.5: return gl(psi,BASE,V0)
        if sigma<=1.2: return gl(psi,MED,V_MED)
        if sigma<=2.0: return reson(psi,HIGH,.03,V_HIGH,NOISE_CENTER,NOISE_WIDTH,False)
        return reson(psi,EXT,.02,V_HIGH,NOISE_CENTER,NOISE_WIDTH,False)
    use_bg=mode in ("B_bg","D_bg_adapt","E_full"); use_ad=mode in ("C_adapt","D_bg_adapt","E_full")
    if use_ad: c,wid,inten,steps,V=params(sigma,psi)
    else:
        c,wid,inten=NOISE_CENTER,NOISE_WIDTH,.03 if sigma<=2 else .02; steps=MED if sigma<=1.2 else HIGH if sigma<=2 else EXT; V=V_MED if sigma<=1.2 else V_HIGH
    if sigma<=.5:
        return gl(project(psi,V0,BG_BLEND*.5) if use_bg else psi,BASE,V0)
    return reson(psi,steps,inten,V,c,wid,use_bg)

def write_segmented(N,seeds,signs):
    k=GRID//N; field=torch.complex(torch.full((GRID,GRID,GRID),V0,device=device),torch.zeros(GRID,GRID,GRID,device=device)); ch=[]
    for i in range(N):
        zs=i*k; ze=(i+1)*k if i<N-1 else GRID; cx,cy=gen_centerline(ze-zs,int(seeds[i]))
        for zl in range(ze-zs):
            dx=(Xg-cx[zl]).expand(1,GRID,GRID).squeeze(0); dy=(Yg-cy[zl]).expand(1,GRID,GRID).squeeze(0); r=torch.sqrt(dx*dx+dy*dy+1e-12)
            field[:,:,zs+zl]=V0*torch.tanh(r/XI)*torch.exp(1j*signs[i]*torch.atan2(dy,dx))
        ch.append(dict(z_s=zs,z_e=ze,k=ze-zs,cx=cx,cy=cy,sign=signs[i]))
    return field,ch

def write_unified(N,seeds,signs):
    phase=torch.zeros((GRID,GRID,GRID),device=device); amp=torch.full((GRID,GRID,GRID),V0,device=device); ch=[]
    for i,off in enumerate(offsets(N,18)):
        cx,cy=gen_centerline(GRID,int(seeds[i]),off,1.0)
        for z in range(GRID):
            dx=(Xg-cx[z]).expand(1,GRID,GRID).squeeze(0); dy=(Yg-cy[z]).expand(1,GRID,GRID).squeeze(0); r=torch.sqrt(dx*dx+dy*dy+1e-12)
            phase[:,:,z]=phase[:,:,z]+signs[i]*torch.atan2(dy,dx)
            amp[:,:,z]=amp[:,:,z]*torch.clamp(torch.tanh(r/XI),min=.08)
        ch.append(dict(z_s=0,z_e=GRID,k=GRID,cx=cx,cy=cy,sign=signs[i]))
    amp=torch.clamp(amp,min=.04,max=V0)
    return torch.complex(amp*torch.cos(phase),amp*torch.sin(phase)).detach(),ch

def read_w(field,ch,robust=True):
    seg=field[:,:,ch['z_s']:ch['z_e']]; u=seg/(torch.abs(seg)+EPS); th=torch.linspace(0,2*math.pi,M+1,device=device)[:-1]; vals=[]
    for z in range(ch['k']):
        xr=ch['cx'][z]+RING_R*torch.cos(th); yr=ch['cy'][z]+RING_R*torch.sin(th); grid=torch.stack([(yr/(GRID-1))*2-1,(xr/(GRID-1))*2-1],-1).view(1,1,-1,2)
        ur=F.grid_sample(u.real[:,:,z].view(1,1,GRID,GRID),grid,mode='bilinear',padding_mode='border',align_corners=True)[0,0,0]
        ui=F.grid_sample(u.imag[:,:,z].view(1,1,GRID,GRID),grid,mode='bilinear',padding_mode='border',align_corners=True)[0,0,0]
        nm=torch.sqrt(ur*ur+ui*ui+1e-12); ur,ui=ur/nm,ui/nm; vals.append(torch.atan2(torch.roll(ui,-1)*ur-torch.roll(ur,-1)*ui,torch.roll(ur,-1)*ur+torch.roll(ui,-1)*ui).sum()/(2*math.pi))
    v=torch.stack(vals)
    if robust and v.numel()>=5:
        m=v.median(); keep=torch.abs(v-m)<=.65
        if keep.any(): v=v[keep]
        return v.median().item()
    return v.mean().item()

def noisy(f,s): return f.clone() if s<=0 else f+s*(torch.randn_like(f.real)+1j*torch.randn_like(f.real))

def topo_trial(N,sigma,mode,storage='seg',bits=1,seed=0):
    rng=np.random.default_rng(seed); seeds=rng.integers(10,9990,N).tolist(); alph=[-2.,-1.,1.,2.] if bits>1 else [-1.,1.]; signs=[alph[int(rng.integers(0,len(alph)))] for _ in range(N)]
    f,ch=(write_unified if storage=='uni' else write_segmented)(N,seeds,signs); f=gl(f,BASE,V0); fn=repair(noisy(f,sigma),sigma,mode)
    tol=.55 if bits>1 else .4; return sum(abs(read_w(fn,c,mode=='E_full')-c['sign'])<tol for c in ch)/N

def rag_cos(db,s,scale=.8):
    dn=F.normalize(db+(s*scale*torch.randn_like(db) if s>0 else 0),dim=-1); return (torch.matmul(db,dn.T).argmax(1)==torch.arange(db.shape[0],device=device)).float().mean().item()

def rag_aw(db,s,samples=7,scale=.8):
    if s<=0: return 1.0
    acc=torch.zeros(db.shape[0],db.shape[0],device=device)
    for _ in range(samples): acc+=torch.matmul(db,F.normalize(db+s*scale*torch.randn_like(db),dim=-1).T)
    return (acc.argmax(1)==torch.arange(db.shape[0],device=device)).float().mean().item()

def agg(xs): return float(np.mean(xs)), float(1.96*np.std(xs,ddof=1)/math.sqrt(len(xs))) if len(xs)>1 else 0.0

def ablation(N=8,runs=8,storage='seg'):
    sig=[1.5,2.0,2.5,3.0]; res={m:{s:[] for s in sig} for m in MODES}
    for r in range(runs):
        for m in MODES:
            for s in sig: res[m][s].append(topo_trial(N,s,m,storage,1,100*r+int(10*s)))
    return {m:{s:agg(v) for s,v in d.items()} for m,d in res.items()}

def ai_fair(N=8,runs=10,dim=64,storage='seg'):
    import torch.nn as nn
    sig=[0,.5,1.2,1.5,1.8,2.0,2.5,3.0]; topo={s:[] for s in sig}; rag={s:[] for s in sig}; aw={s:[] for s in sig}
    net=nn.Sequential(nn.Linear(dim,64),nn.ReLU(),nn.Linear(64,1)).to(device); opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    for _ in range(250):
        e=torch.randn(32,dim,device=device); y=(e[:,0]>0).float(); loss=F.binary_cross_entropy_with_logits(net(e).squeeze(-1),y); opt.zero_grad(); loss.backward(); opt.step()
    rng=np.random.default_rng(7)
    for r in range(runs):
        emb=F.normalize(torch.randn(N,dim,device=device),dim=-1); signs=torch.where(net(emb).squeeze(-1)>=0,1.,-1.).tolist(); seeds=rng.integers(10,9990,N).tolist(); f,ch=(write_unified if storage=='uni' else write_segmented)(N,seeds,signs); f=gl(f,BASE,V0)
        for s in sig:
            fn=repair(noisy(f,s),s,'E_full'); topo[s].append(sum(abs(read_w(fn,c,True)-c['sign'])<.4 for c in ch)/N); rag[s].append(rag_cos(emb,s)); aw[s].append(rag_aw(emb,s))
    return {s:agg(v) for s,v in topo.items()},{s:agg(v) for s,v in rag.items()},{s:agg(v) for s,v in aw.items()},sig

def unified_sanity(N=8,runs=3):
    vals=[]
    for r in range(runs): vals.append(topo_trial(N,0.0,'E_full','uni',1,700+r))
    return agg(vals)

def multibit(N=4,runs=8,storage='seg'):
    sig=[.5,1.2,2.0]; return {s:agg([topo_trial(N,s,'E_full',storage,2,900+r*13+int(10*s)) for r in range(runs)]) for s in sig}

def print_ab(res,title):
    sig=list(next(iter(res.values())).keys()); print('\n'+title); print('  mode'.ljust(14)+''.join([f' | s={s:>3}' for s in sig])); print('  '+'-'*(13+10*len(sig)))
    for m,d in res.items(): print('  '+m.ljust(14)+''.join([f' | {d[s][0]:.3f}' for s in sig]))

def plot_fair(topo,rag,aw,sig,path):
    fig,ax=plt.subplots(figsize=(9,5.4),dpi=150); ax.plot(sig,[topo[s][0] for s in sig],'s-',label='TideMemory'); ax.plot(sig,[rag[s][0] for s in sig],'o--',label='RAG cosine'); ax.plot(sig,[aw[s][0] for s in sig],'^--',label='RAG noise-aware')
    ax.set_title('TideMemory v3 vs fairer RAG'); ax.set_xlabel('noise sigma'); ax.set_ylabel('Top-1'); ax.set_ylim(-.05,1.1); ax.grid(axis='y',color='lightgray',lw=.5); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)

if __name__=='__main__':
    out=os.path.join(os.path.dirname(__file__),'figures'); os.makedirs(out,exist_ok=True); t=time.time(); print('='*70); print('TideMemory v3 validation suite'); print('='*70)
    us=unified_sanity(8,3); print(f'\n[0] unified clean sanity N=8: {us[0]:.3f} ± {us[1]:.3f}')
    seg=ablation(8,8,'seg'); print_ab(seg,'[1] A-E ablation, segmented N=8')
    uni=ablation(8,6,'uni'); print_ab(uni,'[4] A-E ablation, unified multi-vortex N=8')
    topo,rag,aw,sig=ai_fair(8,10,64,'seg'); print('\n[2-3] stable AI task + fairer RAG'); print('  sigma | TideMemory | RAG | RAG-aware'); print('  '+'-'*42)
    for s in sig: print(f'  {s:5.2f} | {topo[s][0]:10.3f} | {rag[s][0]:.3f} | {aw[s][0]:.3f}')
    plot_fair(topo,rag,aw,sig,os.path.join(out,'fig10_v3_fair_rag.png'))
    mb1=multibit(4,8,'seg'); mb2=multibit(4,6,'uni'); print('\n[5] multi-symbol {-2,-1,+1,+2}, N=4'); print('  sigma | segmented | unified')
    for s in mb1: print(f'  {s:5.2f} | {mb1[s][0]:9.3f} | {mb2[s][0]:7.3f}')
    print('\nDone in %.1fs'% (time.time()-t)); print('Saved -> figures/fig10_v3_fair_rag.png')
