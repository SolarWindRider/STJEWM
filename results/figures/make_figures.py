#!/usr/bin/env python3
"""ST-JEWM NMI-style figure suite (Fig.1-5, 19 panels).

Figure specification: 180 mm double-column, white bg, thin axes, sans-serif,
three-colour model hierarchy (ST-JEWM accent / spiking / continuous).
No proxy metric anywhere. Data: v2 retrain落盘 + 现采轨迹.
"""
import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch, Circle
from pathlib import Path

OUT = Path("/home/lx/snn/results/figures"); OUT.mkdir(parents=True, exist_ok=True)
TRJ = json.load(open("/data/lx/tmp/results/fig_trajectories/cartpole_traces.json"))

# ---- global style ----
C_STJ   = "#1b4f72"   # ST-JEWM accent (deep blue)
C_SPK   = "#4c9f70"   # spiking baselines (muted green)
C_CON   = "#9a3b3b"   # continuous baselines (muted red)
C_GRA   = "#777777"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 0.9, "lines.markersize": 3.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})
MM = 1/25.4

def panel_label(ax, s, dx=-0.12, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")

def newfig(w_in, h_in, ncols, nrows, wspace=0.55, hspace=0.5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w_in, h_in))
    fig.subplots_adjust(wspace=wspace, hspace=hspace, left=0.09, right=0.97, top=0.90, bottom=0.13)
    return fig, np.atleast_1d(axes).ravel()

# =============== data (v2 retrain, audited) ===============
E1_COS = {"STJEWM-trace":0.246,"STJEWM-spike":0.233,"STJEWM-rate":0.257,"STJEWM-no_trace":0.262,
          "STJEWM-leak":0.239,"STJEWM-membrane":0.267,"ALIF":0.128,"SLIF-trace":0.073,"SLIF-free":0.097,
          "LeWM":0.196,"GRU":0.148,"MLP":0.029,"LIF-Tx":0.000}
E2_SEED = {"STJEWM-trace":(.2682,.0151),"STJEWM-spike":(.2270,.0203),"STJEWM-rate":(.2630,.0074),
           "STJEWM-no_trace":(.2529,.0366),"STJEWM-membrane":(.2687,.0241),"STJEWM-leak":(.2707,.0192),
           "ALIF":(.1560,.0308),"SLIF-trace":(.0550,.0184),"SLIF-free":(.2019,.0159),
           "LeWM":(.1803,.0577),"GRU":(.1530,.0243),"MLP":(.1817,.0809),"LIF-Tx":(.0130,.0207)}
E6_REG = {"STJEWM-trace":(0.207,0.0142,0.9993),"STJEWM-spike":(0.200,0.0136,0.9999),
          "STJEWM-rate":(0.207,0.0156,0.9993),"STJEWM-no_trace":(0.205,0.0138,0.9984),
          "STJEWM-leak":(0.201,0.0144,0.9996),"STJEWM-membrane":(0.213,0.0122,0.9999),
          "ALIF":(0.202,0.0140,0.9364),"SLIF-trace":(0.388,0.0150,0.9297),
          "SLIF-free":(0.381,0.0139,0.9677),"LeWM":(21.5,0.207,0.2237),
          "GRU":(74.7,0.030,0.1492),"MLP":(0.0,0.0,0.3664)}
E8 = {"LeWM":(.6489,.2237),"GRU":(.4634,.1492),"SLIF-free":(.3296,.9677),
      "STJEWM-leak":(.2942,.9996),"STJEWM-trace":(.2714,.9993),"STJEWM-no_trace":(.2676,.9984),
      "SLIF-trace":(.2674,.9297),"STJEWM-membrane":(.2613,.9999),"STJEWM-spike":(.2612,.9999),
      "ALIF":(.2529,.9364),"STJEWM-rate":(.2467,.9993),"MLP":(-.0011,.3664),"LIF-Tx":(-.0026,-.0247)}
E5_RESP = {"STJEWM-trace":(0.207,0.205,0.207),"LeWM":(26.4,28.0,21.5),
           "GRU":(82.6,84.0,74.7),"MLP":(0,0,0)}
E7_COS = {0:0.0931,0.001:0.2505,0.01:0.2443,0.09:0.2520}
E7_DIV = {0.09:0.00004,0:0.00001}
E9 = {"cartpole_2d":{k:v for k,v in [("baseline",0.0592),("event\nwindow",0.0592),("non-event",0.0592),
        ("random",0.0592),("ablate all",0.0592),("CEM\nrollout",0.1137)]},
      "cheetah":{k:v for k,v in [("baseline",0.2358),("event\nwindow",0.2358),("non-event",0.2358),
        ("random",0.2358),("ablate all",0.2358),("CEM\nrollout",0.1766)]}}
E10 = {"cheetah":{"STJEWM-trace":[0.0754,0.0211,0.0208,0.0340,0.0419],
                  "STJEWM-rate":[0.0491,0.0336,0.0123,0.0154,0.0071]},
       "walker":{"STJEWM-trace":[0.1691,0.1032,0.1001,0.0963,0.0945]},
       "reacher":{"STJEWM-trace":[0.6496,0.6623,0.6608,0.6363,0.5999]}}
PARAMS = {"STJEWM":5.06,"ALIF":4.98,"SLIF-trace":5.11,"SLIF-free":5.05,"LeWM":4.97,
          "GRU":5.13,"MLP":5.00,"LIF-Tx":5.12}

def col_of(name):
    if name.startswith("STJEWM"): return C_STJ
    if name in ("ALIF","SLIF-trace","SLIF-free"): return C_SPK
    return C_CON

# =================== FIGURE 1 ===================
def fig1():
    fig = plt.figure(figsize=(180*MM, 150*MM))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30, left=0.05, right=0.96, top=0.93, bottom=0.07)
    ax_a = fig.add_subplot(gs[0,0]); ax_b = fig.add_subplot(gs[0,1])
    ax_c = fig.add_subplot(gs[1,0]); ax_d = fig.add_subplot(gs[1,1])
    for ax,lab in [(ax_a,"a"),(ax_b,"b"),(ax_c,"c"),(ax_d,"d")]:
        panel_label(ax, lab); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

    # ---- a: membrane-forbidden interface ----
    ax_a.set_xlim(0,10); ax_a.set_ylim(0,10)
    def box(ax,x,y,w,h,txt,fc):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.12",
                       fc=fc,ec="#333333",lw=0.7))
        ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=6.5)
    box(ax_a,0.3,6.6,2.2,1.6,"Environment","#f2f2f2")
    box(ax_a,3.9,5.4,3.4,3.8,"Spiking\nworld model","#e8eef4")
    box(ax_a,7.9,6.6,1.9,1.6,"Planner","#f2f2f2")
    ax_a.annotate("",xy=(3.85,7.4),xytext=(2.6,7.4),arrowprops=dict(arrowstyle="->",lw=0.8,color="#333"))
    ax_a.text(3.2,7.7,"$o_t$",fontsize=6.5,ha="center")
    ax_a.annotate("",xy=(9.8,7.4),xytext=(7.4,7.4),arrowprops=dict(arrowstyle="->",lw=0.8,color="#333"))
    # forbidden membrane path
    ax_a.plot([5.6,5.6],[5.30,4.60],color=C_CON,lw=1.0,ls=(0,(4,2)))
    ax_a.plot([5.25,5.95],[4.85,4.15],color=C_CON,lw=1.0)
    ax_a.plot([5.25,5.95],[4.15,4.85],color=C_CON,lw=1.0)
    ax_a.text(6.15,4.45,"membrane potential\nnot exposed",fontsize=6,color=C_CON,va="center")
    # legal paths
    for y,lab in [(4.7,"spikes $s_t$ (discrete events)"),(3.6,"trace $r_t$ (bounded)")]:
        ax_a.annotate("",xy=(9.8,y),xytext=(7.4,y),
                      arrowprops=dict(arrowstyle="->",lw=0.8,color=C_STJ))
        ax_a.text(8.6,y+0.25,lab,fontsize=5.6,ha="center",color=C_STJ)
    ax_a.text(5.6,1.9,"spike-derived temporal interface = legal read-out",fontsize=6.2,
              ha="center",color=C_STJ)

    # ---- b: JEPA objective ----
    ax_b.set_xlim(0,10); ax_b.set_ylim(0,10)
    box(ax_b,0.4,7.2,2.1,1.5,"$o_t$","#f2f2f2")
    box(ax_b,0.4,3.2,2.1,1.5,"$o_{t+1}$","#f2f2f2")
    box(ax_b,3.3,7.2,2.6,1.5,"Online\nencoder","#e8eef4")
    box(ax_b,3.3,3.2,2.6,1.5,"Target\nencoder (EMA)","#e8eef4")
    box(ax_b,6.6,7.2,2.0,1.5,"Predictor","#e8eef4")
    ax_b.annotate("",xy=(3.25,7.95),xytext=(2.55,7.95),arrowprops=dict(arrowstyle="->",lw=0.8))
    ax_b.annotate("",xy=(3.25,3.95),xytext=(2.55,3.95),arrowprops=dict(arrowstyle="->",lw=0.8))
    ax_b.annotate("",xy=(6.55,7.95),xytext=(5.95,7.95),arrowprops=dict(arrowstyle="->",lw=0.8))
    ax_b.text(4.55,6.9,"$z_t$",fontsize=6.5,ha="center")
    ax_b.text(4.55,2.9,"$z_{t+1}$",fontsize=6.5,ha="center")
    ax_b.text(7.6,6.9,"$\\hat z_{t+1}$",fontsize=6.5,ha="center")
    # loss
    ax_b.add_patch(FancyBboxPatch((1.2,0.6),7.6,1.7,boxstyle="round,pad=0.12",fc="#f7f7f7",ec="#333",lw=0.7))
    ax_b.text(5.0,1.45,r"$\mathcal{L}=\mathcal{L}_{\rm pred}+0.09\,\mathcal{L}_{\rm SIGReg}+0.5\,\mathcal{L}_{\rm goal}$",
              fontsize=6.8,ha="center",va="center")
    ax_b.text(5.0,9.3,"ST-JEWM: JEPA-style latent prediction\non a spiking substrate (5.06M trainable)",
              fontsize=6.2,ha="center")

    # ---- c: three axes ----
    ax_c.set_xlim(0,10); ax_c.set_ylim(0,10)
    subs = [(0.3,"Diversity",r"$\mathrm{div}=\frac{1}{D}\sum_d\mathrm{Std}_t(z_{t,d})$"),
            (3.6,"Responsiveness",r"$\mathrm{resp}=\frac{\mathbb{E}\|\Delta z_t\|}{\mathbb{E}\|\Delta o_t\|}$"),
            (6.9,"Event sync.",r"$\rho_{\rm event}=\mathrm{corr}(\|\Delta o_t\|,\|\Delta z_t\|)$")]
    for x0,ttl,form in subs:
        ax_c.text(x0+1.4,9.3,ttl,fontsize=6.8,ha="center",fontweight="bold")
        ax_c.text(x0+1.4,0.4,form,fontsize=6.2,ha="center")
        ax_c.plot([x0,x0+2.8],[2.2,2.2],color="#999",lw=0.5)
    # diversity mini
    xs=np.linspace(0,2.6,100)
    ax_c.plot(xs+x0 if False else xs,[1.0+0.5*math.sin(6*x)+0.1*r for x,r in zip(xs,np.random.RandomState(1).rand(100)-0.5)][:len(xs)] if False else
              1.0+0.5*np.sin(6*xs)+0.08*np.random.RandomState(1).standard_normal(100), color=C_STJ, lw=0.8)
    ax_c.plot(xs, np.full_like(xs,3.1), color=C_CON, lw=1.1)
    ax_c.text(1.4,4.0,"healthy",fontsize=5.8,color=C_STJ,ha="center")
    ax_c.text(1.4,3.5,"collapse",fontsize=5.8,color=C_CON,ha="center")
    # responsiveness mini
    for y,lab,c in [(3.4,"no response",C_CON),(2.6,"calibrated",C_STJ),(1.8,"overreaction",C_CON)]:
        ax_c.plot([3.7,4.6],[y,y+0.5],color=c,lw=0.9)
        ax_c.plot([4.6,5.4],[y+0.5,y],color=c,lw=0.9)
        ax_c.text(6.3 if False else 5.6,y+0.2,lab,fontsize=5.6,color=c,va="center")
    # event sync mini
    tt=np.linspace(7.0,9.8,200)
    sig=0.6+0.4*((np.sin(8*tt)>0).astype(float))
    ax_c.plot(tt,4.6+0.9*(np.sin(8*tt)>0).astype(float),color="#555",lw=0.9)
    ax_c.plot(tt,2.6+0.9*(np.sin(8*tt+0.15)>0).astype(float),color=C_STJ,lw=0.9)
    ax_c.text(8.4,6.0,r"$\|\Delta o_t\|$",fontsize=5.8,ha="center")
    ax_c.text(8.4,2.2,r"$\|\Delta z_t\|$",fontsize=5.8,color=C_STJ,ha="center")

    # ---- d: regime map ----
    ax_d.set_xlim(-0.2,10.2); ax_d.set_ylim(-0.2,10.2)
    ax_d.set_xlabel("Responsiveness",fontsize=7)
    ax_d.set_ylabel("Diversity",fontsize=7)
    ax_d.set_xticks([]); ax_d.set_yticks([])
    for s in ax_d.spines.values(): s.set_visible(True)
    ax_d.annotate("",xy=(9.8,0.4),xytext=(0.4,0.4),arrowprops=dict(arrowstyle="->",lw=0.8))
    ax_d.annotate("",xy=(0.4,9.8),xytext=(0.4,0.4),arrowprops=dict(arrowstyle="->",lw=0.8))
    # regions
    ax_d.add_patch(Rectangle((0.6,0.7),3.2,3.0,fc="#f6d7d7",ec="none"))
    ax_d.add_patch(Rectangle((7.0,5.6),2.8,3.8,fc="#f6ddd0",ec="none"))
    ax_d.add_patch(Rectangle((4.0,0.7),3.0,3.0,fc="#eeeeee",ec="none"))
    ax_d.add_patch(Circle((5.6,7.2),0.45,fc=C_STJ,ec="none"))
    ax_d.text(2.2,2.2,"Collapse",fontsize=7,ha="center",color=C_CON)
    ax_d.text(8.4,7.4,"Overreaction",fontsize=7,ha="center",color=C_CON)
    ax_d.text(5.5,2.2,"Desynchronized",fontsize=6.4,ha="center",color="#555")
    ax_d.text(5.6,8.0,"Event-calibrated",fontsize=6.6,ha="center",color=C_STJ,fontweight="bold")
    ax_d.text(5.6,6.4,"★",fontsize=13,ha="center",color=C_STJ)
    ax_d.text(9.9,0.05,"resp",fontsize=6,ha="right")
    ax_d.text(0.05,9.6,"div",fontsize=6,va="top")
    ax_d.text(5.0,-0.15,"marker size / opacity: event synchronization",fontsize=5.6,ha="center",color="#555")
    fig.savefig(OUT/"Fig1.pdf"); fig.savefig(OUT/"Fig1.png")
    plt.close(fig)

# =================== FIGURE 2 ===================
def fig2():
    fig = plt.figure(figsize=(180*MM, 155*MM))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.46, left=0.115, right=0.965, top=0.94, bottom=0.115)
    ax_a, ax_b = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])
    ax_c, ax_d = fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])
    panel_label(ax_a,"a"); panel_label(ax_b,"b"); panel_label(ax_c,"c"); panel_label(ax_d,"d")

    # a: parameter matching
    groups=[("ST-JEWM\n(6 readouts)",[PARAMS["STJEWM"]]*6,C_STJ),
            ("ALIF",[PARAMS["ALIF"]],C_SPK),("SLIF-trace",[PARAMS["SLIF-trace"]],C_SPK),
            ("SLIF-free",[PARAMS["SLIF-free"]],C_SPK),("LeWM",[PARAMS["LeWM"]],C_CON),
            ("GRU",[PARAMS["GRU"]],C_CON),("MLP",[PARAMS["MLP"]],C_CON),("LIF-Tx",[PARAMS["LIF-Tx"]],C_CON)]
    for i,(lab,vals,c) in enumerate(groups):
        for v in vals:
            ax_a.plot(v,-i,"o",color=c,ms=3.5)
    ax_a.axvspan(4.97,5.13,color="#e8eef4",zorder=0)
    ax_a.set_yticks(range(0,-8,-1)); ax_a.set_yticklabels([g[0] for g in groups])
    ax_a.set_xlabel("Trainable parameters (M)"); ax_a.set_xlim(4.85,5.25)
    ax_a.text(5.05,0.55,"4.97–5.13 M\n(±3.2%)",fontsize=6,ha="center",color="#33608c")

    # b: cos_dist dot
    order=sorted(E1_COS, key=E1_COS.get)
    for i,name in enumerate(order):
        ax_b.plot(E1_COS[name],-i,"o",color=col_of(name),ms=3.5)
    ax_b.set_yticks(range(0,-13,-1)); ax_b.set_yticklabels(order)
    ax_b.set_xlabel("cos_dist to goal  (lower ≠ better)")
    ax_b.text(0.97,-12.4,"Predictive agreement alone does not\nidentify a calibrated latent state",
              fontsize=5.6,ha="right",style="italic",color="#555")

    # c: regime map — one label per family cluster
    for name,(resp,dv,rho) in E6_REG.items():
        size = 20 + 80*max(rho,0)
        ax_c.plot(resp,dv,"o",color=col_of(name),ms=math.sqrt(size),alpha=0.8)
    # family labels (no per-readout clutter)
    ax_c.annotate("ST-JEWM family\n(6 readouts, ρ≥0.9984)",(0.22,0.0141),xytext=(0.30,0.060),
                  fontsize=5.6,color=C_STJ,arrowprops=dict(arrowstyle="->",lw=0.5,color=C_STJ))
    ax_c.annotate("SLIF",(0.388,0.0150),xytext=(3.2,0.0175),fontsize=5.4,color=C_SPK)
    ax_c.annotate("ALIF",(0.202,0.0140),xytext=(0.05,0.0195),fontsize=5.4,color=C_SPK)
    ax_c.annotate("LeWM",(21.5,0.207),xytext=(30,0.215),fontsize=5.4,color=C_CON)
    ax_c.annotate("GRU",(74.7,0.030),xytext=(38,0.048),fontsize=5.4,color=C_CON)
    ax_c.annotate("MLP",(0,0),xytext=(1.2,0.006),fontsize=5.4,color=C_CON)
    ax_c.annotate("LIF-Tx",(0,0.0),xytext=(1.2,0.012),fontsize=5.4,color=C_CON)
    ax_c.set_xscale("symlog",linthresh=1)
    ax_c.set_xlabel("Responsiveness"); ax_c.set_ylabel("Diversity")
    ax_c.text(0.03,0.05,"marker size: event synchronization",transform=ax_c.transAxes,fontsize=5.4,color="#555")
    ax_c.text(0.97,0.95,"regime coordinates from dedicated\ndiagnostic experiments (E5/E6)",
              transform=ax_c.transAxes,fontsize=5.0,color="#555",ha="right",style="italic")

    # d: four representative trajectory mini-axes (nested)
    ax_d.set_xticks([]); ax_d.set_yticks([])
    for s in ax_d.spines.values(): s.set_visible(False)
    ax_d.set_xlim(0,1); ax_d.set_ylim(0,1)
    cols=[("Collapse","MLP",C_CON),("Overreaction","GRU",C_CON),
          ("Desync.","LeWM",C_CON),("Calibrated","STJEWM-trace",C_STJ)]
    for k,(ttl,m,c) in enumerate(cols):
        rect=[0.02+k*0.25, 0.34, 0.21, 0.58]
        sub = ax_d.inset_axes(rect)
        dz=np.array(TRJ[m]["dz"]); dn=np.array(TRJ[m]["do"])
        idx=np.linspace(0,len(dz)-1,min(len(dz),300)).astype(int)
        xx=np.linspace(0,1,len(idx))
        sub.plot(xx,dz[idx]/(dz.max()+1e-9),color=c,lw=0.6)
        idx2=np.linspace(0,len(dn)-1,min(len(dn),300)).astype(int)
        sub.plot(xx,0.35+0.6*(dn[idx2]/(dn.max()+1e-9)),color="#555",lw=0.6,ls=(0,(3,2)))
        sub.set_xticks([]); sub.set_yticks([])
        for s2 in sub.spines.values(): s2.set_linewidth(0.4)
        sub.text(0.5,0.90,ttl,fontsize=6.0,ha="center",fontweight="bold",transform=sub.transAxes)
        sub.text(0.5,0.78,m,fontsize=5.6,ha="center",color="#555",transform=sub.transAxes)
        sub.text(0.03,0.06,r"$\|\Delta z\|$",fontsize=5.2,color=c,transform=sub.transAxes)
        sub.text(0.03,0.62,r"$\|\Delta o\|$",fontsize=5.2,color="#555",transform=sub.transAxes)
    ax_d.text(0.5,0.20,"200-step random-policy episodes (cartpole); traces normalised per axis",
              transform=ax_d.transAxes,fontsize=5.2,ha="center",color="#555")
    fig.savefig(OUT/"Fig2.pdf"); fig.savefig(OUT/"Fig2.png"); plt.close(fig)

# =================== FIGURE 3 ===================
def fig3():
    fig = plt.figure(figsize=(180*MM, 145*MM))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.42, left=0.10, right=0.97, top=0.93, bottom=0.10)
    ax_a,ax_b,ax_c,ax_d = (fig.add_subplot(gs[i]) for i in range(4))
    panel_label(ax_a,"a"); panel_label(ax_b,"b"); panel_label(ax_c,"c"); panel_label(ax_d,"d")

    # a: seed robustness
    order=sorted(E2_SEED,key=lambda k:E2_SEED[k][0])
    for i,name in enumerate(order):
        mu,sd=E2_SEED[name]
        ci=4.303*sd/math.sqrt(3)
        ax_a.errorbar(mu,-i,yerr=[[ci],[ci]],fmt="o",color=col_of(name),ms=3.5,capsize=1.6,lw=0.8)
    ax_a.set_yticks(range(0,-13,-1)); ax_a.set_yticklabels(order)
    ax_a.set_xlabel("mean cos_dist (3 seeds, F1)")

    # b: readout robustness
    ro=[("no_trace",0.9984),("rate",0.9993),("trace",0.9993),("leak",0.9996),("membrane",0.9999),("spike",0.9999)]
    for i,(n,r) in enumerate(ro):
        ax_b.plot(i,r,"o",color=C_STJ,ms=4)
    ax_b.set_xticks(range(6)); ax_b.set_xticklabels([n for n,_ in ro],rotation=20,ha="right")
    ax_b.set_ylim(0.998,1.0005); ax_b.set_ylabel(r"$\rho_{\rm event}$")
    ax_b.axhline(0.9984,color="#999",lw=0.5,ls=(0,(3,2)))
    ax_b.text(2.5,0.99845,"all readouts ≥ 0.9984",fontsize=5.8,color="#555")

    # c: state vs pixel regime (div vs resp)
    pix={"STJEWM-trace":(0.207,0.0134),"STJEWM-spike":(0.200,0.0135),"STJEWM-rate":(0.207,0.0122),
         "STJEWM-no_trace":(0.205,0.0113),"STJEWM-leak":(0.210,0.0148),"STJEWM-membrane":(0.207,0.0136),
         "ALIF":(18.9,0.0527),"SLIF-trace":(61.4,0.0103),"SLIF-free":(175.6,0.0268),
         "LeWM":(49.8,0.0230),"GRU":(1176.8,0.7837),"MLP":(0.0,0.0),"LIF-Tx":(11.8,0.0033)}
    for name,(resp,dv,_r) in E6_REG.items():
        ax_c.plot(resp,dv,"o",color=col_of(name),ms=3.5)
    for name,(resp,dv) in pix.items():
        ax_c.plot(resp,dv,"x",color=col_of(name),ms=4,mew=1.0)
    ax_c.set_xscale("symlog",linthresh=1); ax_c.set_yscale("symlog",linthresh=0.005)
    ax_c.set_xlabel("Responsiveness"); ax_c.set_ylabel("Diversity")
    ax_c.plot([],[],"o",color="#555",label="state"); ax_c.plot([],[],"x",color="#555",label="pixel")
    ax_c.legend(loc="upper left",frameon=False)
    ax_c.annotate("7/7 baselines collapse on pixels\n6/6 ST-JEWM readouts remain non-degenerate",
                  (0.97,0.04),xycoords="axes fraction",fontsize=5.6,ha="right",color=C_STJ)

    # d: scale invariance + OOD
    xs=[4,8,16]
    for name,c in [("STJEWM-trace",C_STJ),("LeWM",C_CON),("GRU",C_CON),("MLP",C_SPK)]:
        ax_d.plot(xs,E5_RESP[name],"o-",color=c,ms=3.2,lw=0.8)
        ax_d.annotate(name,(16,E5_RESP[name][2]),fontsize=5.2,xytext=(4,-2),textcoords="offset points",color=c)
    ax_d.set_yscale("symlog",linthresh=1)
    ax_d.set_xticks(xs); ax_d.set_xlabel("Training environments"); ax_d.set_ylabel("Responsiveness")
    ax_d.text(0.03,0.92,"scale invariance",transform=ax_d.transAxes,fontsize=5.8,color="#555")
    fig.savefig(OUT/"Fig3.pdf"); fig.savefig(OUT/"Fig3.png"); plt.close(fig)

# =================== FIGURE 4 ===================
def fig4():
    fig = plt.figure(figsize=(180*MM, 155*MM))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.45, left=0.09, right=0.97, top=0.93, bottom=0.10)
    ax_a,ax_b,ax_c,ax_d=(fig.add_subplot(gs[i]) for i in range(4))
    panel_label(ax_a,"a"); panel_label(ax_b,"b"); panel_label(ax_c,"c"); panel_label(ax_d,"d")

    # a: scatter
    for name,(r2,rho) in E8.items():
        ax_a.plot(r2,rho,"o",color=col_of(name),ms=4)
    ax_a.annotate("ST-JEWM\n(6 readouts)",(0.272,0.9995),xytext=(0.30,0.82),
                  fontsize=5.6,color=C_STJ,arrowprops=dict(arrowstyle="->",lw=0.5,color=C_STJ))
    for name,dxdy in [("SLIF-free",(0.015,0.03)),("SLIF-trace",(0.015,-0.06)),("ALIF",(0.015,0.02)),
                      ("GRU",(0.012,0.03)),("MLP",(0.012,0.03)),("LIF-Tx",(0.012,-0.05))]:
        r2,rho=E8[name]
        ax_a.annotate(name,(r2,rho),xytext=(r2+dxdy[0],rho+dxdy[1]),fontsize=5.2)
    ax_a.annotate("high position decodability\n$R^2=0.649$",xy=(0.6489,0.2237),xytext=(0.36,0.52),
                  fontsize=5.8,arrowprops=dict(arrowstyle="->",lw=0.6))
    ax_a.annotate("weak event sync.\n$\\rho=0.224$",xy=(0.6489,0.2237),xytext=(0.44,0.02),
                  fontsize=5.8,arrowprops=dict(arrowstyle="->",lw=0.6,color=C_CON))
    ax_a.set_xlabel("position-$R^2$ (linear probe)")
    ax_a.set_ylabel(r"$\rho_{\rm event}$")
    ax_a.set_xlim(-0.08,0.75); ax_a.set_ylim(-0.12,1.12)
    ax_a.axhline(0,color="#ccc",lw=0.5)

    # b: collapse trajectories
    for k,(m,lab) in enumerate([("MLP","MLP"),("LIF-Tx","LIF-Tx")]):
        x0=0.08+k*0.5
        ax_b.add_patch(Rectangle((x0-0.04,0.08),0.52,0.84,transform=ax_b.transAxes,fc="#fafafa",ec="#ccc",lw=0.5))
        zn=np.array(TRJ[m]["z_norm"])
        xx=np.linspace(0,1,min(len(zn),200)); idx=np.linspace(0,len(zn)-1,200).astype(int)
        ax_b.plot(xx,zn[idx],color=C_CON,lw=0.7)
        ax_b.text(x0+0.22,0.86,f"{lab}   $z_t$ (near-constant)",fontsize=6.0,transform=ax_b.transAxes,ha="center")
        r2=E8[m][0]
        ax_b.text(x0+0.22,0.14,f"position probe $R^2={r2:.4f}$",fontsize=6.0,transform=ax_b.transAxes,ha="center")
    ax_b.set_xticks([]); ax_b.set_yticks([])
    for s in ax_b.spines.values(): s.set_visible(False)
    ax_b.text(0.5,1.0,"independent linear-probe validation of collapse",transform=ax_b.transAxes,
              fontsize=6.0,ha="center",color="#555")

    # c: LeWM explanatory panel
    ax_c.set_xlim(0,10); ax_c.set_ylim(0,10); ax_c.set_xticks([]); ax_c.set_yticks([])
    for s in ax_c.spines.values(): s.set_visible(False)
    ax_c.add_patch(FancyBboxPatch((0.4,4.6),3.4,3.4,boxstyle="round,pad=0.15",fc="#f6ddd0",ec=C_CON,lw=0.8))
    ax_c.add_patch(FancyBboxPatch((6.2,4.6),3.4,3.4,boxstyle="round,pad=0.15",fc="#f6d7d7",ec=C_CON,lw=0.8))
    ax_c.text(2.1,7.1,"Physical information",fontsize=6.8,ha="center",fontweight="bold")
    ax_c.text(2.1,5.9,"$R^2=0.649$",fontsize=9,ha="center")
    ax_c.text(7.9,7.1,"Temporal organization",fontsize=6.8,ha="center",fontweight="bold")
    ax_c.text(7.9,5.9,r"$\rho_{\rm event}=0.224$",fontsize=9,ha="center")
    ax_c.text(2.1,3.6,"“where”  ✓",fontsize=9,ha="center",color="#2e7d32")
    ax_c.text(7.9,3.6,"“when”  ✗",fontsize=9,ha="center",color=C_CON)
    ax_c.plot([5.0,5.0],[3.2,7.2],color="#999",lw=0.6,ls=(0,(3,2)))
    ax_c.text(5.0,1.6,"Encoding state information does not imply temporal calibration",
              fontsize=6.6,ha="center",style="italic")
    ax_c.text(5.0,9.2,"LeWM counterexample",fontsize=7.0,ha="center",fontweight="bold")

    # d: diagnostic matrix (table)
    ax_d.set_xlim(0,10); ax_d.set_ylim(0,10); ax_d.set_xticks([]); ax_d.set_yticks([])
    for s in ax_d.spines.values(): s.set_visible(False)
    cols=["Regime","Diversity","Resp.","Event sync."]
    rows=[("Collapse","↓↓","↓↓","—"),
          ("Overreaction","↑","↑↑","↓ / var."),
          ("Desync.","finite","finite","↓"),
          ("Calibrated","finite","moderate","↑↑")]
    xw=[2.6,2.2,2.2,2.6]; x0=0.2
    for j,(cw,ct) in enumerate(zip(xw,cols)):
        ax_d.text(x0+sum(xw[:j])+cw/2,9.0,ct,fontsize=6.4,ha="center",fontweight="bold")
    for r,(row) in enumerate(rows):
        y=7.4-r*1.7
        if row[0]=="Calibrated":
            ax_d.add_patch(Rectangle((x0,y-0.55),sum(xw),1.5,fc="#e8eef4",ec="none"))
        for j,val in enumerate(row):
            ax_d.text(x0+sum(xw[:j])+xw[j]/2,y,val,fontsize=6.4,ha="center",
                      fontweight="bold" if j==0 or row[0]=="Calibrated" else "normal")
    ax_d.text(5.0,0.4,"definitional schematic — not a statistical result",fontsize=5.4,ha="center",color="#555")
    fig.savefig(OUT/"Fig4.pdf"); fig.savefig(OUT/"Fig4.png"); plt.close(fig)

# =================== FIGURE 5 ===================
def fig5():
    fig = plt.figure(figsize=(180*MM, 110*MM))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.07, right=0.97, top=0.90, bottom=0.16)
    ax_a,ax_b,ax_c=(fig.add_subplot(gs[i]) for i in range(3))
    panel_label(ax_a,"a"); panel_label(ax_b,"b"); panel_label(ax_c,"c")

    # a: SIGReg sweep (two aligned: cos + div)
    lam=[0,0.001,0.01,0.09]
    cos=[E7_COS[l] for l in lam]
    ax_a.plot(lam,cos,"o-",color=C_STJ,ms=4,lw=0.9)
    for l,c in zip(lam,cos): ax_a.annotate(f"{c:.3f}",(l,c),xytext=(2,5),textcoords="offset points",fontsize=5.4)
    ax_a.set_xscale("symlog",linthresh=0.001)
    ax_a.set_xticks([0,0.001,0.01,0.09]); ax_a.set_xticklabels(["0","$10^{-3}$","$10^{-2}$","$9\\times10^{-2}$"])
    ax_a.set_xlabel(r"$\lambda_{\rm SIGReg}$"); ax_a.set_ylabel("cos_dist to goal")
    ax_a.annotate("lower cosine distance\nis misleading here",(0.09,0.11),xytext=(0.004,0.14),
                  fontsize=5.8,arrowprops=dict(arrowstyle="->",lw=0.6,color=C_CON))
    ax_a.annotate("",xy=(0,0.045),xytext=(0,0.09),arrowprops=dict(arrowstyle="-",lw=6,color=C_CON,alpha=0.35))
    ax_a.text(0.003,0.045,"low-variance regime:\ndiv drops 4× (0.00004→0.00001)",fontsize=5.6)

    # b: trace causality paired dot
    for k,(env,modes) in enumerate(E9.items()):
        base_y = 1-k*1.35
        for j,(mode,v) in enumerate(modes.items()):
            ax_b.plot(v, base_y - j*0.22, "o", color=C_STJ if mode!="CEM\nrollout" else C_CON, ms=4)
            ax_b.text(v+0.004, base_y - j*0.22, mode, fontsize=5.4, va="center",
                      color=C_CON if mode=="CEM\nrollout" else "#444")
    ax_b.set_xlim(0,0.30); ax_b.set_ylim(-1.1,1.4)
    ax_b.set_yticks([1.0,-0.35]); ax_b.set_yticklabels(["cartpole_2d","cheetah"])
    ax_b.set_xlabel("terminal cos_dist")
    ax_b.text(0.29,1.25,"No detectable effect of\ntrace-window content",fontsize=5.8,ha="right",color=C_STJ)
    ax_b.text(0.29,-0.95,"CEM information flow\nis sensitive",fontsize=5.8,ha="right",color=C_CON)

    # c: horizon lines
    H=[1,3,5,10,20]
    for (env,m),c,ls in [(("cheetah","STJEWM-trace"),C_STJ,"-"),(("cheetah","STJEWM-rate"),C_STJ,(0,(4,2))),
                         (("walker","STJEWM-trace"),C_SPK,"-"),(("reacher","STJEWM-trace"),C_CON,"-")]:
        ax_c.plot(H,E10[env][m],"o-",color=c,ms=3.2,lw=0.9,ls=ls,
                  label=f"{m.split('-')[0] if m.startswith('STJEWM') else m} · {env}")
    ax_c.set_xscale("log"); ax_c.set_xticks(H); ax_c.set_xticklabels(H)
    ax_c.set_xlabel("Planning horizon $H$ (model steps)")
    ax_c.set_ylabel("terminal cos_dist")
    ax_c.legend(frameon=False,fontsize=5.2,loc="upper right")
    ax_c.text(0.03,0.93,"optimal horizon depends on task and readout",transform=ax_c.transAxes,fontsize=5.8,color="#555")
    fig.savefig(OUT/"Fig5.pdf"); fig.savefig(OUT/"Fig5.png"); plt.close(fig)

if __name__ == "__main__":
    fig1(); print("Fig1 done")
    fig2(); print("Fig2 done")
    fig3(); print("Fig3 done")
    fig4(); print("Fig4 done")
    fig5(); print("Fig5 done")
    print("ALL FIGURES ->", OUT)
