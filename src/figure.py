import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
A='../results/'; T=json.load(open(A+'targets.json'))
INK='#0b0b0b'; INK2='#52514e'; GRID='#e6e5e1'; B='#2a78d6'; O='#eb6834'
plt.rcParams.update({'font.size':9,'axes.edgecolor':INK2,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,
  'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb'})
def fit(s,v): return json.load(open(f'{A}fit_s{s}_{v}_nolag.json'))
g=np.arange(1,8)
fig,ax=plt.subplots(2,3,figsize=(11,6.6))
lab={'V':'Vincent judged to have ADHD\n(share rating ≥ 4)','S':'"I believe I have ADHD"\n(share rating ≥ 4)','P':'Estimated prevalence\n(mean log-odds)'}
def traj(a,s,k,title):
    t=T[str(s)]; m=np.array(t[k+'_mean']); se=np.array(t[k+'_se'])
    a.errorbar(g,m,1.96*se,fmt='o',color=INK,ms=4,lw=1,capsize=0,label='Observed (95% CI)',zorder=3)
    a.plot(g,fit(s,'balanced')['pred'][k+'_mean'],color=B,lw=2,label='Chain model (lineage transmission)')
    a.plot(g,fit(s,'register')['pred'][k+'_mean'],color=O,lw=2,ls='--',label='Shared lay boundary (no lineage)')
    a.set_title(title,loc='left',fontsize=9.5,color=INK); a.set_ylabel(lab[k]); a.set_xticks(g); a.set_xlabel('Generation')
    a.grid(axis='y',color=GRID,lw=.8)
traj(ax[0,0],3,'V','a  Study 3: Vincent vignette'); traj(ax[0,1],3,'S','b  Study 3: self-diagnosis'); traj(ax[0,2],3,'P','c  Study 3: prevalence estimate')
traj(ax[1,0],2,'V','d  Study 2: Vincent vignette (weak drift)')
ax[0,0].legend(frameon=False,fontsize=8,loc='lower right')
pr=json.load(open(A+'profile_s3.json')); e=[r['eta'] for r in pr]
a=ax[1,1]; L=np.array([r['loss'] for r in pr]); a.plot(e,L-L.min(),color=B,lw=2,marker='o',ms=5)
a.axhline(3.84,color=INK2,lw=.8,ls=':'); a.text(0.8,5.5,'≈95% cut-off',ha='right',color=INK2,fontsize=8)
a.set_ylim(0,70); a.set_xlabel('Learning rate η'); a.set_ylabel('Misfit above best (Δ loss)')
a.set_title('e  Study 3: how well each η fits',loc='left',fontsize=9.5,color=INK); a.grid(axis='y',color=GRID,lw=.8)
a=ax[1,2]
a.plot(e,[r['b_lag1'] for r in pr],color=B,lw=2,marker='o',ms=5,label='Boundaries (latent)')
a.plot(e,[r['V_lag1'] for r in pr],color=O,lw=2,marker='o',ms=5,label='Vincent ratings (observable)')
o=T['3']['V_lag1']; se=1/np.sqrt(T['3']['V_lag1_n'])
a.axhspan(o-1.96*se,o+1.96*se,color=GRID,zorder=0); a.text(0.8,o-1.96*se-0.04,'grey band: observed rating correlation, 95% CI',ha='right',color=INK2,fontsize=8); a.set_ylim(-0.12,0.7)
a.set_xlabel('Learning rate η'); a.set_ylabel('Lag-1 correlation within chain')
a.set_title('f  Lineage signal: latent vs observable',loc='left',fontsize=9.5,color=INK); a.legend(frameon=False,fontsize=8,loc='upper left',bbox_to_anchor=(0,0.88))
a.grid(axis='y',color=GRID,lw=.8)
fig.tight_layout(); fig.savefig('../figures/'+'chain_model_fit.png',dpi=170)
