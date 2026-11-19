import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
A='../results/'; N=json.load(open(A+'network.json'))
INK='#0b0b0b'; INK2='#52514e'; GRID='#e6e5e1'; B='#2a78d6'; G='#a8a79f'
plt.rcParams.update({'font.size':9,'axes.edgecolor':INK2,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,
  'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb'})
fig,ax=plt.subplots(2,2,figsize=(10,7.2)); pc=lambda v:100*np.array(v)
def ttl(a,t): a.set_title(t,loc='left',fontsize=9.5,color=INK); a.grid(axis='y',color=GRID,lw=.8)
# a ratchet
a=ax[0,0]; qs=['base',0.5,0.9]; x=np.arange(3)
for i,(lam,c,l) in enumerate([(0.1,B,'reversion 0.1 per round'),(0.0,G,'no reversion')]):
    v=[N['ratchet'][f'{q}|{lam}|complete'][0] for q in qs]
    a.bar(x+(i-.5)*.36,pc(v),.32,color=c,label=l)
    for xi,vi in zip(x,v): a.text(xi+(i-.5)*.36,100*vi+.6,f'{100*vi:.0f}%',ha='center',fontsize=8,color=INK)
a.axhline(20.9,color=INK2,lw=.8,ls=':'); a.text(-0.45,31,'dotted line: lay prior, 21%',ha='left',fontsize=8,color=INK2)
a.set_xticks(x); a.set_xticklabels(['population\nbase rate','half','nine in ten']); a.set_xlabel('Share of examples in a post that are positive cases')
a.set_ylabel('Implied prevalence after 300 rounds (%)'); a.legend(frameon=False,fontsize=8,loc='upper left'); a.set_ylim(0,42)
ttl(a,'a  Peer talk alone broadens the category\n   when posts over-represent positive cases')
# b experts
a=ax[0,1]; FS=[0,0.02,0.05,0.1,0.2,0.4]
for k,c,ls in [('complete',B,'-'),('ring',INK,'--'),('scalefree',G,'-')]:
    a.plot(pc(FS),pc([N['experts'][f'0.5|{k}|{f}'][0] for f in FS]),color=c,ls=ls,lw=2,marker='o',ms=4,label={'complete':'all-pairs','ring':'ring','scalefree':'scale-free'}[k])
a.axhline(5,color=INK2,lw=.8,ls=':'); a.text(40,5.8,'DSM boundary 5%',ha='right',fontsize=8,color=INK2)
a.set_ylim(0,25); a.set_xlabel('Share of agents fixed at the DSM boundary (%), placed at random'); a.set_ylabel('Implied prevalence among lay agents (%)')
a.legend(frameon=False,fontsize=8,loc='lower left',bbox_to_anchor=(0,0.28)); ttl(a,'b  Randomly placed experts pull weakly,\n   and topology barely matters')
# c placement
a=ax[1,0]; labs=['periphery','random','hubs']; x=np.arange(3)
for i,(k,c,l) in enumerate([('scalefree',B,'scale-free'),('star',G,'hub-spoke')]):
    v=[N['placement'][f'{k}|{p}|0.1'][0] for p in labs]
    a.bar(x+(i-.5)*.36,pc(v),.32,color=c,label=l)
    for xi,vi in zip(x,v): a.text(xi+(i-.5)*.36,100*vi+.5,f'{100*vi:.0f}%',ha='center',fontsize=8,color=INK)
a.set_xticks(x); a.set_xticklabels(labs); a.set_xlabel('Where the 5% experts sit'); a.set_ylabel('Implied prevalence among lay agents (%)'); a.set_ylim(0,27)
a.legend(frameon=False,fontsize=8,loc='upper right'); ttl(a,'c  Where experts sit matters far more\n   than how many there are')
# d campaign
a=ax[1,1]; t=np.arange(61)
for lam,c,l in [(0.0,B,'no reversion'),(0.05,INK,'reversion 0.05'),(0.2,G,'reversion 0.2')]:
    d=pc(N['campaign'][f'complete|{lam}']['without'])-pc(N['campaign'][f'complete|{lam}']['with'])
    a.plot(t,d,color=c,lw=2,label=l)
a.set_xlabel('Rounds of peer interaction after everyone reads the DSM once'); a.set_ylabel('Campaign effect: points of prevalence\nbelow a no-campaign population')
a.legend(frameon=False,fontsize=8); a.set_ylim(-.3,5.6); ttl(a,'d  Whether a one-off DSM exposure lasts depends\n   on reversion, which the chains cannot identify')
fig.tight_layout(); fig.savefig('../figures/'+'network_predictions.png',dpi=170)
