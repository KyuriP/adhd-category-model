"""Writes supp.tex (supplementary tables) from ../results. Figures come from make_figs.py."""
import json, numpy as np
R='../results/'; rb=json.load(open(R+'robust.json')); ds=json.load(open(R+'drift_size.json')); ps=json.load(open(R+'param_se.json'))
co=lambda xs,ys: ' '.join(f'({x:g},{y:.3g})' for x,y in zip(xs,ys))
STY='small,title style={font=\\footnotesize,align=left},label style={font=\\scriptsize},tick label style={font=\\scriptsize},legend style={font=\\tiny,draw=none,fill=none},axis lines=left'
S=rb['sens']; none=np.array(S['none']); rnd=np.array(S['random']); hub=np.array(S['hubs']); qi=S['q'].index(0.5); li=S['lam'].index(0.1)
o=['\\newcommand{\\figrobust}{%\n\\begin{tikzpicture}\n\\begin{groupplot}[group style={group size=4 by 1,horizontal sep=1.35cm},width=4.6cm,height=4.6cm,'+STY+',ymin=0]\n']
o.append('\\nextgroupplot[title={(a) Reversion rate},xlabel={Reversion rate $\\lambda$},ylabel={Reduction in implied prevalence (points)},xmode=log,log ticks with fixed point,xtick={0.02,0.05,0.1,0.2,0.4},xticklabels={0.02,0.05,0.1,0.2,0.4},ymax=11,legend style={at={(0.97,0.97)},anchor=north east}]\n')
o.append(f'\\addplot[cA,thick,mark=*,mark size=1.3pt] coordinates {{{co(S["lam"],100*(none-hub)[:,qi])}}};\n\\addplot[black,thick,dashed,mark=*,mark size=1.3pt] coordinates {{{co(S["lam"],100*(none-rnd)[:,qi])}}};\n\\legend{{At hubs,At random}}\n')
o.append('\\nextgroupplot[title={(b) Share of positive cases},xlabel={Positive share $q$ (\\%)},ylabel={Reduction (points)},ymax=11,xmin=10,xmax=100]\n')
o.append(f'\\addplot[cA,thick,mark=*,mark size=1.3pt] coordinates {{{co([100*q for q in S["q"]],100*(none-hub)[li])}}};\n\\addplot[black,thick,dashed,mark=*,mark size=1.3pt] coordinates {{{co([100*q for q in S["q"]],100*(none-rnd)[li])}}};\n')
Z=rb['size']
o.append('\\nextgroupplot[title={(c) Network size},xlabel={Number of agents},ylabel={Implied prevalence, lay agents (\\%)},xmode=log,log ticks with fixed point,xtick={100,200,500,1000},ymin=14,ymax=24,legend style={at={(0.97,0.5)},anchor=east}]\n')
for k,st in (('hubs','cA,thick,mark=*,mark size=1.3pt'),('random','black,thick,dashed,mark=*,mark size=1.3pt'),('none','gray,thick,dotted,mark=*,mark size=1.3pt')):
    o.append(f'\\addplot[{st}] coordinates {{{co(Z["N"],[100*v[0] for v in Z[k]])}}};\n')
o.append('\\legend{At hubs,At random,No experts}\n')
Tp=rb['topo']; names=[('scalefree','scale-free'),('sfclustered','clustered'),('modular','modular')]
o.append('\\nextgroupplot[title={(d) Network type},ybar,bar width=6pt,symbolic x coords={scale-free,clustered,modular},xtick=data,enlarge x limits=0.25,ylabel={Reduction (points)},ymax=7,legend style={at={(0.97,0.97)},anchor=north east}]\n')
o.append('\\addplot[fill=cA,draw=none] coordinates {'+' '.join(f'({n},{100*(Tp[k]["none"][0]-Tp[k]["hubs"][0]):.2g})' for k,n in names)+'};\n')
o.append('\\addplot[fill=cC,draw=none] coordinates {'+' '.join(f'({n},{100*(Tp[k]["none"][0]-Tp[k]["random"][0]):.2g})' for k,n in names)+'};\n\\legend{Most connected,At random}\n\\end{groupplot}\n\\end{tikzpicture}}\n\n')
rows=ds['rows']
o.append('\\newcommand{\\figpred}{%\n\\begin{tikzpicture}\n\\begin{axis}[width=6.5cm,height=6.5cm,'+STY+',axis lines=box,xlabel={Predicted shift of mean offset},ylabel={Simulated shift of mean offset},xmin=-0.45,xmax=0.45,ymin=-0.12,ymax=0.12,legend style={at={(0.03,0.97)},anchor=north west},scaled ticks=false,tick label style={font=\\scriptsize,/pgf/number format/fixed}]\n')
o.append('\\addplot[only marks,mark=o,mark size=1.4pt,cC] coordinates {'+' '.join(f'({r["closed"]:.3g},{r["sim"]:.3g})' for r in rows)+'};\n')
o.append('\\addplot[only marks,mark=*,mark size=1.4pt,cA] coordinates {'+' '.join(f'({r["pair"]:.3g},{r["sim"]:.3g})' for r in rows)+'};\n')
o.append('\\addplot[gray,dashed,domain=-0.12:0.12,samples=2] {x};\n\\legend{Closed form,Pair-integrated,Equality}\n\\end{axis}\n\\end{tikzpicture}}\n')
# figs2.tex is written by make_figs.py

L=lambda ds,v: json.load(open(f'{R}fit_s{ds}_{v}_lag.json'))
t=['\\clearpage\n\\appendix\n\\renewcommand{\\thetable}{S\\arabic{table}}\\renewcommand{\\thefigure}{S\\arabic{figure}}\\setcounter{table}{0}\\setcounter{figure}{0}\n\\section*{Supplementary Information}\n\n']
t.append('\\begin{table}[htbp]\n\\centering\n\\caption{Parameter estimates for Study~3 (adjusted means, 31 moments) with approximate standard errors in parentheses. Standard errors use a local sandwich covariance, combining the simulated-moment Jacobian with the joint covariance from 2,000 whole-chain bootstrap resamples. They omit simulation and model-selection uncertainty.}\n\\label{tab:params}\n\\begin{tabular}{lccc}\n\\toprule\nParameter & Half positive & Positive only & Base rate \\\\\n\\midrule\n')
lab={'mu_b':'Lay prior offset $\\mu_b$','s_b':'Spread of prior $s_b$','eta':'Learning rate $\\eta$','sig_p':'Noise in prevalence reports $\\sigma_p$','v':'Position of vignette $v$','sig_r':'Noise in ratings $\\sigma_r$','c_s':'Self-rating offset $c_s$'}
for k in lab:
    t.append(lab[k]+' & '+' & '.join(f'${ps[v]["est"][k]:.2f}$ ({ps[v]["se"][k]:.2f})' for v in ('balanced','positive','base'))+' \\\\\n')
t.append('Prevalence at mean prior offset (\\%) & '+' & '.join(f'{100*ps[v]["prior_prev"]:.1f} ({100*ps[v]["prior_prev_se"]:.1f})' for v in ('balanced','positive','base'))+' \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n')
t.append('\\begin{table}[htbp]\n\\centering\n\\caption{Loss on 31 moments for each dataset and model version, or 23 for generations 1 to 5 (lower is better). The learning rate $\\eta$ of the half-positive version is given for reference. Losses use a diagonal bootstrap weight matrix and are descriptive, not likelihood-ratio statistics. Dashes indicate fits not reported.}\n\\label{tab:losses}\n\\begin{tabular}{lccccc}\n\\toprule\nDataset & Half positive & Positive only & Base rate & Constant source & $\\eta$ \\\\\n\\midrule\n')
for dsn,name in (('3adj','Study 3, adjusted means'),('3','Study 3, unadjusted means'),('3clean','Study 3, adjusted, AI-flagged removed'),('2','Study 2, unadjusted means'),('3adj_gens5','Study 3, generations 1 to 5')):
    cells=[]
    for v in ('balanced','positive','base','register'):
        try: cells.append(f'{L(dsn,v)["loss"]:.1f}')
        except FileNotFoundError: cells.append('--')
    t.append(f'{name} & '+' & '.join(cells)+f' & {L(dsn,"balanced")["params"]["eta"]:.2f} \\\\\n')
t.append('\\bottomrule\n\\end{tabular}\n\\end{table}\n\n')
t.append('\\begin{table}[htbp]\n\\centering\n\\caption{Reduction in implied prevalence among lay agents (percentage points) produced by experts on 5\\% of nodes of a scale-free network, at the most connected nodes / at random nodes, by reversion rate $\\lambda$ and share of positive cases $q$.}\n\\label{tab:sens}\n\\begin{tabular}{l'+'c'*len(S['q'])+'}\n\\toprule\n$\\lambda$ & '+' & '.join(f'$q={100*q:.0f}\\%$' for q in S['q'])+' \\\\\n\\midrule\n')
for i,lam in enumerate(S['lam']):
    t.append(f'{lam:g} & '+' & '.join(f'{100*(none-hub)[i,j]:.1f} / {100*(none-rnd)[i,j]:.1f}' for j in range(len(S['q'])))+' \\\\\n')
t.append('\\bottomrule\n\\end{tabular}\n\\end{table}\n\n')
t.append('\\begin{figure}[htbp]\n\\centering\n\\figpred\n\\caption{Predicting the size of the drift. Each point is one combination of category prevalence and positive share from Figure~\\ref{fig:drift}b. The closed-form expression for small differences between agents matches the sign away from $\\pi=q$ and closely tracks the ordering but overstates the shift. A deterministic mean-update approximation averaged over aligned boundary pairs tracks the simulation closely. It is not the exact expectation of the nonlinear sequential process.}\n\\label{fig:pred}\n\\end{figure}\n')
t.append('\\begin{figure}[htbp]\n\\centering\n\\figprofile\n\\caption{Loss as a function of the learning rate, with the other parameters re-optimised at each value (Study~3, unadjusted means, 28 moments without adjacent-participant correlations). Zero learning and the largest rate examined give higher loss. This descriptive profile is not a calibrated confidence interval.}\n\\label{fig:profile}\n\\end{figure}\n')
text = ''.join(t)
text = text.replace(r'\setcounter{figure}{0}', r'\setcounter{figure}{0}'+'\n'+r'\renewcommand{\theHtable}{S\arabic{table}}\renewcommand{\theHfigure}{S\arabic{figure}}')
import re
text = re.sub(r'(\\begin\{tabular\}\{lccccc\}.*?\\end\{tabular\})',lambda m:r'\begingroup\small\setlength{\tabcolsep}{3.5pt}'+'\n'+m.group(1)+r'\endgroup',text,flags=re.S)
cal=json.load(open(R+'calibration_sensitivity.json'))
rows=[('Baseline parameters','none','hubs2','random10','hubs5')]
baseline=json.load(open(R+'drift.json'))['dose']
text += r"""
\begin{table}[htbp]
\centering
\caption{Mean implied prevalence for different expert placements using baseline parameters and parameters from the main chain fit. Entries are mean implied prevalence among lay agents (percent) over the last 60 of 300 rounds, averaged across 20 scale-free networks of 200 agents, with $q=0.5$ and $\lambda=0.1$.}
\label{tab:calibration}
\begin{tabular}{lrrrr}
\toprule
Parameters & No experts & 2\% at hubs & 10\% random & 5\% at hubs \\
\midrule
"""
values=[baseline['none'],baseline['hubs'][1],baseline['random'][3],baseline['hubs'][2]]
text += 'Baseline parameters & '+' & '.join(f'{100*v:.2f}' for v in values)+r' \\'+'\n'
values=[cal['cases'][k]['mean'] for k in ['none','hubs2','random10','hubs5']]
text += 'Main chain fit & '+' & '.join(f'{100*v:.2f}' for v in values)+r' \\'+'\n'
text += r'\bottomrule\end{tabular}\end{table}'+'\n'
# same-day checks
C=json.load(open(R+'conditions.json'))
text += r'''
\begin{table}[htbp]
\centering
\caption{Comparisons within the same collection days. Study~2 entries are the difference in generation slopes between examples and words, with chain-clustered 95\% intervals. Study~4 entries compare readers of a generation~7 text with readers of the diagnostic criteria, with normal-approximation 95\% intervals. Model predictions compare an eighth reader in a simulated chain with a reader of the criteria and omit parameter uncertainty. Other- and self-diagnosis use percentage points, and reported prevalence uses log odds.}
\label{tab:conditions}
\begin{tabular}{lrrr}
\toprule
Comparison & Other-diagnosis & Self-diagnosis & Prevalence \\
\midrule
'''
def fmt_ci(v,scale=1,digits=2):
    return f'{scale*v[0]:.{digits}f} ({scale*v[1]:.{digits}f}, {scale*v[2]:.{digits}f})'
text += 'Study 2, slope difference & '+' & '.join(fmt_ci(C['study2'][y]['diff'],100 if y in 'VS' else 1,1 if y in 'VS' else 3) for y in 'VSP')+r' \\'+'\n'
text += 'Study 4, observed difference & '+' & '.join(fmt_ci(C['study4']['observed'][y],100 if y in 'VS' else 1,1 if y in 'VS' else 2) for y in 'VSP')+r' \\'+'\n'
for key,name in [('2','Study 2 fit, predicted difference'),('3adj','Study 3 fit, predicted difference')]:
    text += name+' & '+' & '.join(f'{C["study4"]["predicted"][key][y]*(100 if y in "VS" else 1):.2f}' for y in 'VSP')+r' \\'+'\n'
text+=r'\bottomrule\end{tabular}\end{table}'+'\n'
# recovery
from pathlib import Path
rec=json.load(open(R+'recovery.json')) if Path(R+'recovery.json').exists() else []
text += r'''
\begin{table}[htbp]
\centering
\caption{Conditional model recovery from ten synthetic experiments of 220 chains per generating model. A recovery counts when the generating model has lower loss than the other fitted model. The simulation check re-evaluates both fitted models with three new seeds and 20,000 chains per seed. The range gives constant-source loss minus partial-copying loss in that check, so positive values favour partial copying. Generating parameters and observed-data weights are held fixed. Counts also depend on the finite optimiser budget. The changing-source alternative is not tested here.}
\label{tab:recovery}
\begin{tabular}{lrrrr}
\toprule
Generating model & $N$ & Fit recoveries & Check recoveries & Check loss gap \\
\midrule
'''
for g,name in [('partial','Partial copying'),('lineage_free','Constant source')]:
    pairs=[]
    for rep in sorted({a['rep'] for a in rec if a['generated_by']==g}):
        pair={a['fit_with']:a for a in rec if a['generated_by']==g and a['rep']==rep}
        if len(pair)==2:pairs.append(pair)
    n=len(pairs)
    fwin=sum((a['partial']['loss']<a['lineage_free']['loss'])==(g=='partial') for a in pairs)
    diffs=[a['lineage_free']['simulation_check']['mean_loss']-a['partial']['simulation_check']['mean_loss'] for a in pairs]
    cwin=sum((d>0)==(g=='partial') for d in diffs)
    rg=f'{min(diffs):.1f} to {max(diffs):.1f}' if diffs else '--'
    text += f'{name} & {n} & {fwin} & {cwin} & {rg}'+r' \\'+'\n'
text+=r'\bottomrule\end{tabular}'+'\n'
matched=[a for a in rec if a['generated_by']=='partial' and a['fit_with']=='partial']
if matched:
    text+=r'\par\smallskip\begin{minipage}{0.95\textwidth}\small'+'\n'
    ranges=[]
    for k,name,fun,scale in [('eta',r'$\eta$',lambda x:x,1),('mu_b','prevalence at the mean prior offset',lambda x:1-__import__('scipy').stats.norm.cdf(x),100),('s_b',r'$s_b$',lambda x:x,1)]:
        vals=[scale*fun(a['est'][k]) for a in matched];true=scale*fun(matched[0]['true'][k])
        unit=r'\%' if k=='mu_b' else ''
        ranges.append(f'{name} from {min(vals):.2f} to {max(vals):.2f}{unit} (generating value {true:.2f}{unit})')
    text+='When partial copying generated the data, refitting it gave '+', '.join(ranges[:-1])+' and '+ranges[-1]+'. These ranges describe ten synthetic datasets and are not confidence intervals.\n'+r'\end{minipage}'+'\n'
text+=r'\end{table}'+'\n'
# no-lineage parameter estimates
text+=r'''
\begin{table}[htbp]
\centering
\caption{No-lineage parameter estimates on adjusted Study~3 moments. The changing-source model uses $b_L+\delta(g-2)$ for generations $g\geq2$. Generation~1 receives the diagnostic source in both models. These alternatives require different reader priors from partial copying. Parameters are conditional fits, not direct measurements of pre-reading beliefs.}
\label{tab:alternative}
\begin{tabular}{lrr}
\toprule
Parameter & Constant source & Changing source \\
\midrule
'''
constant=L('3adj','register')['params'];trend=L('3adj','register_trend')['params']
for k,name in list(lab.items())+[('b_lay',r'Initial lay source offset $b_L$'),('trend',r'Source change per generation $\delta$')]:
    text+=name+' & '+(f'{constant[k]:.3f}' if k in constant else '--')+f' & {trend[k]:.3f}'+r' \\'+'\n'
text+=r'\bottomrule\end{tabular}\end{table}'+'\n'
# keep the wide table inside the margins
text=re.sub(r'(\\begin\{tabular\}\{lrrr\}.*?\\end\{tabular\})',lambda m:r'\begingroup\small\setlength{\tabcolsep}{3.5pt}'+'\n'+m.group(1)+r'\endgroup',text,flags=re.S)

open('supp.tex','w').write(text)
print(len(''.join(o)), len(''.join(t)))
