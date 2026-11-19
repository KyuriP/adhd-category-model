"""Fig 3 (learning rules) and the related supplementary tables, from saved results."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]/'results'
LABELS={'copy':'Copying','perceptron':'Fixed step','margin':'Scaled by error'}
COLOURS={'copy':'cA','perceptron':'cB','margin':'cD'}
def load():
 a=json.loads((R/'repeated_sensitivity.json').read_text());b=json.loads((R/'reference_sensitivity.json').read_text())
 assert len(a['cases'])==len(a['planned_scenarios'])==97 and len({x['key'] for x in a['cases']})==97
 assert len(b['fits'])==6 and len(b['networks'])==3
 return a,b

def figures():
 a,_=load();out=[r'''\newcommand{\figlearningrules}{%
\begin{tikzpicture}[font=\sffamily\footnotesize]
\begin{groupplot}[pub,group style={group size=2 by 1,horizontal sep=2.0cm},width=7.5cm,height=6.0cm,xlabel={Positive examples (\%)},xmin=0,xmax=100,xtick={5,20,50,90},legend style={draw=none,fill=none,font=\scriptsize}]
''']
 for panel in ('drift','sources'):
  if panel=='drift':out.append(r'\nextgroupplot[title={\textbf{a}\quad Change across learning rules},ylabel={Change in category breadth (points)},ymin=-7,ymax=12,legend style={at={(0.03,0.97)},anchor=north west}]'+'\n')
  else:out.append(r'\nextgroupplot[title={\textbf{b}\quad Source influence depends on learning},ylabel={Broad minus strict influence},ymin=-0.07,ymax=0.2]'+'\n')
  out.append(r'\draw[gray,dashed] (axis cs:0,0)--(axis cs:100,0);'+'\n')
  for rule in LABELS:
   cases=sorted([c for c in a['cases'] if c['settings']['group']==panel and c['settings']['rule']==rule and not c['settings']['aligned'] and c['settings']['lam']==.1],key=lambda c:c['settings']['q'])
   out.append('\\addplot['+COLOURS[rule]+',thick,mark=*,mark size=1.7pt,error bars/.cd,y dir=both,y explicit] coordinates {')
   for c in cases:
    m,lo,hi=c['change_pp'] if panel=='drift' else c['sources']['hubs']['broad_minus_strict']
    out.append(f'({100*c["settings"]["q"]:g},{m:.5f}) +- (0,{(hi-lo)/2:.5f}) ')
   out.append('};\n')
  if panel=='drift':out.append(r'\legend{Copying,Fixed step,Scaled by error}'+'\n')
 out.append('\\end{groupplot}\n\\end{tikzpicture}}\n');return ''.join(out)

def table(caption,label,columns,head,rows):
 return ('\\begin{table}[htbp]\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n'+
 '\\begingroup\\small\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{'+columns+'}\n\\toprule\n'+head+r' \\'+'\n\\midrule\n'+
 '\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\\endgroup\n\\end{table}\n\n')

def supplement():
 a,b=load();text=r'''
\clearpage
\section*{Supplementary Note 1}
\subsection*{Approximation to the size of the change}
For aligned boundaries with small offset differences and negligible change within a reading, the expected change in mean offset over one round is
\[
n\eta s_b^2\varphi(b)\frac{\pi-q}{\pi(1-\pi)},
\]
where $n$ is the number of examples, $s_b$ the spread of offsets and $\varphi$ the standard normal density. The expression correlates at 0.97 with the simulated shifts across the 64 settings, but overstates their size about fourfold. Corrections during a reading reduce the gap between reader and source and hence the chance of later corrections. Finite heterogeneity and orientation differences also lie outside the approximation. A deterministic mean-update approximation averaged over pairs of aligned boundaries gives correlation 0.999 and regression slope 0.91 (Supplementary Figure~\ref{fig:pred}). It is not an exact expectation of the sequential stochastic process.

\subsection*{Learning from examples and labels alone}
Let the label $y\in\{-1,+1\}$ indicate whether the source classifies a case $x$ as having the condition. A learner uses the score $z=w^\top x-b$ and updates only if its classification disagrees with $y$. The fixed-step learner updates the unnormalised classifier by
\[
w' = w+\alpha yx,\qquad b'=b-\alpha y.
\]
The error-scaled learner moves part of the way towards a classifier that places the current example on its boundary,
\[
c=\frac{\eta z}{1+\lVert x\rVert^2},\qquad w'=w-cx,\qquad b'=b+c.
\]
It then has unnormalised score $(1-\eta)z$ for that example. Both learners divide $w'$ and $b'$ by $\lVert w'\rVert$ to express the classifier as a unit normal and an offset. This rescaling does not change classifications. Neither rule uses the source's offset or orientation. After each reading, the angle and offset return a fraction $\lambda$ towards their initial values, with the shortest angular difference used for the return.

For the aligned threshold checks, orientation stays fixed. The fixed-step learner lowers $b$ by $\alpha$ after a missed positive and raises it after a false positive. The example-directed threshold learner moves $b$ a fraction $\eta$ towards the example's projection on the common normal. The two-dimensional error-scaled update above extends this correction to both weights and offset. These are explicitly different implementations in the two orientation settings.

The main repeated-learning comparisons use $\alpha=0.05$ and $\eta=0.28$. Further checks halve or double these rates. They are not calibrated to equal learning speed, so differences in effect size should not be interpreted as measured differences between human learning strategies. The direction checks test narrowing at $q=0.05$ and broadening at $q=0.5$ and $0.9$. The additional $q=0.2$ setting lies close to the initial category breadth and is not assumed to be an exact balance point.

All planned cases are reported. Twenty independent populations are simulated per case. Averages over rounds 481 to 600 are finite-time summaries. The software includes every population's outcome summary, trajectories sampled every 20 rounds and the difference from rounds 361 to 480. In source comparisons, the no-source baseline excludes the same people who are fixed sources in the comparison condition. This prevents changes in the set of lay agents from entering the estimated source effect. The intervals below describe between-population variation at fixed parameters, rather than uncertainty about the model or empirical effects.

'''
 rows=[]
 for ref in (.025,.05,.1):
  for var,name in [('balanced','Copying'),('register_trend','Changing source')]:
   x=next(x for x in b['fits'] if x['reference']==ref and x['variant']==var)
   rows.append([f'{100*ref:g}',name,f'{x["loss"]:.2f}',f'{x["simulation_check"]["mean_loss"]:.2f}',f'{100*x["prior_at_mean"]:.2f}',f'{x["params"]["eta"]:.3f}'])
 text+=table(r'Refitting at different diagnostic references. Each fit uses the same adjusted Study~3 targets and a 60-iteration differential-evolution search. The 5\% rows reproduce the main fits. The simulation check averages losses over three independent seeds with 20,000 chains each. Prior breadth is prevalence at the mean prior offset. Losses are descriptive and do not penalise the changing-source model\textquotesingle s two extra parameters. No fit reached the convergence tolerance within the stated search budget.',
 'tab:reference','rlrrrr',r'Reference (\%) & Model & Fit loss & Check loss & Prior (\%) & $\eta$',rows)
 rows=[]
 for x in sorted(b['networks'],key=lambda x:x['reference']):
  v=x['hubs2_advantage_pp'];rows.append([f'{100*x["reference"]:g}']+[f'{100*np.mean(x["cases"][k]):.2f}' for k in ('none','hubs2','random10','hubs5')]+[f'{v[0]:.2f} ({v[1]:.2f}, {v[2]:.2f})'])
 text+=table(r'Expert placement after refitting at each diagnostic reference. Mean implied prevalence (percent) averages the last 120 of 600 rounds in 20 scale-free networks of 200 agents, with $q=0.5$ and $\lambda=0.1$. The final column is the additional reduction from experts on 2\% of nodes at hubs compared with 10\% at random, in percentage points with a paired normal-approximation 95\% interval.',
 'tab:reference_network','rrrrrr',r'Reference (\%) & No experts & 2\% hubs & 10\% random & 5\% hubs & Advantage',rows)
 rows=[]
 for aligned in (True,False):
  for rule in LABELS:
   for lam in (0.,.02,.1):
    cases={x['settings']['q']:x for x in a['cases'] if x['settings']['group']=='drift' and x['settings']['aligned']==aligned and x['settings']['rule']==rule and x['settings']['lam']==lam}
    rows.append(['Aligned' if aligned else 'Two traits',LABELS[rule],f'{lam:g}']+[f'{cases[q]["change_pp"][0]:.2f}' for q in (.05,.2,.5,.9)])
 text+=table(r'Repeated exchange across learning rules, orientation settings and return rates. Entries are changes in mean implied prevalence from the initial population, in percentage points. Positive values indicate broadening. Each entry averages 20 fully connected populations of 200 agents over rounds 481 to 600. The initial mean implied prevalence is approximately 21\%. These are finite-time outcomes, not equilibrium estimates.',
 'tab:repeated','llrrrrr',r'Orientation & Rule & $\lambda$ & $q=0.05$ & $q=0.2$ & $q=0.5$ & $q=0.9$',rows)
 fmt=lambda v:f'{v[0]:.3f} ({v[1]:.3f}, {v[2]:.3f})'
 rows=[]
 for rule in LABELS:
  for q in (.2,.5,.9):
   x=next(x for x in a['cases'] if x['settings']['group']=='sources' and x['settings']['rule']==rule and x['settings']['q']==q)
   rows.append([LABELS[rule],f'{q:g}',fmt(x['sources']['hubs']['broad_minus_strict']),fmt(x['sources']['random']['broad_minus_strict'])])
 text+=table(r'Broad minus strict source influence with different learning rules. Entries are differences in the magnitude of the mean-offset shift, with paired normal-approximation 95\% intervals across 20 scale-free networks. Positive values favour broad sources and negative values favour strict sources. Sources occupy 5\% of nodes and lie equally far from the mean prior on the offset scale. All cases have differently oriented boundaries and $\lambda=0.1$, and average rounds 481 to 600.',
 'tab:source_rules','lrll',r'Rule & $q$ & Sources at hubs & Sources at random',rows)
 rows=[]
 for rule,param,values in [('perceptron','step',(.025,.05,.1)),('margin','eta',(.14,.28,.56))]:
  for val in values:
   baseline=val in (.05,.28);group='drift' if baseline else 'step'
   cases={x['settings']['q']:x for x in a['cases'] if x['settings']['group']==group and x['settings']['rule']==rule and not x['settings']['aligned'] and x['settings']['lam']==.1 and (baseline or x['settings'].get(param)==val)}
   sg='sources' if baseline else 'sources_step'
   src=next(x for x in a['cases'] if x['settings']['group']==sg and x['settings']['rule']==rule and x['settings']['q']==.5 and (baseline or x['settings'].get(param)==val))
   rows.append([LABELS[rule],f'{val:g}']+[f'{cases[q]["change_pp"][0]:.2f}' for q in (.05,.5,.9)]+[fmt(src['sources']['hubs']['broad_minus_strict'])])
 text+=table(r'Sensitivity to the strength of label-only learning. The first three outcome columns give change in implied prevalence (percentage points) in fully connected populations. The last gives broad minus strict source influence at hubs at $q=0.5$ on scale-free networks, in offset units with a paired 95\% interval. All settings use two traits, $\lambda=0.1$, 20 populations and rounds 481 to 600. Strength is $\alpha$ for fixed-step learning and $\eta$ for error-scaled learning.',
 'tab:steps','lrrrrl',r'Rule & Strength & $q=0.05$ & $q=0.5$ & $q=0.9$ & Broad minus strict at hubs',rows)
 return text


NETWORK_SUPPLEMENT_FIGURE = r"""\begin{figure}[htbp]
\centering
\resizebox{0.92\textwidth}{!}{\figrobust}
\caption{Expert placement and source influence under partial copying, with experts on 5\% of nodes. (a) Reduction in implied prevalence among lay agents produced by experts at the hubs, divided by the reduction produced by the same number at random positions, for different shares of positive examples $q$ and rates of return towards initial views $\lambda$, which the chain data cannot identify (scale-free network, 200 agents). (b) Reduction with experts at random positions (grey) and at the hubs (blue) for scale-free networks of different sizes, a scale-free network with clustering, and a modular network in which agents have similar numbers of connections. (c) Shift in mean lay offset relative to a no-source condition, measured after round 300, for strict sources (left) and broad sources (right) on 5\% of nodes. Source offsets are equally distant from the mean prior offset. Lines are 95\% intervals for the mean across 40 networks, with $\lambda=0.1$. Numbers give the broadening-to-narrowing ratio.}
\label{fig:robust}
\end{figure}
"""
