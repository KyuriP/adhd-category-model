"""Writes figures.tex (all figures, pgfplots/TikZ) from ../results/*.json.
Colours: blue = model / hubs, orange = alternative / strict, grey = baseline.
"""
import json
import numpy as np
from scipy.stats import norm

# Fig 1 is hand-drawn
CONCEPT = r'''\newcommand{\figconcept}{%
\begin{tikzpicture}[font=\sffamily\scriptsize,>=stealth]
\node[anchor=west,font=\sffamily\footnotesize] at (0,4.75) {\textbf{Partial copying.} Each reader combines the explanation with their own initial view.};
\node[draw,fill=gray!15,minimum width=1.3cm,minimum height=0.7cm] (p0) at (1.1,3.7) {criteria};
\foreach \i/\x/\w/\l in {0/1.1/0.10/5,1/3.6/0.31/15,2/6.1/0.39/19,3/8.6/0.42/21,4/11.1/0.43/21,5/13.6/0.43/21} {
  \fill[cA!60] (\x-0.8,2.75) rectangle (\x-0.8+1.6*\w,3.0);
  \draw[gray] (\x-0.8,2.75) rectangle (\x+0.8,3.0);
  \node at (\x,2.5) {\l\%};
}
\foreach \i/\x in {1/3.6,2/6.1,3/8.6,4/11.1,5/13.6} {\node[circle,draw,minimum size=0.7cm] (p\i) at (\x,3.7) {\i};}
\foreach \a/\b in {0/1,1/2,2/3,3/4,4/5} {\draw[->,thick] (p\a) -- (p\b);}
\node[anchor=west,text=gray] at (0.3,2.1) {bars show the share of people each reader counts as having the condition (full bar = 50\%)};
\node[anchor=west,font=\sffamily\footnotesize] at (0,1.25) {\textbf{Repeated exchange.} The cases people discuss can encourage broader definitions.};
\draw[->] (0.5,0) -- (14.3,0);
\foreach \p in {0,10,20,30,40,50,60} {\draw (0.5+0.22*\p,0) -- (0.5+0.22*\p,-0.08) node[below] {\p\%};}
\draw[thick] (1.6,0) -- (1.6,0.3) node[above] {diagnostic reference};
\draw[thick] (4.9,0) -- (4.9,0.3) node[above] {readers' initial views};
\draw[thick,dashed] (11.5,0) -- (11.5,0.3) node[above,align=center] {share of discussed cases\\said to have the condition, $q$};
\draw[->,very thick,cB] (1.6,-0.85) -- (4.9,-0.85) node[midway,below,text=black] {chain model fit};
\draw[->,very thick,cA] (4.9,-0.85) -- (8.2,-0.85);
\draw[cA,thick,dotted] (8.2,-0.85) -- (11.5,-0.85);
\node[below] at (8.2,-0.85) {predicted broadening};
\draw[<-,thick,gray] (5.5,-1.75) -- (7.3,-1.75) node[right,text=black] {more connections can increase an expert's influence};
\end{tikzpicture}}
'''

R = '../results/'
T = json.load(open(R + 'targets.json')); D = json.load(open(R + 'drift.json')); P = json.load(open(R + 'profile_s3.json'))
RB = json.load(open(R + 'robust.json')); DS = json.load(open(R + 'drift_size.json'))
F = lambda v: json.load(open(f'{R}fit_s3adj_{v}_lag.json'))
co = lambda xs, ys: ' '.join(f'({x:g},{y:.3g})' for x, y in zip(xs, ys))
g = list(range(1, 8)); A = T['3adj']; ch = F('balanced'); nl = F('register'); trend = F('register_trend')
SF = '\\sffamily\\scriptsize'
head = lambda letter, text: f'title={{\\textbf{{{letter}}}\\hspace{{0.6em}}{text}}}'
lab = lambda x, y, text, color='black', anchor='west', extra='': f'\\node[font={SF},text={color},anchor={anchor},inner sep=1.5pt{extra}] at (axis cs:{x},{y}) {{{text}}};\n'

STYLE = ('\\pgfplotsset{pub/.style={axis lines=left,axis line style={-,line width=0.4pt,black!75},tick style={line width=0.4pt,black!75},'
         'tick align=outside,major tick length=2pt,tick label style={font=\\sffamily\\scriptsize,/pgf/number format/assume math mode=true},every axis label/.append style={font=\\sffamily\\footnotesize},log ticks with fixed point,'
         'title style={font=\\sffamily\\footnotesize,anchor=south west,at={(0,1.03)},align=left},'
         'legend style={draw=none,fill=none,font=\\sffamily\\scriptsize,cells={anchor=west}},'
         'every axis plot/.append style={line width=0.9pt},clip=false,xlabel near ticks,ylabel near ticks}}\n\n')


def data_series(k, scale):
    m = np.array(A[k + '_mean']) * scale; e = 1.96 * np.array(A[k + '_se']) * scale
    s = ('\\addplot[black,only marks,mark=*,mark size=1.3pt,line width=0.4pt,error bars/.cd,y dir=both,y explicit,error bar style={line width=0.4pt,black!60},error mark=none] coordinates {'
         + ' '.join(f'({x},{y:.3g}) +- (0,{d:.3g})' for x, y, d in zip(g, m, e)) + '};\n')
    return s, (m - e).min(), (m + e).max()


def traj(k, scale, letter, title, ylabel, labels=False, extra=''):
    ob, lo, hi = data_series(k, scale); pad = 0.08 * (hi - lo)
    a = f'\\nextgroupplot[{head(letter, title)},ylabel={{{ylabel}}},xlabel={{Generation}},xtick={{1,...,7}},xmin=0.6,xmax=7.4,ymin={lo - pad:.3g},ymax={hi + pad:.3g}{extra}]\n'
    mc = np.array(ch['pred'][k + '_mean']) * scale; mn = np.array(nl['pred'][k + '_mean']) * scale
    a += f'\\addplot[cB,dashed] coordinates {{{co(g, mn)}}};\n\\addplot[cA] coordinates {{{co(g, mc)}}};\n' + ob
    if labels:
        a += lab(7.55, mc[-1] + 0.02 * (hi - lo), 'partial copying', 'cA') + lab(7.55, mn[-1] - 0.03 * (hi - lo), 'no lineage', 'cB')
    return a


o = [STYLE]

# drift figure
pp = (norm.cdf(1.5) - norm.cdf(1)) / (1 - norm.cdf(1)); pn = (norm.cdf(1) - norm.cdf(0.5)) / norm.cdf(1)
o.append('\\newcommand{\\figasym}{%\n\\begin{tikzpicture}[font=\\sffamily\\footnotesize]\n\\begin{groupplot}[pub,group style={group size=3 by 1,horizontal sep=2.1cm},width=5.9cm,height=5.4cm]\n')
o.append(f'\\nextgroupplot[{head("a", "Who gets corrected")},xlabel={{Trait, along the direction the boundary cuts}},axis y line=none,xmin=-3,xmax=3,ymin=0,ymax=0.52,xtick={{-2,0,2}},xmin=-3.4,xmax=3.6]\n')
o.append('\\addplot[draw=none,fill=cA!40,domain=0.5:1,samples=12,line width=0pt] {exp(-x^2/2)/2.5066} \\closedcycle;\n\\addplot[draw=none,fill=cB!65,domain=1:1.5,samples=12,line width=0pt] {exp(-x^2/2)/2.5066} \\closedcycle;\n\\addplot[black!75,domain=-3:3,samples=60,line width=0.6pt] {exp(-x^2/2)/2.5066};\n')
o.append('\\draw[black,line width=0.9pt] (axis cs:1,0) -- (axis cs:1,0.45);\n\\draw[black!60,dashed,line width=0.5pt] (axis cs:0.5,0) -- (axis cs:0.5,0.41);\n\\draw[black!60,dashed,line width=0.5pt] (axis cs:1.5,0) -- (axis cs:1.5,0.41);\n')
o.append(lab(1, 0.465, 'source', anchor='south') + lab(0.5, 0.415, 'broader reader', anchor='south east') + lab(1.5, 0.415, 'stricter reader', anchor='south west'))
o.append(lab(1.62, 0.2, f'{100 * pp:.0f}\\% of positive\\\\cases land here', 'cB!85!black', extra=',align=left') + lab(-0.35, 0.1, f'{100 * pn:.0f}\\% of counter-\\\\examples land here', 'cA!85!black', anchor='east', extra=',align=right'))
o.append('\\draw[cA!85!black,line width=0.4pt] (axis cs:-0.35,0.1) -- (axis cs:0.72,0.14);\n')
G = D['grid']; db = -np.array(G['db']); m = float(np.abs(db).max()); tl = ','.join(f'{100 * x:g}' for x in G['pi'])
o.append(f'\\nextgroupplot[{head("b", "Direction of drift")},axis lines=box,xlabel={{Category prevalence $\\pi$ (\\%)}},ylabel={{Positive share $q$ (\\%)}},clip marker paths=true,xtick={{1,...,8}},ytick={{1,...,8}},xticklabels={{{tl}}},yticklabels={{{tl}}},xmin=0.4,xmax=8.6,ymin=0.4,ymax=8.6,tick align=outside,colormap={{dv}}{{color=(cB) color=(white) color=(cA)}},point meta min={-m:.3g},point meta max={m:.3g}]\n')
o.append('\\addplot[scatter,only marks,mark=square*,mark size=5.2pt,point meta=explicit,line width=0pt] coordinates {' + ' '.join(f'({j + 1},{i + 1}) [{db[i][j]:.3g}]' for i in range(8) for j in range(8)) + '};\n')
o.append('\\draw[black,dashed,line width=0.6pt] (axis cs:0.4,0.4) -- (axis cs:8.6,8.6);\n')
o.append(f'\\node[font={SF},fill=white,fill opacity=0.85,text opacity=1,inner sep=2pt] at (axis cs:2.3,7.3) {{broadens}};\n\\node[font={SF},fill=white,fill opacity=0.85,text opacity=1,inner sep=2pt] at (axis cs:6.7,1.7) {{narrows}};\n')
E = D['equil']; qx = [100 * q for q in E['q']]
o.append(f'\\nextgroupplot[{head("c", "Repeated exchange, no experts")},xlabel={{Positive share of examples $q$ (\\%)}},ylabel={{Implied prevalence (\\%)}},xmin=0,xmax=95,ymin=14,ymax=31]\n')
o.append(f'\\addplot[black!45,densely dotted] coordinates {{(0,{100 * E["prior"]:.3g}) (95,{100 * E["prior"]:.3g})}};\n\\addplot[cA!55,mark=*,mark size=1.3pt] coordinates {{{co(qx, [100 * x for x in E["0.1"]])}}};\n\\addplot[cA,mark=*,mark size=1.3pt] coordinates {{{co(qx, [100 * x for x in E["0.02"]])}}};\n')
o.append(lab(92, 100 * E['0.02'][-1], '$\\lambda=0.02$', 'cA') + lab(92, 100 * E['0.1'][-1], '$\\lambda=0.1$', 'cA!70') + lab(2, 100 * E['prior'] + 0.2, 'initial views', 'black!60', anchor='south west'))
o.append('\\end{groupplot}\n\\end{tikzpicture}}\n\n')

# chain figure: distributions by generation + data against simulated experiments
FD = json.load(open(R + 'figure_data.json'))
RG = FD['ridge']; rx = [100 * x for x in RG['x']]; top = max(max(d) for d in RG['dens'] + [RG['prior']]); sc = 1.45 / top
o.append('\\newcommand{\\figmech}{%\n\\begin{tikzpicture}[font=\\sffamily\\footnotesize]\n\\begin{groupplot}[pub,group style={group size=2 by 2,horizontal sep=2.4cm,vertical sep=2.1cm},width=7.2cm,height=5.8cm]\n')
o.append(f'\\nextgroupplot[{head("a", "Each reader\'s category, by generation")},xlabel={{Share of people the reader counts as having ADHD (\\%)}},xmin=0,xmax=45,ymin=-0.3,ymax=9.9,ytick={{0,1,...,7}},yticklabels={{prior (inferred),1,2,3,4,5,6,7}},axis y line*=left,y axis line style={{draw=none}},ytick style={{draw=none}},ylabel={{Generation}},clip=false]\n')
def ridge(y0, dens, fill, draw='black!70'):
    up = ' '.join(f'({x:.1f},{y0 + sc * d:.3f})' for x, d in zip(rx, dens) if x <= 45)
    return f'\\addplot[fill={fill},draw={draw},line width=0.45pt] coordinates {{(0,{y0}) {up} (45,{y0})}} -- cycle;\n'
for gg in range(6, -1, -1):   # generation 7 at top, drawn first; lower ridges overlap from below
    y0 = gg + 1
    o.append(ridge(y0, RG['dens'][gg], f'cA!{18 + 11 * gg}!white'))
o.append(ridge(0, RG['prior'], 'black!12', 'black!55'))
lp = 100 * float(np.median(1 - norm.cdf(ch['params']['mu_b'] + ch['params']['s_b'] * np.random.default_rng(1).standard_normal(4000))))
o.append(f'\\draw[black,line width=0.7pt] (axis cs:5,-0.3) -- (axis cs:5,9.1);\n\\draw[black!60,dashed,line width=0.6pt] (axis cs:{lp:.1f},-0.3) -- (axis cs:{lp:.1f},9.1);\n')
o.append(lab(5, 9.1, 'criteria', anchor='south') + lab(lp, 9.1, 'initial views', 'black!65', anchor='south'))
for k, (letter, ttl, yl, sc_) in {'V': ('b', 'Vincent judged to have ADHD', 'Participants (\\%)', 100), 'S': ('c', 'Participants who think they have ADHD', 'Participants (\\%)', 100), 'P': ('d', 'Estimated prevalence of ADHD', 'Estimate (\\%, logit scale)', 1)}.items():
    B = FD['band'][k]; ob, lo, hi = data_series(k, sc_)
    lo = min(lo, min(B['5']) * sc_, min(trend['pred'][k+'_mean'])*sc_); hi = max(hi, max(B['95']) * sc_, max(trend['pred'][k+'_mean'])*sc_); pad = 0.06 * (hi - lo)
    extra = ''
    if k == 'P':
        lg = lambda q: np.log(q / (1 - q)); extra = f',ytick={{{lg(0.15):.3f},{lg(0.2):.3f},{lg(0.25):.3f},{lg(0.3):.3f}}},yticklabels={{15,20,25,30}}'
    o.append(f'\\nextgroupplot[{head(letter, ttl)},ylabel={{{yl}}},xlabel={{Generation}},xtick={{1,...,7}},xmin=0.6,xmax=7.4,ymin={lo - pad:.3g},ymax={hi + pad:.3g}{extra}]\n')
    band = lambda u, l, c: f'\\addplot[fill={c},draw=none,line width=0pt] coordinates {{' + ' '.join(f'({x},{y * sc_:.4g})' for x, y in zip(g, B[u])) + ' ' + ' '.join(f'({x},{y * sc_:.4g})' for x, y in zip(g[::-1], B[l][::-1])) + '} -- cycle;\n'
    o.append(band('95', '5', 'cA!13') + band('75', '25', 'cA!32'))
    o.append(f'\\addplot[cB,dashed,line width=0.7pt] coordinates {{{co(g, np.array(nl["pred"][k + "_mean"]) * sc_)}}};\n\\addplot[cD,densely dotted,line width=1pt] coordinates {{{co(g, np.array(trend["pred"][k + "_mean"]) * sc_)}}};\n' + ob)
    if k == 'V':
        o.append(lab(7.55, 31, 'partial copying\\\\simulation bands', 'cA', extra=',align=left')
                 + lab(7.55, 40.5, 'constant\\\\source', 'cB', extra=',align=left')
                 + lab(7.55, 50, 'changing\\\\source', 'cD', extra=',align=left'))
o.append('\\end{groupplot}\n\\end{tikzpicture}}\n\n')

# network figure: one 150-agent scale-free network, experts at the hubs or at random
NS = json.load(open(R + 'net_snapshot.json')); pos = np.array(NS['pos']); deg = np.array(NS['deg'])
pos = (pos - pos.min(0)) / (pos.max(0) - pos.min(0)) * 4.7
o.append('\\newcommand{\\netedges}{' + ''.join(f'\\draw({pos[a][0]:.2f},{pos[a][1]:.2f})--({pos[b][0]:.2f},{pos[b][1]:.2f});' for a, b in NS['edges']) + '}\n')
LO, HI = 0.12, 0.28


def shade(v):
    t = float(np.clip((v - LO) / (HI - LO), 0, 1))
    return f'cA!{int(10 + 90 * t / 0.7)}!white' if t < 0.7 else f'cA!{int(100 - 45 * (t - 0.7) / 0.3)}!black'


def nodes(pl):
    ex = NS[pl]['expert']; s = ''
    for i in np.argsort(deg):
        x, y = pos[i]; r = min(1.3 + 0.45 * np.sqrt(deg[i]), 3.6)
        if ex[i]:
            s += f'\\filldraw[fill=cB,draw=white,line width=0.4pt] ({x:.2f},{y:.2f}) circle (3.0pt);'
        else:
            s += f'\\filldraw[fill={shade(NS[pl]["prev"][i])},draw=white,line width=0.3pt] ({x:.2f},{y:.2f}) circle ({r:.1f}pt);'
    return s


LC = FD['locality']; nlay = sum(v['n'] for v in LC['hubs'].values())
o.append('\\newcommand{\\fignet}{%\n\\begin{tikzpicture}[font=\\sffamily\\footnotesize]\n')
for k, (pl, letter, ttl) in enumerate((('hubs', 'a', 'Experts at the hubs'), ('random', 'b', 'Experts at random'))):
    o.append(f'\\begin{{scope}}[xshift={k * 6.4:.1f}cm]\n\\begin{{scope}}[black!20,line width=0.25pt]\\netedges\\end{{scope}}\n{nodes(pl)}\n'
             f'\\node[anchor=south west,inner sep=0pt] at (0,5.3) {{\\textbf{{{letter}}}\\hspace{{0.6em}}{ttl}}};\n'
             f'\\node[font={SF},anchor=south west,inner sep=0pt,text=black!65] at (0,4.92) {{mean category breadth {100 * NS[pl]["mean"]:.0f}\\%}};\n\\end{{scope}}\n')
o.append(f'\\shade[left color=cA!10!white,right color=cA!55!black] (0.6,-0.6) rectangle (3.2,-0.48);\n\\node[font={SF},anchor=east] at (0.55,-0.54) {{{100 * LO:.0f}\\%}};\\node[font={SF},anchor=west] at (3.25,-0.54) {{{100 * HI:.0f}\\%}};\n'
         f'\\node[font={SF},anchor=west,text=black!65] at (4.0,-0.54) {{share of people a lay agent counts as having ADHD}};\n'
         f'\\filldraw[fill=cB,draw=white] (0.72,-1.0) circle (2.6pt);\\node[font={SF},anchor=west] at (0.85,-1.0) {{expert (5\\% of agents)}};\n'
         f'\\node[font={SF},anchor=west,text=black!65] at (4.0,-1.0) {{symbol size shows the number of connections}};\n')
d = D['dose']; fl = ['1', '2', '5', '10', '20']
xs_ = list(range(1, 6))
o.append(f'\\begin{{axis}}[pub,at={{(0cm,-2.6cm)}},anchor=north west,width=6.4cm,height=5.6cm,{head("c", "Few at the hubs or many at random")},xlabel={{Agents who are experts (\\%)}},ylabel={{Implied prevalence, lay agents (\\%)}},xmin=0.6,xmax=5.4,ymin=12,ymax=23.5,xtick={{1,...,5}},xticklabels={{{",".join(fl)}}},ytick={{12,14,...,22}}]\n')
o.append(f'\\draw[black!45,densely dotted,line width=0.6pt] (axis cs:0.6,{100 * d["none"]:.2f}) -- (axis cs:5.4,{100 * d["none"]:.2f});\n' + lab(5.4, 100 * d['none'] + 0.45, 'no experts', 'black!55', anchor='east'))
o.append(f'\\fill[black!7] (axis cs:0.6,{100*d["hubs"][1]:.2f}) rectangle (axis cs:5.4,{100*d["random"][3]:.2f});\n' if False else '')
o.append(f'\\addplot[black!55,mark=square*,mark size=1.8pt] coordinates {{{co(xs_, 100 * np.array(d["random"]))}}};\n'
         f'\\addplot[cA,mark=*,mark size=2pt,line width=1.2pt] coordinates {{{co(xs_, 100 * np.array(d["hubs"]))}}};\n')
o.append(lab(5.0, 100 * d['random'][4] - 0.35, 'at random', 'black!70', anchor='north') + lab(5.0, 100 * d['hubs'][4] - 0.35, 'at hubs', 'cA', anchor='north'))
h2, r10 = 100 * d['hubs'][1], 100 * d['random'][3]
o.append(f'\\draw[black!70,line width=0.5pt,-latex] (axis cs:2.12,{h2:.2f}) -- (axis cs:3.85,{h2:.2f});\n'
         f'\\draw[black!40,line width=0.4pt,densely dotted] (axis cs:4,{h2:.2f}) -- (axis cs:4,{r10:.2f});\n'
         + lab(0.75, 12.4, '2\\% at hubs have more effect\\\\than 10\\% at random', 'black!70', anchor='south west', extra=',align=left'))
o.append('\\end{axis}\n')
o.append(f'\\begin{{axis}}[pub,at={{(7.6cm,-2.6cm)}},anchor=north west,width=6.4cm,height=5.6cm,{head("d", "Influence stays close to the expert")},xlabel={{Steps to the nearest expert}},ylabel={{Implied prevalence, each lay agent (\\%)}},xmin=0.4,xmax=4.6,ymin=5,ymax=40,xtick={{1,2,3,4}},xticklabels={{1,2,3,$\\geq$4}}]\n')
o.append(f'\\draw[black!45,densely dotted,line width=0.6pt] (axis cs:0.4,{100 * LC["none"]:.2f}) -- (axis cs:4.6,{100 * LC["none"]:.2f});\n')
jit = np.random.default_rng(7)
for pl, off, colr in (('hubs', -0.21, 'cA'), ('random', 0.21, 'black!60')):
    for dd in range(1, 5):
        e = LC[pl][str(dd)]
        if not e['q']:
            continue
        x0 = dd + off; smp = e['sample']
        o.append(f'\\addplot[only marks,mark=*,mark size=0.55pt,{colr},opacity=0.35,line width=0pt] coordinates {{' + ' '.join(f'({x0 + jit.uniform(-0.11, 0.11):.3f},{100 * v:.2f})' for v in smp) + '};\n')
        q = [100 * v for v in e['q']]
        o.append(f'\\draw[{colr},line width=0.5pt] (axis cs:{x0:.2f},{q[0]:.2f}) -- (axis cs:{x0:.2f},{q[4]:.2f});\n\\draw[{colr},fill=white,line width=0.6pt] (axis cs:{x0 - 0.07:.2f},{q[1]:.2f}) rectangle (axis cs:{x0 + 0.07:.2f},{q[3]:.2f});\n\\draw[{colr},line width=1.1pt] (axis cs:{x0 - 0.07:.2f},{q[2]:.2f}) -- (axis cs:{x0 + 0.07:.2f},{q[2]:.2f});\n')
        share_y = 7.5 if pl == 'hubs' else 5.8
        o.append(f'\\node[font=\\sffamily\\tiny,text={colr}] at (axis cs:{x0:.2f},{share_y}) {{{100 * e["n"] / nlay:.0f}\\%}};\n')
o.append(lab(4.65, 32, 'experts at hubs', 'cA') + lab(4.65, 28.6, 'experts at random', 'black!70') + lab(4.65, 100 * LC['none'], 'no experts', 'black!55') + f'\\node[font=\\sffamily\\tiny,text=black!55,anchor=west] at (axis cs:4.62,6.5) {{share of lay agents}};\n')
o.append('\\end{axis}\n\\end{tikzpicture}}\n')

part1 = ''.join(o)

# robustness figure and supplementary figures
S = RB['sens']; none = np.array(S['none']); rnd = np.array(S['random']); hub = np.array(S['hubs']); qi = S['q'].index(0.5); li = S['lam'].index(0.1)
Z = RB['size']; ratio = (none - hub) / (none - rnd); Lm = S['lam']; Qm = S['q']; rmax = float(ratio.max())
o = ['\\newcommand{\\figrobust}{%\n\\begin{tikzpicture}[font=\\sffamily\\footnotesize]\n']
o.append(f'\\begin{{axis}}[pub,axis lines=box,width=6.6cm,height=6.0cm,{head("a", "Effect of experts at hubs relative to\\\\the same number at random")},xlabel={{Positive share of examples $q$ (\\%)}},ylabel={{Reversion rate $\\lambda$}},xmin=0.5,xmax={len(Qm) + 0.5},ymin=0.5,ymax={len(Lm) + 0.5},xtick={{1,...,{len(Qm)}}},ytick={{1,...,{len(Lm)}}},xticklabels={{{",".join(f"{100 * q:g}" for q in Qm)}}},yticklabels={{{",".join(f"{l:g}" for l in Lm)}}},major tick length=0pt,colormap={{seq}}{{color=(cA!8!white) color=(cA)}},point meta min=1.5,point meta max={rmax:.2f},title style={{text width=6cm}}]\n')
o.append('\\addplot[matrix plot*,mesh/cols=' + str(len(Qm)) + ',mesh/rows=' + str(len(Lm)) + ',point meta=explicit,draw=white,line width=0.8pt] coordinates {' + ' '.join(f'({j + 1},{i + 1}) [{ratio[i][j]:.3f}]' for i in range(len(Lm)) for j in range(len(Qm))) + '};\n')
for tc, cond in (('white', lambda r: r > 1 + 0.6 * (rmax - 1)), ('black', lambda r: r <= 1 + 0.6 * (rmax - 1))):
    pts_ = [(j + 1, i + 1, ratio[i][j]) for i in range(len(Lm)) for j in range(len(Qm)) if cond(ratio[i][j])]
    if pts_:
        o.append(f'\\addplot[only marks,mark=none,point meta=explicit symbolic,nodes near coords,nodes near coords align={{center}},every node near coord/.append style={{font=\\sffamily\\scriptsize,text={tc},inner sep=0pt}}] coordinates {{' + ' '.join(f'({x},{y}) [{r:.1f}$\\times$]' for x, y, r in pts_) + '};\n')
o.append('\\end{axis}\n')
rowsD = [('100 agents', Z['none'][0][0], Z['random'][0][0], Z['hubs'][0][0]), ('200 agents', Z['none'][1][0], Z['random'][1][0], Z['hubs'][1][0]), ('500 agents', Z['none'][2][0], Z['random'][2][0], Z['hubs'][2][0]), ('1{,}000 agents', Z['none'][3][0], Z['random'][3][0], Z['hubs'][3][0]),
         ('with clustering', RB['topo']['sfclustered']['none'][0], RB['topo']['sfclustered']['random'][0], RB['topo']['sfclustered']['hubs'][0]), ('modular, even degrees', RB['topo']['modular']['none'][0], RB['topo']['modular']['random'][0], RB['topo']['modular']['hubs'][0])][::-1]
o.append(f'\\begin{{axis}}[pub,at={{(8.9cm,0cm)}},width=6.6cm,height=6.0cm,{head("b", "Across network sizes and types")},xlabel={{Reduction in implied prevalence (points)}},xmin=0,xmax=7,ymin=0.4,ymax={len(rowsD) + 0.6},ytick={{1,...,{len(rowsD)}}},yticklabels={{{",".join("{" + r[0] + "}" for r in rowsD)}}},y axis line style={{draw=none}},ytick style={{draw=none}}]\n')
o.append(f'\\draw[black!15,line width=0.4pt] (axis cs:0,2.5) -- (axis cs:7,2.5);\n' )
for i, (nm_, n0, r0_, h0) in enumerate(rowsD):
    a_, b_ = 100 * (n0 - r0_), 100 * (n0 - h0)
    o.append(f'\\draw[black!25,line width=2.2pt] (axis cs:{a_:.2f},{i + 1}) -- (axis cs:{b_:.2f},{i + 1});\n\\addplot[only marks,mark=*,mark size=2.3pt,black!60,line width=0pt] coordinates {{({a_:.2f},{i + 1})}};\n\\addplot[only marks,mark=*,mark size=2.3pt,cA,line width=0pt] coordinates {{({b_:.2f},{i + 1})}};\n')
top_ = rowsD[-1]
o.append(lab(100 * (top_[1] - top_[2]), len(rowsD) + 0.25, 'at random', 'black!70', anchor='south') + lab(100 * (top_[1] - top_[3]), len(rowsD) + 0.25, 'at hubs', 'cA', anchor='south'))
o.append('\\end{axis}\n')
AS = json.load(open(R + 'asymmetry_b.json'))['rows']; AR = [r for r in AS if r['lam'] == 0.1]
rowsA = []
for r in sorted(AR, key=lambda r: -r['q']):
    for pl, lbl in (('hubs', 'at hubs'), ('random', 'at random')):
        rowsA.append((f"{100 * r['q']:g}\\% positive, {lbl}", r['strict_' + pl] - r['none'], r['none'] - r['broad_' + pl], r['ratio_' + pl], r['strict_shift_' + pl], r['broad_shift_' + pl]))
rowsA = rowsA[::-1]
o.append(f'\\begin{{axis}}[pub,at={{(0cm,-2.4cm)}},anchor=north west,width=13.6cm,height=6.2cm,{head("c", "Broad sources can shift boundaries further")},xlabel={{Shift of lay agents\' boundary (standard deviations of the trait)}},xmin=-0.5,xmax=0.5,ymin=0.4,ymax={len(rowsA) + 0.9},ytick={{1,...,{len(rowsA)}}},yticklabels={{{",".join("{" + r[0] + "}" for r in rowsA)}}},y axis line style={{draw=none}},ytick style={{draw=none}},xtick={{-0.4,-0.2,0,0.2,0.4}},xticklabels={{0.4,0.2,0,0.2,0.4}}]\n')
o.append(f'\\draw[black!50,line width=0.5pt] (axis cs:0,0.4) -- (axis cs:0,{len(rowsA) + 0.5});\n')
for i, (lb, st, br, ra, sci, bci) in enumerate(rowsA):
    y = i + 1
    o.append(f'\\fill[cB] (axis cs:{-st:.3f},{y - 0.3}) rectangle (axis cs:0,{y + 0.3});\n\\fill[cD] (axis cs:0,{y - 0.3}) rectangle (axis cs:{br:.3f},{y + 0.3});\n')
    o.append(f'\\draw[black!75,line width=0.5pt] (axis cs:{-sci[1]:.3f},{y}) -- (axis cs:{-sci[2]:.3f},{y});\n\\draw[black!75,line width=0.5pt] (axis cs:{bci[1]:.3f},{y}) -- (axis cs:{bci[2]:.3f},{y});\n')
    o.append(f'\\node[font=\\sffamily\\scriptsize,anchor=west,text=black!70] at (axis cs:0.52,{y}) {{{ra:.1f}$\\times$}};\n')
    if i % 2 == 1 and i < len(rowsA) - 1:
        o.append(f'\\draw[black!12,line width=0.4pt] (axis cs:-0.5,{y + 0.5}) -- (axis cs:0.5,{y + 0.5});\n')
o.append(lab(-0.03, len(rowsA) + 0.55, '$\\leftarrow$ narrower with strict sources', 'cB!90!black', anchor='east') + lab(0.03, len(rowsA) + 0.55, 'broader with broad sources $\\rightarrow$', 'cD', anchor='west') + f'\\node[font=\\sffamily\\scriptsize,anchor=west,text=black!55] at (axis cs:0.52,{len(rowsA) + 0.55}) {{ratio}};\n')
o.append('\\end{axis}\n\\end{tikzpicture}}\n\n')

rows = DS['rows']
o.append('\\newcommand{\\figpred}{%\n\\begin{tikzpicture}[font=\\sffamily\\footnotesize]\n\\begin{axis}[pub,axis lines=box,width=7cm,height=7cm,xlabel={Predicted shift of mean offset},ylabel={Simulated shift of mean offset},xmin=-0.56,xmax=0.56,ymin=-0.12,ymax=0.12,scaled ticks=false,tick label style={font=\\sffamily\\scriptsize,/pgf/number format/fixed},clip=true]\n')
o.append('\\addplot[black!35,dashed,domain=-0.12:0.12,samples=2,line width=0.5pt] {x};\n')
o.append('\\addplot[only marks,mark=o,mark size=1.5pt,black!55,line width=0.4pt] coordinates {' + ' '.join(f'({r["closed"]:.3g},{r["sim"]:.3g})' for r in rows) + '};\n')
o.append('\\addplot[only marks,mark=*,mark size=1.5pt,cA,line width=0pt] coordinates {' + ' '.join(f'({r["pair"]:.3g},{r["sim"]:.3g})' for r in rows) + '};\n')
o.append(lab(0.03, -0.085, 'averaged over\\\\pairs of agents', 'cA', extra=',align=left') + lab(0.16, 0.02, 'closed form', 'black!65', anchor='north west') + lab(-0.1, -0.112, 'equality', 'black!50', anchor='south east'))
o.append('\\end{axis}\n\\end{tikzpicture}}\n\n')
e = [r['eta'] for r in P]; L = np.array([r['loss'] for r in P]); L = L - L.min()
o.append('\\newcommand{\\figprofile}{%\n\\begin{tikzpicture}[font=\\sffamily\\footnotesize]\n\\begin{axis}[pub,width=7cm,height=5.2cm,xlabel={Learning rate $\\eta$},ylabel={Loss above the best fit},ymin=0,ymax=70,xmin=-0.02,xmax=0.83]\n'
         + f'\\addplot[cA,mark=*,mark size=1.3pt] coordinates {{{co(e, L)}}};\n' + '\\end{axis}\n\\end{tikzpicture}}\n')
part2 = ''.join(o).replace('ymax=70', f'ymax={max(10, float(L.max()) * 1.15):.2f}')
# Derive the conceptual chain bars from the current fit rather than an earlier calibration.
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from chain_model import simulate
cp=json.load(open(R+'fit_s3adj_balanced_lag.json'))['params']
cs=simulate([cp[k] for k in ('mu_b','s_b','eta','sig_p','v','sig_r','c_s')],chains=40000,seed=11)
means=(1-norm.cdf(cs['b'])).mean(axis=0)
old_bars='0/1.1/0.10/5,1/3.6/0.31/15,2/6.1/0.39/19,3/8.6/0.42/21,4/11.1/0.43/21,5/13.6/0.43/21'
new_bars='0/1.1/0.10/5,'+','.join(f'{i}/{1.1+2.5*i:.1f}/{2*v:.5f}/{100*v:.0f}' for i,v in enumerate(means[:5],1))
assert old_bars in CONCEPT
CONCEPT=CONCEPT.replace(old_bars,new_bars)
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from sensitivity_material import figures as sensitivity_figures
open('figures.tex', 'w').write(part1 + part2 + CONCEPT + sensitivity_figures())
print('figures.tex', len(part1 + part2 + CONCEPT))
