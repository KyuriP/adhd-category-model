<<<<<<< HEAD
"""Repeated exchange under three rules: boundary copying, and two learners that only see
examples and labels (fixed step, and step scaled by the error). The label-only rules never
look at the source's boundary. In 2D they update a linear classifier (w, b), normalise
both together, then move back toward the initial angle and offset.
=======
"""Repeated exchange with boundary copying and two learners that use labels alone.

Uses the original network sampling scheme but gives every source condition the
same initial people and graphs. The label-only rules never inspect source
parameters. Two-dimensional rules update a linear classifier (w,b), normalise w
and b together, then revert toward the initial angle and offset as in the model.
>>>>>>> 20bc25dc208597d275872f1132daa9fbab6be5c8
"""
from pathlib import Path
import sys
import numpy as np
from scipy.special import ndtr, ndtri

sys.path.insert(0, str(Path(__file__).resolve().parent))
from network_model import build, place_experts


def learn(b, phi, x, labels, rule, eta=.28, step=.05, aligned=False, bt=None, pt=None):
    for j in range(labels.shape[1]):
        u=np.stack([np.cos(phi),np.sin(phi)],axis=-1)
        projected=(u*x[:,j]).sum(axis=-1)
        score=projected-b
        disagree=(score>0)!=labels[:,j]
        if rule=='copy':
            new_b=b+eta*(bt-b);new_phi=phi+eta*(pt-phi)
        elif aligned:
            new_b=(b+np.where(labels[:,j],-step,step) if rule=='perceptron'
                   else b+eta*(projected-b))
            new_phi=phi
        else:
            if rule=='perceptron':
                direction=np.where(labels[:,j],1.,-1.)
                new_w=u+step*direction[:,None]*x[:,j]
                new_b=b-step*direction
            elif rule=='margin':
                correction=eta*score/(1+(x[:,j]**2).sum(axis=-1))
                new_w=u-correction[:,None]*x[:,j]
                new_b=b+correction
            else:raise ValueError(rule)
            scale=np.linalg.norm(new_w,axis=-1)
            if np.any(scale<1e-12):raise FloatingPointError('zero classifier normal')
            new_b=new_b/scale
            angle=np.arctan2(new_w[:,1],new_w[:,0])
            new_phi=phi+np.arctan2(np.sin(angle-phi),np.cos(angle-phi))
        b=np.where(disagree,new_b,b);phi=np.where(disagree,new_phi,phi)
    return b,phi


def run(rule='copy', aligned=False, kind='complete', N=200, R=20, T=600,
        q=.5, lam=.1, f=0., placement='random', source_b=None,
        mu=.85, spread=.31, eta=.28, step=.05, seed=71):
    graphs,deg,nb=build(kind,N,R,seed)
    ex=place_experts(graphs,deg,f,placement,np.random.default_rng(seed+10000)).ravel()
    rng=np.random.default_rng(seed)
    M=N*R
    prior_b=mu+spread*rng.standard_normal(M)
    prior_phi=np.pi/4+np.deg2rad(15)*rng.standard_normal(M)
    if aligned:prior_phi[:]=np.pi/4
    b=prior_b.copy();phi=prior_phi.copy()
    if np.any(ex):
        if source_b is None:source_b=ndtri(.95)
        b[ex]=source_b;phi[ex]=np.pi/4
    start_prev=ndtr(-prior_b).reshape(R,N)
    window=120
    totals_b=np.zeros(M);totals_p=np.zeros(M)
    previous_p=np.zeros(M);trace=[]
    for t in range(T+1):
        if t%20==0:
            trace.append(dict(round=t,mean_offset=float(b[~ex].mean()),
                mean_prevalence=float(ndtr(-b[~ex]).mean())))
        if T-window<t<=T:
            totals_b+=b;totals_p+=ndtr(-b)
        if T-2*window<t<=T-window:previous_p+=ndtr(-b)
        if t==T:break
        teacher=nb[np.arange(M),(rng.random(M)*deg.ravel()).astype(int)]
        teacher=np.where(deg.ravel()==0,np.arange(M),teacher)
        bt,pt=b[teacher],phi[teacher]
        labels=rng.random((M,10))<q
        pb=ndtr(bt)[:,None];U=rng.random((M,10))
        probability=np.where(labels,pb+U*(1-pb),U*pb)
        along=ndtri(np.clip(probability,1e-15,1-1e-15))
        across=rng.standard_normal((M,10))
        u=np.stack([np.cos(pt),np.sin(pt)],axis=-1)
        w=np.stack([-np.sin(pt),np.cos(pt)],axis=-1)
        x=along[...,None]*u[:,None,:]+across[...,None]*w[:,None,:]
        b2,phi2=learn(b,phi,x,labels,rule,eta,step,aligned,bt,pt)
        b2+=lam*(prior_b-b2)
        phi2+=lam*np.arctan2(np.sin(prior_phi-phi2),np.cos(prior_phi-phi2))
        b=np.where(ex,b,b2);phi=np.where(ex,phi,phi2)
        if not np.isfinite(b).all():raise FloatingPointError('non-finite boundary')
    return dict(b=(totals_b/window).reshape(R,N),prev=(totals_p/window).reshape(R,N),
                prev_previous=(previous_p/window).reshape(R,N),initial_prev=start_prev,
                expert=ex.reshape(R,N),trace=trace)
