"""Small-domain matched-physics recovery using the current AK170 mapping.

This is a numerical/identifiability experiment, not a recovery test of NCC
geography.  It changes the domain and observation geometry, not the 170-mode
mapping, material bounds, source budgets, constitutive law or forward solver.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import argparse
import json
import time
import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from observations import PointSampler
from experiments.model import (
    BoundedProperties, BudgetProperties, K_BOUNDS, A_BOUNDS,
    build_solver, InverseProblem,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = (ROOT/'experiments/results/synthetic')
SEED = 917170


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


class Problem(InverseProblem):
    def __init__(self):
        y, x = np.meshgrid(np.arange(11)*20000., np.arange(19)*20000., indexing='ij')
        z = np.arange(0, 100.01, .5)
        u, v = x/x.max(), y/y.max()
        sediment = 4.+2.*np.sin(np.pi*u)*np.sin(np.pi*v)
        moho = 38.+3.*np.cos(np.pi*u)
        interfaces = np.stack([sediment, sediment+.32*(moho-sediment),
                               sediment+.75*(moho-sediment), moho], axis=-1)
        above = np.clip((interfaces[...,None]-z)/.5+.5, 0, 1)
        masks = np.moveaxis(np.concatenate([above[...,:1,:], np.diff(above,axis=-2),
                                            1-above[...,-1:,:]], axis=-2), -2, 0)
        shape = (len(y), x.shape[1], len(z))
        kr = np.broadcast_to(np.array([2.4,2.9,2.4,2.15,3.8])[:,None,None,None], masks.shape).copy()
        ar = np.broadcast_to(np.array([1.2,1.4,.75,.25,.03])[:,None,None,None], masks.shape).copy()
        p = BoundedProperties(masks,z,kr,ar,dc=True,vary_K=True)
        self.volume = np.broadcast_to(np.r_[.25,np.full(len(z)-2,.5),.25],shape).copy()*400
        self.properties = BudgetProperties(p,self.volume)
        p = self.properties
        assert p.n == 170 and sum(s.startswith('A_') for s in p.names) == 99
        self.b = dict(x=x,y=y,z_km=z,masks=masks,domain_mask=np.ones(x.shape,bool),
            core=np.ones(x.shape,bool),horizontal_scale=np.ones(x.shape),
            surface_T=12+2*np.cos(np.pi*v),T_bot=1200+80*np.cos(np.pi*u)*np.cos(np.pi*v),
            A=p.A,K_sun_C=p.K20)
        self.n = p.n; self.law = p; self.regularization = 4.
        self.solver = build_solver(self.b,p,tolerance_C=3e-7)
        self.solver.linear_solver.cg_rtol = 1e-12
        rng = np.random.default_rng(SEED)
        def xy(n):
            return np.column_stack([rng.uniform(10000,y.max()-10000,n),rng.uniform(10000,x.max()-10000,n)])
        gxy,qxy,cxy = xy(160),xy(120),xy(80)
        axes = (y[:,0],x[0],z)
        top = rng.uniform(.1,1.,len(gxy)); bottom=top+rng.uniform(.5,1.5,len(gxy))
        lo=PointSampler.build(axes,np.column_stack([gxy,top])); hi=PointSampler.build(axes,np.column_stack([gxy,bottom]))
        self.g_op=PointSampler(shape,(hi.matrix-lo.matrix).multiply(1/(bottom-top)[:,None]).tocsr())
        self.q_op=PointSampler.build(axes[:2],qxy)
        self.curie_op=PointSampler.build(axes[:2],cxy)
        self.records=pd.DataFrame(dict(gradient_C_km=np.zeros(len(gxy))))
        self.heatflow=pd.DataFrame(dict(q_mW_m2=np.zeros(len(qxy))))
        self.curie=pd.DataFrame(dict(observed=np.full(len(cxy),30.)))
        self.error=np.full(len(gxy),1.); self.qsigma=np.full(len(qxy),3.)
        self.gweight=1/self.error**2; self.qweight=1/self.qsigma**2
        self.precision=np.eye(len(cxy))
        self.last=None
        self.coordinates=dict(gxy=gxy,qxy=qxy,cxy=cxy,gradient_top_km=top,gradient_bottom_km=bottom)


def snapshot(obj):
    return {k:np.asarray(v).copy() for k,v in obj.last.items()} | dict(Kref=obj.properties.K20.copy())


def metrics(state,truth,reference,weights):
    result={}
    for key,label,scale in [('T','T_C',1),('A','A_uW_m3',1e6),('Kref','Kref_W_mK',1),('K','K_in_situ_W_mK',1)]:
        p,t,r=state[key]*scale,truth[key]*scale,reference[key]*scale
        mse=np.average((p-t)**2,weights=weights); initial=np.average((r-t)**2,weights=weights)
        result[label+'_RMSE']=float(np.sqrt(mse))
        result[label+'_bias']=float(np.average(p-t,weights=weights))
        result[label+'_error_reduction']=float(1-np.sqrt(mse/initial))
        # Recovery of anomalies relative to the shared reference avoids credit
        # for the imposed vertical layering and fixed temperature boundaries.
        anomaly_p=(p-r).ravel(); anomaly_t=(t-r).ravel()
        result[label+'_anomaly_r']=float(np.corrcoef(anomaly_p,anomaly_t)[0,1]) if np.std(anomaly_p)>0 else 0.
    for key,label in [('gradient','gradient_C_km'),('heatflow','heatflow_mW_m2'),('depth','Curie_km')]:
        result[label+'_truth_RMSE']=float(np.sqrt(np.mean((state[key]-truth[key])**2)))
    return result


def main(steps, benchmark=False, output=None):
    global OUT
    if output is None: raise ValueError("A fresh output directory is required")
    OUT = Path(output)
    OUT.mkdir(parents=True, exist_ok=False)

    tic=time.perf_counter(); obj=Problem(); p=obj.properties
    rng=np.random.default_rng(SEED+1)
    truth_c=rng.normal(0,.016,p.n)*p.scales
    specified={'A_sediment_y1_x1':.28,'A_upper_crust_y0_x1':-.35,
        'A_upper_crust_y1_x2':.18,'A_middle_crust_y1_x0':.35,
        'A_middle_crust_y1_x2':-.25,'A_lower_crust_y0_x1':.28,
        'K_sediment_y0_x0':.12,'K_sediment_y1_x1':-.30,
        'K_upper_crust_y0_x1':.18,'K_middle_crust_y1_x0':-.20,
        'K_lower_crust_y0_x1':.17,'K_mantle_y0_x0':.08}
    for name,value in specified.items(): truth_c[p.names.index(name)]=value
    obj.evaluate(np.zeros(p.n),gradient=False,warm=False); reference=snapshot(obj)
    obj.evaluate(truth_c,gradient=False,warm=False); truth=snapshot(obj)
    noise_rng=np.random.default_rng(SEED+2)
    noise=dict(gradient=noise_rng.normal(0,1,160),heatflow=noise_rng.normal(0,3,120),depth=noise_rng.normal(0,1,80))
    def data(case):
        noisy=case!='joint_noisefree'
        obj.records['gradient_C_km']=truth['gradient']+(noise['gradient'] if noisy else 0)
        obj.heatflow['q_mW_m2']=truth['heatflow']+(noise['heatflow'] if noisy else 0)
        obj.curie['observed']=truth['depth']+(noise['depth'] if noisy else 0)
        obj.gweight=np.ones(160)
        obj.qweight=np.full(120,0. if case=='no_heatflow' else 1/9)
        obj.precision=np.eye(80)*(0. if case=='no_Curie' else 1.)
        obj.last=None
    data('joint'); start=time.perf_counter(); value,grad=obj.evaluate(np.zeros(p.n),warm=False)
    seconds=time.perf_counter()-start
    direction=rng.normal(size=p.n);direction/=np.linalg.norm(direction);h=2e-5
    vp=obj.evaluate(h*direction,False,warm=False);vm=obj.evaluate(-h*direction,False,warm=False)
    finite=(vp-vm)/(2*h); exact=float(grad@direction)
    rel=abs(finite-exact)/max(1,abs(finite));assert rel<.003,(finite,exact,rel)
    check=dict(evaluation_seconds=seconds,directional_finite_difference=finite,directional_adjoint=exact,
        relative_error=rel,n_parameters=p.n,shape=list(obj.b['A'].shape))
    write((OUT/'benchmark.json'),check); print(json.dumps(check),flush=True)
    if benchmark:return
    sources=[Path(__file__), Path(__file__).with_name('model.py')]+[ROOT/name for name in
        ['heat_solver.py', 'nonlinear_heat_solver.py', 'observations.py', 'spatial_basis.py', 'heat_flux.py']]
    write((OUT/'manifest.json'),dict(seed=SEED,shape=list(obj.b['A'].shape),horizontal_spacing_km=20,
        vertical_spacing_km=.5,extent_km=[360,200,100],n_parameters=p.n,n_A=99,n_K=71,
        layer_names=['sediment','upper_crust','middle_crust','lower_crust','mantle'],
        Kref_bounds=K_BOUNDS.tolist(),A_bounds=A_BOUNDS.tolist(),reference_A_budget=p.reference_means.tolist(),
        truth_coefficients=truth_c.tolist(),coefficient_names=p.names,
        noise_sigma=dict(gradient_C_km=1,heatflow_mW_m2=3,Curie_km=1),
        noise_assumption='Independent Gaussian synthetic perturbations; not empirical uncertainty estimates for NCC.',
        n_observations=dict(gradient=160,heatflow=120,Curie=80),regularization=4.,steps=steps,
        profile_locations_xy_km=[[180,100],[120,60]],
        selected_model_changed=False,initialization='All coefficients zero for every case',
        limitations='Matched-physics small-domain experiment; same 170 modes and physical mapping but shorter wavelengths and idealized geography/observations. No model-error test or regional posterior coverage.',
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}))
    np.savez_compressed((OUT/'setup.npz'),**obj.b,**obj.coordinates,truth_coefficients=truth_c,
        noise_gradient=noise['gradient'],noise_heatflow=noise['heatflow'],noise_Curie=noise['depth'])
    np.savez_compressed((OUT/'reference.npz'),**reference);np.savez_compressed((OUT/'truth.npz'),**truth)
    rows=[]; profiles=[]
    def add_profiles(case,state):
        for label,iy,ix in [('center',5,9),('southwest',3,6)]:
            for iz,depth in enumerate(obj.b['z_km']):
                profiles.append(dict(case=case,location=label,x_km=float(obj.b['x'][iy,ix]/1000),
                    y_km=float(obj.b['y'][iy,ix]/1000),depth_km=float(depth),
                    T_C=float(state['T'][iy,ix,iz]),Kref_W_mK=float(state['Kref'][iy,ix,iz]),
                    K_in_situ_W_mK=float(state['K'][iy,ix,iz]),A_uW_m3=float(state['A'][iy,ix,iz]*1e6)))
    add_profiles('truth',truth);add_profiles('reference',reference)
    for case in ['joint','no_Curie','no_heatflow','joint_noisefree']:
        data(case);history=[];ct=time.perf_counter()
        def fun(c):
            value,gradient=obj.evaluate(c)
            history.append(dict(evaluation=len(history)+1,objective=value,seconds=time.perf_counter()-ct))
            if len(history)%10==0: print(case,json.dumps(history[-1]),flush=True)
            return value,gradient
        fit=minimize(fun,np.zeros(p.n),jac=True,method='L-BFGS-B',bounds=[(-4,4)]*p.n,
            options=dict(maxiter=steps,maxfun=steps*4,maxls=25,maxcor=25,ftol=2e-9,gtol=1e-4))
        obj.evaluate(fit.x,warm=False);state=snapshot(obj)
        np.savez_compressed(OUT/(case+'.npz'),**state)
        add_profiles(case,state);pd.DataFrame(profiles).to_csv((OUT/'profiles.csv'),index=False)
        write(OUT/(case+'_history.json'),history)
        row=dict(case=case,iterations=int(fit.nit),evaluations=int(fit.nfev),success=bool(fit.success),
            message=str(fit.message),seconds=time.perf_counter()-ct,objective=float(fit.fun),
            coefficient_RMSE=float(np.sqrt(np.mean((fit.x-truth_c)**2))),
            PDE_residual=obj.solver.info['nonlinear_relative_residual'],**metrics(state,truth,reference,obj.volume))
        for depth in [20,40,60]:
            iz=int(depth/.5)
            row[f'T_{depth}km_RMSE_C']=float(np.sqrt(np.mean((state['T'][...,iz]-truth['T'][...,iz])**2)))
        crust_weights=obj.volume*np.sum(obj.b['masks'][:4],axis=0)
        row['crust_A_RMSE_uW_m3']=float(np.sqrt(np.average(((state['A']-truth['A'])*1e6)**2,weights=crust_weights)))
        row['crust_A_reference_RMSE_uW_m3']=float(np.sqrt(np.average(((reference['A']-truth['A'])*1e6)**2,weights=crust_weights)))
        for key,table,col in [('gradient',obj.records,'gradient_C_km'),('heatflow',obj.heatflow,'q_mW_m2'),('depth',obj.curie,'observed')]:
            row[key+'_data_RMSE']=float(np.sqrt(np.mean((state[key]-table[col].to_numpy())**2)))
        for key in ['A','K']:
            lo,hi=p.bounds[key].T
            assert np.all(p.values[key]>=lo[:,None,None,None]) and np.all(p.values[key]<=hi[:,None,None,None])
        row['physical_bound_violations']=0
        row['max_layer_A_budget_error_uW_m3']=float(max(abs(w@p.values['A'][i].ravel()[ids]-p.reference_means[i])
            for i,(ids,w) in enumerate(zip(p.budget_indices,p.budget_weights))))
        rows.append(row);pd.DataFrame(rows).to_csv((OUT/'metrics.csv'),index=False)
        print(json.dumps(row),flush=True)
    write((OUT/'complete.json'),dict(complete=True,seconds=time.perf_counter()-tic,main_model_changed=False))
    plot()


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    truth=dict(np.load((OUT/'truth.npz'))); ref=dict(np.load((OUT/'reference.npz')))
    cases=['truth','joint','no_Curie','no_heatflow']
    fields=[dict(np.load(OUT/(case+'.npz'))) for case in cases]
    fig,axes=plt.subplots(3,4,figsize=(12,6.7),layout='constrained')
    for row,(key,depth,scale,label) in enumerate([('T',20,1,'T at 20 km (°C)'),
            ('A',20,1e6,'A at 20 km (μW m$^{-3}$)'),
            ('Kref',2,1,'K$_{ref}$ at 2 km (W m$^{-1}$ K$^{-1}$)')]):
        iz=int(depth/.5); arrays=[d[key][...,iz]*scale for d in fields]
        vmin=min(a.min() for a in arrays);vmax=max(a.max() for a in arrays)
        for col,arr in enumerate(arrays):
            ax=axes[row,col];im=ax.imshow(arr,origin='lower',extent=[0,360,0,200],cmap='turbo',vmin=vmin,vmax=vmax,aspect='auto')
            if row==0:ax.set_title(['Synthetic truth','Joint constraints','Without Curie depth','Without heat flow'][col],fontsize=10)
            ax.set_xticks([0,180,360]);ax.set_yticks([0,100,200])
            if col==0:ax.set_ylabel('Northing (km)')
            if row==2:ax.set_xlabel('Easting (km)')
            ax.text(.025,.97,chr(97+row*4+col),transform=ax.transAxes,va='top',fontweight='bold',bbox=dict(facecolor='white',alpha=.65,edgecolor='none',pad=1.5))
        cb=fig.colorbar(im,ax=axes[row,:],fraction=.016,pad=.012)
        cb.set_label(label,fontsize=9);cb.ax.tick_params(labelsize=9)
    fig.savefig((OUT/'synthetic_recovery.png'),dpi=220)
    fig.savefig((OUT/'synthetic_recovery.svg'))
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=60);parser.add_argument('--benchmark',action='store_true');parser.add_argument('--plot-only',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.plot_only:plot()
    else:
        if args.output is None:parser.error('--output is required for a new synthetic run')
        main(args.steps,args.benchmark,output=args.output)
