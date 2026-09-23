"""Numerical summaries, physical checks and exports for the paper experiments."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy.spatial import cKDTree
from project import ROOT, write_json, digest
from experiments.model import LAYERS, K_BOUNDS, A_BOUNDS
from heat_flux import upward_flux
from fit_metrics import metrics


def column_budgets(T,K,A,z_km,basement_km,moho_km,solver):
    z=np.asarray(z_km)*1000
    edges=np.r_[z[0],(z[:-1]+z[1:])/2,z[-1]]
    widths=np.diff(edges)
    if not (np.all(basement_km>=0) and np.all(moho_km>=basement_km) and np.all(moho_km<=z_km[-1])):
        raise ValueError('Interfaces outside the model or in the wrong order')
    divergence=np.zeros_like(T)
    for axis,spacing,weight in [(0,solver.dy,solver.horizontal_face_y),(1,solver.dx,solver.horizontal_face_x)]:
        aa,bb=[slice(None)]*3,[slice(None)]*3
        aa[axis],bb[axis]=slice(None,-1),slice(1,None)
        a,b=tuple(aa),tuple(bb)
        transfer=weight/spacing**2*2*K[a]*K[b]/(K[a]+K[b])*(T[a]-T[b])
        divergence[a]+=transfer;divergence[b]-=transfer
    divergence/=solver.horizontal_area_factor
    faceq=2*K[...,:-1]*K[...,1:]/(K[...,:-1]+K[...,1:])*np.diff(T,axis=-1)/np.diff(z)*1000
    source=A-divergence
    top=faceq[...,0]+widths[0]*source[...,0]*1000
    bottom=faceq[...,-1]-widths[-1]*source[...,-1]*1000
    nodes=np.concatenate([top[...,None],faceq,bottom[...,None]],axis=-1)
    interfaces=[np.zeros_like(basement_km),basement_km*1000,moho_km*1000,np.full_like(moho_km,z[-1])]
    flux=[]
    for depth in interfaces:
        idx=np.clip(np.searchsorted(edges,depth,side='right')-1,0,len(z)-1)
        qa=np.take_along_axis(nodes,idx[...,None],axis=-1)[...,0]
        qb=np.take_along_axis(nodes,(idx+1)[...,None],axis=-1)[...,0]
        frac=(depth-edges[idx])/widths[idx]
        flux.append(qa+(qb-qa)*frac)
    production=[];lateral=[]
    for lo,hi in zip(interfaces[:-1],interfaces[1:]):
        overlap=np.maximum(0,np.minimum(edges[1:],hi[...,None])-np.maximum(edges[:-1],lo[...,None]))
        production.append(np.sum(A*overlap,axis=-1)*1000)
        lateral.append(-np.sum(divergence*overlap,axis=-1)*1000)
    flux=np.array(flux);production=np.array(production);lateral=np.array(lateral)
    closure=flux[:-1]-flux[1:]-production-lateral
    return dict(interface_flux=flux,production=production,lateral_gain=lateral,closure=closure,
                surface_halfcell_lateral_correction=-widths[0]*divergence[...,0]*1000)


def interface_budgets(obj,s,dest):
    dest.mkdir(exist_ok=False);b=obj.b;core=b['core'];active=b['domain_mask']
    r=column_budgets(s['T'],s['K'],s['A'],b['z_km'],b['basement_depth_km'],b['moho_depth_km'],obj.solver.linear_solver)
    error=float(abs(r['closure'][:,active]).max());assert error<2e-5,error
    np.testing.assert_allclose(r['interface_flux'][0],s['heatflow_field']+r['surface_halfcell_lateral_correction'],atol=1e-10)
    np.savez_compressed(dest/'budgets.npz',**r)
    fields={name:field for name,field in zip(['q_surface','q_basement','q_moho','q_bottom100'],r['interface_flux'])}
    for i,layer in enumerate(['sediment','crust','mantle']):
        fields['A_'+layer]=r['production'][i];fields['lateral_'+layer]=r['lateral_gain'][i]
        fields['net_'+layer]=r['production'][i]+r['lateral_gain'][i]
    q=obj.heatflow.copy();q['prediction_original']=s['heatflow'];qobs=q.q_mW_m2.to_numpy()
    a=(400/b['horizontal_scale'][core]**2);a/=a.sum();rows=[]
    for name,field in fields.items():
        pred=obj.q_op.sample(field);q[name]=pred
        stat=metrics(qobs,pred)
        rows.append(dict(field=name,mean=float(a@field[core]),SD=float(np.sqrt(a@(field[core]-a@field[core])**2)),
            correlation_with_observed_surface_q=stat['r'],p05=float(np.quantile(field[core],.05)),p95=float(np.quantile(field[core],.95))))
    for layer,top,bottom in [('sediment','q_surface','q_basement'),('crust','q_basement','q_moho'),('mantle','q_moho','q_bottom100')]:
        np.testing.assert_allclose(q[top]-q[bottom],q['net_'+layer],atol=2e-5)
    q['surface_error']=q.q_surface-q.q_mW_m2
    q['group']=np.select([q.surface_error<=-20,q.surface_error>=20],['under20','over20'],default='other')
    q.to_csv(dest/'heatflow_sites.csv',index=False);pd.DataFrame(rows).to_csv(dest/'components.csv',index=False)
    q.groupby('group')[['q_mW_m2',*fields]].mean().to_csv(dest/'residual_group_means.csv')
    area=400e6/b['horizontal_scale']**2
    global_lateral=float(np.sum(r['lateral_gain'][:,active]*area[active])/np.sum(area[active]))
    assert abs(global_lateral)<1e-8
    info=dict(max_budget_closure_mW_m2=error,whole_domain_lateral_mean_mW_m2=global_lateral,
        top_operator_correction_max_core_mW_m2=float(abs(r['surface_halfcell_lateral_correction'][core]).max()),
        top_operator_correction_RMS_core_mW_m2=float(np.sqrt(a@r['surface_halfcell_lateral_correction'][core]**2)),
        flux_definition='Vertical upward flux at local B/M per horizontal area, not normal flux through sloping interfaces',
        correlation_definition='Component correlation with raw observed surface q; components are not independent interface observations',
        region_averaging='Area weighted target core, boundaries include lateral exchange with outside target',
        source='Production integrated in geometric depth intervals; distinct from smoothed thermal-unit material attribution')
    write_json(dest/'audit.json',info)
    return rows,info


def branch_summary(obj,T=None):
    p,b=obj.properties,obj.b;active=b['domain_mask'];area=400/b['horizontal_scale']**2
    volume=area[...,None]*np.r_[.25,np.full(len(b['z_km'])-2,.5),.25]
    rows=[]
    fields=dict(A_uW_m3=p.values['A'],K20_W_mK=p.values['K'])
    if T is not None:fields['K_in_situ_W_mK']=p.values['K']*p.factors(T)[0]
    for key,array in fields.items():
        for i,layer in enumerate(LAYERS):
            present=(b['masks'][i]>1e-8)&active[...,None]
            core=present&b['core'][...,None];v=array[i][present]
            w=(volume*b['masks'][i])[core];w/=w.sum()
            row=dict(property=key,layer=layer,min=float(v.min()),p05=float(np.quantile(v,.05)),
                median=float(np.median(v)),p95=float(np.quantile(v,.95)),max=float(v.max()),core_volume_mean=float(w@array[i][core]))
            if key!='K_in_situ_W_mK':
                lo,hi=(A_BOUNDS if key=='A_uW_m3' else K_BOUNDS)[i]
                row.update(lower=float(lo),upper=float(hi),violations=int(((v<lo-1e-12)|(v>hi+1e-12)).sum()),
                    near_limit_fraction=float(((v<lo+.02*(hi-lo))|(v>hi-.02*(hi-lo))).mean()))
                assert row['violations']==0
            rows.append(row)
    return rows


def export(obj,dest,fit):
    s,b,p=obj.last,obj.b,obj.properties;active=b['domain_mask']
    cold,_,_=obj.solver.solve(s['A'],b['T_bot'],b['surface_T'],None)
    cold_error=float(abs(cold[active]-s['T'][active]).max());assert cold_error<5e-5
    matrix,rhs=obj.solver.linear_solver._build_matrix_and_rhs(s['K'],s['A'],s['T'],b['T_bot'])
    keep=np.broadcast_to(active[...,None],s['T'][...,1:-1].shape).ravel()
    residual=float(np.linalg.norm((matrix@s['T'][...,1:-1].ravel()-rhs)[keep])/np.linalg.norm(rhs[keep]));assert residual<2e-10
    area=400e6/b['horizontal_scale']**2;flux=upward_flux(s['T'],s['K'],.5)
    qt,qb=(np.sum(flux[active,i]*area[active])/1000 for i in [0,-1]);generated=np.sum(s['A'][active,1:-1]*area[active,None])*500
    energy=float(abs(qt-qb-generated)/max(abs(qt),1));assert energy<1e-7
    for iz,key in [(0,'surface_T'),(-1,'T_bot')]:np.testing.assert_array_equal(s['T'][...,iz],b[key])
    ranges=branch_summary(obj,s['T']);pd.DataFrame(ranges).to_csv(dest/'material_ranges.csv',index=False)
    info=dict(case=obj.case,n_parameters=p.n,n_A=sum(n.startswith('A_') for n in p.names),
        n_K=sum(n.startswith('K_') for n in p.names),fold=obj.fold,regularization=obj.regularization,qfactor=obj.qfactor,
        iterations=int(fit.nit),evaluations=int(fit.nfev),optimizer_success=bool(fit.success),message=str(fit.message),
        coefficient_gradient_inf=float(abs(s['objective_gradient']).max()),
        gradient=metrics(obj.records.gradient_C_km.to_numpy(),s['gradient'],obj.records.cell_weight.to_numpy()),
        heatflow=metrics(obj.heatflow.q_mW_m2.to_numpy(),s['heatflow']),
        curie_RMSE_km=float(np.sqrt(np.mean((s['depth']-obj.curie.observed.to_numpy())**2))),
        objective=s['objective'],losses={k:s[k] for k in ['gradient_loss','heatflow_loss','curie_loss','regularization_loss']},
        cold_replay_C=cold_error,PDE_residual=residual,energy_imbalance=energy,
        T_active_range_C=[float(s['T'][active].min()),float(s['T'][active].max())],physical_range_violations=0,
        coefficient_bound_hits=int((abs(fit.x)>4-1e-7).sum()),main_model_replaced=False,uncertainty_computed=False)
    if obj.fold>=0:
        for key,table,mask in [('q',obj.heatflow,obj.qtest),('g',obj.records,obj.gtest)]:
            obs=table.q_mW_m2.to_numpy() if key=='q' else table.gradient_C_km.to_numpy()
            pred=s['heatflow'] if key=='q' else s['gradient']
            info[key+'_holdout']=metrics(obs[mask],pred[mask])
    write_json(dest/'summary.json',info);np.savez_compressed(dest/'solution.npz',**s)
    pd.DataFrame(dict(name=p.names,coefficient=fit.x,regularization_scale=p.scales)).to_csv(dest/'coefficients.csv',index=False)
    for key,table in [('gradient',obj.records),('heatflow',obj.heatflow)]:
        frame=table.copy();frame['prediction']=s[key];frame['train']=obj.gtrain if key=='gradient' else obj.qtrain
        frame['test']=obj.gtest if key=='gradient' else obj.qtest;frame.to_csv(dest/(key+'_predictions.csv'),index=False)
    pd.DataFrame(dict(observed=obj.curie.observed,predicted=s['depth'])).to_csv(dest/'curie_predictions.csv',index=False)
    with h5py.File(dest/'fields.h5','x') as h:
        h.attrs.update(case=obj.case,main_model_replaced=False,uncertainty='not computed',K20_definition='20 C, zero depth factor; mantle normalized at20 C')
        for key in ['x','y','longitude','latitude','z_km','core','domain_mask','masks','thermal_interfaces_km']:
            h.create_dataset('grid/'+key,data=b[key],compression='gzip')
        for key,array in [('T_C',s['T']),('K_W_mK',s['K']),('A_uW_m3',s['A']*1e6),('K20_W_mK',p.K20),('q_surface_mW_m2',s['heatflow_field'])]:
            h.create_dataset('solution/'+key,data=array,compression='gzip')
        for key in ['A','K']:h.create_dataset('materials/'+key,data=p.values[key],compression='gzip')
    interface_budgets(obj,s,dest/'interfaces')
    write_json(dest/'complete.json',dict(complete=True,solution_sha256=digest(dest/'solution.npz'),fields_sha256=digest(dest/'fields.h5')))
    print(json.dumps(info),flush=True)


def summarize(paths):
    rows=[]
    for path in paths:
        path=Path(path)
        record=json.loads((path/'summary.json').read_text())
        rows.append(dict(run=str(path), parameters=record['n_parameters'],
            objective=record['objective'], PDE_residual=record['PDE_residual'],
            heatflow_RMSE=record['heatflow']['RMSE'], gradient_RMSE=record['gradient']['RMSE'],
            curie_RMSE_km=record['curie_RMSE_km']))
    return pd.DataFrame(rows)


def baselines(bundle, split, output):
    """Training-only mean and IDW32 on the recorded physical-model folds."""
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    rows=[];predictions=[]
    with np.load(split) as folds:
        for observable,filename,column,prefix in [('heatflow','heatflow.csv','q_mW_m2','q'),
                                                 ('gradient','gradient.csv','gradient_C_km','g')]:
            table=pd.read_csv(Path(bundle)/filename)
            y=table[column].to_numpy()
            weights=np.ones(len(y)) if prefix=='q' else table.cell_weight.to_numpy()
            xy=table[['X','Y']].to_numpy()
            mean=np.full(len(y),np.nan);idw=np.full(len(y),np.nan)
            for fold in range(3):
                train=folds[f'{prefix}train_{fold}'];test=folds[f'{prefix}test_{fold}']
                assert not np.any(train&test) and not np.any(np.isfinite(mean[test]))
                mean[test]=np.average(y[train],weights=weights[train])
                distance,index=cKDTree(xy[train]).query(xy[test],k=32)
                assert distance.min()>40000
                w=distance**-2;w/=w.sum(axis=1,keepdims=True)
                idw[test]=np.sum(w*y[train][index],axis=1)
            assert np.isfinite(mean).all() and np.isfinite(idw).all()
            for name,prediction in [('training_mean',mean),('IDW32',idw)]:
                rows.append(dict(model=name,observable=observable,**metrics(y,prediction,weights)))
                predictions.append(pd.DataFrame(dict(model=name,observable=observable,row=np.arange(len(y)),observed=y,predicted=prediction)))
    pd.DataFrame(rows).to_csv(output/'conditional_data_baselines.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(output/'conditional_data_baseline_predictions.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    table=summarize(args.runs)
    if args.output is not None:
        if args.output.exists(): raise FileExistsError(args.output)
        table.to_csv(args.output,index=False)
    print(table.to_string(index=False))


if __name__=='__main__':main()
