"""NCC paper model: material assumptions, observation losses and inverse cases.

Five material units, their bounds, spatial mode layouts, reference-production
budgets, magnetic isotherm and basal scenarios are experiment definitions here.
The generic heat solver receives fields and a constitutive law explicitly.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import brentq
from scipy.sparse import load_npz
from scipy.special import expit, logit
from heat_solver import HeatSolver
from nonlinear_heat_solver import NonlinearHeatSolver
from spatial_basis import DCTBasis
from observations import PointSampler
from heat_flux import upward_flux, conductivity_pullback, explicit_flux, surface_flux_temperature_gradient
from project import ROOT, digest



THERMAL_LAYER_NAMES = ['sediment', 'upper_crust', 'middle_crust', 'lower_crust', 'mantle']


LAYERS = ['sediment', 'upper_crust', 'middle_crust', 'lower_crust', 'mantle']


K_BOUNDS = np.array([[1.2, 4.0], [2.5, 3.3], [2.0, 2.8], [1.8, 2.5], [3.0, 4.5]])


A_BOUNDS = np.array([[.2, 3.0], [.5, 3.0], [.2, 1.5], [.05, .8], [.005, .05]])


BETA = np.array([.001, .001, .0005, .00015])


def mantle_factor(T):
    tk = np.asarray(T) + 273.15
    if np.any(tk <= 0):
        raise ValueError('Nonphysical absolute temperature')
    f = 1.23e-10*tk**3 + 1/(.518+7.89e-4*tk)
    df = 3*1.23e-10*tk**2 - 7.89e-4/(.518+7.89e-4*tk)**2
    f20 = 1.23e-10*293.15**3 + 1/(.518+7.89e-4*293.15)
    return f/f20, df/f20


class BoundedProperties:
    def __init__(self, masks, z, K_reference, A_reference, dc=False,
                 vary_K=False, fine_K=False):
        self.masks = np.asarray(masks)
        self.z = np.asarray(z)
        self.raw = dict(K=np.asarray(K_reference), A=np.asarray(A_reference))
        self.bounds = dict(K=K_BOUNDS, A=A_BOUNDS)
        self.offset = {}
        for key in ['A', 'K']:
            lo, hi = self.bounds[key].T
            ref = self.raw[key]
            if ref.shape != masks.shape or not np.isfinite(ref).all():
                raise ValueError('Expected finite five-material 3D references')
            fraction = (ref-lo[:,None,None,None])/(hi-lo)[:,None,None,None]
            # A declared initialization projection only, not a posterior clip.
            self.offset[key] = logit(np.clip(fraction, .05, .95))
        self.names, self.blocks, scales = [], [], []
        ny, nx = masks.shape[1:3]
        layouts = dict(A=[(3,5),(5,9),(4,7),(3,5),None],
                       K=[(9,17) if fine_K else (5,9),(5,9) if fine_K else (3,5),(2,3),(2,2),(1,1)])
        for key in ['A', 'K']:
            if key == 'K' and not vary_K:
                continue
            for layer, shape in enumerate(layouts[key]):
                if shape is None:
                    continue
                my, mx = shape
                modes = [(y,x) for y in range(my) for x in range(mx) if dc or (y,x)!=(0,0)]
                if not modes:
                    continue
                ids = [y*mx+x for y,x in modes]
                basis = DCTBasis(ny,nx,my,mx).decode_layers(np.eye(my*mx))[ids]
                start = len(self.names)
                self.names += [f'{key}_{LAYERS[layer]}_y{y}_x{x}' for y,x in modes]
                self.blocks.append((key,layer,slice(start,len(self.names)),basis))
                # Penalize short wavelengths with a fixed physical scale, not
                # number-of-modes-dependent coefficient envelope allocation.
                scales.extend([(1.0 if key=='A' else .7)/np.sqrt(1+(y/4)**2+(x/8)**2)
                               for y,x in modes])
        self.n = len(self.names)
        self.scales = np.array(scales)
        self.set(np.zeros(self.n))

    def set(self,c):
        c = np.asarray(c)
        if c.shape != (self.n,) or not np.isfinite(c).all():
            raise ValueError('Wrong finite coefficient vector')
        fields = {key:np.zeros(self.masks.shape[:3]) for key in ['A','K']}
        for key,layer,sl,basis in self.blocks:
            fields[key][layer] = np.einsum('i,ijk->jk',c[sl],basis)
        self.values, self.derivatives = {}, {}
        for key in ['A','K']:
            lo,hi = self.bounds[key].T
            fraction = expit(self.offset[key]+fields[key][...,None])
            self.values[key] = lo[:,None,None,None]+(hi-lo)[:,None,None,None]*fraction
            self.derivatives[key] = (hi-lo)[:,None,None,None]*fraction*(1-fraction)
        self.A = np.sum(self.masks*self.values['A'],axis=0)*1e-6
        self.K20 = 1/np.sum(self.masks/self.values['K'],axis=0)
        self.fields = fields

    def factors(self,T):
        denominator = 1+BETA[:,None,None,None]*(T[None]-20)
        if np.any(denominator<=0):
            raise ValueError('Nonpositive crustal conductivity denominator')
        pressure = 1+.0015*self.z
        crust = pressure/denominator
        crust_T = -BETA[:,None,None,None]*pressure/denominator**2
        mantle, mantle_T = mantle_factor(T)
        return np.concatenate([crust,mantle[None]]),np.concatenate([crust_T,mantle_T[None]])

    def evaluate(self,T):
        factors,dt = self.factors(T)
        branch = self.values['K']*factors
        K = 1/np.sum(self.masks/branch,axis=0)
        KT = K*K*np.sum(self.masks*dt/(self.values['K']*factors**2),axis=0)
        return K,KT

    def pullback(self,T,K,ga,gK,qadj,flux_derivative):
        factors,_ = self.factors(T)
        result = np.zeros(self.n)
        for key,layer,sl,basis in self.blocks:
            if key=='A':
                column = np.sum(ga*self.masks[layer]*self.derivatives['A'][layer],axis=-1)*1e-6
            else:
                dk = K*K*self.masks[layer]*self.derivatives['K'][layer]/(self.values['K'][layer]**2*factors[layer])
                column = np.sum(gK*dk,axis=-1)+qadj*flux_derivative(T,K,dk,.5)[...,0]
            result[sl] = np.einsum('ijk,jk->i',basis,column)
        return result


class BudgetProperties(BoundedProperties):
    def __init__(self,parent,volume_weights):
        self.__dict__.update(parent.__dict__)
        weights=self.masks*volume_weights[None]
        self.budget_indices=[];self.budget_weights=[];self.reference_means=[]
        for i in range(4):
            indices=np.flatnonzero(weights[i].ravel()>0);w=weights[i].ravel()[indices];w/=w.sum()
            self.budget_indices.append(indices);self.budget_weights.append(w)
            self.reference_means.append(float(w@self.values['A'][i].ravel()[indices]))
        self.reference_means=np.array(self.reference_means)
        # A DC would be exactly canceled by the budget correction: remove it.
        names=[];blocks=[];scales=[]
        for key,layer,sl,basis in self.blocks:
            keep=np.array([not(key=='A' and n.endswith('_y0_x0')) for n in self.names[sl]])
            if not keep.any():continue
            start=len(names);names.extend(np.array(self.names[sl])[keep].tolist());scales.extend(self.scales[sl][keep])
            blocks.append((key,layer,slice(start,len(names)),basis[keep].copy()))
        self.names=names;self.blocks=blocks;self.scales=np.array(scales);self.n=len(names)
        self.set(np.zeros(self.n))

    def set(self,c):
        super().set(c);self.budget_shifts=np.zeros(4);self.budget_basis_means={}
        for i in range(4):
            lo,hi=A_BOUNDS[i];ids=self.budget_indices[i];w=self.budget_weights[i]
            logits=self.offset['A'][i]+self.fields['A'][i,...,None]
            v=logits.ravel()[ids];target=(self.reference_means[i]-lo)/(hi-lo)
            def fun(shift):return float(w@expit(v+shift)-target)
            shift=0. if abs(fun(0.))<1e-14 else brentq(fun,-float(v.max())-40,-float(v.min())+40,xtol=1e-12)
            fraction=expit(logits+shift)
            self.values['A'][i]=lo+(hi-lo)*fraction
            self.derivatives['A'][i]=(hi-lo)*fraction*(1-fraction)
            self.budget_shifts[i]=shift;self.fields['A'][i]+=shift
            np.testing.assert_allclose(w@self.values['A'][i].ravel()[ids],self.reference_means[i],atol=2e-12)
        self.A=np.sum(self.masks*self.values['A'],axis=0)*1e-6

    def pullback(self,T,K,ga,gK,qadj,flux_derivative):
        result=super().pullback(T,K,ga,gK,qadj,flux_derivative)
        for key,layer,sl,basis in self.blocks:
            if key!='A':continue
            ids=self.budget_indices[layer];w=self.budget_weights[layer]
            sensitivity=self.derivatives['A'][layer]
            denominator=float(w@sensitivity.ravel()[ids])
            if denominator<=0:raise ValueError('Degenerate mean-conserving mapping')
            weighted=np.zeros_like(sensitivity);weighted.ravel()[ids]=w*sensitivity.ravel()[ids]/denominator
            basis_average=np.einsum('ijk,jk->i',basis,weighted.sum(axis=-1))
            total=float(np.sum(ga*self.masks[layer]*sensitivity))*1e-6
            result[sl]-=total*basis_average
        return result


def refine_A(properties):
    p=properties;ny,nx=p.masks.shape[1:3];names=[];blocks=[];scales=[]
    target={1:(9,17),2:(5,9)}
    dc=any(n.endswith('_y0_x0') for n in p.names if n.startswith('A_'))
    for key,layer,sl,basis in p.blocks:
        block_names=list(p.names[sl]);block_scales=list(p.scales[sl]);basis=basis.copy()
        if key=='A' and layer in target:
            my,mx=target[layer];modes=[(y,x) for y in range(my) for x in range(mx) if dc or (y,x)!=(0,0)]
            extra=[(y,x) for y,x in modes if f'A_{LAYERS[layer]}_y{y}_x{x}' not in block_names]
            allbasis=DCTBasis(ny,nx,my,mx).decode_layers(np.eye(my*mx))
            basis=np.concatenate([basis,allbasis[[y*mx+x for y,x in extra]]])
            block_names += [f'A_{LAYERS[layer]}_y{y}_x{x}' for y,x in extra]
            block_scales += [1/np.sqrt(1+(y/4)**2+(x/8)**2) for y,x in extra]
        start=len(names);names+=block_names;scales+=block_scales
        blocks.append((key,layer,slice(start,len(names)),basis))
    p.names=names;p.blocks=blocks;p.scales=np.array(scales);p.n=len(names);p.set(np.zeros(p.n))
    return p


def curie_value_gradient(profiles, depths, observed, precision):
    """Depth residual and its gradient with respect to interpolated profiles."""
    above = profiles >= 580.
    j = above.argmax(axis=1)
    if not np.all(above.any(axis=1) & (j > 0)):
        raise ValueError('Fixed boundaries did not provide a supported Curie crossing')
    row = np.arange(len(j))
    lo, hi = profiles[row, j-1], profiles[row, j]
    fraction = (580-lo)/(hi-lo)
    dz = depths[j]-depths[j-1]
    prediction = depths[j-1]+fraction*dz
    residual = prediction-observed
    weighted = precision@residual
    gradient = np.zeros_like(profiles)
    gradient[row, j-1] = -weighted*dz*(1-fraction)/(hi-lo)
    gradient[row, j] = -weighted*dz*fraction/(hi-lo)
    return .5*residual@weighted, gradient, prediction


def bottom_scenario(field, core, active, area, case, spacing_km=20.):
    """Preserve the core area mean while removing or smoothing basal anomalies."""
    mean = float(np.sum(field[core]*area[core])/area[core].sum())
    if case == 'original':
        return field.copy()
    if case == 'uniform':
        return np.full_like(field, mean)
    if case == 'smooth300':
        weight = area*active
        sigma = 300/spacing_km
        denominator = gaussian_filter(weight, sigma, mode='constant')
        if np.any(denominator <= 0):
            raise ValueError('No support for normalized basal smoothing')
        smoothed = gaussian_filter(field*weight, sigma, mode='constant')/denominator
        smoothed += mean-np.sum(smoothed[core]*area[core])/area[core].sum()
        return smoothed
    raise ValueError(case)


def new_solver(b, K, surface=None):
    return HeatSolver(x=b["x"], y=b["y"], z_nodes_km=b["z_km"],
        surface_temp=b["surface_T"] if surface is None else surface,
        K0_ref=K, c_ref=np.zeros_like(K), A_ref=np.zeros_like(K),
        horizontal_scale=b["horizontal_scale"], domain_mask=b.get('domain_mask'), cg_rtol=1e-11)


def build_solver(background, law, tolerance_C=2e-5, residual_tolerance=2e-10, max_iterations=45):
    """Bind this experiment's prepared grid to the generic nonlinear solver."""
    linear = new_solver(background, np.full_like(background['A'], 2.))
    return NonlinearHeatSolver(linear, law, tolerance_C, residual_tolerance, max_iterations)



class InverseProblem:
    """A fixed-boundary regional inverse problem loaded from explicit inputs."""

    def __init__(self, bundle, *, variant='AK170_budget', regularization=None,
                 fold=-1, split=None, qfactor=1., boundary='original'):
        bundle = Path(bundle)
        manifest = json.loads((bundle/'manifest.json').read_text())
        for name, checksum in manifest['data_sha256'].items():
            if digest(bundle/name) != checksum:
                raise ValueError(f'Input checksum mismatch: {name}')
        self.b = dict(np.load(bundle/'background.npz'))
        b = self.b
        cases = {'A99': (False, False), 'A224_bounded': (False, False),
                 'AK165': (False, True), 'AK174': (True, True),
                 'AK299': (True, True), 'AK170_budget': (True, True),
                 'AK295_budget': (True, True)}
        dc, vary = cases[variant]
        with np.load(bundle/'material_references.npz') as raw:
            p = BoundedProperties(b['masks'], b['z_km'], raw['K'], raw['A'], dc=dc, vary_K=vary)
            if variant in ['A224_bounded', 'AK299', 'AK295_budget']:
                refine_A(p)
            if variant in ['AK170_budget', 'AK295_budget']:
                volume = (400/b['horizontal_scale']**2*b['core'])[...,None]*np.r_[.25,np.full(len(b['z_km'])-2,.5),.25]
                p = BudgetProperties(p, volume)
                np.testing.assert_array_equal(p.reference_means, raw['reference_means'])
        self.properties = self.law = p
        self.n = p.n
        if variant == 'AK170_budget':
            assert p.names == manifest['names'] and p.n == manifest['n']
        self.original_bottom = b['T_bot'].copy()
        b['T_bot'] = bottom_scenario(self.original_bottom, b['core'], b['domain_mask'],
                                    400/b['horizontal_scale']**2, boundary)
        self.case = variant
        self.reference = {'bundle': str(bundle), 'manifest_sha256': digest(bundle/'manifest.json')}
        self.regularization = manifest['regularization'] if regularization is None else regularization
        self.fold, self.qfactor = fold, qfactor
        self.solver = build_solver(b, p, tolerance_C=3e-7)
        self.solver.linear_solver.cg_rtol = 1e-12
        for name, shape in [('g', b['A'].shape), ('q', b['core'].shape), ('curie', b['core'].shape)]:
            setattr(self, name+'_op', PointSampler(shape, load_npz(bundle/(name+'_operator.npz'))))
        self.records = pd.read_csv(bundle/'gradient.csv')
        self.heatflow = pd.read_csv(bundle/'heatflow.csv')
        self.curie = pd.read_csv(bundle/'curie.csv')
        with np.load(bundle/'observation_weights.npz') as arrays:
            for name in arrays.files:
                setattr(self, name, arrays[name])
        self.qtrain = np.ones(len(self.heatflow), bool)
        self.gtrain = np.ones(len(self.records), bool)
        self.qtest, self.gtest = ~self.qtrain, ~self.gtrain
        if fold >= 0:
            with np.load(split) as arrays:
                for name in ['qtrain', 'gtrain', 'qtest', 'gtest']:
                    setattr(self, name, arrays[f'{name}_{fold}'])
            if np.any(self.qtrain & self.qtest) or np.any(self.gtrain & self.gtest):
                raise ValueError('Training and held-out sets overlap')
        self.qweight = self.qtrain/self.qsigma**2*qfactor
        self.gweight = self.gtrain/self.error**2
        with np.load(bundle/'selected_state.npz') as selected:
            self.initial_coefficients = selected['coefficients'].copy()
            self.initial_temperature = selected['T'].copy()
        self.last = None



    def evaluate(self,c,gradient=True,warm=True):
        p,b=self.properties,self.b;p.set(c);A=p.A
        initial=self.last['T'] if warm and self.last is not None else None
        T,K,KT=self.solver.solve(A,b['T_bot'],b['surface_T'],initial)
        pred=self.g_op.sample(T);rg=pred-self.records.gradient_C_km.to_numpy()
        profiles=self.curie_op.matrix@T.reshape(-1,T.shape[-1])
        closs,gprofile,depth=curie_value_gradient(profiles,b['z_km'],self.curie.observed.to_numpy(),self.precision)
        q=upward_flux(T,K,.5)[...,0]+250000*A[...,0]
        qp=self.q_op.sample(q);rq=qp-self.heatflow.q_mW_m2.to_numpy()
        lg=.5*float(self.gweight@rg**2);lq=.5*float(self.qweight@rq**2)
        penalty=.5*self.regularization*float(np.sum((c/p.scales)**2))
        value=lg+lq+closs+penalty
        self.last=dict(A=A,K=K,K_T=KT,T=T,top=b['surface_T'],bottom=b['T_bot'],coefficients=c.copy(),
            gradient=pred,heatflow=qp,heatflow_field=q,depth=depth,profiles=profiles,
            gradient_loss=lg,heatflow_loss=lq,curie_loss=float(closs),regularization_loss=penalty,objective=value)
        if not gradient:return value
        qadj=(self.q_op.matrix.T@(self.qweight*rq)).reshape(T.shape[:2])
        grad_T=(self.g_op.matrix.T@(self.gweight*rg)).reshape(T.shape)+(self.curie_op.matrix.T@gprofile).reshape(T.shape)
        grad_T+=surface_flux_temperature_gradient(T,K,KT,qadj,.5)
        ga,_,_=self.solver.adjoint(T,K,KT,grad_T)
        lam=np.zeros_like(T);lam[...,1:-1]=ga[...,1:-1]/self.solver.linear_solver.horizontal_area_factor
        gK=conductivity_pullback(T,K,lam,self.solver) if any(n.startswith('K_') for n in p.names) else np.zeros_like(T)
        ga[...,0]+=250000*qadj
        g=p.pullback(T,K,ga,gK,qadj,explicit_flux)+self.regularization*c/p.scales**2
        self.last['objective_gradient']=g
        return value,g
