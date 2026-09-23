import numpy as np

from experiments.model import BoundedProperties, K_BOUNDS, A_BOUNDS, mantle_factor

from heat_flux import explicit_flux

from heat_flux import upward_flux

from experiments.model import BudgetProperties

from experiments.model import refine_A

import unittest

from spatial_basis import DCTBasis

def example(dc=True,vary=True,fine=False):
    rng=np.random.default_rng(916)
    masks=rng.uniform(.05,1,(5,10,18,6));masks/=masks.sum(axis=0)
    K=np.broadcast_to(K_BOUNDS.mean(axis=1)[:,None,None,None],masks.shape).copy()
    A=np.broadcast_to(A_BOUNDS.mean(axis=1)[:,None,None,None],masks.shape).copy()
    A[4]=.03
    return BoundedProperties(masks,np.arange(6)*.5,K,A,dc,vary,fine)

def test_bounds_hold_for_extreme_fields_and_mantle_A_stays_fixed():
    p=example();rng=np.random.default_rng(91);p.set(rng.normal(0,20,p.n))
    for key in ['K','A']:
        lo,hi=p.bounds[key].T
        assert np.all(p.values[key]>=lo[:,None,None,None])
        assert np.all(p.values[key]<=hi[:,None,None,None])
    np.testing.assert_allclose(p.values['A'][4],.03)
    np.testing.assert_allclose(p.A,np.sum(p.values['A']*p.masks,axis=0)*1e-6)
    assert p.n==174 and example(False,True).n==165
    assert example(False,False).n==99 and example(True,True,True).n==312

def test_room_definition_pressure_and_temperature_derivative():
    p=example();T=np.full(p.A.shape,20.)
    factors,dt=p.factors(T)
    np.testing.assert_allclose(factors[4],1.)
    np.testing.assert_allclose(factors[:4],np.broadcast_to(1+.0015*p.z,factors[:4].shape))
    np.testing.assert_allclose(mantle_factor(20)[0],1.)
    K,KT=p.evaluate(T);h=1e-3
    np.testing.assert_allclose((p.evaluate(T+h)[0]-p.evaluate(T-h)[0])/(2*h),KT,rtol=1e-7,atol=1e-10)
    np.testing.assert_allclose(K[...,0],p.K20[...,0])

def test_property_pullback_including_explicit_surface_flux():
    p=example();rng=np.random.default_rng(17);c=rng.normal(0,.04,p.n);p.set(c)
    T=np.broadcast_to(np.arange(6)*20.,p.A.shape).copy();K,_=p.evaluate(T)
    ga=rng.normal(size=T.shape)*1e6;gK=rng.normal(size=T.shape);qadj=rng.normal(size=T.shape[:2])
    def objective(coeff):
        p.set(coeff);kk,_=p.evaluate(T)
        return np.sum(ga*p.A)+np.sum(gK*kk)+np.sum(qadj*upward_flux(T,kk,.5)[...,0])
    exact=p.pullback(T,K,ga,gK,qadj,explicit_flux)
    for key in ['A','K']:
        d=np.array([rng.normal() if n.startswith(key+'_') else 0 for n in p.names]);d/=np.linalg.norm(d)
        h=1e-5;fd=(objective(c+h*d)-objective(c-h*d))/(2*h)
        np.testing.assert_allclose(fd,exact@d,rtol=1e-6,atol=2e-7)

def test_no_constant_modes_and_zero_projection_exactly_preserves_interior():
    p=example(False,False)
    assert not any(n.endswith('_y0_x0') for n in p.names)
    for key in ['K','A']:np.testing.assert_allclose(p.values[key],p.raw[key],atol=1e-14)

def test_budget_is_preserved_at_large_spatial_perturbations():
    old=example();volume=np.random.default_rng(17).uniform(.1,2,old.A.shape)
    p=BudgetProperties(old,volume);assert p.n==170
    assert not any(n.startswith('A_') and n.endswith('_y0_x0') for n in p.names)
    p.set(np.random.default_rng(3).normal(0,1.3,p.n))
    for i in range(4):
        w=p.masks[i]*volume
        np.testing.assert_allclose(np.sum(w*p.values['A'][i])/w.sum(),p.reference_means[i],atol=2e-12)
        assert p.values['A'][i].min()>=p.bounds['A'][i,0]
        assert p.values['A'][i].max()<=p.bounds['A'][i,1]

def test_pullback_includes_implicit_budget_shift():
    old=example();rng=np.random.default_rng(19);volume=rng.uniform(.1,2,old.A.shape)
    p=BudgetProperties(old,volume);c=rng.normal(0,.15,p.n);p.set(c)
    T=np.broadcast_to(np.arange(6)*20.,p.A.shape).copy();K,_=p.evaluate(T)
    ga=rng.normal(size=T.shape)*1e6;gK=rng.normal(size=T.shape);qadj=rng.normal(size=T.shape[:2])
    gradient=p.pullback(T,K,ga,gK,qadj,explicit_flux)
    def f(x):
        p.set(x);k,_=p.evaluate(T)
        return np.sum(ga*p.A)+np.sum(gK*k)+np.sum(qadj*upward_flux(T,k,.5)[...,0])
    for key in ['A','K']:
        d=np.array([rng.normal() if n.startswith(key+'_') else 0 for n in p.names]);d/=np.linalg.norm(d)
        h=1e-5;fd=(f(c+h*d)-f(c-h*d))/(2*h)
        np.testing.assert_allclose(fd,gradient@d,rtol=2e-6,atol=5e-7)

def test_total_production_derivative_is_zero_and_zero_state_preserved():
    old=example();A=old.A.copy();volume=np.ones_like(A);p=BudgetProperties(old,volume)
    np.testing.assert_allclose(p.A,A,atol=1e-20)
    p.set(np.random.default_rng(111).normal(0,.3,p.n));T=np.zeros_like(A)+20;K,_=p.evaluate(T)
    g=p.pullback(T,K,volume,np.zeros_like(A),np.zeros(A.shape[:2]),explicit_flux)
    np.testing.assert_allclose(g,0,atol=1e-18)

def test_nested_refinement_preserves_old_fields_and_penalty():
    old=example();fine=refine_A(example());rng=np.random.default_rng(299)
    c=rng.normal(0,.05,old.n);mapping=dict(zip(old.names,c));expanded=np.array([mapping.get(n,0.) for n in fine.names])
    old.set(c);fine.set(expanded)
    assert fine.n==299
    for key in ['A','K']:np.testing.assert_allclose(fine.values[key],old.values[key],rtol=2e-15,atol=2e-15)
    np.testing.assert_allclose(np.sum((c/old.scales)**2),np.sum((expanded/fine.scales)**2),atol=1e-14)
    assert refine_A(example(False,False)).n==224
    assert refine_A(example(True,True,True)).n==437

def test_each_layer_has_one_block_and_all_modes_are_unique():
    p=refine_A(example())
    assert len(p.names)==len(set(p.names))
    assert len([(k,l) for k,l,_,_ in p.blocks])==len(set((k,l) for k,l,_,_ in p.blocks))
    assert all(len(p.names[sl])==len(basis) for _,_,sl,basis in p.blocks)
