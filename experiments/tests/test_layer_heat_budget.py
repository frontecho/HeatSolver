from types import SimpleNamespace
import numpy as np
from experiments.analyze import column_budgets


def test_curved_interface_constant_source_analytic_budget():
    z=np.linspace(0,10,21);shape=(3,4,len(z));A=np.full(shape,1.3e-6);K=np.full(shape,2.6)
    T=np.broadcast_to(12+.08/2.6*z*1000-1.3e-6/(2*2.6)*(z*1000)**2,shape).copy()
    B=np.linspace(.17,2.97,12).reshape(3,4);M=B+5
    solver=SimpleNamespace(dx=20000,dy=20000,horizontal_face_y=1.,horizontal_face_x=1.,horizontal_area_factor=np.ones((3,4,1)))
    result=column_budgets(T,K,A,z,B,M,solver)
    expected=np.stack([np.full_like(B,80),80-1.3*B,80-1.3*M,np.full_like(B,67)])
    np.testing.assert_allclose(result['interface_flux'],expected,atol=1e-12)
    np.testing.assert_allclose(result['closure'],0,atol=1e-12)
    np.testing.assert_array_equal(result['lateral_gain'],0.)


def test_lateral_budget_integrates_to_zero_with_insulated_sides():
    rng=np.random.default_rng(916);z=np.arange(5.);shape=(3,4,5)
    solver=SimpleNamespace(dx=20000,dy=20000,horizontal_face_y=1.,horizontal_face_x=1.,horizontal_area_factor=np.full((3,4,1),.7))
    r=column_budgets(rng.uniform(10,100,shape),rng.uniform(2,3,shape),np.zeros(shape),z,np.ones((3,4)),np.full((3,4),3),solver)
    np.testing.assert_allclose(r['lateral_gain'].sum(axis=0).sum(),0,atol=1e-13)
    # Random T is not a PDE solution; closure must reveal that discrepancy.
    assert np.max(abs(r['closure']))>1
