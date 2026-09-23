"""Generic nonlinear solver checks with no regional configuration or inputs."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
from heat_solver import HeatSolver
from nonlinear_heat_solver import NonlinearHeatSolver


def solver(z, law):
    x,y=np.meshgrid(np.arange(4)*1500.,np.arange(3)*1700.)
    shape=(*x.shape,len(z))
    linear=HeatSolver(x=x,y=y,z_nodes_km=z,surface_temp=np.full(x.shape,11.),
        K0_ref=np.full(shape,2.5),c_ref=np.zeros(shape),A_ref=np.zeros(shape),cg_rtol=1e-13)
    return NonlinearHeatSolver(linear,law,tolerance_C=1e-10,residual_tolerance=1e-12)


class Conductivity:
    def __init__(self,c):self.c=c
    def evaluate(self,T):
        K=2.5/(1+self.c*T)
        return K,-self.c*K/(1+self.c*T)


def test_constant_conductivity_with_nonzero_depth_origin():
    z=np.linspace(12.,17.,11)
    model=solver(z,Conductivity(0.))
    shape=model.linear_solver.field_shape
    top=np.full(shape[:2],11.);bottom=np.full(shape[:2],136.)
    T,K,_=model.solve(np.zeros(shape),bottom,top)
    np.testing.assert_allclose(T,np.broadcast_to(11.+25.*(z-12.),shape),atol=1e-8)
    np.testing.assert_array_equal(K,np.full(shape,2.5))
    assert model.info['nonlinear_relative_residual']<1e-12


def test_nonlinear_source_adjoint_with_supplied_constitutive_law():
    model=solver(np.linspace(0.,8.,17),Conductivity(.001))
    shape=model.linear_solver.field_shape
    A=np.full(shape,1.2e-6)
    top=np.full(shape[:2],11.);bottom=np.full(shape[:2],300.)
    T,K,KT=model.solve(A,bottom,top)
    rng=np.random.default_rng(43);gT=rng.normal(size=shape)
    gA,_,_=model.adjoint(T,K,KT,gT)
    direction=rng.normal(size=shape)*1e-6;h=1e-4
    plus=model.solve(A+h*direction,bottom,top)[0]
    minus=model.solve(A-h*direction,bottom,top)[0]
    finite=np.sum(gT*(plus-minus))/(2*h)
    np.testing.assert_allclose(finite,np.sum(gA*direction),rtol=2e-6,atol=2e-6)


def test_solver_import_has_no_repository_file_dependencies(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    root=Path(__file__).resolve().parents[1]
    for filename, symbol in [('heat_solver', 'HeatSolver'),
                             ('nonlinear_heat_solver', 'NonlinearHeatSolver'),
                             ('observations', 'PointSampler'), ('spatial_basis', 'DCTBasis'),
                             ('heat_flux', 'upward_flux'), ('fit_metrics', 'metrics')]:
        name='standalone_'+filename
        spec=importlib.util.spec_from_file_location(name,root/(filename+'.py'))
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        try:
            spec.loader.exec_module(module)
            assert symbol in vars(module)
            assert not set(['ROOT','DATA_LOCATIONS','K_BOUNDS','A_BOUNDS','NCCProblem','Fixed60HeatSolver','SunPicard']) & vars(module).keys()
        finally:
            del sys.modules[name]
