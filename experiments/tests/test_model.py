import numpy as np
from experiments.model import InverseProblem
from experiments.synthetic import Problem


def test_current_objective_adjoint_matches_directional_difference():
    obj=Problem()
    assert isinstance(obj,InverseProblem) and obj.n==170
    rng=np.random.default_rng(923)
    c=rng.normal(0,.03,obj.n)
    value,gradient=obj.evaluate(c,warm=False)
    assert np.isfinite(value) and obj.solver.info['nonlinear_relative_residual']<2e-10
    direction=rng.normal(size=obj.n);direction/=np.linalg.norm(direction)
    cross=(obj.last['profiles']>=580).argmax(axis=1)
    step=2e-5
    plus=obj.evaluate(c+step*direction,False,warm=False)
    np.testing.assert_array_equal(cross,(obj.last['profiles']>=580).argmax(axis=1))
    minus=obj.evaluate(c-step*direction,False,warm=False)
    np.testing.assert_array_equal(cross,(obj.last['profiles']>=580).argmax(axis=1))
    finite=(plus-minus)/(2*step)
    np.testing.assert_allclose(finite,gradient@direction,rtol=3e-5,atol=3e-5)
    np.testing.assert_array_equal(obj.last['T'][...,0],obj.b['surface_T'])
    np.testing.assert_array_equal(obj.last['T'][...,-1],obj.b['T_bot'])
