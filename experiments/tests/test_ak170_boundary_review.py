"""The review scenarios must change spatial structure without mean drift."""
import numpy as np
from experiments.model import bottom_scenario


def test_bottom_scenarios_preserve_area_mean_and_do_not_modify_input():
    y,x=np.mgrid[:35,:57];field=1250.+3*x+35*np.sin(y*.5)
    original=field.copy();core=(y>3)&(y<30)&(x>4)&(x<50)
    active=(y>1)&(y<33)&(x>1)&(x<55);area=1+.01*x
    w=area[core]/area[core].sum();mean=w@field[core]
    uniform=bottom_scenario(field,core,active,area,'uniform')
    smooth=bottom_scenario(field,core,active,area,'smooth300')
    np.testing.assert_array_equal(field,original)
    np.testing.assert_array_equal(bottom_scenario(field,core,active,area,'original'),field)
    np.testing.assert_allclose(uniform,mean,atol=1e-10)
    np.testing.assert_allclose(w@smooth[core],mean,atol=1e-10)
    assert np.std(smooth[core])<np.std(field[core])


def test_smoothing_keeps_a_constant_boundary_constant():
    field=np.full((12,20),1325.);core=np.ones_like(field,dtype=bool)
    np.testing.assert_allclose(bottom_scenario(field,core,core,np.ones_like(field),'smooth300'),field,atol=1e-10)
