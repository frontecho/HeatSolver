from types import SimpleNamespace
import numpy as np
import pytest
from project import digest
from experiments.run import initialize


def test_holdout_rejects_a_fitted_initial_state():
    obj=SimpleNamespace(n=2,fold=0)
    with pytest.raises(ValueError,match='untrained reference'):
        initialize(obj,{'initial_coefficients':[.1,0.]})


def test_boundary_warm_start_is_verified_and_adjusted(tmp_path):
    source=tmp_path/'state.npz'
    T=np.arange(12,dtype=float).reshape(2,2,3)
    np.savez(source,T=T)
    config={'initial_coefficients':[0.,0.], 'initial_temperature':{
        'path':str(source),'sha256':digest(source),'adjust_boundary':True}}
    obj=SimpleNamespace(n=2,fold=-1,original_bottom=np.zeros((2,2)),
        b={'T_bot':np.full((2,2),20.),'z_km':np.array([0.,50.,100.]),'A':np.zeros_like(T)},last=None)
    initialize(obj,config)
    np.testing.assert_array_equal(obj.last['T'][...,0],T[...,0])
    np.testing.assert_array_equal(obj.last['T'][...,1],T[...,1]+10.)
    np.testing.assert_array_equal(obj.last['T'][...,-1],T[...,-1]+20.)
    np.savez(source,T=T+1)
    with pytest.raises(ValueError,match='checksum'):
        initialize(obj,config)
