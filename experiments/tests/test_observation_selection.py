import numpy as np
import pandas as pd
import pytest
from Data.processed.ncc_preparation.sampling import select_gradients, gradient_errors


def source():
    return pd.DataFrame(dict(id=range(6),g=['100-500','1～5000','-','900-400','0-5100','20-200'],
        h=[20,30,25,20,30,None],i=[.2,8,None,1,1,None],X=[0,1000,0,0,0,0],Y=0,
        Heat_Flow=[50,60,70,80,90,100]))


def test_selection_is_depth_matched_and_independent_of_heatflow_value():
    raw=source()
    audit, rows=select_gradients(raw)
    assert rows.id.tolist()==[0,1]
    assert audit.reason.iloc[2]=='invalid_or_missing_depth_interval;'
    assert 'below_5km' in audit.reason.iloc[4]
    raw['Heat_Flow']=np.nan
    again=select_gradients(raw)[1]
    np.testing.assert_array_equal(again.id,rows.id)
    np.testing.assert_array_equal(again.gradient_C_km,rows.gradient_C_km)
    np.testing.assert_allclose(rows.cell_weight,[.5,.5])
    sigma,effective=gradient_errors(rows)
    np.testing.assert_allclose(sigma,[5,8])
    np.testing.assert_allclose(effective,np.array([5,8])*np.sqrt(2))


def test_repeating_a_cell_does_not_multiply_likelihood_precision():
    raw=source().iloc[[0]].copy()
    one=select_gradients(raw)[1]
    many=select_gradients(pd.concat([raw]*10,ignore_index=True))[1]
    assert np.sum(1/gradient_errors(many)[1]**2)==pytest.approx(np.sum(1/gradient_errors(one)[1]**2))
