import numpy as np

import pytest

from experiments.exploratory.observation_constraints.modules.gradient_observations import select_gradients

from observations import gradient_operator

from observations import Observations, PointSampler, equal_group_weights

def test_interval_gradient_units_and_surface_offset_cancellation():
    axes=(np.array([0.,20000]),np.array([0.,20000]),np.array([0.,.5,1.,2.]))
    y,x,z=np.meshgrid(*axes,indexing='ij')
    op=gradient_operator(axes,[[10000,6000],[1000,9000]],[.073,.4],[.394,1.7])
    field=11+25*z+1e-4*x+2e-4*y
    np.testing.assert_allclose(op.sample(field),25,atol=1e-12)
    np.testing.assert_allclose(op.sample(np.ones_like(field)),0,atol=1e-12)
    rng=np.random.default_rng(1)
    perturb=rng.normal(size=field.shape)
    v=np.array([2.,-3.])
    np.testing.assert_allclose(v@op.sample(perturb),np.sum(op.transpose(v)*perturb),atol=1e-12)
    with pytest.raises(ValueError):
        gradient_operator(axes,[[0,0]],[.5],[.4])

def test_trilinear_multiaffine_field_and_boundary_points():
    axes = (np.array([-2., 0., 3.]), np.array([1., 4., 9.]), np.array([0., .2, 1.]))
    y, x, z = np.meshgrid(*axes, indexing='ij')
    field = 3 + y + 2*x - z + y*x + 2*x*z + y*x*z
    points = np.array([[-1, 2, .1], [3, 9, 1], [-2, 1, 0], [1, 6, .6]])
    p, q, r = points.T
    expected = 3 + p + 2*q - r + p*q + 2*q*r + p*q*r
    np.testing.assert_allclose(PointSampler.build(axes, points).sample(field), expected, atol=1e-12)

def test_sampling_transpose_accumulates_repeated_points():
    rng = np.random.default_rng(5)
    axes = (np.arange(3.), np.arange(4.))
    points = np.array([[.5, .4], [.5, .4], [2, 3]])
    op = PointSampler.build(axes, points)
    field, gradient = rng.normal(size=(3,4)), rng.normal(size=3)
    np.testing.assert_allclose(op.sample(field) @ gradient, np.sum(field * op.transpose(gradient)), atol=1e-14)

def test_weighted_misfit_gradient_and_group_balance():
    axes = (np.arange(3.), np.arange(3.))
    op = PointSampler.build(axes, np.array([[0, 0], [1, 1], [.2, .5]]))
    weights = equal_group_weights(np.array(['a', 'a', 'b']))
    np.testing.assert_allclose(weights, [.5, .5, 1])
    obs = Observations(op, np.array([2., 3., 4.]), np.array([1., 2., 3.]), weights)
    field = np.arange(9.).reshape(3,3)
    direction = np.linspace(-1, 1, 9).reshape(3,3)
    _, gradient, _ = obs.loss(field)
    step = 1e-5
    fd = (obs.loss(field + step*direction)[0] - obs.loss(field - step*direction)[0])/(2*step)
    np.testing.assert_allclose(np.sum(gradient * direction), fd, rtol=1e-8)

def test_reject_outside_domain_and_zero_uncertainty():
    axes = (np.arange(3.), np.arange(3.))
    with pytest.raises(ValueError, match='outside'):
        PointSampler.build(axes, np.array([[3., 1.]]))
    op = PointSampler.build(axes, np.array([[1., 1.]]))
    with pytest.raises(ValueError, match='positive'):
        Observations(op, np.ones(1), np.zeros(1), np.ones(1))
