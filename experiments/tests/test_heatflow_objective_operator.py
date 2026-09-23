"""Surface observation derivative includes temperature-dependent harmonic K."""
import numpy as np
from heat_flux import surface_flux_temperature_gradient
from heat_flux import upward_flux


def test_surface_flux_transpose_temperature_derivative():
    rng = np.random.default_rng(236)
    T = rng.uniform(5, 200, (3, 4, 5))
    K0 = rng.uniform(1, 4, T.shape)
    beta = .001
    K = K0/(1+beta*T)
    KT = -beta*K0/(1+beta*T)**2
    weights = rng.normal(size=T.shape[:2])
    direction = rng.normal(size=T.shape)
    gradient = surface_flux_temperature_gradient(T, K, KT, weights, .5)
    def value(t):
        return np.sum(weights*upward_flux(t, K0/(1+beta*t), .5)[..., 0])
    eps = 1e-4
    finite = (value(T+eps*direction)-value(T-eps*direction))/(2*eps)
    np.testing.assert_allclose(np.sum(gradient*direction), finite, rtol=1e-8)
    np.testing.assert_array_equal(gradient[..., 2:], np.zeros_like(gradient[..., 2:]))
