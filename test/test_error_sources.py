import numpy as np
import pandas as pd


def publication_covariance(raw, heatflow, gradient, covariance, fraction):
    """Within-type publication effects replacing a fraction of independent variance.

    Heat-flow mixtures use the same local median influence as the data product.
    Missing/unpublished references receive record-specific groups. Heat-flow and
    gradient publication effects are separate, so no cross-type sign is assumed.
    """
    if not 0 <= fraction <= 1:
        raise ValueError('The shared variance fraction must lie in [0, 1].')
    raw = raw.copy()
    raw.index = raw.index.astype(str)
    reference = {i: str(v).strip() if pd.notna(v) and str(v).strip() not in ['', '未发表'] else 'unknown:'+i
                 for i, v in raw.n.items()}
    groups = {v: i for i, v in enumerate(sorted(set(reference.values())))}
    result = covariance.copy()
    start = 0
    for kind, table in [('heatflow', heatflow), ('gradient', gradient)]:
        factors = np.zeros((len(table), len(groups)))
        sigma = np.sqrt(np.diag(covariance)[start:start+len(table)])
        for i, row in table.iterrows():
            ids = str(row.ids).split(';') if kind == 'heatflow' else [str(table.id.iloc[i])]
            weights = median_influence(pd.to_numeric(raw.loc[ids, 'o'])) if kind == 'heatflow' else [1.]
            for record, weight in zip(ids, weights):
                factors[i, groups[reference[record]]] += weight
            factors[i] *= np.sqrt(fraction)*sigma[i]/np.linalg.norm(factors[i])
        result[start:start+len(table), start:start+len(table)] = factors@factors.T+(1-fraction)*np.diag(sigma**2)
        start += len(table)
    return result

def temperature_functional_gradient(problem, weight, depth_index):
    """Adjoint derivative of an area mean/contrast with respect to all coefficients."""
    T, K = problem.last['T'], problem.last['K']
    _, KT = problem.p.evaluate(T)
    target = np.zeros_like(T)
    target[..., depth_index] = weight
    ga = problem.solver.adjoint(T, K, KT, target)
    lam = np.zeros_like(T)
    lam[..., 1:-1] = ga[..., 1:-1]/problem.solver.area
    gK = problem.solver.conductivity_pullback(T, K, lam)
    return problem.p.pullback(T, K, ga, gK, np.zeros(T.shape[:2]), problem.b['z_km'][1]-problem.b['z_km'][0])

def beta_conductivity_direction(problem, layer, change):
    """Explicit dK from a Chapman beta change, including the harmonic layer mixture."""
    T, K, p = problem.last['T'], problem.last['K'], problem.p
    factors, _ = p.factors(T)
    denominator = 1+p.beta[layer]*(T-20)
    return -K**2*p.masks[layer]*(T-20)*change/(p.values['K'][layer]*factors[layer]*denominator)



from uncertainty import Tangent, coupled_heatflow_gradient, median_influence
from test.test_uncertainty import small_problem




def test_median_derivative_and_ties():
    for values in [[1., 3., 8.], [1., 3., 5., 8.]]:
        direction = np.array([.2, -.3, .7, .8])[:len(values)]
        step = 1e-5
        fd = (np.median(values+step*direction)-np.median(values-step*direction))/(2*step)
        np.testing.assert_allclose(median_influence(values)@direction, fd, atol=1e-10)
    np.testing.assert_allclose(median_influence([1, 3, 3, 8]), [0, .5, .5, 0])


def test_shared_observation_covariance_preserves_marginals_and_matches_simulation():
    raw = pd.DataFrame(dict(o=[20., 40., 45.], h=[10., 20., 15.], j=[2., 2., 3.], n=['paper', 'paper', 'other']), index=['1', '2', '3'])
    q = pd.DataFrame(dict(ids=['1;2', '3']))
    g = pd.DataFrame(dict(id=[1, 2, 3]))
    covariance = np.diag([100., 144., 25., 36., 49.])
    coupled = coupled_heatflow_gradient(raw, q, g, covariance, .5)
    np.testing.assert_allclose(np.diag(coupled), np.diag(covariance))
    assert np.linalg.eigvalsh(coupled).min() > 0
    # Independent latent primitive errors reconstruct the specified q/g correlations.
    rng = np.random.default_rng(194)
    z = rng.normal(size=(5, 100000))
    q1 = np.sqrt(50)*(10*z[2]+12*z[3])/np.sqrt(244)+np.sqrt(50)*z[0]
    q2 = np.sqrt(72)*(z[4]+z[1])
    draws = np.vstack([q1, q2, 5*z[2], 6*z[3], 7*z[4]])
    np.testing.assert_allclose(np.cov(draws), coupled, rtol=.025, atol=1.)
    pub = publication_covariance(raw, q, g, covariance, .5)
    np.testing.assert_allclose(np.diag(pub), np.diag(covariance))
    assert pub[2, 3] > 0 and pub[2, 4] == 0
    assert np.linalg.eigvalsh(pub).min() > 0


def test_temperature_functional_adjoint_and_beta_against_nonlinear_forward(tmp_path, monkeypatch):
    problem = small_problem(tmp_path)
    coefficients = np.linspace(-.1, .15, len(problem.p.names))
    problem.evaluate(coefficients)
    tangent = Tangent(problem)
    weight = problem.b['core'].astype(float)
    weight /= weight.sum()
    depth_index = len(problem.b['z_km'])//2
    gradient = temperature_functional_gradient(problem, weight, depth_index)
    direction = np.arange(1, len(coefficients)+1, dtype=float)
    direction /= np.linalg.norm(direction)
    dT, _ = tangent.solve(*tangent.properties(coefficients=direction))
    np.testing.assert_allclose(gradient@direction, np.sum(weight*dT[..., depth_index]), rtol=2e-8, atol=1e-8)
    baseline_beta = problem.p.beta.copy()
    change = .1*baseline_beta[1]
    explicit = beta_conductivity_direction(problem, 1, change)
    expected, _ = tangent.solve(np.zeros_like(dT), explicit)
    temperatures, conductivities = [], []
    for sign in [1, -1]:
        perturbed = baseline_beta.copy()
        perturbed[1] += sign*1e-3*change
        monkeypatch.setattr(problem.p, 'beta', perturbed)
        conductivities.append(problem.p.evaluate(tangent.T)[0])
        T, _, _ = problem.solver.solve(problem.p.A, problem.b['surface_T'], problem.b['T_bot'], tangent.T)
        temperatures.append(T)
    np.testing.assert_allclose((conductivities[0]-conductivities[1])/.002, explicit, rtol=2e-7, atol=1e-10)
    np.testing.assert_allclose((temperatures[0]-temperatures[1])/.002, expected, rtol=2e-5, atol=1e-5)
