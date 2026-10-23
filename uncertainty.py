"""Propagate observation, boundary and reference errors to the fitted T, K(T), and A fields.

usage: python uncertainty.py RESULTS

Writes final fields and statistics to RESULTS/uncertainty, with intermediate
linear, inputs and external products in its subdirectories.
"""
import runtime  # Set BLAS thread counts before importing NumPy.
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.linalg import block_diag, cholesky, cho_factor, cho_solve, solve_triangular
from scipy.linalg.lapack import dgttrf, dgttrs
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.spatial.distance import cdist
from scipy.special import logit

from config import A_BOUNDS, BETA, K_BOUNDS
from data_io import load_model, write_json
from geometry import composition_profiles, idw4
from observations import load as load_observations
from objective import Problem
from references import depth_kernel, heat_production_curve
from solver import surface_flux_conductivity, surface_flux_gradient

KEYS = ['heatflow', 'gradient', 'curie']
FIELDS = ['T', 'K', 'A']
UNITS = dict(T='degree_C', K='W/(m K)', A='uW/m3')


def observation_errors(problem):
    """Separate physical noise from gradient cell weighting; retain the full Curie covariance."""
    variances = []
    for key in KEYS[:2]:
        table = problem.data[key]
        if 'noise_sigma' in table:
            sigma = table.noise_sigma.to_numpy()
        elif key == 'gradient':
            cells = np.floor((table[['X', 'Y']].to_numpy()+1e4)/2e4).astype(int)
            count = table.groupby([cells[:, 0], cells[:, 1]])['observed'].transform('size').to_numpy()
            sigma = table.sigma.to_numpy()/np.sqrt(count)
        else:
            sigma = table.sigma.to_numpy()
        variances.append(np.diag(sigma**2))
    precision = problem.b['curie_precision']
    curie_covariance = cho_solve(cho_factor(precision), np.eye(len(precision)))
    W = block_diag(*(np.diag(problem.weights[key]) for key in KEYS[:2]), precision)
    return W, block_diag(*variances, curie_covariance)


def covariances(J, W, noise, prior_precision):
    """Return GN curvature, propagated-noise covariance, and conditional Gaussian covariance."""
    H = J.T@W@J+np.diag(prior_precision)
    factor = cho_factor(H)
    response = cho_solve(factor, J.T@W)
    propagated = response@noise@response.T
    gaussian = cho_solve(factor, np.eye(len(H)))
    return H, (propagated+propagated.T)/2, (gaussian+gaussian.T)/2


def standard_deviation(derivative, covariance):
    variance = np.sum((derivative@covariance)*derivative, axis=-1)
    return np.sqrt(np.maximum(variance, 0.))


class Tangent:
    """One fixed PDE linearization, shared by coefficient and external-input directions."""

    def __init__(self, problem):
        self.problem = problem
        self.T, self.K = problem.last['T'], problem.last['K']
        _, self.KT = problem.p.evaluate(self.T)
        self.matrix = problem.solver.jacobian(self.T, self.K, self.KT)
        factors = dgttrf(self.matrix.diagonal(-1), self.matrix.diagonal(), self.matrix.diagonal(1))
        if factors[-1] != 0:
            raise RuntimeError(f'Tangent preconditioner factorization failed: {factors[-1]}')
        self.preconditioner = LinearOperator(self.matrix.shape,
            matvec=lambda v: dgttrs(*factors[:5], np.asarray(v)[:, None])[0][:, 0])
        self.dz = problem.b['z_km'][1]-problem.b['z_km'][0]
        self.flux_T = surface_flux_gradient(self.T, self.K, self.KT, np.ones(self.T.shape[:2]), self.dz)
        self.factors = problem.p.factors(self.T)[0]
        profiles = problem.ops['curie']@self.T.reshape(-1, self.T.shape[-1])
        self.crossing = (profiles >= problem.curie_C).argmax(axis=1)
        if np.any(self.crossing == 0):
            raise ValueError('Every Curie profile must cross inside the grid.')
        self.rows = np.arange(len(self.crossing))
        lo, hi = profiles[self.rows, self.crossing-1], profiles[self.rows, self.crossing]
        self.fraction = (problem.curie_C-lo)/(hi-lo)
        self.curie_scale = -(problem.b['z_km'][self.crossing]-problem.b['z_km'][self.crossing-1])/(hi-lo)

    def properties(self, coefficients=None, offsets=None):
        """Explicit dA,dK from coefficient changes and/or reference-logit offsets."""
        p = self.problem.p
        fields = {key: np.zeros(p.masks.shape[:3]) for key in ['A', 'K']}
        if coefficients is not None:
            for key, layer, sl, basis in p.blocks:
                fields[key][layer] = np.einsum('i,ijk->jk', coefficients[sl], basis)
        offsets = {} if offsets is None else offsets
        dA, dK = np.zeros_like(self.T), np.zeros_like(self.T)
        for layer in range(len(p.masks)):
            da = fields['A'][layer, ..., None]
            dk = fields['K'][layer, ..., None]
            if 'A' in offsets:
                da = da+offsets['A'][layer]
            if 'K' in offsets:
                dk = dk+offsets['K'][layer]
            dA += p.masks[layer]*p.derivatives['A'][layer]*da*1e-6
            dK += p.masks[layer]*p.derivatives['K'][layer]*dk/(p.values['K'][layer]**2*self.factors[layer])
        dK *= self.K**2
        return dA, dK

    def solve(self, dA, dK, boundary=None):
        rhs = self.problem.solver.tangent_rhs(self.T, self.K, dA, dK, boundary, self.KT)
        interior, info = gmres(self.matrix, rhs, M=self.preconditioner, rtol=2e-10,
                               atol=0., restart=12, maxiter=100)
        if info != 0:
            raise RuntimeError(f'Tangent solve failed: {info}')
        dT = np.zeros_like(self.T) if boundary is None else boundary.copy()
        dT[..., 1:-1] = interior.reshape(*self.T.shape[:2], -1)
        dq = np.sum(self.flux_T*dT, axis=-1)+surface_flux_conductivity(self.T, self.K, dK, self.dz)+.5*self.dz*1e6*dA[..., 0]
        dp = self.problem.ops['curie']@dT.reshape(-1, self.T.shape[-1])
        dc = self.curie_scale*((1-self.fraction)*dp[self.rows, self.crossing-1]+self.fraction*dp[self.rows, self.crossing])
        return dT, np.r_[self.problem.ops['heatflow']@dq.ravel(), self.problem.ops['gradient']@dT.ravel(), dc]

    def external(self, response, offsets=None, boundary=None):
        """Effective derivative after a GN refit, including the direct input effect."""
        dA, dK = self.properties(offsets=offsets)
        direct_T, direct_observations = self.solve(dA, dK, boundary)
        dc = -response@direct_observations
        ca, ck = self.properties(coefficients=dc)
        effective_T, effective_observations = self.solve(dA+ca, dK+ck, boundary)
        return effective_T, effective_observations, dc, direct_T


def linearize(problem, coefficients, temperature_jacobian):
    """Coefficient derivatives of temperature and the three observation types."""
    problem.predict(coefficients)
    tangent = Tangent(problem)
    J = np.empty((sum(len(problem.data[key]) for key in KEYS), len(coefficients)))
    direction = np.zeros(len(coefficients))
    for i in range(len(coefficients)):
        direction[i] = 1.
        dT, J[:, i] = tangent.solve(*tangent.properties(coefficients=direction))
        temperature_jacobian[i] = dT.ravel()
        direction[i] = 0.
    return J


def analyze(problem, coefficients, output, regions=None):
    """Write local uncertainty fields without forming a full temperature covariance matrix."""
    if np.any(np.abs(coefficients) >= 4-1e-6):
        raise ValueError('This unconstrained local approximation requires coefficients inside their bounds.')
    output.mkdir(parents=True)
    problem.solver.tolerance, problem.solver.residual_tolerance = 1e-9, 1e-12
    shape = (*problem.b['x'].shape, len(problem.b['z_km']))
    temporary = output/'temperature_jacobian.npy'
    temperature_jacobian = np.lib.format.open_memmap(temporary, mode='w+', dtype='float64',
        shape=(len(coefficients), int(np.prod(shape))))
    J = linearize(problem, coefficients, temperature_jacobian)
    W, noise = observation_errors(problem)
    prior_precision = problem.regularization/problem.p.scales**2
    H, propagated, gaussian = covariances(J, W, noise, prior_precision)
    covariance = dict(observation_noise=propagated, conditional_gaussian=gaussian)
    np.savez_compressed(output/'parameters.npz', names=problem.p.names, coefficients=coefficients,
        observation_jacobian=J, observation_precision=W, noise_covariance=noise,
        prior_precision=prior_precision, gn_hessian=H, covariance_noise=propagated, covariance_gaussian=gaussian)
    table = pd.DataFrame(dict(name=problem.p.names, coefficient=coefficients))
    for name, C in covariance.items():
        table[name+'_sd'] = np.sqrt(np.diag(C))
    table.to_csv(output/'parameters.csv', index=False)
    start = 0
    for key in KEYS:
        stop = start+len(problem.data[key])
        table = problem.data[key].assign(predicted=problem.last[key])
        for name, C in covariance.items():
            table[name+'_prediction_sd'] = standard_deviation(J[start:stop], C)
        table.to_csv(output/f'{key}.csv', index=False)
        start = stop

    b, T = problem.b, problem.last['T']
    weights = b['core']/b['horizontal_scale']**2
    weights = dict(NCC=weights/weights.sum(), **({} if regions is None else regions))
    if 'ENCC' in weights and 'WNCC' in weights:
        weights['ENCC minus WNCC'] = weights['ENCC']-weights['WNCC']
    if 'N Shanxi' in weights and 'Ordos interior' in weights:
        weights['N Shanxi minus Ordos interior'] = weights['N Shanxi']-weights['Ordos interior']
    profiles = []
    for region, weight in weights.items():
        G = np.einsum('yx,iyxz->zi', weight, temperature_jacobian.reshape(len(coefficients), *shape))
        table = pd.DataFrame(dict(region=region, depth_km=b['z_km'], temperature_C=np.einsum('yx,yxz->z', weight, T)))
        for name, C in covariance.items():
            table[name+'_sd_C'] = standard_deviation(G, C)
        profiles.append(table)
    pd.concat(profiles, ignore_index=True).to_csv(output/'profiles.csv', index=False)

    depth_statistics = []
    with h5py.File(output/'fields.h5', 'w') as h:
        h.attrs.update(method='local Gauss-Newton', field_axis_order='y,x,z', temperature_units='degree_C',
            fixed_inputs='geometry, boundaries, reference A/K, conductivity laws, mode layout, regularization',
            noise_model='Independent heat flow/gradient errors; full declared Curie covariance; cross-type errors independent')
        for key in ['x', 'y', 'z_km', 'domain_mask', 'core', 'horizontal_scale', 'longitude', 'latitude']:
            if key in b:
                h.create_dataset('grid/'+key, data=b[key])
        h.create_dataset('temperature_mean', data=T, compression='gzip', shuffle=True)
        datasets = {name: h.create_dataset('temperature_sd/'+name, shape=shape, dtype='float64',
                    compression='gzip', shuffle=True) for name in covariance}
        for y in range(shape[0]):
            begin_cell, end_cell = y*shape[1]*shape[2], (y+1)*shape[1]*shape[2]
            G = np.array(temperature_jacobian[:, begin_cell:end_cell].T)
            for name, C in covariance.items():
                datasets[name][y] = standard_deviation(G, C).reshape(shape[1:])
        for z in [20., 40., 60., 80.]:
            if z > b['z_km'][-1]:
                continue
            index = np.argmin(abs(b['z_km']-z))
            for name, dataset in datasets.items():
                values = dataset[..., index][b['core']]
                depth_statistics.append(dict(depth_km=float(b['z_km'][index]), covariance=name,
                    median_sd_C=float(np.median(values)), p05_sd_C=float(np.quantile(values, .05)),
                    p95_sd_C=float(np.quantile(values, .95)), max_sd_C=float(values.max())))
    pd.DataFrame(depth_statistics).to_csv(output/'depth_statistics.csv', index=False)
    temperature_jacobian.flush()
    del temperature_jacobian
    temporary.unlink()


def run_linear(results, output):
    from config import A_BOUNDS, BETA, K_BOUNDS, SOURCES
    config = json.loads((results/'fit/training_config.json').read_text(encoding='utf-8'))
    problem = Problem(load_model(results/'problem'), load_observations(results/'problem'), config['modes'], config['constant_modes'],
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA,
                      regularization=config['regularization'])
    with np.load(results/'fit/result.npz') as saved:
        coefficients = saved['coefficients']
        problem.last = dict(T=saved['T'])
    regions = None
    if 'longitude' in problem.b:
        from si.regions import region_weights
        regions = region_weights(problem.b, SOURCES['regions'])
    analyze(problem, coefficients, output, regions)


def covariance_modes(draws, retained=.99):
    """Principal square-root covariance columns retaining the specified trace fraction."""
    covariance = np.cov(draws, rowvar=False)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = np.maximum(eigenvalues[order], 0.), vectors[:, order]
    count = np.searchsorted(np.cumsum(eigenvalues), retained*eigenvalues.sum())+1
    return vectors[:, :count]*np.sqrt(eigenvalues[:count])


def heat_production_bootstrap(table, centres, replicates, rng):
    """Resample whole publications jointly for both curves and every overlapping window."""
    groups, index = np.unique(table.study, return_inverse=True)
    counts = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=replicates)
    curves = []
    for metamorphic in [False, True]:
        for centre in centres:
            selected = (abs(table.sio2.to_numpy()-centre) <= 2+1e-10)
            if metamorphic:
                selected &= table.rock.eq('MET').to_numpy()
            rows = np.flatnonzero(selected)
            rows = rows[np.argsort(table.A.to_numpy()[rows])]
            weights = counts[:, index[rows]]
            cumulative = np.cumsum(weights, axis=1)
            total = cumulative[:, -1]
            if np.any(total == 0):
                raise ValueError('A bootstrap window has no samples; increase the calibration support.')
            # Median of the repeated records, including the midpoint for an even count.
            left = np.argmax(cumulative > ((total-1)//2)[:, None], axis=1)
            right = np.argmax(cumulative > (total//2)[:, None], axis=1)
            values = table.A.to_numpy()[rows]
            curves.append((values[left]+values[right])/2)
    return np.array(curves).T


def deep_conductivity_bootstrap(table, replicates, rng):
    """Publication/sample multiplier bootstrap, equal rock groups as in the calibration.

    Shared publications and specimens receive the same Exp(1) weight in every
    layer. Positive weights retain each rock group, including the small mantle
    calibration. This is a Bayesian/multiplier bootstrap, not rock scatter.
    """
    refs, ri = np.unique(table.reference, return_inverse=True)
    specimens, si = np.unique(table.reference.astype(str)+':'+table['sample'].astype(str), return_inverse=True)
    rw, sw = rng.exponential(size=(replicates, len(refs))), rng.exponential(size=(replicates, len(specimens)))
    result = []
    for layer in ['upper_crust', 'middle_crust', 'lower_crust', 'mantle']:
        part = table.layer.eq(layer).to_numpy()
        rocks = []
        for rock in sorted(table.loc[part, 'group'].unique()):
            ref_means, ref_weights = [], []
            for ref in np.unique(ri[part & table.group.eq(rock).to_numpy()]):
                selected = part & table.group.eq(rock).to_numpy() & (ri == ref)
                weight = sw[:, si[selected]]
                ref_means.append((weight@table.log_K0.to_numpy()[selected])/weight.sum(axis=1))
                ref_weights.append(rw[:, ref])
            weight = np.array(ref_weights).T
            rocks.append(np.sum(weight*np.array(ref_means).T, axis=1)/weight.sum(axis=1))
        result.append(np.mean(rocks, axis=0))
    return np.array(result).T


def shallow_design(table):
    """Original GLiM/depth design at conductivity interval midpoints."""
    classes = ['su', 'ss', 'py', 'sm', 'sc', 'ev', 'va', 'vi', 'vb', 'pa', 'pi', 'pb', 'mt', 'wb', 'ig', 'nd']
    merge = np.eye(6)[[0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5]]
    a, b = table.depth_top_m.to_numpy()/1000, table.depth_bottom_m.to_numpy()/1000
    z = (a+b)/2
    lithology = merge[[classes.index(c) for c in table.glim]]
    F = np.column_stack([np.ones(len(z)), np.log1p(z), *(lithology*np.exp(-z/2)[:, None]).T])
    return F


def shallow_gp(table):
    """Reconstruct the saved ridge mean and fixed-kernel training covariance."""
    a, b = table.depth_top_m.to_numpy()/1000, table.depth_bottom_m.to_numpy()/1000
    xy, F = table[['X', 'Y']].to_numpy(), shallow_design(table)
    weights = 1/table.groupby('site_id').record_id.transform('size').to_numpy()
    penalty = np.diag([1e-8, 2., *([12.]*6)])
    beta = np.linalg.solve(F.T@(weights[:, None]*F)+penalty, F.T@(weights*np.log(table.k_W_mK)))
    residual = np.log(table.k_W_mK)-F@beta
    variance = max(float(np.average(residual**2, weights=weights)), 1e-6)
    correlation = np.exp(-cdist(xy, xy)/150)*depth_kernel(a[:, None], b[:, None], a[None], b[None])
    correlation[np.diag_indices_from(correlation)] += 1/3
    return xy, a, b, variance, beta, correlation


def conditional_gp_draws(query_xy, depths, train_xy, a, b, correlation, count, rng):
    """Exact fixed-mean GP conditional draws using separable grid covariance.

    No full (x,y,z) covariance is formed. First simulate the query grid using
    spatial/depth Cholesky factors, then training intervals conditional on that
    grid, then apply the usual Gaussian conditioning correction. The training
    nugget is 1/3 of the latent variance (0.25/0.75 in the original model).
    All returned draws have unit latent prior variance; multiply by sqrt(.75 v).
    """
    spatial = cdist(query_xy, query_xy)
    spatial *= -1/150
    np.exp(spatial, out=spatial)
    Lxy = cholesky(spatial.T, lower=True, overwrite_a=True, check_finite=False)
    del spatial
    cross_xy = np.exp(-cdist(train_xy, query_xy)/150)
    Wxy = cho_solve((Lxy, True), cross_xy.T, check_finite=False).T
    depth = np.exp(-abs(depths[:, None]-depths[None])/.5)
    Lz = cholesky(depth, lower=True)
    cross_z = depth_kernel(a[:, None], b[:, None], depths[None], depths[None])
    Wz = cho_solve((Lz, True), cross_z.T).T
    conditional = correlation-(Wxy@cross_xy.T)*(Wz@cross_z.T)
    conditional = (conditional+conditional.T)/2
    Ltrain = cholesky(conditional, lower=True, overwrite_a=True, check_finite=False)
    Cfactor = cho_factor(correlation, lower=True)
    shape = len(query_xy), len(depths), count
    prior = (Lxy@rng.normal(size=(len(query_xy), len(depths)*count))).reshape(shape)
    del Lxy
    prior = np.einsum('zj,qjn->qzn', Lz, prior)
    train = np.einsum('tz,tzn->tn', Wz, (Wxy@prior.reshape(len(query_xy), -1)).reshape(len(a), len(depths), count))
    train += Ltrain@rng.normal(size=(len(a), count))
    del Wxy, Ltrain, conditional
    alpha = cho_solve(Cfactor, train)
    for iz in range(len(depths)):
        prior[:, iz] -= cross_xy.T@(cross_z[:, iz, None]*alpha)
    return prior


def reference_logit_direction(problem, key, derivative):
    """Apply the derivative of reference clipping and logit normalization exactly."""
    lo, hi = problem.p.bounds[key].T
    span = (hi-lo)[:, None, None, None]
    fraction = (problem.b[key+'_reference']-lo[:, None, None, None])/span
    active = (fraction > .05) & (fraction < .95)
    denominator = span*np.clip(fraction, .05, .95)*(1-np.clip(fraction, .05, .95))
    return np.where(active, derivative/denominator, 0.)


def median_influence(values):
    """Local derivative of a median, distributing tied central values equally."""
    values = np.asarray(values, dtype=float)
    ordered = np.sort(values)
    result = np.zeros(len(values))
    for centre in [ordered[(len(values)-1)//2], ordered[len(values)//2]]:
        selected = values == centre
        result[selected] += .5/selected.sum()
    return result


def coupled_heatflow_gradient(raw, heatflow, gradient, covariance, fraction):
    """Share a fraction of heat-flow variance with its constituent gradients.

    Only median-active records satisfying corrected q ~= K*g (within 5%)
    contribute. K*sigma_g sets relative loadings, then each cell is normalised
    to the specified fraction of its existing variance. This preserves all
    marginal variances; it does not infer the fraction from the q=K*g identity.
    """
    if not 0 <= fraction <= 1:
        raise ValueError('The shared variance fraction must lie in [0, 1].')
    nq, ng = len(heatflow), len(gradient)
    raw = raw.copy()
    raw.index = raw.index.astype(str)
    lookup = {str(v): i for i, v in enumerate(gradient.id)}
    sigma = np.sqrt(np.diag(covariance))
    load = np.zeros((nq+ng, ng))
    load[nq:] = np.diag(sigma[nq:nq+ng])
    for i, row in heatflow.iterrows():
        ids = str(row.ids).split(';')
        records = raw.loc[ids]
        q, g, k = (pd.to_numeric(records[c], errors='coerce').to_numpy() for c in ['o', 'h', 'j'])
        influence = median_influence(q)
        for record, weight, qi, gi, ki in zip(ids, influence, q, g, k):
            if record in lookup and weight > 0 and np.isfinite([qi, gi, ki]).all() and gi*ki > 0 and abs(qi/(gi*ki)-1) <= .05:
                j = lookup[record]
                load[i, j] += weight*ki*sigma[nq+j]
        norm = np.linalg.norm(load[i])
        if norm > 0:
            load[i] *= np.sqrt(fraction)*sigma[i]/norm
    result = covariance.copy()
    block = load@load.T
    residual = sigma[:nq+ng]**2-np.diag(block)
    if residual.min() < -1e-9:
        raise ValueError('Shared factors exceed marginal variances.')
    block += np.diag(np.maximum(residual, 0.))
    result[:nq+ng, :nq+ng] = block
    return result


def prepare_inputs(results, output, config):
    output.mkdir(parents=True)
    rng = np.random.default_rng(config['seed'])
    references = results/'references'
    b = dict(np.load(results/'problem/bundle.npz'))
    centres = pd.read_csv(references/'heat_production_relation.csv').sio2_wt_percent.to_numpy()
    a = heat_production_bootstrap(pd.read_csv(references/'georoc_ncc.csv'), centres, config['bootstrap_replicates'], rng)
    k = deep_conductivity_bootstrap(pd.read_csv(references/'deep_conductivity_samples.csv'), config['bootstrap_replicates'], rng)
    amodes = covariance_modes(a, config['calibration_retained_fraction'])
    residual = a-a.mean(axis=0)
    basis = amodes/np.linalg.norm(amodes, axis=0)
    residual -= (residual@basis)@basis.T
    tail_rng = np.random.default_rng(np.random.SeedSequence([config['seed'], 1]))
    tail_count = config['calibration_tail_draws']
    tail = residual.T@tail_rng.normal(size=(len(a), tail_count))/np.sqrt((len(a)-1)*tail_count)
    kmodes = covariance_modes(k, 1.-1e-12)
    table = pd.read_csv(references/'shallow_conductivity_samples.csv')
    xy, top, bottom, variance, beta, correlation = shallow_gp(table)
    depths = np.arange(0, 7.5001, .5)
    draws = conditional_gp_draws(np.c_[b['x'].ravel(), b['y'].ravel()]/1000,
        depths, xy, top, bottom, correlation, config['gp_draws'], rng)
    draws *= np.sqrt(.75*variance)
    with h5py.File(output/'calibration.h5', 'w') as h:
        h['A/centres'] = centres
        h['A/bootstrap_curves'] = a
        h['A/modes'] = amodes
        h['A/tail_modes'] = tail
        h['deep_K/bootstrap_log_K0'] = k
        h['deep_K/modes'] = kmodes
        h['GP/beta'] = beta
        h['GP/residual_variance'] = variance
        h['GP/depths'] = depths
        h.create_dataset('GP/log_K_draws', data=draws.transpose(2, 0, 1).reshape(config['gp_draws'], *b['x'].shape, len(depths)),
                         compression='gzip', shuffle=True)
    write_json(output/'summary.json', dict(settings=config))


def curve_mapper(x, y, z, profiles, forward):
    query = np.column_stack([x.ravel(), y.ravel()])
    index, weight = idw4(np.column_stack(forward.transform([p[0] for p in profiles], [p[1] for p in profiles])), query)
    used = np.unique(index)

    def map_curve(centres, values, extrapolate):
        transform = (lambda silica: heat_production_curve(silica, centres, values)) if extrapolate else (
            lambda silica: np.interp(silica, centres, values).astype('f4'))
        columns = np.array([np.interp(z, profiles[i][2][:, 0], transform(profiles[i][2][:, 1])) for i in used])
        return np.sum(columns[np.searchsorted(used, index)]*weight[..., None], axis=1).reshape(*x.shape, -1)
    return map_curve


def input_modes(problem, calibration, boundary_sd, A_scale_sd, profiles, forward):
    """Yield unit covariance columns, or explicitly labelled unit scenario responses."""
    b, shape = problem.b, problem.last['T'].shape
    map_curve = curve_mapper(b['x'], b['y'], b['z_km'], profiles, forward)
    centres = calibration['A/centres'][:]
    for source, dataset in [('A_curve', 'A/modes'), ('A_curve_tail', 'A/tail_modes')]:
        for i, mode in enumerate(calibration[dataset][:].T):
            derivative = np.zeros_like(b['A_reference'])
            derivative[1] = map_curve(centres, mode[:len(centres)], True)
            derivative[2] = derivative[3] = map_curve(centres, mode[len(centres):], False)
            derivative[b['masks'] <= 1e-30] = 0.
            yield source, f'{source}_{i+1}', dict(A=reference_logit_direction(problem, 'A', derivative)), None, 1.
    z = b['z_km']
    blend = np.exp(-np.maximum(z-5, 0)/.5)
    for i, mode in enumerate(calibration['deep_K/modes'][:].T):
        derivative = np.zeros_like(b['K_reference'])
        derivative[1] = b['K_reference'][1]*(1-blend)*mode[0]
        derivative[2:] = b['K_reference'][2:]*mode[1:, None, None, None]
        yield 'deep_K', f'deep_K_{i+1}', dict(K=reference_logit_direction(problem, 'K', derivative)), None, 1.
    count = len(calibration['GP/log_K_draws'])
    depth_index = np.minimum(np.arange(len(z)), len(calibration['GP/depths'])-1)
    for i, draw in enumerate(calibration['GP/log_K_draws']):
        log_change = draw[..., depth_index]/np.sqrt(count)
        derivative = np.zeros_like(b['K_reference'])
        derivative[0] = b['K_reference'][0]*log_change
        derivative[1] = b['K_reference'][1]*blend*log_change
        yield 'shallow_GP', f'shallow_GP_{i+1}', dict(K=reference_logit_direction(problem, 'K', derivative)), None, 1.
    # Unit RMS patterns over the area-weighted NCC. All pattern amplitudes remain explicit scenarios.
    area = b['core']/b['horizontal_scale']**2
    area /= area.sum()
    patterns = dict(offset=np.ones(shape[:2]))
    for name, coordinate in [('east_west', b['x']), ('north_south', b['y'])]:
        pattern = coordinate-np.sum(area*coordinate)
        patterns[name] = pattern/np.sqrt(np.sum(area*pattern**2))
    for source, index in [('surface', 0), ('bottom', -1)]:
        for name, pattern in patterns.items():
            boundary = np.zeros(shape)
            boundary[..., index] = pattern
            amplitude = boundary_sd[source][name]
            yield 'boundary_scenario', source+'_'+name+'_per_C', {}, boundary, amplitude
    for layer, name in [(0, 'sediment'), (4, 'mantle')]:
        derivative = np.zeros_like(b['A_reference'])
        derivative[layer] = b['A_reference'][layer]*.01
        yield 'A_scale_scenario', name+'_A_per_percent', dict(A=reference_logit_direction(problem, 'A', derivative)), None, A_scale_sd[name]


def validate_cross_covariance(noise, cross):
    """Cross=Cov(observation errors, independent unit external modes); require joint PSD."""
    whitened = solve_triangular(np.linalg.cholesky(noise), cross, lower=True)
    norm = float(np.linalg.svd(whitened, compute_uv=False)[0]) if cross.size else 0.
    if norm > 1+1e-10:
        raise ValueError(f'The joint observation/input covariance is not positive semidefinite: norm={norm}')
    return norm


def propagate_inputs(results, linear, inputs, output, cross_file=None):
    from config import A_BOUNDS, BETA, CRS, K_BOUNDS, SOURCES
    output.mkdir(parents=True)
    settings = json.loads((inputs/'summary.json').read_text(encoding='utf-8'))
    config = settings['settings']
    training = json.loads((results/'fit/training_config.json').read_text(encoding='utf-8'))
    problem = Problem(load_model(results/'problem'), load_observations(results/'problem'), training['modes'], training['constant_modes'],
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA,
                      regularization=training['regularization'])
    cached = np.load(linear/'parameters.npz')
    with np.load(results/'fit/result.npz') as fit:
        c = fit['coefficients']
        problem.last = dict(T=fit['T'])
    if np.any(abs(c) >= 4-1e-6):
        raise ValueError('Local unconstrained propagation needs interior coefficients.')
    problem.solver.tolerance, problem.solver.residual_tolerance = 1e-9, 1e-12
    problem.predict(c)
    W, noise = observation_errors(problem)
    response = cho_solve(cho_factor(cached['gn_hessian']), cached['observation_jacobian'].T@W)
    tangent = Tangent(problem)
    b, T = problem.b, problem.last['T']
    area = b['core']/b['horizontal_scale']**2
    weights = dict(NCC=area/area.sum())
    if 'longitude' in b:
        from si.regions import region_weights
        weights.update(region_weights(b, SOURCES['regions']))
        weights['ENCC minus WNCC'] = weights['ENCC']-weights['WNCC']
    reference_sources = ['A_curve', 'A_curve_tail', 'deep_K', 'shallow_GP']
    variances = {key: np.zeros_like(T) for key in [*reference_sources, 'boundary_scenario', 'A_scale_scenario', 'observation_reference_cross']}
    profiles, parameters, observations, rows = [], [], [], []
    cross, cross_names = None, None
    if cross_file is not None:
        with np.load(cross_file) as joint:
            cross, cross_names = joint['cross_covariance'], joint['mode_names'].tolist()
        with h5py.File(inputs/'calibration.h5') as calibration:
            expected = [f'{source}_{i+1}' for source, dataset in [
                ('A_curve', 'A/modes'), ('A_curve_tail', 'A/tail_modes'), ('deep_K', 'deep_K/modes')]
                for i in range(calibration[dataset].shape[1])]
            expected += [f'shallow_GP_{i+1}' for i in range(len(calibration['GP/log_K_draws']))]
        if len(set(cross_names)) != len(expected) or set(cross_names) != set(expected) or cross.shape != (len(W), len(expected)):
            raise ValueError('Cross-covariance mode names/shape do not match this calibration.')
        validate_cross_covariance(noise, cross)
    csrm_profiles = composition_profiles(SOURCES['csrm'])
    forward = Transformer.from_crs(4326, CRS, always_xy=True)
    with h5py.File(inputs/'calibration.h5') as calibration, h5py.File(output/'responses.h5', 'w') as h:
        for source, name, offsets, boundary, amplitude in input_modes(problem, calibration, config['boundary_scenario_sd_C'], config['A_scale_scenario_sd_percent'], csrm_profiles, forward):
            effective_T, effective_obs, dc, direct_T = tangent.external(response, offsets, boundary)
            h.create_dataset('temperature/'+name, data=effective_T, compression='gzip', shuffle=True)
            h.create_dataset('direct_temperature/'+name, data=direct_T, compression='gzip', shuffle=True)
            variances[source] += (amplitude*effective_T)**2
            cross_T = np.zeros_like(T)
            if cross is not None and source in reference_sources:
                column = cross[:, cross_names.index(name)]
                cross_T, _ = tangent.solve(*tangent.properties(coefficients=response@column))
                variances['observation_reference_cross'] += 2*effective_T*cross_T
            for region, weight in weights.items():
                profiles.append(dict(source=source, mode=name, region=region,
                    response=np.einsum('yx,yxz->z', weight, effective_T),
                    cross_response=np.einsum('yx,yxz->z', weight, cross_T), amplitude=amplitude))
            parameters.append(dc)
            observations.append(effective_obs)
            rows.append(dict(source=source, name=name, scenario_amplitude=amplitude))
    pd.DataFrame(rows).to_csv(output/'modes.csv', index=False)
    np.savez_compressed(output/'responses.npz', mode_names=[r['name'] for r in rows],
        sources=[r['source'] for r in rows], coefficient_responses=np.array(parameters), observation_responses=np.array(observations))
    with h5py.File(linear/'fields.h5') as old, h5py.File(output/'fields.h5', 'w') as h:
        old.copy('grid', h)
        h['temperature_mean'] = T
        reference = sum(variances[key] for key in reference_sources)
        noise_variance = old['temperature_sd/observation_noise'][:]**2
        joint_variance = noise_variance+reference+variances['observation_reference_cross']
        if np.min(joint_variance) < -1e-8:
            raise ValueError('Negative total variance: check the joint covariance.')
        all_variances = dict(variances, reference_calibration=reference, observation_noise=noise_variance,
            observation_plus_reference=joint_variance,
            illustrative_scenario=joint_variance+variances['boundary_scenario']+variances['A_scale_scenario'])
        for name, value in all_variances.items():
            h.create_dataset('temperature_variance/'+name, data=value, compression='gzip', shuffle=True)
            if name != 'observation_reference_cross':
                h.create_dataset('temperature_sd/'+name, data=np.sqrt(np.maximum(value, 0.)), compression='gzip', shuffle=True)
        h.attrs.update(method='local Gauss-Newton external-input propagation',
            reference_observation_cross_covariance='provided' if cross_file else 'assumed zero',
            scenario='Illustrative amplitudes from settings; not calibrated probability intervals.',
            GP='Fixed mean/kernel; Monte Carlo conditional covariance on original grid.')
        stats = []
        for depth in [0., 20., 40., 60., 80., 100.]:
            iz = np.argmin(abs(b['z_km']-depth))
            for name in all_variances:
                if name == 'observation_reference_cross':
                    continue
                v = h['temperature_sd/'+name][..., iz][b['core']]
                stats.append(dict(source=name, depth_km=b['z_km'][iz], median_sd_C=np.median(v), p05_sd_C=np.quantile(v, .05), p95_sd_C=np.quantile(v, .95)))
    pd.DataFrame(stats).to_csv(output/'depth_statistics.csv', index=False)
    baseline_profiles = pd.read_csv(linear/'profiles.csv')
    profile_rows = []
    for region in weights:
        subset = [p for p in profiles if p['region'] == region]
        old = baseline_profiles[baseline_profiles.region.eq(region)]
        pv = {source: sum((p['amplitude']*p['response'])**2 for p in subset if p['source'] == source) for source in variances if source != 'observation_reference_cross'}
        pv['observation_reference_cross'] = sum(2*p['response']*p['cross_response'] for p in subset)
        pv['observation_noise'] = old.observation_noise_sd_C.to_numpy()**2
        pv['reference_calibration'] = sum(pv[key] for key in reference_sources)
        pv['observation_plus_reference'] = pv['observation_noise']+pv['reference_calibration']+pv['observation_reference_cross']
        pv['illustrative_scenario'] = pv['observation_plus_reference']+pv['boundary_scenario']+pv['A_scale_scenario']
        for source, v in pv.items():
            for depth, value in zip(b['z_km'], v):
                profile_rows.append(dict(region=region, source=source, depth_km=depth, variance_C2=value,
                    sd_C=np.sqrt(max(value, 0.)) if source != 'observation_reference_cross' else np.nan))
    pd.DataFrame(profile_rows).to_csv(output/'profiles.csv', index=False)
    write_json(output/'summary.json', dict(scenario=config))


def field_response(tangent, coefficients, temperature, offsets=None):
    """Include direct/reference, inverse-compensation, and K(T) contributions."""
    dA, dK = tangent.properties(coefficients=coefficients, offsets=offsets)
    return dict(T=temperature, K=dK+tangent.KT*temperature, A=dA*1e6)


def property_jacobians(tangent, temperature_jacobian, row):
    """Coefficient derivatives for one horizontal grid row; all values in output units."""
    p, K = tangent.problem.p, tangent.K
    n = len(p.names)
    GT = np.asarray(temperature_jacobian[:, row])
    GA, GK = np.zeros_like(GT), np.zeros_like(GT)
    for key, layer, sl, basis in p.blocks:
        if key == 'A':
            explicit = p.masks[layer, row]*p.derivatives['A'][layer, row]
            GA[sl] = basis[:, row, :, None]*explicit
        else:
            explicit = K[row]**2*p.masks[layer, row]*p.derivatives['K'][layer, row]/(
                p.values['K'][layer, row]**2*tangent.factors[layer, row])
            GK[sl] = basis[:, row, :, None]*explicit
    GK += tangent.KT[row]*GT
    return {name: value.reshape(n, -1).T for name, value in zip(FIELDS, [GT, GK, GA])}


def depth_statistics(mean, variance, area):
    """Local-error magnitude and spatial spread, distinct from error of an area mean."""
    valid = area > 0
    weights = area[valid]/area.sum()
    values = mean[valid]
    sd = np.sqrt(np.maximum(variance[valid], 0.))
    avg = np.sum(weights[:, None]*values, axis=0)
    rms = np.sqrt(np.sum(weights[:, None]*sd**2, axis=0))
    order = np.argsort(sd, axis=0)
    sorted_sd = np.take_along_axis(sd, order, axis=0)
    cumulative = np.cumsum(weights[order], axis=0)
    quantiles = [np.take_along_axis(sorted_sd, np.minimum(np.sum(cumulative < q, axis=0), len(weights)-1)[None], axis=0)[0]
                 for q in [.05, .5, .95]]
    return dict(mean=avg, spatial_sd=np.sqrt(np.sum(weights[:, None]*(values-avg)**2, axis=0)),
        point_sd_mean=np.sum(weights[:, None]*sd, axis=0), point_sd_rms=rms,
        point_sd_median=quantiles[1], point_sd_p05=quantiles[0], point_sd_p95=quantiles[2], relative_rms_percent=100*rms/abs(avg))


def propagate_fields(results, linear, inputs, external, config, output):
    from config import A_BOUNDS, BETA, CRS, K_BOUNDS, LAYERS, SOURCES
    output.mkdir(parents=True, exist_ok=True)
    training = json.loads((results/'fit/training_config.json').read_text(encoding='utf-8'))
    problem = Problem(load_model(results/'problem'), load_observations(results/'problem'), training['modes'], training['constant_modes'],
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA,
                      regularization=training['regularization'])
    saved, cached = np.load(results/'fit/result.npz'), np.load(linear/'parameters.npz')
    coefficients = saved['coefficients']
    problem.last = dict(T=saved['T'])
    problem.solver.tolerance, problem.solver.residual_tolerance = 1e-9, 1e-12
    shape = saved['T'].shape
    temporary = output/'coefficient_temperature_jacobian.npy'
    jacobian = np.lib.format.open_memmap(temporary, mode='w+', dtype='float64', shape=(len(coefficients), *shape))
    J = linearize(problem, coefficients, jacobian.reshape(len(coefficients), -1))
    W, noise = observation_errors(problem)
    H = J.T@W@J+np.diag(cached['prior_precision'])
    B = cho_solve(cho_factor(H), J.T@W)
    tangent = Tangent(problem)
    b = problem.b
    area = b['core']/b['horizontal_scale']**2
    area /= area.sum()
    q, g = problem.data['heatflow'], problem.data['gradient']
    raw = pd.DataFrame(json.loads(SOURCES['heatflow'].read_text(encoding='utf-8-sig'))).set_index('a')
    joint_noise = coupled_heatflow_gradient(raw, q, g, noise, config['q_gradient_shared_fraction'])
    split = len(q)+len(g)
    covariances = {}
    for name, a, z in [('heatflow_gradient', 0, split), ('curie', split, len(J))]:
        covariances[name] = B[:, a:z]@joint_noise[a:z, a:z]@B[:, a:z].T
    variances = {source: {field: np.zeros(shape) for field in FIELDS} for source in [
        'heatflow_gradient', 'curie', 'A_calibration', 'K_calibration', 'shallow_GP',
        'surface_boundary', 'bottom_boundary', 'A_representativeness', 'K_representativeness']}
    profile_var = {source: {field: np.zeros(shape[-1]) for field in FIELDS} for source in variances}
    profile_jacobian = {field: np.zeros((shape[-1], len(coefficients))) for field in FIELDS}
    for row in range(shape[0]):
        matrices = property_jacobians(tangent, jacobian, row)
        for field, G in matrices.items():
            profile_jacobian[field] += np.einsum('x,xzi->zi', area[row], G.reshape(shape[1], shape[2], -1))
            for source, C in covariances.items():
                v = np.sum((G@C)*G, axis=1)
                variances[source][field][row] = v.reshape(shape[1:])
    for source, C in covariances.items():
        for field, G in profile_jacobian.items():
            profile_var[source][field] = np.sum((G@C)*G, axis=1)

    def add(source, response, amplitude):
        for field in FIELDS:
            d = response[field]*amplitude
            variances[source][field] += d**2
            profile_var[source][field] += np.einsum('yx,yxz->z', area, d)**2

    old_parameters = np.load(external/'responses.npz')
    mode_names = old_parameters['mode_names'].tolist()
    old_config = json.loads((external/'summary.json').read_text(encoding='utf-8'))['scenario']
    rms_fractions = np.array(config['spatial_variance_fractions'])
    if len(rms_fractions) != 3 or np.any(rms_fractions < 0) or not np.isclose(rms_fractions.sum(), 1.):
        raise ValueError('Offset/EW/NS variance fractions must be nonnegative and sum to one.')
    old_source_map = dict(A_curve='A_calibration', A_curve_tail='A_calibration', deep_K='K_calibration', shallow_GP='shallow_GP')
    profiles = composition_profiles(SOURCES['csrm'])
    forward = Transformer.from_crs(4326, CRS, always_xy=True)
    with h5py.File(inputs/'calibration.h5') as calibration, h5py.File(external/'responses.h5') as responses:
        for source, name, offsets, _, _ in input_modes(problem, calibration, old_config['boundary_scenario_sd_C'], old_config['A_scale_scenario_sd_percent'], profiles, forward):
            if source == 'A_scale_scenario':
                continue
            dc = old_parameters['coefficient_responses'][mode_names.index(name)]
            dT = responses['temperature/'+name][:]
            if source == 'boundary_scenario':
                location = name.split('_')[0]
                pattern = 'offset' if '_offset_' in name else 'east_west' if '_east_west_' in name else 'north_south'
                index = ['offset', 'east_west', 'north_south'].index(pattern)
                amplitude = config['boundary_rms_sd_C'][location]*np.sqrt(rms_fractions[index])
                destination = location+'_boundary'
            else:
                amplitude, destination = 1., old_source_map[source]
            add(destination, field_response(tangent, dc, dT, offsets), amplitude)
    patterns = [np.ones(shape[:2])]
    for coordinate in [b['x'], b['y']]:
        pattern = coordinate-np.sum(area*coordinate)
        patterns.append(pattern/np.sqrt(np.sum(area*pattern**2)))
    response_rows = []
    for key in ['A', 'K']:
        for layer, layer_name in enumerate(LAYERS):
            fraction = config['reference_additional_relative_sd'][key][layer]
            for index, (pattern, share) in enumerate(zip(patterns, rms_fractions)):
                derivative = np.zeros_like(b[key+'_reference'])
                derivative[layer] = b[key+'_reference'][layer]*fraction*np.sqrt(share)*pattern[..., None]
                offsets = {key: reference_logit_direction(problem, key, derivative)}
                direct_T, direct_obs = tangent.solve(*tangent.properties(offsets=offsets))
                dc = -B@direct_obs
                dT = direct_T+(dc@jacobian.reshape(len(coefficients), -1)).reshape(shape)
                response = field_response(tangent, dc, dT, offsets)
                name = f'{key}_{layer_name}_{index}'
                add(key+'_representativeness', response, 1.)
                response_rows.append(dict(name=name, key=key, layer=layer_name,
                    pattern=['offset', 'east_west', 'north_south'][index], relative_amplitude=fraction*np.sqrt(share)))
    pd.DataFrame(response_rows).to_csv(output/'additional_reference_modes.csv', index=False)
    calibration_sources = ['heatflow_gradient', 'curie', 'A_calibration', 'K_calibration', 'shallow_GP', 'surface_boundary', 'bottom_boundary']
    source_sets = dict(calibration_and_boundaries=calibration_sources,
        typical_reference=[*calibration_sources, 'A_representativeness', 'K_representativeness'])
    for name, sources in source_sets.items():
        variances[name] = {field: sum(variances[source][field] for source in sources) for field in FIELDS}
        profile_var[name] = {field: sum(profile_var[source][field] for source in sources) for field in FIELDS}
    base_variance = variances['typical_reference']
    for multiplier, name in [(.5, 'reference_half'), (2., 'reference_double')]:
        variances[name] = {field: variances['calibration_and_boundaries'][field]+multiplier**2*(
            variances['A_representativeness'][field]+variances['K_representativeness'][field]) for field in FIELDS}
        profile_var[name] = {field: profile_var['calibration_and_boundaries'][field]+multiplier**2*(
            profile_var['A_representativeness'][field]+profile_var['K_representativeness'][field]) for field in FIELDS}
    for basal_sd, name in [(50., 'bottom_50C'), (150., 'bottom_150C')]:
        scale = (basal_sd/config['boundary_rms_sd_C']['bottom'])**2-1
        variances[name] = {field: base_variance[field]+scale*variances['bottom_boundary'][field] for field in FIELDS}
        profile_var[name] = {field: profile_var['typical_reference'][field]+scale*profile_var['bottom_boundary'][field] for field in FIELDS}
    means = dict(T=problem.last['T'], K=problem.last['K'], A=problem.p.A*1e6)
    rows = []
    for source, fields in variances.items():
        for field, variance in fields.items():
            values = depth_statistics(means[field], variance, area)
            for i, depth in enumerate(b['z_km']):
                row = dict(source=source, field=field, units=UNITS[field], depth_km=float(depth),
                    area_mean_sd=float(np.sqrt(max(profile_var[source][field][i], 0.))))
                row.update({key: float(v[i]) for key, v in values.items()})
                if field == 'T':
                    row['relative_rms_kelvin_percent'] = 100*values['point_sd_rms'][i]/(values['mean'][i]+273.15)
                rows.append(row)
    statistics = pd.DataFrame(rows)
    statistics.to_csv(output/'depth_statistics.csv', index=False)
    with h5py.File(output/'fields.h5', 'w') as h:
        h.attrs.update(method='local Gauss-Newton', field_axis_order='y,x,z',
            reported_mean='Saved fitted field; first-order zero-mean perturbations preserve this central value.',
            statistics='Pointwise errors distinct from uncertainty of the area mean; all spatial summaries use NCC area weights.',
            cross_source_covariance='q/gradient shared factors included; other source groups independent by assumption.')
        for key in ['x', 'y', 'z_km', 'domain_mask', 'core', 'horizontal_scale', 'longitude', 'latitude']:
            h['grid/'+key] = b[key]
        for field, mean in means.items():
            h.create_dataset('mean/'+field, data=mean, compression='gzip', shuffle=True).attrs['units'] = UNITS[field]
        for source, fields in variances.items():
            for field, variance in fields.items():
                h.create_dataset(f'variance/{source}/{field}', data=variance, compression='gzip', shuffle=True).attrs['units'] = UNITS[field]+' squared'
                if source in source_sets:
                    h.create_dataset(f'std/{source}/{field}', data=np.sqrt(np.maximum(variance, 0.)), compression='gzip', shuffle=True).attrs['units'] = UNITS[field]
    jacobian.flush()
    jacobian._mmap.close()
    del jacobian
    temporary.unlink()
    summary = dict(settings=config, source_sets=source_sets)
    write_json(output/'summary.json', summary)
    return summary


def nonlinear_checks(results):
    output = results/'uncertainty'
    external = results/'uncertainty/external'
    training = json.loads((results/'fit/training_config.json').read_text(encoding='utf-8'))
    config = json.loads((output/'summary.json').read_text(encoding='utf-8'))['settings']
    problem = Problem(load_model(results/'problem'), load_observations(results/'problem'),
        training['modes'], training['constant_modes'], A_BOUNDS, K_BOUNDS, BETA, training['regularization'])
    saved = np.load(results/'fit/result.npz')
    coefficients = saved['coefficients']
    problem.solver.tolerance, problem.solver.residual_tolerance = 1e-9, 1e-12
    problem.last = dict(T=saved['T'])
    problem.predict(coefficients)
    tangent = Tangent(problem)
    b = problem.b
    area = b['core']/b['horizontal_scale']**2
    area /= area.sum()
    means = dict(T=problem.last['T'], K=problem.last['K'], A=problem.p.A*1e6)
    linear = np.load(results/'uncertainty/linear/parameters.npz')
    response = cho_solve(cho_factor(linear['gn_hessian']), linear['observation_jacobian'].T@linear['observation_precision'])
    offsets = {key: value.copy() for key, value in problem.p.offset.items()}
    parameters = np.load(external/'responses.npz')
    rows = []
    for key, layer, name in [('A', 1, 'A_upper_crust'), ('K', 1, 'K_upper_crust'),
                              ('A', 4, 'A_mantle'), ('bottom', None, 'bottom_east_west')]:
        boundary = np.zeros_like(means['T'])
        if key == 'bottom':
            amplitude = config['boundary_rms_sd_C']['bottom']*np.sqrt(config['spatial_variance_fractions'][1])
            mode = 'bottom_east_west_per_C'
            index = parameters['mode_names'].tolist().index(mode)
            dc = amplitude*parameters['coefficient_responses'][index]
            with h5py.File(external/'responses.h5') as h:
                dT = amplitude*h['temperature/'+mode][:]
            predicted = field_response(tangent, dc, dT)
            pattern = b['x']-np.sum(area*b['x'])
            boundary[..., -1] = amplitude*pattern/np.sqrt(np.sum(area*pattern**2))
        else:
            derivative = np.zeros_like(b[key+'_reference'])
            derivative[layer] = (b[key+'_reference'][layer]*config['reference_additional_relative_sd'][key][layer]
                                 *np.sqrt(config['spatial_variance_fractions'][0]))
            direction = {key: reference_logit_direction(problem, key, derivative)}
            dT, _, dc, _ = tangent.external(response, direction)
            predicted = field_response(tangent, dc, dT, direction)
        states = []
        for sign in [1., -1.]:
            if key != 'bottom':
                lo, hi = problem.p.bounds[key][layer]
                ref = b[key+'_reference'][layer]+sign*derivative[layer]
                problem.p.offset[key][layer] = logit(np.clip((ref-lo)/(hi-lo), .05, .95))
            problem.p.set(coefficients+sign*dc)
            T, K, _ = problem.solver.solve(problem.p.A, b['surface_T'], b['T_bot']+sign*boundary[..., -1], means['T'])
            states.append(dict(T=T, K=K, A=problem.p.A*1e6))
            if key != 'bottom':
                problem.p.offset[key][layer] = offsets[key][layer]
        problem.p.set(coefficients)
        for field in ['T', 'K', 'A']:
            central = (states[0][field]-states[1][field])/2
            scale = np.sqrt(np.sum(area[..., None]*predicted[field]**2))
            remainder = max(np.sqrt(np.sum(area[..., None]*(state[field]-means[field]-sign*predicted[field])**2))
                            for sign, state in zip([1., -1.], states))
            rows.append(dict(source=name, field=field,
                one_sd_secant_relative_error=float(np.sqrt(np.sum(area[..., None]*(central-predicted[field])**2))/scale) if scale else 0.,
                max_one_sided_relative_remainder=float(remainder/scale) if scale else 0.))
    pd.DataFrame(rows).to_csv(output/'nonlinear_checks.csv', index=False)


def run(results):
    from config import CALIBRATION_SETTINGS, FIELD_SETTINGS

    output = results/'uncertainty'
    output.mkdir(parents=True)
    print('[1/5] Propagating observation errors', flush=True)
    run_linear(results, output/'linear')
    print('[2/5] Preparing reference calibration errors', flush=True)
    prepare_inputs(results, output/'inputs', CALIBRATION_SETTINGS)
    print('[3/5] Propagating boundary and reference errors', flush=True)
    propagate_inputs(results, output/'linear', output/'inputs', output/'external')
    print('[4/5] Combining T, K and A uncertainty fields', flush=True)
    summary = propagate_fields(results, output/'linear', output/'inputs', output/'external', FIELD_SETTINGS, output)
    print('[5/5] Checking nonlinear perturbations', flush=True)
    nonlinear_checks(results)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    run(parser.parse_args().results)
