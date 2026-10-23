"""Objective, observation losses, regularization and adjoint gradient for fitting."""
import numpy as np
from model import Model
from solver import interval_gradient, sampler, surface_flux_gradient

CURIE_C = 580.


def curie_misfit(profiles, z, observed, precision):
    """Isotherm depth, its weighted misfit and the gradient with respect to the profiles."""
    above = profiles >= CURIE_C
    j = above.argmax(axis=1)
    row = np.arange(len(j))
    lo, hi = profiles[row, j-1], profiles[row, j]
    fraction, dz = (CURIE_C-lo)/(hi-lo), z[j]-z[j-1]
    depth = z[j-1]+fraction*dz
    weighted = precision@(depth-observed)
    gradient = np.zeros_like(profiles)
    gradient[row, j-1] = -weighted*dz*(1-fraction)/(hi-lo)
    gradient[row, j] = -weighted*dz*fraction/(hi-lo)
    return .5*(depth-observed)@weighted, gradient, depth


class Problem(Model):
    curie_C = CURIE_C

    def __init__(self, b, data, layout, constant_modes, a_bounds, k_bounds, beta, regularization):
        super().__init__(b, layout, constant_modes, a_bounds, k_bounds, beta)
        self.regularization = np.array([regularization[name.split('_')[0]] for name in self.p.names])
        self.data = d = data
        axes = b['y'][:, 0], b['x'][0]
        self.ops = dict(heatflow=sampler(axes, d['heatflow'][['Y', 'X']].to_numpy()),
                        curie=sampler(axes, d['curie'][['Y', 'X']].to_numpy()),
                        gradient=interval_gradient((*axes, b['z_km']), d['gradient'][['Y', 'X']].to_numpy(),
                                                   d['gradient'].top_km.to_numpy(), d['gradient'].bottom_km.to_numpy()))
        self.weights = {key: 1/d[key].sigma.to_numpy()**2 for key in ['heatflow', 'gradient']}
        self.last = None

    def metrics(self):
        """Statistics over all observations, shared by iteration logs and final evaluation."""
        metrics = {}
        for key, table in self.data.items():
            observed, predicted = table.observed.to_numpy(), self.last[key]
            residual = predicted-observed
            metrics[key] = dict(n=len(table), rmse=float(np.sqrt(np.mean(residual**2))),
                                bias=float(residual.mean()), r=float(np.corrcoef(observed, predicted)[0, 1]))
        return metrics

    def predict(self, c):
        b, ops = self.b, self.ops
        state = self.forward(c, None if self.last is None else self.last['T'])
        T = state['T']
        lc, gc, depth = curie_misfit(ops['curie']@T.reshape(-1, T.shape[-1]), b['z_km'],
                                   self.data['curie'].observed.to_numpy(), b['curie_precision'])
        self.last = dict(state, heatflow=ops['heatflow']@state['q'].ravel(), gradient=ops['gradient']@T.ravel(), curie=depth)
        return lc, gc

    def evaluate(self, c):
        b, p, ops, data = self.b, self.p, self.ops, self.data
        lc, gc = self.predict(c)
        state = self.last
        T, K, KT, q = (state[key] for key in ['T', 'K', 'KT', 'q'])
        dz = b['z_km'][1]-b['z_km'][0]
        rq = state['heatflow']-data['heatflow'].observed.to_numpy()
        rg = state['gradient']-data['gradient'].observed.to_numpy()
        lq, lg = .5*self.weights['heatflow']@rq**2, .5*self.weights['gradient']@rg**2
        penalty = .5*np.sum(self.regularization*(c/p.scales)**2)
        self.last['losses'] = dict(heatflow=lq, gradient=lg, curie=lc, regularization=penalty)
        qadj = (ops['heatflow'].T@(self.weights['heatflow']*rq)).reshape(q.shape)
        grad_T = (ops['gradient'].T@(self.weights['gradient']*rg)+(ops['curie'].T@gc).ravel()).reshape(T.shape)
        grad_T += surface_flux_gradient(T, K, KT, qadj, dz)
        ga = self.solver.adjoint(T, K, KT, grad_T)
        lam = np.zeros_like(T)
        lam[..., 1:-1] = ga[..., 1:-1]/self.solver.area
        gK = self.solver.conductivity_pullback(T, K, lam)
        ga[..., 0] += .5*dz*1e6*qadj
        gradient = p.pullback(T, K, ga, gK, qadj, dz)+self.regularization*c/p.scales**2
        return lq+lg+lc+penalty, gradient
