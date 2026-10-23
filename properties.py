"""Bounded A/K parameterization and temperature-dependent conductivity laws."""
import numpy as np
from scipy.fft import idct
from scipy.special import expit, logit
from solver import surface_flux_conductivity


def dct_modes(n, m):
    return np.sqrt(n)*idct(np.eye(n, min(m, n)), axis=0, norm='ortho')


def mantle_factor(T):
    """Doin and Fleitout (1996) lattice plus radiative conductivity, normalised to 20 C."""
    tk = T+273.15
    f20 = 1.23e-10*293.15**3+1/(.518+7.89e-4*293.15)
    return (1.23e-10*tk**3+1/(.518+7.89e-4*tk))/f20, (3*1.23e-10*tk**2-7.89e-4/(.518+7.89e-4*tk)**2)/f20


class Properties:
    """Bounded layer values lo+(hi-lo)*expit(logit(reference)+sum c_j b_j(x,y))."""

    def __init__(self, masks, z, K_reference, A_reference, layout, constant_modes, a_bounds, k_bounds, beta):
        layers = ['sediment', 'upper_crust', 'middle_crust', 'lower_crust', 'mantle']
        self.masks, self.z = masks, z
        self.bounds, self.beta = dict(A=a_bounds, K=k_bounds), beta
        self.offset = {key: logit(np.clip((ref-lo[:, None, None, None])/(hi-lo)[:, None, None, None], .05, .95))
                       for key, ref in [('A', A_reference), ('K', K_reference)] for lo, hi in [self.bounds[key].T]}
        ny, nx = masks.shape[1:3]
        self.names, self.blocks, scales = [], [], []
        for key in ['A', 'K']:
            for layer, shape in enumerate(layout[key]):
                if shape is None:
                    continue
                by, bx = dct_modes(ny, shape[0]), dct_modes(nx, shape[1])
                modes = [(y, x) for y in range(by.shape[1]) for x in range(bx.shape[1])
                         if (y, x) != (0, 0) or constant_modes[key][layer]]
                if not modes:
                    continue
                start = len(self.names)
                self.names += [f'{key}_{layers[layer]}_y{y}_x{x}' for y, x in modes]
                basis = np.array([np.outer(by[:, y], bx[:, x]) for y, x in modes])
                self.blocks.append((key, layer, slice(start, len(self.names)), basis))
                scales += [(1. if key == 'A' else .7)/np.sqrt(1+(y/4)**2+(x/8)**2) for y, x in modes]
        self.scales = np.array(scales)
        self.set(np.zeros(len(self.names)))

    def set(self, c):
        fields = {key: np.zeros(self.masks.shape[:3]) for key in ['A', 'K']}
        for key, layer, sl, basis in self.blocks:
            fields[key][layer] = np.einsum('i,ijk->jk', c[sl], basis)
        self.values, self.derivatives = {}, {}
        for key in ['A', 'K']:
            lo, hi = self.bounds[key].T
            fraction = expit(self.offset[key]+fields[key][..., None])
            self.values[key] = lo[:, None, None, None]+(hi-lo)[:, None, None, None]*fraction
            self.derivatives[key] = (hi-lo)[:, None, None, None]*fraction*(1-fraction)
        self.A = np.sum(self.masks*self.values['A'], axis=0)*1e-6

    def factors(self, T):
        """Crust: Chapman (1986) form with pressure term; mantle: Doin and Fleitout (1996)."""
        pressure = 1+.0015*self.z
        denominator = 1+self.beta[:, None, None, None]*(T[None]-20)
        mantle, mantle_T = mantle_factor(T)
        return (np.concatenate([pressure/denominator, mantle[None]]),
                np.concatenate([-self.beta[:, None, None, None]*pressure/denominator**2, mantle_T[None]]))

    def evaluate(self, T):
        """Harmonic mixture of layer conductivities in each cell and its temperature derivative."""
        factors, dt = self.factors(T)
        branch = self.values['K']*factors
        K = 1/np.sum(self.masks/branch, axis=0)
        return K, K*K*np.sum(self.masks*dt/(self.values['K']*factors**2), axis=0)

    def pullback(self, T, K, ga, gK, qadj, dz_km):
        factors, _ = self.factors(T)
        result = np.zeros(len(self.names))
        for key, layer, sl, basis in self.blocks:
            if key == 'A':
                sensitivity = self.derivatives['A'][layer]
                column = np.sum(ga*self.masks[layer]*sensitivity, axis=-1)*1e-6
                result[sl] = np.einsum('ijk,jk->i', basis, column)
            else:
                dk = K*K*self.masks[layer]*self.derivatives['K'][layer]/(self.values['K'][layer]**2*factors[layer])
                column = np.sum(gK*dk, axis=-1)+qadj*surface_flux_conductivity(T, K, dk, dz_km)
                result[sl] = np.einsum('ijk,jk->i', basis, column)
        return result


