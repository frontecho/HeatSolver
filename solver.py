"""Finite-volume steady conduction -div(K(T) grad T) = A with prescribed top and bottom T."""
from itertools import product
import numpy as np
import scipy.sparse as sp
from scipy.linalg.lapack import dgttrf, dgttrs, dpttrf, dpttrs
from scipy.sparse.linalg import LinearOperator, cg, gmres


class Solver:
    def __init__(self, x, y, z_km, horizontal_scale, active, law, tolerance=3e-7, residual_tolerance=2e-10):
        self.ny, self.nx, self.nz = *x.shape, len(z_km)
        self.dx, self.dy, self.dz = x[0, 1]-x[0, 0], y[1, 0]-y[0, 0], (z_km[1]-z_km[0])*1e3
        self.z, self.law = z_km, law
        self.tolerance, self.residual_tolerance = tolerance, residual_tolerance
        # Conformal map factor h: div_xy(K grad_xy T) + d_z(K/h^2 d_z T) + A/h^2 = 0.
        self.area = (1/horizontal_scale**2)[..., None]
        # Faces leaving the active domain are insulated.
        self.face_y = (active[:-1] & active[1:])[..., None]
        self.face_x = (active[:, :-1] & active[:, 1:])[..., None]
        self.faces = [(0, self.dy, self.face_y), (1, self.dx, self.face_x), (2, self.dz, self.area)]
        ny, nx, n = self.ny, self.nx, self.nz-2
        i, j, k = np.meshgrid(np.arange(ny), np.arange(nx), np.arange(n), indexing='ij')
        flat = lambda i, j, k: (i*nx*n+j*n+k).ravel()
        row = flat(i, j, k)
        self.neighbour = [(k > 0).ravel(), (k < n-1).ravel(), (i > 0).ravel(), (i < ny-1).ravel(),
                          (j > 0).ravel(), (j < nx-1).ravel()]
        shifts = [(0, 0, -1), (0, 0, 1), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]
        self.rows = np.concatenate([row[m] for m in self.neighbour]+[row])
        self.cols = np.concatenate([flat(i+a, j+b, k+c)[m] for (a, b, c), m in zip(shifts, self.neighbour)]+[row])
        self.size = ny*nx*n

    def matrix(self, minus, plus, diagonal):
        """Assemble from per-axis couplings to the lower (minus) and upper (plus) neighbour."""
        inner = lambda a: a[..., 1:-1].ravel()
        values = [inner(minus[2])[self.neighbour[0]], inner(plus[2])[self.neighbour[1]],
                  inner(minus[0])[self.neighbour[2]], inner(plus[0])[self.neighbour[3]],
                  inner(minus[1])[self.neighbour[4]], inner(plus[1])[self.neighbour[5]], inner(diagonal)]
        return sp.csr_matrix((np.concatenate(values), (self.rows, self.cols)), shape=(self.size, self.size))

    def system(self, K, A, top, bottom):
        shape = K.shape
        minus, plus, diagonal = [np.zeros(shape) for _ in range(3)], [np.zeros(shape) for _ in range(3)], np.zeros(shape)
        for axis, spacing, weight in self.faces:
            a, b = [slice(None)]*3, [slice(None)]*3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            c = 2*K[a]*K[b]/(K[a]+K[b])*weight/spacing**2
            plus[axis][a], minus[axis][b] = -c, -c
            diagonal[a] += c
            diagonal[b] += c
        matrix = self.matrix(minus, plus, diagonal)
        rhs = A[..., 1:-1]*self.area
        rhs[..., 0] -= minus[2][..., 1]*top
        rhs[..., -1] -= plus[2][..., -2]*bottom
        return matrix, rhs.ravel()

    def linear(self, K, A, top, bottom, initial):
        matrix, rhs = self.system(K, A, top, bottom)
        d, e, _ = dpttrf(matrix.diagonal(), matrix.diagonal(-1))
        column = LinearOperator(matrix.shape, matvec=lambda v: dpttrs(d, e, np.asarray(v)[:, None])[0][:, 0])
        interior, info = cg(matrix, rhs, x0=initial[..., 1:-1].ravel(), M=column, rtol=1e-12, maxiter=500)
        assert info == 0
        T = initial.copy()
        T[..., 1:-1] = interior.reshape(self.ny, self.nx, self.nz-2)
        return T

    def solve(self, A, top, bottom, initial=None):
        if initial is None:
            f = (self.z-self.z[0])/(self.z[-1]-self.z[0])
            initial = top[..., None]*(1-f)+bottom[..., None]*f
        T = initial.copy()
        T[..., 0], T[..., -1] = top, bottom
        for _ in range(45):
            updated = self.linear(self.law.evaluate(T)[0], A, top, bottom, T)
            change = np.max(abs(updated-T))
            T = updated
            if change <= self.tolerance:
                K, KT = self.law.evaluate(T)
                matrix, rhs = self.system(K, A, top, bottom)
                if np.linalg.norm(matrix@T[..., 1:-1].ravel()-rhs)/np.linalg.norm(rhs) <= self.residual_tolerance:
                    return T, K, KT
        raise RuntimeError('Picard iteration did not converge')

    def jacobian(self, T, K, KT):
        """Interior Jacobian of the nonlinear residual with fixed boundary temperatures."""
        shape = T.shape
        minus, plus, diagonal = [np.zeros(shape) for _ in range(3)], [np.zeros(shape) for _ in range(3)], np.zeros(shape)
        for axis, spacing, weight in self.faces:
            a, b = [slice(None)]*3, [slice(None)]*3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            ka, kb, dt = K[a], K[b], T[a]-T[b]
            face = 2*ka*kb/(ka+kb)
            dfa = weight/spacing**2*(face+dt*2*kb**2/(ka+kb)**2*KT[a])
            dfb = weight/spacing**2*(-face+dt*2*ka**2/(ka+kb)**2*KT[b])
            diagonal[a] += dfa
            diagonal[b] -= dfb
            plus[axis][a], minus[axis][b] = dfb, -dfa
        return self.matrix(minus, plus, diagonal)

    def tangent_rhs(self, T, K, dA, dK, boundary=None, KT=None):
        """Interior RHS for explicit properties and prescribed boundary-temperature changes.

        boundary has full field shape and is zero at interior nodes. KT is needed
        with boundary because the boundary conductivity also changes with T.
        """
        if boundary is not None:
            dK = dK+KT*boundary
        rhs = np.broadcast_to(dA*self.area, T.shape).copy()
        for axis, spacing, weight in self.faces:
            a, b = [slice(None)]*3, [slice(None)]*3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            face = 2*(K[b]**2*dK[a]+K[a]**2*dK[b])/(K[a]+K[b])**2
            flux = weight/spacing**2*face*(T[a]-T[b])
            if boundary is not None:
                flux += weight/spacing**2*2*K[a]*K[b]/(K[a]+K[b])*(boundary[a]-boundary[b])
            rhs[a] -= flux
            rhs[b] += flux
        return rhs[..., 1:-1].ravel()

    def adjoint(self, T, K, KT, grad_T):
        """Solve J^T lam = dL/dT for the nonlinear residual; return lam scaled as dL/dA."""
        J = self.jacobian(T, K, KT).T.tocsr()
        factors = dgttrf(J.diagonal(-1), J.diagonal(), J.diagonal(1))
        column = LinearOperator(J.shape, matvec=lambda v: dgttrs(*factors[:5], np.asarray(v)[:, None])[0][:, 0])
        lam, info = gmres(J, grad_T[..., 1:-1].ravel(), M=column, rtol=2e-10, atol=0, restart=35, maxiter=50)
        assert info == 0
        grad_A = np.zeros_like(T)
        grad_A[..., 1:-1] = lam.reshape(self.ny, self.nx, self.nz-2)*self.area
        return grad_A

    def conductivity_pullback(self, T, K, lam):
        """Explicit derivative of lam^T residual with respect to cell conductivity."""
        result = np.zeros_like(K)
        for axis, spacing, weight in self.faces:
            a, b = [slice(None)]*3, [slice(None)]*3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            common = -2*weight/spacing**2*(lam[a]-lam[b])*(T[a]-T[b])/(K[a]+K[b])**2
            result[a] += common*K[b]**2
            result[b] += common*K[a]**2
        return result


def sampler(axes, points):
    """Sparse multilinear interpolation from grid nodes (axes in field order) to points."""
    lower, fraction = [], []
    for axis, p in zip(axes, points.T):
        i = np.clip(np.searchsorted(axis, p, side='right')-1, 0, len(axis)-2)
        lower.append(i)
        fraction.append((p-axis[i])/(axis[i+1]-axis[i]))
    shape, rows, cols, values = tuple(len(a) for a in axes), [], [], []
    for corner in product([0, 1], repeat=len(axes)):
        rows.append(np.arange(len(points)))
        cols.append(np.ravel_multi_index(tuple(i+c for i, c in zip(lower, corner)), shape))
        values.append(np.prod([f if c else 1-f for f, c in zip(fraction, corner)], axis=0))
    return sp.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(len(points), int(np.prod(shape))))


def interval_gradient(axes, yx, top, bottom):
    """Operator for (T(bottom)-T(top))/(bottom-top) in C/km."""
    upper, lower = (sampler(axes, np.column_stack([yx, z])) for z in (top, bottom))
    return (lower-upper).multiply(1/(bottom-top)[:, None]).tocsr()


def surface_flux(T, K, A, dz_km):
    """Upward heat flow at the surface in mW/m2, including heat generated in the upper half-cell."""
    a, b = K[..., 0], K[..., 1]
    return 2*a*b/(a+b)*(T[..., 1]-T[..., 0])/dz_km+.5*dz_km*1e6*A[..., 0]


def surface_flux_gradient(T, K, KT, weights, dz_km):
    """Transpose derivative of surface_flux with respect to T, including K(T)."""
    a, b, dt = K[..., 0], K[..., 1], T[..., 1]-T[..., 0]
    face = 2*a*b/(a+b)
    result = np.zeros_like(T)
    result[..., 0] = weights*(-face+dt*2*b*b/(a+b)**2*KT[..., 0])/dz_km
    result[..., 1] = weights*(face+dt*2*a*a/(a+b)**2*KT[..., 1])/dz_km
    return result


def surface_flux_conductivity(T, K, dK, dz_km):
    """Derivative of the surface conduction term with respect to a conductivity perturbation dK."""
    a, b = K[..., 0], K[..., 1]
    return 2*(b*b*dK[..., 0]+a*a*dK[..., 1])/(a+b)**2*(T[..., 1]-T[..., 0])/dz_km
