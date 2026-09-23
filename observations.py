"""Grid sampling, interval gradients and weighted observation losses."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class PointSampler:
    shape: tuple[int, ...]
    matrix: csr_matrix

    @classmethod
    def build(cls, axes: tuple[np.ndarray, ...], points: np.ndarray) -> 'PointSampler':
        """Axes and point columns follow field order: (y,x,z) or (y,x).

        Units of points must equal their corresponding axis units. No extrapolation.
        """
        axes = tuple(np.asarray(a, dtype=float) for a in axes)
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != len(axes) or not np.all(np.isfinite(p)):
            raise ValueError('points must be finite with one column per grid axis')
        lower, fraction = [], []
        for dim, axis in enumerate(axes):
            if axis.ndim != 1 or len(axis) < 2 or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
                raise ValueError('grid axes must be finite and strictly increasing')
            if np.any(p[:, dim] < axis[0]) or np.any(p[:, dim] > axis[-1]):
                raise ValueError('observation lies outside the model grid')
            index = np.clip(np.searchsorted(axis, p[:, dim], side='right') - 1, 0, len(axis) - 2)
            lower.append(index)
            fraction.append((p[:, dim] - axis[index]) / (axis[index + 1] - axis[index]))
        shape = tuple(len(a) for a in axes)
        row, column, value = [], [], []
        for corner in product([0, 1], repeat=len(axes)):
            indices = tuple(lower[d] + bit for d, bit in enumerate(corner))
            weights = np.prod([fraction[d] if bit else 1 - fraction[d] for d, bit in enumerate(corner)], axis=0)
            row.append(np.arange(len(p)))
            column.append(np.ravel_multi_index(indices, shape))
            value.append(weights)
        matrix = csr_matrix((np.concatenate(value), (np.concatenate(row), np.concatenate(column))),
                            shape=(len(p), int(np.prod(shape))))
        return cls(shape, matrix)

    def sample(self, field: np.ndarray) -> np.ndarray:
        if np.shape(field) != self.shape:
            raise ValueError('field shape does not match observation grid')
        return self.matrix @ np.asarray(field).ravel()

    @classmethod
    def vertical_average(cls, axes: tuple[np.ndarray, ...], xy: np.ndarray,
                         top: np.ndarray, bottom: np.ndarray) -> 'PointSampler':
        """Exact vertical integral of the trilinear interpolant divided by interval length.

        xy columns are (y,x), depths use the same units as axes[2].
        """
        xy, top, bottom = np.asarray(xy),np.asarray(top),np.asarray(bottom)
        if len(axes) != 3 or xy.shape != (len(top),2) or bottom.shape != top.shape or np.any(bottom <= top):
            raise ValueError('invalid vertical observation intervals')
        points, weights, rows = [],[],[]
        for i,(lo,hi) in enumerate(zip(top,bottom)):
            nodes = np.r_[lo,axes[2][(axes[2]>lo)&(axes[2]<hi)],hi]
            segment = np.diff(nodes)/(hi-lo)
            w = np.r_[segment,0]/2 + np.r_[0,segment]/2
            points.extend(np.column_stack([np.full(len(nodes),xy[i,0]),np.full(len(nodes),xy[i,1]),nodes]))
            weights.extend(w)
            rows.extend([i]*len(nodes))
        op = cls.build(axes,np.asarray(points))
        average = csr_matrix((weights,(rows,np.arange(len(weights)))),shape=(len(top),len(weights)))
        return cls(op.shape,(average@op.matrix).tocsr())

    def transpose(self, point_gradient: np.ndarray) -> np.ndarray:
        gradient = np.asarray(point_gradient, dtype=float)
        if gradient.shape != (self.matrix.shape[0],):
            raise ValueError('point gradient shape does not match observations')
        return np.asarray(self.matrix.T @ gradient).reshape(self.shape)


@dataclass(frozen=True)
class Observations:
    sampler: PointSampler
    values: np.ndarray
    sigma: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        for name in ['values', 'sigma', 'weights']:
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != (self.sampler.matrix.shape[0],) or not np.all(np.isfinite(array)):
                raise ValueError(f'{name} must be finite and match observation count')
            object.__setattr__(self, name, array)
        if np.any(self.sigma <= 0) or np.any(self.weights <= 0):
            raise ValueError('observation sigma and weights must be positive')

    def loss(self, field: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        value, point_gradient, residual = self.loss_values(self.sampler.sample(field))
        return value, self.sampler.transpose(point_gradient), residual

    def loss_values(self, prediction: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        prediction = np.asarray(prediction, dtype=float)
        if prediction.shape != self.values.shape or not np.all(np.isfinite(prediction)):
            raise ValueError('predictions must be finite and match observations')
        residual = prediction - self.values
        value = .5 * np.sum(self.weights * (residual / self.sigma)**2)
        point_gradient = self.weights * residual / self.sigma**2
        return float(value), point_gradient, residual


def equal_group_weights(groups: np.ndarray) -> np.ndarray:
    """Give each well/site total weight one; not a claim of independent samples."""
    _, inverse, counts = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def gradient_operator(axes, yx, top_km, bottom_km):
    """(T_bottom-T_top)/(z_bottom-z_top), Celsius/km, trilinear T."""
    yx = np.asarray(yx, dtype=float)
    top, bottom = np.asarray(top_km, dtype=float), np.asarray(bottom_km, dtype=float)
    if yx.shape != (len(top),2) or bottom.shape != top.shape or np.any(bottom <= top):
        raise ValueError('Invalid gradient depth intervals')
    upper = PointSampler.build(axes, np.column_stack([yx,top]))
    lower = PointSampler.build(axes, np.column_stack([yx,bottom]))
    matrix = (lower.matrix-upper.matrix).multiply(1/(bottom-top)[:,None]).tocsr()
    return PointSampler(upper.shape,matrix)
