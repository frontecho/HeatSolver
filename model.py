"""Forward conduction model from geometry, reference properties and boundaries."""
from properties import Properties
from solver import Solver, surface_flux


class Model:
    def __init__(self, b, layout, constant_modes, a_bounds, k_bounds, beta):
        self.b = b
        self.p = Properties(b['masks'], b['z_km'], b['K_reference'], b['A_reference'],
                            layout, constant_modes, a_bounds, k_bounds, beta)
        self.solver = Solver(b['x'], b['y'], b['z_km'], b['horizontal_scale'], b['domain_mask'], self.p)

    def forward(self, c, initial=None):
        b, p = self.b, self.p
        p.set(c)
        T, K, KT = self.solver.solve(p.A, b['surface_T'], b['T_bot'], initial)
        q = surface_flux(T, K, p.A, b['z_km'][1]-b['z_km'][0])
        return dict(T=T, K=K, KT=KT, A=p.A, q=q)
