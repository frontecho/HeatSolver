"""Heat flux and conductivity/temperature derivatives of finite-volume operators."""
from __future__ import annotations

import numpy as np


def upward_flux(T, K, dz_km):
    return 2*K[..., :-1]*K[..., 1:]/(K[..., :-1]+K[..., 1:])*np.diff(T, axis=-1)/dz_km


def flux_tangent(T, K, derivative, tangent, dz_km):
    ka, kb = K[..., :-1], K[..., 1:]
    face = 2*ka*kb/(ka+kb)
    dface = (2*kb**2*derivative[..., :-1]*tangent[..., :-1]+
             2*ka**2*derivative[..., 1:]*tangent[..., 1:])/(ka+kb)**2
    return (dface*np.diff(T, axis=-1)+face*np.diff(tangent, axis=-1))/dz_km


def conductivity_pullback(T,K,lam,solver):
    """Transpose of explicit_residual w.r.t. cell K, including boundary faces.

lam is zero at prescribed T nodes. One face traversal replaces a separate
3-D residual allocation for every horizontal K basis mode.
"""
    result = np.zeros_like(K); s = solver.linear_solver
    for axis,spacing,weight in [(0,s.dy,s.horizontal_face_y),(1,s.dx,s.horizontal_face_x),(2,s.dz,s.horizontal_area_factor)]:
        aa,bb = [slice(None)]*3,[slice(None)]*3
        aa[axis],bb[axis] = slice(None,-1),slice(1,None)
        a,b = tuple(aa),tuple(bb); ka,kb = K[a],K[b]
        common = -2*weight/spacing**2*(lam[a]-lam[b])*(T[a]-T[b])/(ka+kb)**2
        result[a] += common*kb**2; result[b] += common*ka**2
    return result


def explicit_residual(T,K,dK,solver):
    """Explicit derivative of the frozen-coefficient finite-volume residual."""
    result=np.zeros_like(T)
    s=solver.linear_solver
    for axis,spacing,weight in [(0,s.dy,s.horizontal_face_y),
            (1,s.dx,s.horizontal_face_x),(2,s.dz,s.horizontal_area_factor)]:
        aa,bb=[slice(None)]*3,[slice(None)]*3
        aa[axis],bb[axis]=slice(None,-1),slice(1,None)
        a,b=tuple(aa),tuple(bb)
        ka,kb=K[a],K[b]
        dface=2*(kb*kb*dK[a]+ka*ka*dK[b])/(ka+kb)**2
        flux=weight/spacing**2*dface*(T[a]-T[b])
        result[a]+=flux
        result[b]-=flux
    return result


def explicit_flux(T,K,dK,dz_km):
    ka,kb=K[...,:-1],K[...,1:]
    dface=2*(kb*kb*dK[...,:-1]+ka*ka*dK[...,1:])/(ka+kb)**2
    return dface*np.diff(T,axis=-1)/dz_km


def surface_flux_temperature_gradient(T, K, KT, weights, dz_km):
    """Transpose derivative of upward top flux, including K(T), in mW/m2."""
    a, b = K[..., 0], K[..., 1]
    dt = T[..., 1]-T[..., 0]
    face = 2*a*b/(a+b)
    result = np.zeros_like(T)
    result[..., 0] = weights*(-face+dt*2*b*b/(a+b)**2*KT[..., 0])/dz_km
    result[..., 1] = weights*(face+dt*2*a*a/(a+b)**2*KT[..., 1])/dz_km
    return result
