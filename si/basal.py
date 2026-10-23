"""Compare original, smoothed, uniform and published 100-km basal temperatures.

usage: python -m si.basal RESULTS

Writes boundaries, paired fits and comparison tables to RESULTS/si/basal.
"""
import runtime  # Set BLAS thread counts before importing NumPy.
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import gaussian_filter

from config import A_BOUNDS, BETA, INPUT, K_BOUNDS, SOURCES
from data_io import load_model, write_json
from fit import fit
from observations import load as load_observations
from objective import Problem
from si.regions import region_weights

CASES = ['original', 'smoothed', 'uniform', 'sun2022_c', 'zhang2025', 'yan2019']
LABELS = dict(original='Sun2022 Case B', smoothed='300-km smoothed', uniform='Uniform',
              sun2022_c='Sun2022 Case C + adiabat', zhang2025='Zhang2025', yan2019='Yan2019')
FILES = dict(sun2022_c=INPUT/'models/sun2022/sun2022_C.txt',
             zhang2025=SOURCES['Zhang2025'],
             yan2019=INPUT/'models/Yan2019_Velocity_and_Temperature_model.txt')


def boundary(b, case):
    T, core, area = b['T_bot'], b['core'], 400/b['horizontal_scale']**2
    mean = lambda f: np.sum((f*area)[core])/np.sum(area[core])
    if case == 'uniform':
        return np.full_like(T, mean(T))
    if case == 'smoothed':
        weight = area*b['domain_mask']
        smooth = gaussian_filter(T*weight, 300/20, mode='constant')/gaussian_filter(weight, 300/20, mode='constant')
        return smooth+mean(T)-mean(smooth)
    return T


def native_slice(source, case):
    """Extract an exact native depth; return lon, lat and temperature in source units."""
    if case == 'zhang2025':
        data = pd.read_excel(source, sheet_name='Temperature').to_numpy()
        return data[data[:, 2] == 100][:, [0, 1, 3]]
    data = np.loadtxt(source, skiprows=1)
    return data[data[:, 2] == 100][:, [0, 1, 5]]


def case_c_slice(data):
    """Interpolate each column; below its base, add 0.5 K/km to that native basal temperature."""
    columns = data.reshape(25, -1, 5)
    elevation, temperature = columns[:, :, 2], columns[:, :, 3]
    j = np.arange(elevation.shape[1])
    i = np.clip(np.sum(elevation < -100000, axis=0)-1, 0, len(columns)-2)
    fraction = (-100000-elevation[i, j])/(elevation[i+1, j]-elevation[i, j])
    T = temperature[i, j]*(1-fraction)+temperature[i+1, j]*fraction
    extended = elevation[0] > -100000
    T[extended] = temperature[0, extended]+.5*(100+elevation[0, extended]/1000)
    return np.column_stack([columns[0, :, :2], T]), extended


def interpolate_boundary(nodes, longitude, latitude):
    """Linear lon/lat interpolation, with explicit nearest-node extension outside the hull."""
    T = LinearNDInterpolator(nodes[:, :2], nodes[:, 2])(longitude, latitude)
    extended = ~np.isfinite(T)
    T[extended] = NearestNDInterpolator(nodes[:, :2], nodes[:, 2])(
        longitude[extended], latitude[extended])
    return T, extended


def moho_flux(z, moho, T, K):
    """Upward heat flow at the Moho in mW/m2, interpolated between cell faces."""
    q = 2*K[..., :-1]*K[..., 1:]/(K[..., :-1]+K[..., 1:])*np.diff(T, axis=-1)/np.diff(z)
    mid = (z[1:]+z[:-1])/2
    i = np.clip(np.searchsorted(mid, moho)-1, 0, len(mid)-2)[..., None]
    w = (moho[..., None]-mid[i])/(mid[i+1]-mid[i])
    return (np.take_along_axis(q, i, -1)*(1-w)+np.take_along_axis(q, i+1, -1)*w)[..., 0]


def prepare(results):
    """Prepare all basal fields with the adopted source units and continuation."""
    output = results/'si/basal'
    output.mkdir(parents=True)
    shutil.copy2(SOURCES['regions'], output/'regions.geojson')
    b = dict(np.load(results/'problem/bundle.npz'))
    fields = {case: boundary(b, case) for case in CASES[:3]}
    for case, source in FILES.items():
        if case == 'sun2022_c':
            nodes, below_lab = case_c_slice(np.loadtxt(source))
            table = pd.DataFrame(dict(longitude=nodes[:, 0], latitude=nodes[:, 1],
                temperature_source=nodes[:, 2], below_source_LAB=below_lab))
            fields[case+'_below_lab_fraction'] = LinearNDInterpolator(
                nodes[:, :2], below_lab.astype(float))(b['longitude'], b['latitude'])
        else:
            nodes = native_slice(source, case)
            if case == 'yan2019':
                raw = nodes[:, 2].copy()
                nodes[:, 2] -= 273.15  # Yan2019 Fig. 6: Temperature/K.
                table = pd.DataFrame(dict(longitude=nodes[:, 0], latitude=nodes[:, 1],
                    temperature_K=raw, temperature_C=nodes[:, 2]))
            else:
                table = pd.DataFrame(nodes, columns=['longitude', 'latitude', 'temperature_source'])
        table.to_csv(output/f'{case}_100km.csv', index=False)
        fields[case], fields[case+'_extended'] = interpolate_boundary(nodes, b['longitude'], b['latitude'])
    np.savez_compressed(output/'boundaries.npz', **fields)


def refit(results, case):
    output = results/'si/basal'
    target = output/case
    training = json.loads((results/'fit/training_config.json').read_text(encoding='utf-8'))
    T_bot = np.load(output/'boundaries.npz')[case]
    if case == 'original':
        shutil.copytree(results/'fit', target)
    else:
        problem = Problem(load_model(results/'problem'), load_observations(results/'problem'),
                          training['modes'], training['constant_modes'],
                          a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA,
                          regularization=training['regularization'])
        problem.b['T_bot'] = T_bot
        fit(problem, target, np.zeros(len(problem.p.names)), training['options'])
    np.save(target/'T_bot.npy', T_bot)
    write_json(target/'training_config.json', training)


def analyze(results):
    """Compare all cases using the same regional, observation and property statistics."""
    output = results/'si/basal'
    b = dict(np.load(results/'problem/bundle.npz'))
    boundaries = dict(np.load(output/'boundaries.npz'))
    area = 400/b['horizontal_scale']**2
    weights = dict(NCC=area*b['core']/area[b['core']].sum(), **region_weights(b, output/'regions.geojson'))
    base = dict(np.load(output/'original/result.npz'))
    rows, profiles, properties = [], [], []
    z = b['z_km']
    dz = np.ones(len(z))
    dz[[0, -1]] = .5
    for case in CASES:
        folder = output/case
        s = base if case == 'original' else dict(np.load(folder/'result.npz'))
        summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        row = dict(case=case, label=LABELS[case], objective=summary['objective'],
                   iterations=summary['iterations'], success=summary['success'],
                   stopping_reason=summary['message'], gradient_inf=summary['gradient_inf'],
                   T100_mean_C=float(np.sum(weights['NCC']*boundaries[case])))
        for key, value in summary['misfit'].items():
            row.update({key+'_'+metric: value[metric] for metric in ['n', 'rmse', 'bias', 'r']})
        row.update({key+'_loss': value for key, value in summary['losses'].items()})
        qmoho = moho_flux(z, b['thermal_interfaces_km'][..., 3], s['T'], s['K'])
        regional = {}
        for name, w in weights.items():
            temp = np.einsum('yx,yxz->z', w, s['T'])
            regional[name] = temp
            delta = s['T']-base['T']
            mean_delta = np.einsum('yx,yxz->z', w, delta)
            rms_delta = np.sqrt(np.einsum('yx,yxz->z', w, delta**2))
            for i, depth in enumerate(z):
                profiles.append(dict(case=case, region=name, depth_km=depth, T_C=temp[i],
                    delta_T_mean_C=mean_delta[i], delta_T_rms_C=rms_delta[i],
                    K_W_mK=float(np.sum(w*s['K'][..., i])),
                    A_uW_m3=float(np.sum(w*s['A'][..., i])*1e6)))
            row[name+'_moho_mW_m2'] = float(np.sum(w*qmoho))
        for depth in [20, 30, 40, 60]:
            index = np.flatnonzero(z == depth).item()
            row[f'ENCC_minus_WNCC_{depth}km_C'] = float(regional['ENCC'][index]-regional['WNCC'][index])
            row[f'N_Shanxi_minus_Ordos_{depth}km_C'] = float(regional['N Shanxi'][index]-regional['Ordos interior'][index])
        for i, name in enumerate(['sediment', 'upper_crust', 'middle_crust', 'lower_crust', 'mantle']):
            volume = weights['NCC'][..., None]*b['masks'][i]*dz
            for key in ['A', 'K']:
                value = float(np.sum(volume*s['layer_'+key][i])/volume.sum())
                reference = float(np.sum(volume*base['layer_'+key][i])/volume.sum())
                properties.append(dict(case=case, layer=name, field=key, mean=value,
                                       delta_percent=100*(value/reference-1)))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output/'metrics.csv', index=False)
    pd.DataFrame(profiles).to_csv(output/'depth_profiles.csv', index=False)
    pd.DataFrame(properties).to_csv(output/'layer_properties.csv', index=False)


def run(results):
    prepare(results)
    for case in CASES:
        print(f'Basal temperature: {case}', flush=True)
        refit(results, case)
    analyze(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    run(parser.parse_args().results)
