"""Reference heat production and conductivity from rock samples and measurements.

usage: python -m references RESULTS
Reads RESULTS/problem/bundle.npz; writes regional grids and calibration tables to RESULTS/references/.
"""
import runtime  # Set BLAS thread counts before importing NumPy.
import argparse
import csv
import io
import json
import re
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import pyogrio.raw
import shapely
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from geometry import composition_profiles, idw4, inside, regions

# The historical construction passed tables through CSV files; this keeps their float parsing.
reparse = lambda d: pd.read_csv(io.StringIO(d.to_csv(index=False)), low_memory=False)


def number(value):
    match = re.match(r'^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', unicodedata.normalize('NFKC', str(value)).strip().replace('−', '-'))
    return float(match[1]) if match else np.nan


def union(pairs, n):
    parent = np.arange(n)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in pairs:
        a, b = root(i), root(j)
        parent[max(a, b)] = min(a, b)
    return np.array([root(i) for i in range(n)])


def sample_heat_production(k2o, thorium, uranium):
    """McDonough et al. (2020), K/Th/U terms at 2700 kg/m3; inputs: K2O wt%, Th/U ppm."""
    potassium = np.asarray(k2o)*(2*39.0983/(2*39.0983+15.999))*.01
    return 2700*(3.387e-3*potassium+26.18*np.asarray(thorium)*1e-6+98.29*np.asarray(uranium)*1e-6)


def heat_production(output, georoc, region_path):
    """SiO2-A relations: median A of NCC whole-rock samples in 2 wt% SiO2 windows."""
    rows, citations = [], {}
    for path in georoc:
        text = path.read_text(encoding='cp1252')
        citations.update({m[1]: m[3].lower().rstrip('.,;') for line in text.splitlines()
                          if (m := re.match(r'^"?\[(\d+)\]\s+([^,].*doi:\s*(10\.[^\s"<>]+))', line, re.I))})
        rows += [r for r in csv.DictReader(io.StringIO(text)) if 'WR' in (r.get('MATERIAL') or '')]
    raw = pd.DataFrame(rows).drop_duplicates('UNIQUE_ID')
    value = lambda c: pd.to_numeric(raw[c], errors='coerce')
    lon0, lon1, lat0, lat1 = (value(c) for c in ['LONGITUDE MIN', 'LONGITUDE MAX', 'LATITUDE MIN', 'LATITUDE MAX'])
    d = pd.DataFrame(dict(
        unique_id=raw.UNIQUE_ID, sio2=value('SIO2(WT%)'), k2o=value('K2O(WT%)'), th=value('TH(PPM)'), u=value('U(PPM)'),
        lon=(lon0+lon1)/2, lat=(lat0+lat1)/2, span=np.maximum(lon1-lon0, lat1-lat0), rock=raw['ROCK TYPE'],
        study=raw.CITATIONS.map(lambda v: '|'.join(sorted(citations.get(i, 'georoc-ref-'+i) for i in re.findall(r'\[(\d+)\]', v)))
                                or 'georoc-reference-missing'),
        sample=raw['SAMPLE NAME'].fillna('').str.replace(r'\[[^]]+\]', '', regex=True).str.lower()
        .str.replace(r'^s_', '', regex=True).str.replace(r'[^a-z0-9]', '', regex=True)))
    d['A'] = sample_heat_production(d.k2o, d.th, d.u)  # uW/m3
    d = d[d.rock.isin(['PLU', 'VOL', 'MET']) & d.sio2.between(40, 85) & d.k2o.between(0, 20) & (d.th > 0) & (d.u > 0)
          & ~raw['ROCK NAME'].fillna('').str.contains(
              r'carbonat|quartzite|serpent|peridot|harzburg|lherzolit|dunite|kimberlit|pegmat|\bore\b|\bvein\b', case=False)
          & ~raw.ALTERATION.isin(['E', 'S', 'M'])].copy()
    # Records citing a shared paper form one publication group.
    tokens = sorted({t for s in d.study for t in s.split('|')})
    index = {t: i for i, t in enumerate(tokens)}
    groups = union([(index[s.split('|')[0]], index[t]) for s in d.study for t in s.split('|')[1:]], len(tokens))
    d['study'] = [tokens[groups[index[s.split('|')[0]]]] for s in d.study]
    d = d[~d.duplicated(['study', 'sample', 'sio2', 'k2o', 'th', 'u', 'lon', 'lat'])]
    d = d[(d.span.fillna(0) <= .5) & inside(regions(region_path), d.lon, d.lat)]
    d['used_all_rocks'] = d.sio2.between(40, 80)
    d['used_metamorphic'] = d.used_all_rocks & d.rock.eq('MET')
    d.to_csv(output/'georoc_ncc.csv', index=False)
    d = pd.read_csv(output/'georoc_ncc.csv')
    centres = np.arange(42., 78.01, .5)

    def quartiles(d, n):
        windows = [d[np.abs(d.sio2-c) <= 2+1e-10] for c in centres]
        return np.array([np.quantile(w.A, [.25, .5, .75]) if len(w) >= n and w.study.nunique() >= 5 else np.full(3, np.nan)
                         for w in windows])
    all_rocks, metamorphic = quartiles(d, 30), quartiles(d[d.rock == 'MET'], 20)
    pd.DataFrame(dict(sio2_wt_percent=centres,
        all_rocks_q25_uW_m3=all_rocks[:, 0], all_rocks_uW_m3=all_rocks[:, 1], all_rocks_q75_uW_m3=all_rocks[:, 2],
        metamorphic_q25_uW_m3=metamorphic[:, 0], metamorphic_uW_m3=metamorphic[:, 1],
        metamorphic_q75_uW_m3=metamorphic[:, 2])).to_csv(output/'heat_production_relation.csv', index=False)
    return centres, all_rocks, metamorphic


def sediment(source):
    """Near-surface heat production of Sun et al. (2022), Table S1 (values in uW/m3)."""
    with zipfile.ZipFile(source) as archive:
        table = ET.fromstring(archive.read('word/document.xml')).find('.//{*}tbl')
    return np.array([[float(''.join(c.itertext())) for c in row.findall('{*}tc')] for row in table.findall('{*}tr')[1:]])


def deep_conductivity(output, source):
    """P3 laboratory conductivity reduced to 20 C (crust) or to the mantle-law coefficient, per layer."""
    with open(source, encoding='latin1', newline='') as f:
        lines = [r for r in csv.reader(f, delimiter='\t') if r and any(v.strip() for v in r)]
    start = next(i for i, r in enumerate(lines) if r[0].strip() == 'Sample ID')
    p3 = pd.DataFrame(lines[start+1:], columns=[' '.join(v.split()) for v in lines[start]])
    value = lambda c: pd.to_numeric(p3[c].str.replace(',', '.', regex=False), errors='coerce')
    porosity = ['Value total porosity (%)', 'Value effective porosity (%)']
    d = reparse(pd.DataFrame(dict(sample=p3['Sample ID'], reference=p3['Primary Reference (Bibtexkey)'], rock=p3['Petrographic term'],
                                  T_K=value('Temperature (K)'), P_Pa=value('Pressure (Pa)'), fluid=p3['Saturating Fluid'],
                                  K=value('Value bulk thermal conductivity (W/(m·K))'))))
    phi = p3[p3[porosity].apply(lambda c: c.str.strip().ne('')).any(axis=1)].assign(**{c: value(c) for c in porosity})
    d = d[d.K > 0].join(phi.groupby(['Sample ID', 'Primary Reference (Bibtexkey)'])[porosity].median().max(axis=1).rename('phi'),
                        on=['sample', 'reference'])

    def group(rock):
        text = str(rock).lower()
        if re.search(r'serpentin|weather|altered|karbonat,|sediment|\bmixed\b', text):
            return 'other'
        return next((g for g, p in [('felsic', r'granit|granodiorit|tonalit'), ('gneiss', r'gneiss|gneis'),
                                    ('mafic_metamorphic', r'amphibol|granulit'), ('mafic_intrusive', r'gabbro|diorit|anorthosit'),
                                    ('ultramafic', r'peridot|dunit|lherzol|harzburg|wehrlit|pyroxenit')] if re.search(p, text)), 'other')
    d['group'], d['T_C'] = d.rock.map(group), d.T_K-273.15
    keep = d.group.ne('other') & d.T_C.between(0, 1400) & ~d.P_Pa.gt(1e6) & ~d.phi.gt(5) & ~d.fluid.fillna('').str.contains('isooctane', case=False)
    d = reparse(d)[keep.to_numpy()]
    # A multi-temperature curve republished under another reference keeps the earliest reference.
    year = lambda r: int(m.group()) if (m := re.search(r'(?:19|20)\d{2}', r)) else 9999
    seen, drop = {}, set()
    for (ref, sample), part in d.groupby(['reference', 'sample']):
        points = part.groupby('T_C').K.median()
        if len(points) < 3 or points.nunique() < 3:
            continue
        key = (part.group.iloc[0], tuple(np.round(points.index.to_numpy(), 3)), tuple(np.round(points.to_numpy(), 5)))
        if key in seen and seen[key][0] != ref:
            seen[key], dropped = sorted([seen[key], (ref, sample)], key=lambda s: year(s[0]))
            drop.add(dropped)
        else:
            seen[key] = (ref, sample)
    d = d[[k not in drop for k in zip(d.reference, d['sample'])]]
    result, records, samples, parameters = [], [], [], []
    # Upper, middle and lower crust with their Chapman (1986) coefficients; mantle with the Doin-Fleitout law.
    for layer, (rocks, beta) in zip(['upper_crust', 'middle_crust', 'lower_crust', 'mantle'], [(['felsic', 'gneiss'], .001),
            (['gneiss', 'mafic_intrusive', 'mafic_metamorphic'], .0005),
            (['mafic_intrusive', 'mafic_metamorphic'], .00015), (['ultramafic'], None)]):
        part = d[d.group.isin(rocks)]
        tk = part.T_C+273.15
        factor = 1.23e-10*tk**3+1/(.518+7.89e-4*tk) if beta is None else 1/(1+beta*(part.T_C-20))
        part = part.assign(layer=layer, correction_factor=factor, log_K0=np.log(part.K/factor))
        s = part.groupby(['group', 'reference', 'sample']).log_K0.mean().reset_index()
        # Equal weight per rock group, per reference within a group and per sample within a reference.
        w = 1/s.groupby('group').reference.transform('nunique')/s.groupby(['group', 'reference']).log_K0.transform('size')/len(rocks)
        s['weight'] = w/w.sum()
        log_K0 = float(s.weight@s.log_K0)
        result.append(log_K0)
        records.append(part)
        samples.append(s.assign(layer=layer))
        f20 = 1.23e-10*293.15**3+1/(.518+7.89e-4*293.15) if beta is None else 1.
        parameters.append(dict(layer=layer, law='Doin-Fleitout' if beta is None else 'Chapman',
            beta_per_C=beta, log_K0=log_K0, K0=np.exp(log_K0), K20_W_mK=np.exp(log_K0)*f20,
            measurements=len(part), samples=len(s)))
    pd.concat(records, ignore_index=True).to_csv(output/'deep_conductivity_records.csv', index=False)
    pd.concat(samples, ignore_index=True).to_csv(output/'deep_conductivity_samples.csv', index=False)
    pd.DataFrame(parameters).to_csv(output/'deep_conductivity_parameters.csv', index=False)
    return np.array(result)


def conductivity_records(ihfc_source, heatflow_source, forward):
    """Interval conductivity of the fifth Chinese compilation and IHFC (2024), screened and de-duplicated."""
    text = ihfc_source.read_bytes().decode('latin1')
    ihfc = pd.read_csv(io.StringIO(text), sep='\t', low_memory=False,
                       skiprows=next(i for i, line in enumerate(text.splitlines()) if line.startswith('q\tq_uncertainty\t')))
    ihfc = reparse(ihfc[ihfc.long_EW.between(73, 135) & ihfc.lat_NS.between(18, 54)])
    rows = []
    for r in json.loads(heatflow_source.read_text(encoding='utf-8-sig')):
        parts = re.split(r'~|～|–|—|-', unicodedata.normalize('NFKC', str(r['g'])).replace(',', '').replace(' ', ''))
        top, bottom = (number(parts[0]), number(parts[1])) if len(parts) == 2 else (np.nan, number(r['g']))
        rows.append(dict(record_id=f"China5th2024:{r['a']}", source_id='China5th2024', source_locator=str(r['a']), site_name=r['f'],
                         longitude=number(r['b']), latitude=number(r['c']), k_W_mK=number(r['j']), k_sd_W_mK=number(r['k']),
                         depth_top_m=top, depth_bottom_m=bottom, original_reference=r['n'], quality=r['p'], priority=10,
                         status='measured', basis='site'))
    for _, r in ihfc[ihfc.tc_mean.notna()].iterrows():
        source, method, location = str(r.tc_source), str(r.tc_method), str(r.tc_location)
        rows.append(dict(record_id='IHFC2024:'+r.ID, source_id='IHFC2024', source_locator=r.ID, site_name=str(r['name']),
                         longitude=number(r.long_EW), latitude=number(r.lat_NS), k_W_mK=number(r.tc_mean),
                         k_sd_W_mK=number(r.tc_uncertainty), depth_top_m=number(r.q_top), depth_bottom_m=number(r.q_bottom),
                         original_reference=str(r.publication_reference), quality='', priority=40,
                         status='estimated' if 'Assumed' in source or 'Estimation' in method else 'measured',
                         basis='elsewhere' if 'Other location' in location else
                         'literature' if 'Literature' in location and 'Actual heat-flow location' not in location else 'site'))
    d = pd.DataFrame(rows)
    d = reparse(d[d.longitude.between(-180, 180) & d.latitude.between(-90, 90) & d.k_W_mK.gt(0)]).fillna({'site_name': '', 'original_reference': ''})

    # Duplicates within 200 m: equal K (0.015) at equal depth (5 m) within 10 m, or 100 m in the same year; or one numbered well.
    lon, lat = np.deg2rad(d.longitude), np.deg2rad(d.latitude)
    xyz = np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    year = lambda v: m[0] if (m := re.search(r'(?:19|20)\d{2}', str(v))) else ''
    names = d.site_name.astype(str).str.replace(r'\W', '', regex=True).str.lower()
    wells = names.where(names.str.contains(r'\d') & (names.str.len() > 2), '').to_numpy()   # numbered well names
    pairs = []
    for i, j in sorted(cKDTree(xyz).query_pairs(.2/6371)):
        a, b = d.iloc[i], d.iloc[j]
        distance = float(np.linalg.norm(xyz[i]-xyz[j])*6371)
        da, db = np.array([a.depth_top_m, a.depth_bottom_m]), np.array([b.depth_top_m, b.depth_bottom_m])
        both = np.isfinite(da) & np.isfinite(db)
        same_depth = bool(both.all() and np.max(abs(da-db)) <= 5)
        same_year = bool(year(a.original_reference) and year(a.original_reference) == year(b.original_reference))
        same_well = bool(wells[i] and wells[i] == wells[j])
        if (abs(a.k_W_mK-b.k_W_mK) <= .015 and not a.source_id == b.source_id != 'IHFC2024'
                and ((same_depth and (distance <= .01 or (same_year and distance <= .1)))
                     or (same_well and distance <= .2 and (same_depth or not both.any())))):
            pairs.append((i, j))
    d['group'] = union(pairs, len(d))
    members = d.groupby('group').record_id.agg(lambda v: '|'.join(sorted(set(v))))
    d = d.loc[d.sort_values(['priority', 'record_id']).groupby('group', sort=False).head(1).index].sort_values(['source_id', 'record_id'])
    d['merged'] = d.group.map(members)
    # Two fifth-edition records conflict with their source publications.
    d = reparse(d[~d.record_id.isin(['China5th2024:2267', 'China5th2024:1426'])])
    d['X'], d['Y'] = forward.transform(d.longitude, d.latitude)
    lookup = ihfc.set_index('ID')
    d['ratio'] = d.source_locator.where(d.source_id.eq('IHFC2024')).map(lookup.q/(lookup.tc_mean*lookup.T_grad_mean))
    valid = (d.longitude.between(73, 135) & d.latitude.between(18, 54) & d.quality.fillna('').str.upper().ne('D')
             & d.status.eq('measured') & d.basis.eq('site') & ~(d.source_id.eq('IHFC2024') & d.original_reference.eq('Zhang_etal._1992'))
             & ~(d.ratio.notna() & ~d.ratio.between(.5, 2.)))
    for factor in [.4184, 1/.4184, 4.184, 1/4.184, 1000., .001]:   # unit-conversion signatures in q/(K G)
        valid &= ~d.ratio.between(factor*.9, factor*1.1)
    # One source per 1 km site, the fifth compilation first; IHFC repeats of it within 100 m are dropped.
    d['site_id'] = np.floor(d.X/1000).astype(int).astype(str)+':'+np.floor(d.Y/1000).astype(int).astype(str)
    d['rank'] = np.where(d.source_id.eq('China5th2024'), 0, 2)
    valid &= d['rank'].eq(d.site_id.map(d[valid].groupby('site_id')['rank'].min()))
    fifth = d[valid & d.source_id.eq('China5th2024')]
    tree = cKDTree(fifth[['X', 'Y']])
    for idx, r in d[valid & d.source_id.ne('China5th2024')].iterrows():
        years = re.findall(r'(?:19|20)\d{2}', str(r.original_reference))
        for j in tree.query_ball_point([r.X, r.Y], 100):
            o = fifth.iloc[j]
            other = re.findall(r'(?:19|20)\d{2}', str(o.original_reference))
            if (years and other and abs(int(years[-1])-int(other[-1])) <= 1 and np.isfinite([r.depth_bottom_m, o.depth_bottom_m]).all()
                    and abs(r.depth_bottom_m-o.depth_bottom_m) <= 5
                    and (not np.isfinite([r.depth_top_m, o.depth_top_m]).all() or abs(r.depth_top_m-o.depth_top_m) <= 5)):
                valid.loc[idx] = False
                break
    d = reparse(d[valid])
    d[['X', 'Y']] /= 1000
    d['reference_group'] = d.original_reference.where(~(d.original_reference.isna() | d.original_reference.eq('未发表')), 'unknown:'+d.site_id)
    d = reparse(d)
    # Measurement state (in situ, ambient, unknown) from IHFC records merged with equal K.
    states = []
    for r in d.itertuples():
        keys = {v.split(':', 1)[1] for v in r.merged.split('|') if v.startswith('IHFC2024:')} | ({r.source_locator} if r.source_id == 'IHFC2024' else set())
        matched = lookup.loc[sorted(keys)]
        if keys and np.max(abs(matched.tc_mean.to_numpy(float)-r.k_W_mK)) > .01500001:
            matched = lookup.iloc[:0]
        text = ' | '.join(sorted({str(v).strip() for v in matched.tc_pT_conditions
                                  if str(v).strip().lower() not in {'', 'nan', 'none', 'unspecified', '[unspecified]'}})).lower()
        states.append('in_situ' if 'in-situ' in text else 'ambient' if 'ambient' in text else 'unknown')
    d['state'] = states
    return reparse(d)


def depth_kernel(a, b, c, e):
    """Exponential depth correlation (0.5 km) averaged over intervals [a, b] and [c, e]."""
    a, b, c, e = np.broadcast_arrays(a, b, c, e)
    h, j = b-a, e-c
    both = (h > 1e-8) & (j > 1e-8)
    first, second = (h > 1e-8) & ~both, (j > 1e-8) & ~both
    out = np.exp(-abs(a-c)/.5)
    primitive = lambda v: .5*abs(v)+.25*np.exp(-abs(v)/.5)
    integral = lambda v: np.sign(v)*.5*(-np.expm1(-abs(v)/.5))
    out[both] = (primitive(b-c)-primitive(a-c)-primitive(b-e)+primitive(a-e))[both]/(h*j)[both]
    out[first] = (integral(b-c)-integral(a-c))[first]/h[first]
    out[second] = (integral(e-a)-integral(c-a))[second]/j[second]
    return np.clip(out, 0, 1)


def shallow_conductivity(d, x, y, output, glim, inverse):
    """GP of log interval conductivity (150 km, 0.5 km, nugget 0.25) with a GLiM lithology mean; median at 0-7.5 km."""
    classes = ['su', 'ss', 'py', 'sm', 'sc', 'ev', 'va', 'vi', 'vb', 'pa', 'pi', 'pb', 'mt', 'wb', 'ig', 'nd']
    # GLiM classes merged into unconsolidated, siliciclastic, carbonate, igneous, metamorphic and other.
    merge = np.eye(6)[[0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5]]
    lon, lat = inverse.transform(x, y)
    gdb = '/vsizip/'+glim.as_posix()+'/LiMW_GIS 2015.gdb'
    native = Transformer.from_crs(4326, pyogrio.read_info(gdb, layer='GLiM_export')['crs'], always_xy=True)
    px, py = native.transform(np.r_[d.longitude, lon.ravel()], np.r_[d.latitude, lat.ravel()])
    _, _, geometry, fields = pyogrio.raw.read(gdb, layer='GLiM_export', columns=['xx'],
                                              bbox=(px.min()-6e4, py.min()-6e4, px.max()+6e4, py.max()+6e4))
    polygons = shapely.from_wkb(geometry)
    codes, tree, area = np.char.lower(np.asarray(fields[0], dtype=str)), shapely.STRtree(polygons), shapely.area(polygons)

    def classify(lon, lat):
        pairs = tree.query(shapely.points(*native.transform(lon, lat)), predicate='covered_by')
        pairs = pairs[:, np.lexsort((area[pairs[1]], pairs[0]))]   # the smallest covering polygon
        unique, first = np.unique(pairs[0], return_index=True)
        result = np.full(len(lon), classes.index('nd'))
        result[unique] = [classes.index(c) for c in codes[pairs[1, first]]]
        return result
    d['glim'] = np.array(classes)[classify(d.longitude.to_numpy(), d.latitude.to_numpy())]
    d = reparse(d)
    fractions = np.zeros((x.size, len(classes)))   # 5x5 sub-cell lithology fractions of each 20 km cell
    for dy in [-8e3, -4e3, 0, 4e3, 8e3]:
        for dx in [-8e3, -4e3, 0, 4e3, 8e3]:
            fractions[np.arange(x.size), classify(*inverse.transform(x.ravel()+dx, y.ravel()+dy))] += 1/25

    def design(z, lithology):
        """Intercept, log(1+z), and merged lithology groups attenuated as exp(-z/2), at interval midpoints z."""
        return np.column_stack([np.ones(len(z)), np.log1p(z), *(lithology@merge*np.exp(-z/2)[:, None]).T])

    t = d[d.state.ne('in_situ') & d.depth_top_m.ge(0) & d.depth_bottom_m.gt(d.depth_top_m) & d.depth_bottom_m.le(7500)].reset_index(drop=True)
    t.to_csv(output/'shallow_conductivity_samples.csv', index=False)
    a, b, xy = t.depth_top_m.to_numpy(float)/1000, t.depth_bottom_m.to_numpy(float)/1000, t[['X', 'Y']].to_numpy(float)
    F = design((a+b)/2, np.eye(len(classes))[[classes.index(c) for c in t.glim]])
    target = np.log(t.k_W_mK.to_numpy(float))
    weights = 1/t.groupby('site_id').record_id.transform('size').to_numpy()
    # The ridge also fixes the depth extrapolation: intercept, log(1+z) and lithology terms are nearly collinear.
    penalty = np.eye(F.shape[1])*12
    penalty[0, 0], penalty[1, 1] = 1e-8, 2.
    normal = F.T@(weights[:, None]*F)+penalty
    beta = np.linalg.solve(normal, F.T@(weights*target))
    residual = target-F@beta
    variance = max(float(np.average(residual**2, weights=weights)), 1e-6)
    C = np.exp(-cdist(xy, xy)/150)
    C *= depth_kernel(a[:, None], b[:, None], a[None, :], b[None, :])
    C *= variance*.75
    C[np.diag_indices_from(C)] += variance*.25
    alpha = cho_solve(cho_factor(C, lower=True, check_finite=False), residual, check_finite=False)
    query = np.c_[x.ravel(), y.ravel()]/1000
    depths = np.arange(0, 7.5001, .5)   # to the deepest sediment base
    K = np.zeros((x.size, len(depths)))
    for k, depth in enumerate(depths):
        for start in range(0, x.size, 512):
            rows = slice(start, min(start+512, x.size))
            z = np.full(rows.stop-start, depth)
            cross = np.exp(-cdist(query[rows], xy)/150)
            cross *= depth_kernel(z[:, None], z[:, None], a[None, :], b[None, :])
            K[rows, k] = np.exp(design(z, fractions[rows])@beta+variance*.75*cross@alpha)   # predictive median
    return K.reshape(*x.shape, len(depths))


def heat_production_curve(silica, centres, values):
    """Linear interpolation between empirical SiO2 windows, as float32 per CSRM node."""
    right = np.searchsorted(centres, silica, side='right').clip(1, len(centres)-1)
    weight = (silica-centres[right-1])/(centres[right]-centres[right-1])
    return (values[right-1]*(1-weight)+values[right]*weight).astype('f4')


def regional_fields(x, y, z, masks, A_ref, K_ref, profiles, forward, a_bounds):
    """Map the empirical reference products onto the prepared five-layer regional grid."""
    sites, group = np.unique(np.column_stack(forward.transform(A_ref['sediment_longitude'][:], A_ref['sediment_latitude'][:])),
                             axis=0, return_inverse=True)
    medians = np.array([np.median(A_ref['sediment_uW_m3'][:][group.ravel() == i]) for i in range(len(sites))])
    query = np.column_stack([x.ravel(), y.ravel()])
    sediment_A = LinearNDInterpolator(sites, medians)(query)
    missing = ~np.isfinite(sediment_A)
    sediment_A[missing] = medians[cKDTree(sites).query(query[missing])[1]]
    index, weights = idw4(np.column_stack(forward.transform([p[0] for p in profiles], [p[1] for p in profiles])), query)
    sio2, crystalline = A_ref['sio2_wt_percent'][:], []
    for values in [lambda s: heat_production_curve(s, sio2, A_ref['all_rocks_uW_m3'][:]),
                   lambda s: np.interp(s, sio2, A_ref['metamorphic_uW_m3'][:]).astype('f4')]:
        used = np.unique(index)
        columns = np.array([np.interp(z, profiles[i][2][:, 0], values(profiles[i][2][:, 1])) for i in used])
        crystalline.append(np.sum(columns[np.searchsorted(used, index)]*weights[..., None], axis=1).reshape(*x.shape, len(z)))
    A = np.stack([np.broadcast_to(sediment_A.reshape(x.shape)[..., None], crystalline[0].shape), crystalline[0],
                  crystalline[1], crystalline[1], np.full(crystalline[0].shape, A_ref['mantle_uW_m3'][()])])
    A = np.where(masks > 1e-30, A, a_bounds.mean(axis=1)[:, None, None, None])

    shallow = K_ref['shallow_W_mK'][:]
    shallow = shallow[..., np.minimum(np.arange(len(z)), shallow.shape[-1]-1)]
    mu = K_ref['deep_log_K0'][:]
    blend = np.exp(-np.maximum(z-5, 0)/.5)
    K = np.stack([shallow, np.exp(blend*np.log(shallow)+(1-blend)*mu[0]),
                  *[np.full(shallow.shape, np.exp(m)) for m in mu[1:]]])
    K[4] *= 1.23e-10*293.15**3+1/(.518+7.89e-4*293.15)
    return A, K


def build(b, output, georoc, region_path, sediment_source, p3_source, ihfc_source, heatflow_source, glim, csrm, crs, a_bounds):
    layers = ['sediment', 'upper_crust', 'middle_crust', 'lower_crust', 'mantle']
    forward = Transformer.from_crs(4326, crs, always_xy=True)
    inverse = Transformer.from_crs(crs, 4326, always_xy=True)
    output.mkdir()
    centres, all_rocks, metamorphic = heat_production(output, georoc, region_path)
    sun = sediment(sediment_source)
    deep = deep_conductivity(output, p3_source)
    shallow = shallow_conductivity(conductivity_records(ihfc_source, heatflow_source, forward),
                                   b['x'], b['y'], output, glim, inverse)
    with h5py.File(output/'references.h5', 'w') as h:
        h.attrs.update(layers=json.dumps(layers),
                       crs=crs, field_axis_order='y,x,z', layer_field_axis_order='layer,y,x,z')
        for key in ['x', 'y', 'z_km', 'longitude', 'latitude', 'domain_mask', 'core', 'masks', 'thermal_interfaces_km']:
            h.create_dataset('grid/'+key, data=b[key], compression='gzip', shuffle=True)
        h['heat_production/sio2_wt_percent'] = centres
        h['heat_production/all_rocks_uW_m3'] = all_rocks[:, 1]    # median; upper crust
        h['heat_production/metamorphic_uW_m3'] = metamorphic[:, 1]  # median; middle and lower crust
        h['heat_production/all_rocks_quartiles_uW_m3'] = all_rocks[:, [0, 2]]
        h['heat_production/metamorphic_quartiles_uW_m3'] = metamorphic[:, [0, 2]]
        h['heat_production'].attrs.update(window_half_width_wt_percent=2., min_all_rocks_samples=30,
                                         min_metamorphic_samples=20, min_publication_groups=5)
        h['heat_production'].attrs.update(sample_formula_source='10.1029/2019GC008865; K, Th and U terms',
            sample_density_kg_m3=2700., K_mass_fraction_in_K2O=2*39.0983/(2*39.0983+15.999),
            sample_formula='2700*(0.003387*K_mass_fraction+26.18*Th_mass_fraction+98.29*U_mass_fraction)')
        h['heat_production/sediment_longitude'], h['heat_production/sediment_latitude'], h['heat_production/sediment_uW_m3'] = sun.T
        mantle = h.create_dataset('heat_production/mantle_uW_m3', data=.03)
        mantle.attrs.update(units='uW/m3', source='Prescribed uniform mantle heat production')
        h['conductivity/shallow_W_mK'] = shallow
        h['conductivity/shallow_depth_km'] = np.arange(0, 7.5001, .5)
        h['conductivity/deep_log_K0'] = deep
        h['conductivity/deep_log_K0'].attrs['layers'] = json.dumps(layers[1:])
        A, K = regional_fields(b['x'], b['y'], b['z_km'], b['masks'], h['heat_production'], h['conductivity'],
                                composition_profiles(csrm), forward, a_bounds)
        for name, field, units in [('A_reference', A, 'uW/m3'), ('K_reference', K, 'W/m/K'),
                ('A', np.sum(b['masks']*A, axis=0), 'uW/m3'), ('K20', 1/np.sum(b['masks']/K, axis=0), 'W/m/K')]:
            dataset = h.create_dataset('fields/'+name, data=field, compression='gzip', shuffle=True)
            dataset.attrs['units'] = units
        h['fields'].attrs['conductivity_state'] = '20 C base values; the solver applies temperature and depth factors'


if __name__ == '__main__':
    from config import A_BOUNDS, CRS, GEOROC, SOURCES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    cli = parser.parse_args()
    build(dict(np.load(cli.results/'problem/bundle.npz')), cli.results/'references', GEOROC, SOURCES['regions'],
          SOURCES['sediment_A'], SOURCES['P3'], SOURCES['IHFC'], SOURCES['heatflow'], SOURCES['GLiM'], SOURCES['csrm'], CRS, A_BOUNDS)
