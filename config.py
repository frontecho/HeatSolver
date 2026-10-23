"""Inputs, grid, observation selection and model settings."""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
INPUT = HERE/'inputs'

SOURCES = dict(
    heatflow=INPUT/'heatflow/china5_heatData.json',
    curie=INPUT/'curie/cpd_result.xyz',
    csrm=INPUT/'csrm',
    crust1=INPUT/'crust1/crust1.bnds',
    surface_T=INPUT/'surface_T/LSTC_average.tif',
    bottom_T=INPUT/'models/sun2022/sun2022_B.txt',
    sediment_A=INPUT/'models/sun2022/Table S1.docx',
    countries=INPUT/'regions/countries.geojson',
    provinces=INPUT/'regions/global_gprv.kml',
    regions=INPUT/'regions/ncc_regions.geojson',
    P3=INPUT/'p3/P3_PetroPhysicalProperty Database_V1.txt',
    IHFC=INPUT/'ihfc/IHFC_2024_GHFDB_v.2026.03.txt',
    GLiM=INPUT/'glim/LiMW_GIS_2015.gdb.zip',
    Zhang2025=INPUT/'models/Anqi_Zhang_NCC_results.xlsx',
    Yang2026=INPUT/'models/YangXiaoyu_NCClitmod.xlsx',
    relief=INPUT/'relief_noaa/relief_noaa_98_124_30_45.nc'
)
GEOROC = sorted(INPUT.glob('georoc/*/*.csv'))

GMT = HERE/'.tools/gmt-6.7.0/bin/gmt.exe'

CRS = '+proj=lcc +lat_1=30 +lat_2=45 +lat_0=37 +lon_0=113 +datum=WGS84 +units=m +no_defs'
X = np.arange(-1440e3, 1440e3+1, 20e3)
Y = np.arange(-720e3, 880e3+1, 20e3)
Z = np.arange(0, 100.25, .5)

BOX = (100, 122, 33, 44)  # west, east, south, north (degrees)
LAYERS = ['sediment', 'upper_crust', 'middle_crust', 'lower_crust', 'mantle']

K_BOUNDS = np.array([[1.2, 4.0], [2.5, 3.3], [2.0, 2.8], [1.8, 2.5], [3.0, 4.5]])
A_BOUNDS = np.array([[.2, 3.0], [.5, 3.0], [.2, 1.5], [.05, .8], [.005, .05]])
BETA = np.array([.001, .001, .0005, .00015])

# DCT grid (ny, nx), including the (0, 0) slot; None disables the layer.
MODES = dict(A=[(3, 5), (5, 9), (4, 7), (3, 5), None],
             K=[(5, 9), (3, 5), (2, 3), (2, 2), (1, 1)])
# Retain the (0, 0) coefficient independently for each A/K layer.
CONSTANT_MODES = dict(A=[False, False, False, False, False],
                      K=[True, True, True, True, True])

REGULARIZATION = dict(A=4., K=4.)
FIT_OPTIONS = dict(maxiter=60, maxfun=250, maxls=25, maxcor=40, ftol=2e-9, gtol=1e-4)

# Uncertainty sampling and error amplitudes.
CALIBRATION_SETTINGS = dict(
    seed=20261021, bootstrap_replicates=500, calibration_retained_fraction=.99,
    calibration_tail_draws=8, gp_draws=64,
    boundary_scenario_sd_C=dict(
        surface=dict(offset=1.5, east_west=0., north_south=0.),
        bottom=dict(offset=50., east_west=0., north_south=0.)),
    A_scale_scenario_sd_percent=dict(sediment=0., mantle=0.)
)

FIELD_SETTINGS = dict(
    boundary_rms_sd_C=dict(surface=1.5, bottom=100.),
    spatial_variance_fractions=[.5, .25, .25],
    spatial_order=['offset', 'east_west', 'north_south'],
    q_gradient_shared_fraction=.5,
    reference_additional_relative_sd=dict(A=[.2, .2, .2, .2, .5], K=[.1]*5),
    layers=['sediment', 'upper_crust', 'middle_crust', 'lower_crust', 'mantle']
)
