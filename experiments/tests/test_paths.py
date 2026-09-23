import ast
import json
from pathlib import Path

from project import ROOT


def test_runtime_has_no_path_translation_files():
    assert ROOT == Path(__file__).resolve().parents[2]
    assert not (ROOT/'experiments/locations.json').exists()
    assert not (ROOT/'experiments/paths.py').exists()


def test_paper_configs_only_require_local_input_files():
    paper = ROOT/'experiments'
    for path in (paper/'configs').rglob('*.json'):
        config = json.loads(path.read_text())
        if not isinstance(config, dict) or 'action' not in config:
            continue
        inputs = [config[key] for key in ['bundle', 'split'] if key in config]
        if 'initial_temperature' in config:
            inputs.append(config['initial_temperature']['path'])
        for value in inputs:
            target = ROOT/value
            assert target.is_relative_to(paper/'inputs') and target.exists(), (path, target)


def test_preprocessing_has_no_experiment_code_or_result_dependency():
    group = ROOT/'Data/processed/ncc_preparation/ak170'
    specification = json.loads((group/'sources.json').read_text())
    for source in specification['sources'].values():
        target = ROOT/source['path']
        assert target.is_relative_to(ROOT/'Data') and target.is_file()
    for name in ['prepare.py', 'material_reference.py']:
        tree = ast.parse((group/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('experiments')


def test_region_reader_uses_current_original_geometry():
    from Data.processed.thermal_properties.csrm_heatproduction.regions import read_regions
    regions = read_regions()
    assert regions['China'] and regions['NCC'] and regions['Ordos Block']


def test_data_preparation_modules_use_repository_root():
    import Data.processed.ncc_preparation.source.prepare_full as prepare_full
    import Data.processed.thermal_properties.china_crustal_composition.native as native
    import Data.processed.thermal_properties.rock_geochemistry.prepare as geochemistry
    import Data.processed.thermal_properties.thermal_conductivity.prepare_laboratory as conductivity
    for module in [prepare_full, native, geochemistry, conductivity]:
        assert module.ROOT == ROOT
