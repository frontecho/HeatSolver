"""Prepared forward-model input loading and JSON output."""
import json
import h5py
import numpy as np


def load_model(bundle):
    b = dict(np.load(bundle/'bundle.npz'))
    with h5py.File(bundle.parent/'references/references.h5') as references:
        b.update({key: references['fields/'+key][:] for key in ['A_reference', 'K_reference']})
    return b


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
