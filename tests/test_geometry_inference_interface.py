import sys
from types import ModuleType

import pytest

from cycpep_master.core.geometry_inference import infer_monomer_geometry


@pytest.mark.parametrize('mode', ['simple', 'complex'])
def test_opt_in_interface_passes_only_geometry_fields(monkeypatch, mode):
    name = f'cycpep_master.core.geometry_{mode}'
    module = ModuleType(name)
    observed = {}

    def infer(record, **kwargs):
        observed.update(record=record, kwargs=kwargs)
        return {'status': 'unresolved', 'candidates': [], 'search_complete': True}

    module.infer_geometry = infer
    monkeypatch.setitem(sys.modules, name, module)
    result = infer_monomer_geometry(
        {'atoms': [], 'bonds': None, 'total_charge': None, 'reference_smiles': 'secret', 'component_id': 'PTR'},
        mode=mode, timeout_seconds=0.5,
    )
    assert result['status'] == 'unresolved'
    assert set(observed['record']) == {'atoms', 'bonds', 'total_charge'}
    assert observed['kwargs']['timeout_seconds'] == 0.5


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf'), True])
def test_reject_invalid_budget(timeout):
    with pytest.raises(ValueError):
        infer_monomer_geometry({'atoms': []}, timeout_seconds=timeout)


def test_reject_invalid_mode():
    with pytest.raises(ValueError):
        infer_monomer_geometry({'atoms': []}, mode='automatic')
