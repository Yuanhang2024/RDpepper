import json

from cycpep_master.core import geometry_complex, geometry_simple


def test_simple_does_not_override_supplied_total_charge():
    record = {'atoms': [{'id': 0, 'element': 'C', 'xyz': [0.0, 0.0, 0.0]}],
              'bonds': [], 'total_charge': 7}
    result = geometry_simple.infer_geometry(record)
    assert result['status'] == 'unresolved'
    assert result['candidates'] == []
    assert 'total_charge_unmatched' in result['reason_codes']


def test_simple_records_connectivity_source():
    record = {'atoms': [{'id': 0, 'element': 'C', 'xyz': [0.0, 0.0, 0.0]},
                        {'id': 1, 'element': 'O', 'xyz': [1.23, 0.0, 0.0]}],
              'bonds': None, 'total_charge': None}
    result = geometry_simple.infer_geometry(record)
    assert result['evidence']['bonds_supplied'] is False
    assert result['evidence']['geometry_usage']['neighbor_plane_fits_diagnostic_only'] is True


def test_microstate_budget_counts_completed_assignments_not_atom_count():
    atoms = [{'id': i, 'element': 'C', 'xyz': [i * 1.52, (i % 2) * 0.1, 0.0]} for i in range(65)]
    record = {'atoms': atoms, 'bonds': [[i, i+1] for i in range(64)], 'total_charge': 0}
    result = geometry_complex.infer_geometry(record, timeout_seconds=2.0)
    assert result['candidates'], result['reason_codes']
    assert len(result['candidates'][0]['formal_charges']) == 65
    json.dumps(result, allow_nan=False)
