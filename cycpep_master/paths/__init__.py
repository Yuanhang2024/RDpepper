"""SMILES generation paths."""
from .path_a import generate as generate_a
from .path_b import (generate as generate_b, build_helm_from_pdb,
                     build_helm_multichain, generate_multichain)
from .path_c import generate as generate_c
from .path_f import generate_f
from .path_g import generate_g
from .path_h import generate_h


def generate_e(pdb_path, chain_id='L', *, monomer_context=None):
    """Path E = Path A with geometric covalent-radius cyclization fallback.

    Adds missing bonds from geometry one edge at a time while retaining
    mapped explicit connectivity.
    """
    return generate_a(
        pdb_path,
        chain_id=chain_id,
        geometric_cyclization=True,
        monomer_context=monomer_context,
    )


PATH_MAP = {
    'a': generate_a,
    'b': generate_b,
    'c': generate_c,
    'e': generate_e,
    'f': generate_f,
    'g': generate_g,
    'h': generate_h,
}
