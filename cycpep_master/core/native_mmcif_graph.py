"""Read-only native mmCIF evidence for coordinate-to-graph workflows.

The production reconstruction path historically projects mmCIF files to a
legacy PDB representation.  That is useful for compatibility, but it loses
some of the distinctions made by mmCIF (label versus author addresses,
entity membership, and multi-character chain identifiers).  This module is a
small, dependency-light adapter for code that needs those distinctions before
choosing a reconstruction route.

The adapter deliberately does not infer missing chemistry.  ``atom_site``
rows are retained even when an element or coordinate is missing, and issues
are returned as warnings.  ``chem_comp`` rows contribute source-declared
component bonds when both endpoint atoms are present; ``struct_conn`` rows are
kept as explicit cross-component evidence.  No external CCD, PRD, PRDCC, or
benchmark reference is consulted.

Public entry points are :func:`read_native_mmcif` and
:func:`list_peptide_chains`.  The returned objects are immutable records and
have deterministic ``to_dict``/``as_graph`` projections suitable for a later
orchestration layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import gemmi


_MISSING = {None, "", ".", "?"}
_ELEMENTS = {
    "H", "D", "T", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE",
    "NA", "MG", "AL", "SI", "P", "S", "CL", "AR", "K", "CA", "SC",
    "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE",
    "AS", "SE", "BR", "KR", "RB", "SR", "Y", "ZR", "NB", "MO", "TC",
    "RU", "RH", "PD", "AG", "CD", "IN", "SN", "SB", "TE", "I", "XE",
    "CS", "BA", "LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD", "TB",
    "DY", "HO", "ER", "TM", "YB", "LU", "HF", "TA", "W", "RE", "OS",
    "IR", "PT", "AU", "HG", "TL", "PB", "BI", "PO", "AT", "RN", "FR",
    "RA", "AC", "TH", "PA", "U", "NP", "PU", "AM", "CM", "BK", "CF",
    "ES", "FM", "MD", "NO", "LR", "RF", "DB", "SG", "BH", "HS", "MT",
    "DS", "RG", "CN", "NH", "FL", "MC", "LV", "TS", "OG",
}


class NativeMmcifError(ValueError):
    """Raised when the mmCIF document itself cannot be parsed."""


def _value(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw in _MISSING:
        return None
    try:
        return str(gemmi.cif.as_string(raw)).strip()
    except (TypeError, ValueError):
        return raw


def _rows(block: gemmi.cif.Block, category: str) -> list[dict[str, str | None]]:
    """Return a category as lowercase-tag dictionaries without dropping rows."""

    table = block.find_mmcif_category(category)
    if len(table) == 0:
        return []
    tags = [str(tag).lower() for tag in table.tags]
    output: list[dict[str, str | None]] = []
    for row_index, row in enumerate(table, start=1):
        item: dict[str, str | None] = {}
        for index, tag in enumerate(tags):
            try:
                item[tag] = _value(row[index])
            except (IndexError, TypeError):
                item[tag] = None
        item["__row_index"] = str(row_index)
        output.append(item)
    return output


def _clean(value: Any) -> str | None:
    return _value(value)


def _parse_int(value: Any, warnings: list[str], code: str) -> int | None:
    raw = _clean(value)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        warnings.append(f"{code}:{raw}")
        return None


def _parse_float(value: Any, warnings: list[str], code: str) -> float | None:
    raw = _clean(value)
    if raw is None:
        return None
    try:
        parsed = float(raw)
        if not math.isfinite(parsed):
            raise ValueError("non-finite value")
        return parsed
    except (TypeError, ValueError):
        warnings.append(f"{code}:{raw}")
        return None


def _parse_charge(value: Any, warnings: list[str], code: str) -> int | None:
    raw = _clean(value)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        if len(raw) > 1 and raw[-1] in "+-":
            try:
                magnitude = int(raw[:-1])
                return magnitude if raw[-1] == "+" else -magnitude
            except ValueError:
                pass
        if len(raw) > 1 and raw[0] in "+-":
            try:
                return int(raw)
            except ValueError:
                pass
        warnings.append(f"{code}:{raw}")
        return None


def _normalise_element(raw: str | None) -> str | None:
    if raw is None:
        return None
    token = raw.strip().upper()
    if not token:
        return None
    if token not in _ELEMENTS:
        return None
    # The source spelling is retained separately on NativeAtom.  Canonical
    # title-case is useful to downstream RDKit/graph consumers.
    return token[:1] + token[1:].lower()


def _sort_token(value: str | None) -> tuple[int, int | str]:
    if value is None:
        return (1, "")
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (0, str(value))


def _endpoint_value(value: str | None) -> str | None:
    return value if value not in _MISSING else None


def _endpoint_sort_key(endpoint: "MmcifEndpoint") -> tuple[str, ...]:
    """Return a stable address key for canonicalising undirected edges."""

    return tuple(
        str(value) if value is not None else ""
        for value in (
            endpoint.model_num,
            endpoint.label_asym_id,
            endpoint.label_entity_id,
            endpoint.label_seq_id,
            endpoint.label_comp_id,
            endpoint.label_atom_id,
            endpoint.auth_asym_id,
            endpoint.auth_seq_id,
            endpoint.auth_comp_id,
            endpoint.auth_atom_id,
            endpoint.insertion_code,
        )
    )


@dataclass(frozen=True)
class MmcifChainRecord:
    """A source-declared chain/asym and its entity evidence."""

    label_asym_id: str
    auth_asym_id: str | None
    entity_id: str | None
    entity_type: str | None
    entity_poly_type: str | None
    peptide_bearing: bool
    peptide_evidence: tuple[str, ...]
    component_ids: tuple[str, ...]
    residue_names: tuple[str, ...]
    atom_count: int
    residue_count: int

    @property
    def chain_id(self) -> str:
        """Label asym ID, the unambiguous native mmCIF chain key."""

        return self.label_asym_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MmcifEntityRecord:
    """Unique entity-level view assembled from chain records."""

    entity_id: str
    entity_type: str | None
    entity_poly_type: str | None
    label_asym_ids: tuple[str, ...]
    auth_asym_ids: tuple[str, ...]
    peptide_bearing: bool
    peptide_evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MmcifEndpoint:
    """The label/author address carried by one ``struct_conn`` endpoint."""

    label_asym_id: str | None = None
    label_entity_id: str | None = None
    label_seq_id: str | None = None
    label_comp_id: str | None = None
    label_atom_id: str | None = None
    auth_asym_id: str | None = None
    auth_seq_id: str | None = None
    auth_comp_id: str | None = None
    auth_atom_id: str | None = None
    insertion_code: str | None = None
    model_num: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NativeAtom:
    """An ``atom_site`` row retained in the native coordinate graph."""

    atom_id: str
    source_atom_id: str | None
    element: str | None
    element_raw: str | None
    coordinates: tuple[float, float, float] | None
    label_asym_id: str | None
    label_entity_id: str | None
    label_seq_id: str | None
    label_comp_id: str | None
    label_atom_id: str | None
    auth_asym_id: str | None
    auth_seq_id: str | None
    auth_comp_id: str | None
    auth_atom_id: str | None
    insertion_code: str | None
    model_num: int
    alt_id: str | None
    group_pdb: str | None
    formal_charge: int | None
    occupancy: float | None
    b_iso: float | None

    @property
    def chain_id(self) -> str | None:
        return self.label_asym_id or self.auth_asym_id

    def endpoint(self) -> MmcifEndpoint:
        return MmcifEndpoint(
            label_asym_id=self.label_asym_id,
            label_entity_id=self.label_entity_id,
            label_seq_id=self.label_seq_id,
            label_comp_id=self.label_comp_id,
            label_atom_id=self.label_atom_id,
            auth_asym_id=self.auth_asym_id,
            auth_seq_id=self.auth_seq_id,
            auth_comp_id=self.auth_comp_id,
            auth_atom_id=self.auth_atom_id,
            insertion_code=self.insertion_code,
            model_num=self.model_num,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NativeBond:
    """A source-declared component or cross-component edge."""

    atom_id_1: str | None
    atom_id_2: str | None
    order: str | None
    source: str
    connection_id: str | None
    connection_type: str | None
    endpoint_1: MmcifEndpoint
    endpoint_2: MmcifEndpoint
    resolved: bool
    within_selection: bool

    def __post_init__(self) -> None:
        """Canonicalise the two sides of this undirected source edge.

        mmCIF connection and component-bond rows do not assign direction to
        an edge.  Canonicalising the endpoint order makes the graph stable
        when a depositor changes the row/endpoint order, while preserving the
        full endpoint evidence and resolution flags.
        """

        left_key = (
            (0, self.atom_id_1)
            if self.atom_id_1 is not None
            else (1, _endpoint_sort_key(self.endpoint_1))
        )
        right_key = (
            (0, self.atom_id_2)
            if self.atom_id_2 is not None
            else (1, _endpoint_sort_key(self.endpoint_2))
        )
        # Keep struct_conn endpoint 1/2 semantics available to diagnostics;
        # component-bond rows are purely undirected and can be canonicalised.
        if self.source == "chem_comp_bond" and left_key > right_key:
            atom_id_1, atom_id_2 = self.atom_id_1, self.atom_id_2
            endpoint_1, endpoint_2 = self.endpoint_1, self.endpoint_2
            object.__setattr__(self, "atom_id_1", atom_id_2)
            object.__setattr__(self, "atom_id_2", atom_id_1)
            object.__setattr__(self, "endpoint_1", endpoint_2)
            object.__setattr__(self, "endpoint_2", endpoint_1)

    @property
    def a(self) -> str | None:
        return self.atom_id_1

    @property
    def b(self) -> str | None:
        return self.atom_id_2

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_id_1": self.atom_id_1,
            "atom_id_2": self.atom_id_2,
            "order": self.order,
            "source": self.source,
            "connection_id": self.connection_id,
            "connection_type": self.connection_type,
            "endpoint_1": self.endpoint_1.to_dict(),
            "endpoint_2": self.endpoint_2.to_dict(),
            "resolved": self.resolved,
            "within_selection": self.within_selection,
        }


@dataclass(frozen=True)
class NativeMmcifGraph:
    """Selected native mmCIF coordinates plus source-declared edges."""

    source_path: str
    model_num: int | None
    selected_chain_ids: tuple[str, ...]
    chains: tuple[MmcifChainRecord, ...]
    atoms: tuple[NativeAtom, ...]
    bonds: tuple[NativeBond, ...]
    warnings: tuple[str, ...]
    provenance: Mapping[str, Any]

    @property
    def graph(self) -> dict[str, Any]:
        return self.as_graph()

    def as_graph(self) -> dict[str, Any]:
        """Return the compact graph shape used by result-first consumers."""

        return {
            "atoms": [atom.to_dict() for atom in self.atoms],
            "bonds": [bond.to_dict() for bond in self.bonds],
            "chains": [chain.to_dict() for chain in self.chains],
            "entities": [entity.to_dict() for entity in self.entities],
            "selected_chain_ids": list(self.selected_chain_ids),
            "model_num": self.model_num,
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }

    @property
    def entities(self) -> tuple[MmcifEntityRecord, ...]:
        """Return a deterministic entity-level projection of ``chains``."""

        grouped: dict[str, list[MmcifChainRecord]] = {}
        for chain in self.chains:
            if chain.entity_id is not None:
                grouped.setdefault(chain.entity_id, []).append(chain)
        entities: list[MmcifEntityRecord] = []
        for entity_id, chains in sorted(grouped.items()):
            entity_types = {chain.entity_type for chain in chains if chain.entity_type}
            poly_types = {chain.entity_poly_type for chain in chains if chain.entity_poly_type}
            evidence = {
                item for chain in chains for item in chain.peptide_evidence
            }
            entities.append(MmcifEntityRecord(
                entity_id=entity_id,
                entity_type=sorted(entity_types)[0] if entity_types else None,
                entity_poly_type=sorted(poly_types)[0] if poly_types else None,
                label_asym_ids=tuple(sorted(chain.label_asym_id for chain in chains)),
                auth_asym_ids=tuple(sorted({
                    chain.auth_asym_id for chain in chains if chain.auth_asym_id
                })),
                peptide_bearing=any(chain.peptide_bearing for chain in chains),
                peptide_evidence=tuple(sorted(evidence)),
            ))
        return tuple(entities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "model_num": self.model_num,
            "selected_chain_ids": list(self.selected_chain_ids),
            "chains": [chain.to_dict() for chain in self.chains],
            "entities": [entity.to_dict() for entity in self.entities],
            "atoms": [atom.to_dict() for atom in self.atoms],
            "bonds": [bond.to_dict() for bond in self.bonds],
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }


# Upper-case alias keeps the acronym discoverable without forcing callers to
# use a non-PEP8 class spelling.
NativeMMCIFGraph = NativeMmcifGraph


def _read_block(source: str | Path) -> tuple[gemmi.cif.Block, Path]:
    path = Path(source)
    try:
        if path.name.lower().endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                document = gemmi.cif.read_string(handle.read())
        else:
            document = gemmi.cif.read_file(str(path))
    except (OSError, UnicodeError, RuntimeError, ValueError) as exc:
        raise NativeMmcifError(
            f"mmCIF input cannot be parsed: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        if len(document) == 0:
            raise NativeMmcifError("mmCIF document has no data block")
        return document.sole_block(), path
    except NativeMmcifError:
        raise
    except (IndexError, RuntimeError, ValueError) as exc:
        raise NativeMmcifError(
            f"mmCIF document has no unique data block: {type(exc).__name__}: {exc}"
        ) from exc


def _component_metadata(block: gemmi.cif.Block) -> tuple[
    dict[str, dict[str, str | None]],
    dict[str, list[dict[str, str | None]]],
    dict[str, list[dict[str, str | None]]],
]:
    metadata: dict[str, dict[str, str | None]] = {}
    for row in _rows(block, "_chem_comp."):
        comp_id = _clean(row.get("_chem_comp.id"))
        if comp_id is None:
            continue
        key = comp_id.upper()
        metadata.setdefault(key, row)
    atoms: dict[str, list[dict[str, str | None]]] = {}
    for row in _rows(block, "_chem_comp_atom."):
        comp_id = _clean(row.get("_chem_comp_atom.comp_id"))
        if comp_id is None:
            continue
        atoms.setdefault(comp_id.upper(), []).append(row)
    bonds: dict[str, list[dict[str, str | None]]] = {}
    for row in _rows(block, "_chem_comp_bond."):
        comp_id = _clean(row.get("_chem_comp_bond.comp_id"))
        if comp_id is None:
            continue
        bonds.setdefault(comp_id.upper(), []).append(row)
    return metadata, atoms, bonds


def _entity_metadata(block: gemmi.cif.Block) -> tuple[
    dict[str, dict[str, str | None]], dict[str, dict[str, str | None]]
]:
    entities: dict[str, dict[str, str | None]] = {}
    for row in _rows(block, "_entity."):
        entity_id = _clean(row.get("_entity.id"))
        if entity_id is not None:
            entities[entity_id] = row
    entity_poly: dict[str, dict[str, str | None]] = {}
    for row in _rows(block, "_entity_poly."):
        entity_id = _clean(row.get("_entity_poly.entity_id"))
        if entity_id is not None:
            entity_poly[entity_id] = row
    return entities, entity_poly


def _struct_asym_metadata(block: gemmi.cif.Block) -> dict[str, str | None]:
    mapping: dict[str, str | None] = {}
    for row in _rows(block, "_struct_asym."):
        asym_id = _clean(row.get("_struct_asym.id"))
        if asym_id is not None:
            mapping[asym_id] = _clean(row.get("_struct_asym.entity_id"))
    return mapping


def _is_peptide_entity(
    entity_type: str | None,
    poly_type: str | None,
    component_ids: Iterable[str],
    component_metadata: Mapping[str, Mapping[str, str | None]],
) -> tuple[bool, tuple[str, ...]]:
    evidence: list[str] = []
    entity_token = " ".join((entity_type or "").lower().split())
    poly_token = " ".join((poly_type or "").lower().split())
    if "polymer" == entity_token or entity_token.startswith("polymer"):
        evidence.append("entity.type=polymer")
    if "peptide" in poly_token or "polypeptide" in poly_token:
        evidence.append("entity_poly.type=peptide")
    for comp_id in component_ids:
        row = component_metadata.get(comp_id.upper(), {})
        comp_type = " ".join(
            (row.get("_chem_comp.type") or "").lower().split()
        )
        if "peptide" in comp_type or "polypeptide" in comp_type:
            evidence.append(f"chem_comp.type={comp_id}")
            break
    # An entity-poly declaration is the strongest native evidence.  When the
    # declaration is absent, a peptide-linking component is still useful as a
    # positive hint, but a generic polymer is intentionally not called a
    # peptide merely from its coordinates.
    peptide = "peptide" in poly_token or "polypeptide" in poly_token
    peptide = peptide or any(item.startswith("chem_comp.type=") for item in evidence)
    return peptide, tuple(sorted(set(evidence)))


def _endpoint_from_row(
    row: Mapping[str, str | None], prefix: str, *, model_num: int | None
) -> MmcifEndpoint:
    model_value = _clean(
        row.get(f"_struct_conn.pdbx_ptnr{prefix}_pdb_model_num")
    )
    parsed_model: int | None = None
    if model_value is not None:
        try:
            parsed_model = int(model_value)
        except ValueError:
            parsed_model = None
    if parsed_model is None:
        parsed_model = model_num
    return MmcifEndpoint(
        label_asym_id=_endpoint_value(
            row.get(f"_struct_conn.ptnr{prefix}_label_asym_id")
        ),
        label_entity_id=_endpoint_value(
            row.get(f"_struct_conn.ptnr{prefix}_label_entity_id")
        ),
        label_seq_id=_endpoint_value(
            row.get(f"_struct_conn.ptnr{prefix}_label_seq_id")
        ),
        label_comp_id=_endpoint_value(
            row.get(f"_struct_conn.ptnr{prefix}_label_comp_id")
        ),
        label_atom_id=_endpoint_value(
            row.get(f"_struct_conn.ptnr{prefix}_label_atom_id")
        ),
        auth_asym_id=_endpoint_value(
            row.get(f"_struct_conn.ptnr{prefix}_auth_asym_id")
        ),
        auth_seq_id=_endpoint_value(
            row.get(f"_struct_conn.ptnr{prefix}_auth_seq_id")
        ),
        auth_comp_id=_endpoint_value(
            row.get(f"_struct_conn.ptnr{prefix}_auth_comp_id")
        ),
        auth_atom_id=_endpoint_value(
            row.get(f"_struct_conn.ptnr{prefix}_auth_atom_id")
        ),
        insertion_code=_endpoint_value(
            row.get(f"_struct_conn.pdbx_ptnr{prefix}_pdb_ins_code")
        ),
        model_num=parsed_model,
    )


def _atom_matches_endpoint(atom: NativeAtom, endpoint: MmcifEndpoint) -> bool:
    label_fields = (
        endpoint.label_asym_id,
        endpoint.label_seq_id,
        endpoint.label_comp_id,
        endpoint.label_atom_id,
    )
    if all(value is not None for value in label_fields):
        return (
            atom.label_asym_id == endpoint.label_asym_id
            and (
                endpoint.label_entity_id is None
                or atom.label_entity_id == endpoint.label_entity_id
            )
            and atom.label_seq_id == endpoint.label_seq_id
            and atom.label_comp_id == endpoint.label_comp_id
            and atom.label_atom_id == endpoint.label_atom_id
            and (
                endpoint.insertion_code is None
                or atom.insertion_code == endpoint.insertion_code
            )
        )
    auth_fields = (
        endpoint.auth_asym_id,
        endpoint.auth_seq_id,
        endpoint.auth_comp_id,
        endpoint.auth_atom_id,
    )
    if all(value is not None for value in auth_fields):
        return (
            atom.auth_asym_id == endpoint.auth_asym_id
            and atom.auth_seq_id == endpoint.auth_seq_id
            and atom.auth_comp_id == endpoint.auth_comp_id
            and atom.auth_atom_id == endpoint.auth_atom_id
            and (
                endpoint.insertion_code is None
                or atom.insertion_code == endpoint.insertion_code
            )
        )
    # Some deposited connections omit sequence IDs for non-polymer partners.
    # Use the complete subset of the label address, then author address, only
    # when it identifies exactly one atom.
    label_pairs = (
        (endpoint.label_asym_id, atom.label_asym_id),
        (endpoint.label_comp_id, atom.label_comp_id),
        (endpoint.label_atom_id, atom.label_atom_id),
    )
    if endpoint.label_asym_id is not None and endpoint.label_atom_id is not None:
        return all(expected is None or expected == actual for expected, actual in label_pairs)
    auth_pairs = (
        (endpoint.auth_asym_id, atom.auth_asym_id),
        (endpoint.auth_comp_id, atom.auth_comp_id),
        (endpoint.auth_atom_id, atom.auth_atom_id),
    )
    if endpoint.auth_asym_id is not None and endpoint.auth_atom_id is not None:
        return all(expected is None or expected == actual for expected, actual in auth_pairs)
    return False


def _resolve_endpoint(
    endpoint: MmcifEndpoint,
    atoms: Sequence[NativeAtom],
    *,
    model_num: int | None,
    warnings: list[str],
    connection_id: str,
    side: str,
) -> NativeAtom | None:
    candidates = [
        atom for atom in atoms
        if (model_num is None or atom.model_num == model_num)
        and (endpoint.model_num is None or atom.model_num == endpoint.model_num)
        and _atom_matches_endpoint(atom, endpoint)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        warnings.append(f"struct_conn_endpoint_unresolved:{connection_id}:{side}")
    else:
        warnings.append(f"struct_conn_endpoint_ambiguous:{connection_id}:{side}")
    return None


def _source_bond_order(row: Mapping[str, str | None]) -> str | None:
    for key in (
        "_struct_conn.pdbx_value_order",
        "_struct_conn.value_order",
        "_struct_conn.pdbx_bond_order",
    ):
        value = _clean(row.get(key))
        if value is not None:
            return value
    return None


def _make_atom(
    row: Mapping[str, str | None],
    row_index: int,
    *,
    warnings: list[str],
    template_atoms: Mapping[str, Mapping[str, str | None]],
) -> NativeAtom:
    source_id = _clean(row.get("_atom_site.id"))
    atom_id = source_id or f"row:{row_index}"
    if source_id is None:
        warnings.append(f"atom_id_missing:{row_index}")

    label_atom_id = _clean(row.get("_atom_site.label_atom_id"))
    auth_atom_id = _clean(row.get("_atom_site.auth_atom_id"))
    comp_id = _clean(row.get("_atom_site.label_comp_id")) or _clean(
        row.get("_atom_site.auth_comp_id")
    )
    template = template_atoms.get((label_atom_id or auth_atom_id or "").upper())
    element_raw = _clean(row.get("_atom_site.type_symbol"))
    if element_raw is None and template is not None:
        element_raw = _clean(template.get("_chem_comp_atom.type_symbol"))
        if element_raw is not None:
            warnings.append(f"atom_element_from_chem_comp:{atom_id}")
    element = _normalise_element(element_raw)
    if element_raw is None:
        warnings.append(f"atom_element_missing:{atom_id}")
    elif element_raw.upper() not in _ELEMENTS:
        warnings.append(f"atom_element_unrecognized:{atom_id}:{element_raw}")
    template_element = (
        _clean(template.get("_chem_comp_atom.type_symbol"))
        if template is not None else None
    )
    if (
        element_raw is not None
        and template_element is not None
        and element_raw.upper() != template_element.upper()
    ):
        warnings.append(f"atom_element_mismatch_chem_comp:{atom_id}")

    x = _parse_float(
        row.get("_atom_site.cartn_x"), warnings,
        f"atom_coordinate_invalid:{atom_id}:x",
    )
    y = _parse_float(
        row.get("_atom_site.cartn_y"), warnings,
        f"atom_coordinate_invalid:{atom_id}:y",
    )
    z = _parse_float(
        row.get("_atom_site.cartn_z"), warnings,
        f"atom_coordinate_invalid:{atom_id}:z",
    )
    coordinates = (x, y, z) if all(value is not None for value in (x, y, z)) else None
    if coordinates is None:
        warnings.append(f"atom_coordinate_incomplete:{atom_id}")
    model_num = _parse_int(
        row.get("_atom_site.pdbx_pdb_model_num"),
        warnings,
        f"atom_model_invalid:{atom_id}",
    ) or 1
    charge_raw = _clean(row.get("_atom_site.pdbx_formal_charge"))
    if charge_raw is None and template is not None:
        charge_raw = _clean(template.get("_chem_comp_atom.charge"))
        if charge_raw is not None:
            warnings.append(f"atom_charge_from_chem_comp:{atom_id}")
    return NativeAtom(
        atom_id=atom_id,
        source_atom_id=source_id,
        element=element,
        element_raw=element_raw,
        coordinates=coordinates,
        label_asym_id=_clean(row.get("_atom_site.label_asym_id")),
        label_entity_id=_clean(row.get("_atom_site.label_entity_id")),
        label_seq_id=_clean(row.get("_atom_site.label_seq_id")),
        label_comp_id=_clean(row.get("_atom_site.label_comp_id")),
        label_atom_id=label_atom_id,
        auth_asym_id=_clean(row.get("_atom_site.auth_asym_id")),
        auth_seq_id=_clean(row.get("_atom_site.auth_seq_id")),
        auth_comp_id=_clean(row.get("_atom_site.auth_comp_id")),
        auth_atom_id=auth_atom_id,
        insertion_code=_clean(
            row.get("_atom_site.pdbx_pdb_ins_code")
        ),
        model_num=model_num,
        alt_id=_clean(
            row.get("_atom_site.label_alt_id")
            or row.get("_atom_site.pdbx_pdb_alt_id")
        ),
        group_pdb=_clean(row.get("_atom_site.group_pdb")),
        formal_charge=_parse_charge(
            charge_raw,
            warnings,
            f"atom_charge_invalid:{atom_id}",
        ),
        occupancy=_parse_float(
            row.get("_atom_site.occupancy"), warnings, f"atom_occupancy_invalid:{atom_id}"
        ),
        b_iso=_parse_float(
            row.get("_atom_site.b_iso_or_equiv"), warnings, f"atom_bfactor_invalid:{atom_id}"
        ),
    )


def _chain_records(
    block: gemmi.cif.Block,
    atoms: Sequence[NativeAtom],
    *,
    component_metadata: Mapping[str, Mapping[str, str | None]],
    warnings: list[str],
) -> tuple[MmcifChainRecord, ...]:
    entities, entity_poly = _entity_metadata(block)
    struct_asym = _struct_asym_metadata(block)
    rows_by_chain: dict[str, list[NativeAtom]] = {}
    for atom in atoms:
        chain_id = atom.label_asym_id or atom.auth_asym_id
        if chain_id is not None:
            rows_by_chain.setdefault(chain_id, []).append(atom)
    all_chain_ids = set(struct_asym) | set(rows_by_chain)
    output: list[MmcifChainRecord] = []
    for label_asym_id in sorted(all_chain_ids):
        chain_atoms = rows_by_chain.get(label_asym_id, [])
        entity_id = struct_asym.get(label_asym_id)
        if entity_id is None:
            entity_ids = {atom.label_entity_id for atom in chain_atoms if atom.label_entity_id}
            if len(entity_ids) == 1:
                entity_id = next(iter(entity_ids))
        entity_row = entities.get(entity_id or "", {})
        poly_row = entity_poly.get(entity_id or "", {})
        entity_type = _clean(entity_row.get("_entity.type"))
        poly_type = _clean(poly_row.get("_entity_poly.type"))
        auth_ids = sorted({atom.auth_asym_id for atom in chain_atoms if atom.auth_asym_id})
        component_ids = sorted({
            comp for atom in chain_atoms
            for comp in (atom.label_comp_id, atom.auth_comp_id)
            if comp
        })
        residue_keys = {
            (
                atom.label_seq_id or atom.auth_seq_id,
                atom.insertion_code,
                atom.label_comp_id or atom.auth_comp_id,
            )
            for atom in chain_atoms
        }
        peptide_bearing, evidence = _is_peptide_entity(
            entity_type, poly_type, component_ids, component_metadata
        )
        if not chain_atoms:
            warnings.append(f"chain_has_no_atom_site_rows:{label_asym_id}")
        output.append(MmcifChainRecord(
            label_asym_id=label_asym_id,
            auth_asym_id=auth_ids[0] if len(auth_ids) == 1 else None,
            entity_id=entity_id,
            entity_type=entity_type,
            entity_poly_type=poly_type,
            peptide_bearing=peptide_bearing,
            peptide_evidence=evidence,
            component_ids=tuple(component_ids),
            residue_names=tuple(sorted({value for _seq, _ins, value in residue_keys if value})),
            atom_count=len(chain_atoms),
            residue_count=len(residue_keys),
        ))
    return tuple(output)


def _select_chains(
    records: Sequence[MmcifChainRecord],
    chain_ids: Iterable[str] | str | None,
    *,
    peptide_only: bool,
    warnings: list[str],
) -> tuple[str, ...]:
    if isinstance(chain_ids, str):
        requested = {chain_ids}
    elif chain_ids is None:
        requested = None
    else:
        requested = {str(value) for value in chain_ids}
    if requested is None:
        selected = {
            record.label_asym_id for record in records
            if not peptide_only or record.peptide_bearing
        }
        if peptide_only and not selected:
            warnings.append("no_peptide_bearing_chain_found")
        return tuple(sorted(selected))
    selected: set[str] = set()
    for value in requested:
        for record in records:
            if value in {record.label_asym_id, record.auth_asym_id}:
                selected.add(record.label_asym_id)
    missing = sorted(requested - {
        value for record in records
        for value in (record.label_asym_id, record.auth_asym_id)
        if value is not None
    })
    for value in missing:
        warnings.append(f"requested_chain_not_found:{value}")
    return tuple(sorted(selected))


def _atom_sort_key(atom: NativeAtom) -> tuple[Any, ...]:
    return (
        atom.model_num,
        atom.label_asym_id or "",
        _sort_token(atom.label_seq_id),
        atom.insertion_code or "",
        atom.label_atom_id or atom.auth_atom_id or "",
        _sort_token(atom.atom_id),
    )


def _bond_sort_key(bond: NativeBond) -> tuple[Any, ...]:
    return (
        bond.source,
        bond.connection_id or "",
        bond.atom_id_1 or "",
        bond.atom_id_2 or "",
        bond.order or "",
    )


def read_native_mmcif(
    source: str | Path,
    chain_ids: Iterable[str] | str | None = None,
    *,
    model_num: int | None = 1,
    peptide_only: bool = False,
) -> NativeMmcifGraph:
    """Read native mmCIF coordinates and source-declared graph edges.

    ``chain_ids`` accepts either native ``label_asym_id`` values or author
    chain IDs.  With no explicit selection all chains are returned; set
    ``peptide_only=True`` to select peptide-bearing chains by default.  The
    default model is the first model (model 1).  Pass ``model_num=None`` to
    retain every model; callers should then treat repeated atom IDs as
    separate observations.
    """

    block, path = _read_block(source)
    warnings: list[str] = []
    component_metadata, component_atoms, component_bonds = _component_metadata(block)
    atom_rows = _rows(block, "_atom_site.")
    if not atom_rows:
        warnings.append("atom_site_category_missing")

    # Build component atom lookup before materializing atom_site rows.  A
    # missing coordinate element may be recovered from the source-embedded
    # component definition, but the event remains visible in warnings.
    template_lookup: dict[str, dict[str, str | None]] = {}
    for comp_id, rows in component_atoms.items():
        for row in rows:
            atom_id = _clean(row.get("_chem_comp_atom.atom_id"))
            if atom_id is not None:
                template_lookup[f"{comp_id}:{atom_id.upper()}"] = row

    raw_atoms: list[NativeAtom] = []
    seen_ids: set[str] = set()
    for row_index, row in enumerate(atom_rows, start=1):
        comp_id = (
            _clean(row.get("_atom_site.label_comp_id"))
            or _clean(row.get("_atom_site.auth_comp_id"))
            or ""
        )
        atom_name = (
            _clean(row.get("_atom_site.label_atom_id"))
            or _clean(row.get("_atom_site.auth_atom_id"))
            or ""
        )
        template = template_lookup.get(f"{comp_id.upper()}:{atom_name.upper()}")
        atom = _make_atom(
            row, row_index, warnings=warnings,
            template_atoms={atom_name.upper(): template} if template else {},
        )
        if atom.atom_id in seen_ids:
            # A duplicate source ID cannot be used as an unambiguous graph
            # endpoint.  Preserve the row with a deterministic synthetic ID.
            warnings.append(f"atom_id_duplicate:{atom.atom_id}")
            atom = NativeAtom(
                **{
                    **atom.to_dict(),
                    "atom_id": f"{atom.atom_id}@row{row_index}",
                }
            )
        seen_ids.add(atom.atom_id)
        raw_atoms.append(atom)

    available_models = sorted({atom.model_num for atom in raw_atoms})
    selected_model = model_num
    if model_num is not None and available_models and model_num not in available_models:
        warnings.append(f"requested_model_not_found:{model_num}")
        selected_model = available_models[0]
        warnings.append(f"requested_model_fallback:{selected_model}")
    if selected_model is None:
        atoms = list(raw_atoms)
    else:
        atoms = [atom for atom in raw_atoms if atom.model_num == selected_model]
        excluded = len(raw_atoms) - len(atoms)
        if excluded:
            warnings.append(f"atom_rows_other_models_excluded:{excluded}")

    provisional_records = _chain_records(
        block, atoms, component_metadata=component_metadata, warnings=warnings
    )
    selected_chain_ids = _select_chains(
        provisional_records, chain_ids, peptide_only=peptide_only, warnings=warnings
    )
    selected_set = set(selected_chain_ids)
    atoms = [
        atom for atom in atoms
        if (atom.label_asym_id or atom.auth_asym_id) in selected_set
    ]
    records = _chain_records(
        block, atoms, component_metadata=component_metadata, warnings=warnings
    )
    records = tuple(
        record for record in records if record.label_asym_id in selected_set
    )

    # Build internal component bonds from the source-embedded chemical
    # dictionary.  The endpoint mapping intentionally uses observed
    # label_atom_id values and never invents missing coordinates/atoms.
    bonds: list[NativeBond] = []
    component_edge_keys: set[tuple[str, str, str]] = set()
    grouped: dict[tuple[Any, ...], list[NativeAtom]] = {}
    for atom in atoms:
        grouped.setdefault((
            atom.model_num,
            atom.label_asym_id or atom.auth_asym_id,
            atom.label_seq_id or atom.auth_seq_id,
            atom.insertion_code,
            atom.label_comp_id or atom.auth_comp_id,
        ), []).append(atom)
    for group_key, group_atoms in sorted(grouped.items(), key=lambda item: str(item[0])):
        comp_id = str(group_key[-1] or "").upper()
        if not comp_id or comp_id not in component_bonds:
            continue
        atoms_by_name: dict[str, list[NativeAtom]] = {}
        for atom in group_atoms:
            for name in (atom.label_atom_id, atom.auth_atom_id):
                if name:
                    atoms_by_name.setdefault(name.upper(), []).append(atom)
        for row in component_bonds[comp_id]:
            left_name = _clean(row.get("_chem_comp_bond.atom_id_1"))
            right_name = _clean(row.get("_chem_comp_bond.atom_id_2"))
            if left_name is None or right_name is None:
                warnings.append(f"chem_comp_bond_endpoint_missing:{comp_id}")
                continue
            left_copies = atoms_by_name.get(left_name.upper())
            right_copies = atoms_by_name.get(right_name.upper())
            if left_copies is None or right_copies is None:
                missing = left_name if left_copies is None else right_name
                warnings.append(f"chem_comp_bond_atom_unobserved:{comp_id}:{missing}")
                continue
            # Alternate conformers deposit several atom_site rows under one
            # atom name within a residue.  Every observed copy belongs to the
            # same source component, so each copy must carry the component's
            # declared bonds; otherwise the extra copies stay isolated in the
            # graph.  Copies are paired within the same conformer only (an
            # alt_id None row is the shared main copy and pairs with every
            # conformer); cross-conformer pairs are not guessed.
            for left in left_copies:
                for right in right_copies:
                    if (
                        left.alt_id is not None
                        and right.alt_id is not None
                        and left.alt_id != right.alt_id
                    ):
                        continue
                    if left.atom_id == right.atom_id:
                        warnings.append(
                            f"chem_comp_bond_self_edge:{comp_id}:{left_name}"
                        )
                        continue
                    edge_key = tuple(sorted((left.atom_id, right.atom_id)))
                    duplicate_key = (comp_id, *edge_key)
                    if duplicate_key in component_edge_keys:
                        warnings.append(
                            f"chem_comp_bond_duplicate:{comp_id}:{left_name}:{right_name}"
                        )
                        continue
                    component_edge_keys.add(duplicate_key)
                    order = _clean(row.get("_chem_comp_bond.value_order"))
                    if order is None:
                        warnings.append(
                            f"chem_comp_bond_order_missing:{comp_id}:{left_name}:{right_name}"
                        )
                    bonds.append(NativeBond(
                        atom_id_1=left.atom_id,
                        atom_id_2=right.atom_id,
                        order=order,
                        source="chem_comp_bond",
                        connection_id=comp_id,
                        connection_type=None,
                        endpoint_1=left.endpoint(),
                        endpoint_2=right.endpoint(),
                        resolved=True,
                        within_selection=True,
                    ))

    # Explicit cross-component bonds are resolved against the native label or
    # author addresses.  Unresolved rows are retained as evidence records with
    # null atom IDs; this is preferable to silently dropping an explicit link.
    for row_index, row in enumerate(_rows(block, "_struct_conn."), start=1):
        connection_id = _clean(row.get("_struct_conn.id")) or f"row:{row_index}"
        endpoint_1 = _endpoint_from_row(row, "1", model_num=selected_model)
        endpoint_2 = _endpoint_from_row(row, "2", model_num=selected_model)
        if selected_model is not None:
            endpoint_models = {
                value for value in (endpoint_1.model_num, endpoint_2.model_num)
                if value is not None
            }
            if endpoint_models and endpoint_models != {selected_model}:
                warnings.append(f"struct_conn_other_model_excluded:{connection_id}")
                continue
        left = _resolve_endpoint(
            endpoint_1, atoms, model_num=selected_model, warnings=warnings,
            connection_id=connection_id, side="1",
        )
        right = _resolve_endpoint(
            endpoint_2, atoms, model_num=selected_model, warnings=warnings,
            connection_id=connection_id, side="2",
        )
        touches_selection = (
            (left is not None and left.atom_id in {atom.atom_id for atom in atoms})
            or (right is not None and right.atom_id in {atom.atom_id for atom in atoms})
            or (endpoint_1.label_asym_id in selected_set)
            or (endpoint_2.label_asym_id in selected_set)
            or (endpoint_1.auth_asym_id in {
                record.auth_asym_id for record in records if record.auth_asym_id
            })
            or (endpoint_2.auth_asym_id in {
                record.auth_asym_id for record in records if record.auth_asym_id
            })
        )
        if not touches_selection:
            continue
        order = _source_bond_order(row)
        if order is None:
            warnings.append(f"struct_conn_order_missing:{connection_id}")
        bonds.append(NativeBond(
            atom_id_1=left.atom_id if left is not None else None,
            atom_id_2=right.atom_id if right is not None else None,
            order=order,
            source="struct_conn",
            connection_id=connection_id,
            connection_type=_clean(row.get("_struct_conn.conn_type_id")),
            endpoint_1=endpoint_1,
            endpoint_2=endpoint_2,
            resolved=left is not None and right is not None,
            within_selection=(
                left is not None and right is not None
                and left.atom_id in {atom.atom_id for atom in atoms}
                and right.atom_id in {atom.atom_id for atom in atoms}
            ),
        ))

    # Missing component categories are warnings only.  Standard mmCIF files
    # often omit the embedded dictionary and can still provide useful atom and
    # struct_conn evidence.
    chem_comp_ids = set(component_metadata) | set(component_atoms) | set(component_bonds)
    for comp_id in sorted(chem_comp_ids):
        if comp_id not in component_atoms:
            warnings.append(f"chem_comp_atom_category_missing:{comp_id}")
        if comp_id not in component_bonds:
            warnings.append(f"chem_comp_bond_category_missing:{comp_id}")

    unique_warnings = tuple(sorted(set(warnings)))
    atoms = sorted(atoms, key=_atom_sort_key)
    bonds = sorted(bonds, key=_bond_sort_key)
    provenance = {
        "parser": "gemmi.cif.native_mmcif_graph",
        "source_format": "mmcif",
        "source_path": str(path),
        "selected_model": selected_model,
        "available_models": available_models,
        "atom_site_rows": len(atom_rows),
        "selected_atom_count": len(atoms),
        "selected_chain_ids": list(selected_chain_ids),
        "category_rows": {
            "atom_site": len(atom_rows),
            "struct_conn": len(_rows(block, "_struct_conn.")),
            "chem_comp": len(_rows(block, "_chem_comp.")),
            "chem_comp_atom": len(_rows(block, "_chem_comp_atom.")),
            "chem_comp_bond": len(_rows(block, "_chem_comp_bond.")),
        },
    }
    return NativeMmcifGraph(
        source_path=str(path),
        model_num=selected_model,
        selected_chain_ids=selected_chain_ids,
        chains=records,
        atoms=tuple(atoms),
        bonds=tuple(bonds),
        warnings=unique_warnings,
        provenance=provenance,
    )


def inspect_mmcif(source: str | Path) -> tuple[MmcifChainRecord, ...]:
    """List all source-declared chains, including non-peptide chains."""

    graph = read_native_mmcif(source, model_num=1, peptide_only=False)
    return graph.chains


def list_peptide_chains(source: str | Path) -> tuple[MmcifChainRecord, ...]:
    """List chains with native peptide evidence without reading external gold."""

    return tuple(record for record in inspect_mmcif(source) if record.peptide_bearing)


def list_peptide_entities(source: str | Path) -> tuple[MmcifEntityRecord, ...]:
    """List unique peptide-bearing entities and their native chain mappings."""

    graph = read_native_mmcif(source, model_num=1, peptide_only=False)
    return tuple(entity for entity in graph.entities if entity.peptide_bearing)


# Friendly aliases for callers that use ``parse``/``load`` terminology.
parse_native_mmcif = read_native_mmcif
parse_mmcif_graph = read_native_mmcif
load_mmcif_graph = read_native_mmcif
list_peptide_bearing_chains = list_peptide_chains


__all__ = [
    "MmcifChainRecord",
    "MmcifEntityRecord",
    "MmcifEndpoint",
    "NativeAtom",
    "NativeBond",
    "NativeMmcifGraph",
    "NativeMMCIFGraph",
    "NativeMmcifError",
    "inspect_mmcif",
    "list_peptide_chains",
    "list_peptide_bearing_chains",
    "list_peptide_entities",
    "read_native_mmcif",
    "parse_native_mmcif",
    "parse_mmcif_graph",
    "load_mmcif_graph",
]
