"""Background workers so long-running cheminformatics calls never freeze the UI.

generate_*()/run_admet() can take seconds (large PDBs, ADMET model load), so
they run on QThreads and report back via signals.
"""
from PyQt5 import QtCore


class ServiceWorker(QtCore.QThread):
    """Execute one shared application-service call off the UI thread."""

    result_ready = QtCore.pyqtSignal(dict)

    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self._function = function
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._function(*self._args, **self._kwargs)
            if not isinstance(result, dict):
                raise TypeError("application service did not return a result object")
        except Exception as exc:
            result = {
                "operation": getattr(self._function, "__name__", "service"),
                "status": "failed",
                "data": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
        self.result_ready.emit(result)


class ConvertWorker(QtCore.QThread):
    """Run a PDB -> SMILES conversion off the UI thread.

    Emits finished(smiles, error) — error is '' on success.
    """
    done = QtCore.pyqtSignal(str, str)

    def __init__(self, pdb_path, path, chain, multichain, chain_ids=None):
        super().__init__()
        self._pdb = pdb_path
        self._path = path
        self._chain = chain
        self._multichain = multichain
        self._chain_ids = chain_ids

    def run(self):
        try:
            from .. import application

            if self._multichain:
                result = application.reconstruct_multichain(
                    self._pdb, chain_ids=self._chain_ids
                )
            elif self._path == "v6":
                result = application.reconstruct_structure(
                    self._pdb,
                    chain_id=self._chain,
                    mode="strict",
                )
            else:
                if self._path not in application.RECONSTRUCTION_PATHS:
                    raise ValueError(f"unknown diagnostic path: {self._path}")
                result = application.reconstruct_coordinates(
                    [self._pdb],
                    path=self._path,
                    chain_id=self._chain,
                )
            data = result.get("data") or {}
            if self._multichain or self._path == "v6":
                smiles = data.get("smiles") or data.get(
                    "output_smiles"
                )
                warning_codes = list(
                    data.get("warning_codes") or []
                )
            else:
                rows = list(data.get("results") or [])
                row = rows[0] if rows else {}
                smiles = row.get("smiles")
                warning_codes = list(
                    row.get("warning_codes") or []
                )
            err = None
            if result.get("status") != "success" or not smiles:
                err = str(
                    result.get("error")
                    or f"{result.get('status')}: no qualified molecular graph"
                )
                if warning_codes:
                    err += " [" + ",".join(warning_codes) + "]"
            self.done.emit(smiles or "", err or "")
        except Exception as ex:  # never let a worker crash take down the app
            self.done.emit("", str(ex))


class AdmetWorker(QtCore.QThread):
    """Run ADMET prediction off the UI thread.

    Emits done(result_dict, error). result_dict is {} on error.
    """
    done = QtCore.pyqtSignal(dict, str)

    def __init__(self, smiles):
        super().__init__()
        self._smiles = smiles

    def run(self):
        try:
            from .. import application

            result = application.predict_admet([self._smiles])
            predictions = list(
                (result.get("data") or {}).get("predictions") or []
            )
            if result.get("status") != "success" or not predictions:
                self.done.emit(
                    {},
                    str(
                        result.get("error")
                        or "ADMET returned no result"
                    ),
                )
                return
            res = predictions[0]
            if "error" in res:
                self.done.emit({}, str(res["error"]))
            else:
                self.done.emit(res, "")
        except Exception as ex:
            self.done.emit({}, str(ex))
