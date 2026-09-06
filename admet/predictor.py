"""ADMET prediction wrapper."""
def run_admet(smiles_list):
    """Run ADMET prediction on a list of SMILES.

    Returns list of dicts, or list with error dict on failure.
    """
    try:
        from admet_ai import ADMETModel
    except ImportError:
        return [
            {
                'input_smiles': str(value),
                'error': 'admet_ai not installed. pip install admet-ai',
            }
            for value in smiles_list
        ]

    try:
        from rdkit import Chem

        values = [str(value).strip() for value in smiles_list]
        invalid = [value for value in values if Chem.MolFromSmiles(value) is None]
        if invalid:
            message = f"invalid SMILES in ADMET batch: {invalid}"
            return [
                {"input_smiles": value, "error": message}
                for value in values
            ]
        model = ADMETModel()
        df = model.predict(values)
        if len(df) != len(values):
            message = (
                "ADMET output cardinality mismatch: "
                f"{len(df)} predictions for {len(values)} inputs"
            )
            return [
                {"input_smiles": value, "error": message}
                for value in values
            ]
        results = []
        for input_smiles, (_, row) in zip(values, df.iterrows()):
            entry = {"input_smiles": input_smiles}
            for col in df.columns:
                val = row[col]
                if hasattr(val, 'item'):
                    val = val.item()
                entry[col] = val
            results.append(entry)
        return results
    except Exception as e:
        return [
            {"input_smiles": str(value), "error": str(e)}
            for value in smiles_list
        ]
