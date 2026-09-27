"""SMILES -> QPixmap rendering via RDKit's Cairo backend."""
from PyQt5 import QtGui


def smiles_to_pixmap(smiles, width=420, height=320):
    """Render a SMILES string to a QPixmap. Returns None if it cannot be drawn."""
    if not smiles:
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except Exception:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
    except Exception:
        return None
    pix = QtGui.QPixmap()
    if not pix.loadFromData(png, "PNG"):
        return None
    return pix
