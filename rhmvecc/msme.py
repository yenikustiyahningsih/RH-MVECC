"""MSME data loader and shared constants."""
from pathlib import Path
import numpy as np
import pandas as pd

VIEWS = {"V1": [0, 7], "V2": [1, 2], "V3": [3, 4], "V4": [5, 6]}
VIEW_NAMES = {"V1": "Formality & Resource", "V2": "Operational Capacity",
              "V3": "Financial Capacity", "V4": "Digital Capability"}
FEATURES_EN = ["NIB/SKU", "Employees", "Production capacity", "Revenue (log1p)", "Assets (log1p)",
               "Social media", "Marketplace", "Land ownership"]


def load(path):
    """Accepts the original workbook (sheet 'MinMax_2896', optional 'MinMax_Info') or a CSV
    whose first column is the MSME identifier followed by the eight normalized attributes."""
    path = Path(path)
    info = None
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, sheet_name="MinMax_2896", engine="openpyxl")
        try:
            info = pd.read_excel(path, sheet_name="MinMax_Info", engine="openpyxl")
        except ValueError:
            info = None
    else:
        df = pd.read_csv(path)
    ids = df.iloc[:, 0]
    data = df.iloc[:, 1:9].apply(pd.to_numeric, errors="raise")
    x = data.to_numpy(float)
    if x.shape[1] != 8 or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("Expected a finite normalized eight-feature matrix in [0, 1]")
    return ids, data, x, info


def to_original(data, x, info):
    """Invert min-max (and log1p for revenue/assets) using the MinMax_Info sheet."""
    original = data.copy()
    for j, col in enumerate(data.columns):
        rr = info[info.iloc[:, 0] == col].iloc[0]
        lo, hi = float(rr.iloc[1]), float(rr.iloc[2])
        original[col] = x[:, j] * (hi - lo) + lo
        if col in ["Omset_Log1p", "Aset_Log1p"]:
            original[col] = np.expm1(original[col])
    return original
