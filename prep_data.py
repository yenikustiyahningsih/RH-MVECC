#!/usr/bin/env python3
"""
prep_data.py - rebuild the anonymised, min-max normalised MSME matrix from the
ORIGINAL DPMPTSP Sampang Regency records (2,896 MSMEs, extracted 18 Sept 2025).

The original file contains owner and business names and is NOT distributed.
This script is provided so that every preparation step is documented and can be
repeated by anyone who holds an authorised copy of the original file. Names are
never read into the output: only the eight analytical attributes are kept, and
records receive sequential identifiers (UMKM-0001, UMKM-0002, ...).

Steps (identical to Section 4.1 of the manuscript)
  1. Employees: values above 6 are recoded as 6.
  2. Production capacity, revenue and assets: winsorised at the upper Tukey
     fence Q3 + 1.5 * IQR, computed on the ORIGINAL scale.
  3. Revenue and assets: log1p.
  4. Social media and marketplace: number of platforms used, capped at 3.
  5. All eight attributes: min-max normalisation to [0, 1].

Marketplace coding (--marketplace-coding)
  as_analyzed (default): records whose ONLY marketplace is TikTok Shop are coded
      as 2 platforms. This is the coding of the matrix behind every result in the
      manuscript.
  corrected: the same records are coded as 1 platform (sensitivity analysis:
      K = 6, SC 0.4043, DBI 0.9948, CHI 1337.9, VDI 0.7786, ARI 0.984).

Example
  python prep_data.py --input dataset_2896.xlsx --output prep_out/data_prep_2896.xlsx
  python prep_data.py --input dataset_2896.xlsx --output prep_out/corrected.xlsx \
         --marketplace-coding corrected
  python prep_data.py --input dataset_2896.xlsx --output prep_out/data_prep_2896.xlsx \
         --compare data/data_prep_2896_reconstructed.xlsx

Column detection: the original column names are matched by keyword (see
KEYWORDS). If a column is not found, or more than one matches, the script stops
and lists the candidates; pass them explicitly with --map, e.g.
  --map "employees=Jumlah Karyawan" --map "land=Kepemilikan Lahan"
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# canonical attribute order = order of the deposited matrix / Table 4
ATTRS = ["nib", "employees", "capacity", "revenue", "assets",
         "social", "marketplace", "land"]
LABELS = {
    "nib": "NIB/SKU", "employees": "Employees", "capacity": "Production capacity",
    "revenue": "Revenue (log1p)", "assets": "Assets (log1p)",
    "social": "Social media", "marketplace": "Marketplace", "land": "Land ownership",
}
KEYWORDS = {  # lower-case substrings searched in the original column names
    "nib": ["nib", "sku"],
    "employees": ["karyawan", "tenaga kerja", "employee", "pekerja"],
    "capacity": ["kapasitas", "capacity"],
    "revenue": ["omzet", "omset", "revenue", "pendapatan"],
    "assets": ["aset", "asset"],
    "social": ["sosmed", "sosial media", "media sosial", "social"],
    "marketplace": ["marketplace", "market place"],
    "land": ["lahan", "tanah", "land"],
}
NAME_COLUMNS = ["nama pemilik", "nama usaha", "pemilik", "owner", "alamat", "address"]
TRUE_WORDS = {"1", "ya", "y", "ada", "punya", "memiliki", "milik", "sudah", "yes",
              "true", "sendiri", "hak milik", "milik sendiri"}
FALSE_WORDS = {"0", "tidak", "tidak ada", "tidak punya", "tidak memiliki", "belum",
               "no", "false", "-", "", "nan", "none", "bukan"}
NONE_PLATFORM = {"", "-", "0", "tidak", "tidak ada", "belum ada", "nan", "none"}


def is_name_column(lower_name):
    """True for owner/business-name or address columns (never used)."""
    return any(re.search(r"\b" + re.escape(n) + r"\b", lower_name) for n in NAME_COLUMNS)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_any(path, sheet):
    p = Path(path)
    if p.suffix.lower() in (".csv", ".txt"):
        return pd.read_csv(p)
    return pd.read_excel(p, sheet_name=sheet if sheet is not None else 0)


def detect_columns(df, overrides):
    cols = {}
    low = {c: str(c).lower() for c in df.columns}
    for key in ATTRS:
        if key in overrides:
            name = overrides[key]
            if name not in df.columns:
                sys.exit(f"--map {key}={name!r}: column not found. Columns: {list(df.columns)}")
            cols[key] = name
            continue
        hits = [c for c, l in low.items()
                if any(k in l for k in KEYWORDS[key]) and not is_name_column(l)]
        if len(hits) != 1:
            sys.exit(f"Cannot identify the column for '{key}' automatically "
                     f"(candidates: {hits}). Use --map \"{key}=<column name>\". "
                     f"All columns: {list(df.columns)}")
        cols[key] = hits[0]
    if len(set(cols.values())) != len(cols):
        sys.exit(f"Two attributes were mapped to the same column: {cols}")
    return cols


def to_binary(s, name):
    out, bad = [], set()
    for v in s:
        if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
            out.append(1 if v > 0 else 0)
            continue
        t = str(v).strip().lower()
        if t in TRUE_WORDS:
            out.append(1)
        elif t in FALSE_WORDS or pd.isna(v):
            out.append(0)
        else:
            bad.add(t)
            out.append(np.nan)
    if bad:
        sys.exit(f"Unrecognised values in binary column '{name}': {sorted(bad)[:20]}. "
                 f"Edit TRUE_WORDS / FALSE_WORDS in prep_data.py.")
    return np.array(out, dtype=float)


def platform_set(v):
    if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
        return None  # numeric count, handled by caller
    t = str(v).strip().lower()
    if t in NONE_PLATFORM or pd.isna(v):
        return set()
    parts = re.split(r"[,;/+&\n]| dan ", t)
    return {re.sub(r"\s+", " ", p).strip() for p in parts if p.strip() not in NONE_PLATFORM}


def is_tiktok_only(pset):
    if not pset:
        return False
    norm = {re.sub(r"[^a-z]", "", p) for p in pset}
    return norm <= {"tiktokshop", "tiktok"} and "tiktokshop" in norm


def count_platforms(s, cap=3, tiktok_rule=None):
    counts, tiktok_only = [], 0
    for v in s:
        ps = platform_set(v)
        if ps is None:
            n = int(v)
        else:
            n = len(ps)
            if tiktok_rule is not None and is_tiktok_only(ps):
                tiktok_only += 1
                n = tiktok_rule
        counts.append(min(n, cap))
    return np.array(counts, dtype=float), tiktok_only


def tukey_upper(x):
    q1, q3 = np.percentile(x, [25, 75])
    return q3 + 1.5 * (q3 - q1)


def minmax(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x), lo, hi


def build(df, cols, coding):
    report, work = {}, {}
    work["nib"] = to_binary(df[cols["nib"]], cols["nib"])
    work["land"] = to_binary(df[cols["land"]], cols["land"])

    emp = pd.to_numeric(df[cols["employees"]], errors="raise").to_numpy(float)
    report["employees_max_source"] = float(emp.max())
    report["employees_capped"] = int((emp > 6).sum())
    work["employees"] = np.minimum(emp, 6)

    fences = {}
    for key in ("capacity", "revenue", "assets"):
        x = pd.to_numeric(df[cols[key]], errors="raise").to_numpy(float)
        fence = tukey_upper(x)
        fences[key] = fence
        report[f"{key}_fence"] = float(fence)
        report[f"{key}_clipped"] = int((x > fence).sum())
        work[key] = np.minimum(x, fence)
    work["revenue"] = np.log1p(work["revenue"])
    work["assets"] = np.log1p(work["assets"])

    work["social"], _ = count_platforms(df[cols["social"]])
    rule = 2 if coding == "as_analyzed" else 1
    work["marketplace"], n_tt = count_platforms(df[cols["marketplace"]], tiktok_rule=rule)
    report["tiktok_only_records"] = int(n_tt)

    mat, info = {}, []
    for key in ATTRS:
        z, lo, hi = minmax(np.asarray(work[key], float))
        mat[LABELS[key]] = z
        transform = "log1p" if key in ("revenue", "assets") else "none"
        info.append({"Attribute": LABELS[key], "Transform": transform,
                     "Min": lo, "Max": hi})
    return pd.DataFrame(mat), pd.DataFrame(info), report


def compare(new, path):
    ref = pd.read_excel(path, sheet_name="MinMax_2896")
    ref_num = ref.select_dtypes("number")
    if ref_num.shape[1] != 8:  # ID column may be numeric-like; keep last 8
        ref_num = ref_num.iloc[:, -8:]
    print(f"\nComparison with {path}")
    print(f"  rows: new={len(new)} reference={len(ref_num)}")
    print(f"  reference headers: {list(ref_num.columns)}")
    if len(new) != len(ref_num):
        print("  !! row counts differ; per-column comparison skipped")
        return
    for j, c in enumerate(new.columns):
        d = np.abs(new[c].to_numpy() - ref_num.iloc[:, j].to_numpy())
        print(f"  {c:<22s} max|diff|={d.max():.6f}  mean|diff|={d.mean():.6f}")
    try:
        info = pd.read_excel(path, sheet_name="MinMax_Info")
        print(f"  reference MinMax_Info columns: {list(info.columns)}")
    except Exception as e:  # noqa: BLE001
        print(f"  (no MinMax_Info sheet readable: {e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="original records (.xlsx/.csv); never commit this file")
    ap.add_argument("--sheet", default=None, help="sheet name or index of the original file (default: first)")
    ap.add_argument("--output", required=True, help="output .xlsx (sheets MinMax_2896, MinMax_Info, Run_Info)")
    ap.add_argument("--marketplace-coding", choices=["as_analyzed", "corrected"], default="as_analyzed")
    ap.add_argument("--map", action="append", default=[], metavar="KEY=COLUMN",
                    help=f"override column detection; KEY in {ATTRS}")
    ap.add_argument("--round", type=int, default=6, help="decimals stored in the matrix (deposited matrix: 6)")
    ap.add_argument("--compare", default=None, help="deposited matrix to compare with (diagnostic only)")
    a = ap.parse_args()

    overrides = {}
    for m in a.map:
        k, _, v = m.partition("=")
        if k not in ATTRS or not v:
            sys.exit(f"--map expects KEY=COLUMN with KEY in {ATTRS}")
        overrides[k] = v
    sheet = int(a.sheet) if (a.sheet is not None and a.sheet.isdigit()) else a.sheet

    df = read_any(a.input, sheet)
    n = len(df)
    cols = detect_columns(df, overrides)
    print(f"{n} records read. Column mapping:")
    for k in ATTRS:
        print(f"  {LABELS[k]:<22s} <- {cols[k]!r}")

    mat, info, rep = build(df, cols, a.marketplace_coding)
    mat = mat.round(a.round)
    mat.insert(0, "ID", [f"UMKM-{i:04d}" for i in range(1, n + 1)])

    print("\nChecks against Section 4.1 (expected values in brackets):")
    print(f"  records: {n} [2896]")
    print(f"  employees > 6 recoded as 6: {rep['employees_capped']} [29]; source max {rep['employees_max_source']:.0f} [10]")
    print(f"  capacity fence {rep['capacity_fence']:.1f} [1302.5], clipped {rep['capacity_clipped']} [387]")
    print(f"  revenue fence {rep['revenue_fence']:,.1f} [27.5 million], assets fence {rep['assets_fence']:,.1f} [39.5 million]")
    print(f"  records whose only marketplace is TikTok Shop: {rep['tiktok_only_records']} [191]")
    print(f"  marketplace coding: {a.marketplace_coding}")
    print("  attribute means (Table 4): " +
          ", ".join(f"{c}={mat[c].mean():.4f}" for c in mat.columns[1:]))

    run_info = pd.DataFrame([
        ("input_sha256", sha256(a.input)), ("marketplace_coding", a.marketplace_coding),
        ("n_records", n), ("round_decimals", a.round),
        ("employees_recoded", rep["employees_capped"]),
        ("tiktok_only_records", rep["tiktok_only_records"]),
    ], columns=["Key", "Value"])
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(a.output) as xw:
        mat.to_excel(xw, sheet_name="MinMax_2896", index=False)
        info.to_excel(xw, sheet_name="MinMax_Info", index=False)
        run_info.to_excel(xw, sheet_name="Run_Info", index=False)
    print(f"\nWritten: {a.output}  (input SHA-256 {run_info.iloc[0, 1]})")

    if a.compare:
        compare(mat.drop(columns="ID"), a.compare)


if __name__ == "__main__":
    main()