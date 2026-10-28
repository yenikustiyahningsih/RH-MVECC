"""Scan data/ and results/ for owner/business names or address columns before pushing.
Usage (from the repo root):  python check_repo_privacy.py
Exit code 1 if anything suspicious is found."""
import re, sys
from pathlib import Path
import pandas as pd

BAD = re.compile(r"nama\s*pemilik|nama\s*usaha|owner\s*name|business\s*name|alamat|address|\bnik\b|telepon|no\.?\s*hp", re.I)
ID_OK = re.compile(r"^UMKM-\d{4}$")
problems = 0
files = [p for d in ("data", "results") if Path(d).exists() for p in Path(d).rglob("*")
         if p.is_file() and p.suffix.lower() in (".xlsx", ".xlsm", ".xls", ".csv", ".json", ".txt", ".md")]
print(f"{len(files)} files scanned under data/ and results/")
for p in sorted(files):
    if p.suffix.lower() == ".xlsm":
        print(f"[!] macro workbook present: {p}"); problems += 1; continue
    try:
        if p.suffix.lower() in (".xlsx", ".xls"):
            sheets = pd.read_excel(p, sheet_name=None, dtype=str)
        elif p.suffix.lower() == ".csv":
            sheets = {"csv": pd.read_csv(p, dtype=str, low_memory=False)}
        else:
            txt = p.read_text(encoding="utf8", errors="ignore")
            if BAD.search(txt): print(f"[!] {p}: matches a sensitive term"); problems += 1
            continue
    except Exception as e:
        print(f"[?] could not read {p}: {e}"); continue
    for name, df in sheets.items():
        hits = [c for c in df.columns if BAD.search(str(c))]
        if hits: print(f"[!] {p} [{name}]: sensitive column header(s) {hits}"); problems += 1
        # text cells that look like names: any non-numeric string column other than ID/labels
        for c in df.columns:
            col = df[c].dropna().astype(str)
            if col.empty: continue
            if str(c).lower() in ("id", "umkm_id", "unnamed: 0") or col.map(lambda v: bool(ID_OK.match(v))).mean() > .95:
                if not col.map(lambda v: bool(ID_OK.match(v))).all() and "UMKM" in " ".join(col.head(20)):
                    print(f"[!] {p} [{name}] column {c}: IDs not all of the form UMKM-####"); problems += 1
print("\nRESULT:", "OK - nothing suspicious found" if not problems else f"{problems} problem(s) found - fix before pushing")
sys.exit(1 if problems else 0)