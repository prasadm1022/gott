"""
Preprocess raw CSV files into tidy format and merge them into a single Parquet file.
- Reads {ROOT}/data/raw/*.csv
- Writes tidy CSVs to {ROOT}/data/tidy/*_tidy.csv
- Merges all tidy CSVs into {ROOT}/data/processed/all_years_data.parquet
"""

import os
import re

from glob import glob
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------
# Month / period parsing
# ---------------------------
MONTHS = {
    "JAN": 1, "JANUARY": 1,
    "FEB": 2, "FEBRUARY": 2,
    "MAR": 3, "MARCH": 3,
    "APR": 4, "APRIL": 4,
    "MAY": 5,
    "JUN": 6, "JUNE": 6,
    "JUL": 7, "JULY": 7,
    "AUG": 8, "AUGUST": 8,
    "SEP": 9, "SEPT": 9, "SEPTEMBER": 9,
    "OCT": 10, "OCTOBER": 10,
    "NOV": 11, "NOVEMBER": 11,
    "DEC": 12, "DECEMBER": 12,
}
YEAR_RE = re.compile(r"(19|20)\d{2}")
MONTH_TOKEN_RE = re.compile(
    r"(?i)\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b"
)


def _normalize_cols(cols: Iterable[str]) -> List[str]:
    return [str(c).strip() for c in cols]


# Return True if header looks like a month name (e.g., 'Jan', '2020-Jan', 'Aug-24').
def _is_month_header(s: str) -> bool:
    if s is None:
        return False
    s = str(s).strip()
    return bool(MONTH_TOKEN_RE.search(s))


def _parse_year_from_filename(fp: str | Path) -> Optional[int]:
    m = YEAR_RE.search(os.path.basename(str(fp)))
    return int(m.group(0)) if m else None


# Returns (year, month) for labels like: '2020 JAN', 'JAN 2020', '2020-Jan', 'Aug', 'AUG-20', '2021-August'
def _parse_period_label(label: str) -> Tuple[Optional[int], Optional[int]]:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None, None

    s = str(label).strip()

    y = YEAR_RE.search(s)
    year = int(y.group(0)) if y else None

    m = MONTH_TOKEN_RE.search(s)
    month = None

    if m:
        token = m.group(1).upper()
        month = MONTHS.get(token)

    return year, month


# Robust numeric parser:
# - handles commas and spaces
# - handles currency prefixes/suffixes (Rs, $, etc.)
# - parentheses for negatives
# - blanks/None/'-'
def _parse_amount(x) -> Optional[float]:
    if pd.isna(x):
        return None

    s = str(x).strip()
    if s in {"", "-"}:
        return None

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]

    # remove everything except digits, dot, minus
    s = re.sub(r"[^\d.\-]", "", s)
    if s in {"", ".", "-"}:
        return None

    try:
        val = float(s)
        return -val if neg else val
    except ValueError:
        return None


# Read a raw CSV, convert from wide to tidy, and save to output_path.
# Tidy columns: period, Year, Month, Type, Source, Amount
def process_csv(file_path: Path, output_path: Path) -> None:
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    if df.empty:
        print(f"Skipped empty file: {file_path}")
        return

    # Normalize headers; first column is the category
    df.columns = _normalize_cols(df.columns)
    df = df.rename(columns={df.columns[0]: "Category"})

    # Keep month-like columns (drop obvious totals)
    non_month = {"total", "totals", "grand total"}
    month_cols = [
                     c for c in df.columns[1:]
                     if _is_month_header(c) and str(c).strip().lower() not in non_month
                 ] or list(df.columns[1:])  # fallback

    # Wide → long
    df_long = df.melt(
        id_vars=["Category"],
        value_vars=month_cols,
        var_name="Month",
        value_name="AmountRaw",
    )

    # Parse amounts
    df_long["Amount"] = df_long["AmountRaw"].apply(_parse_amount)

    # Split "Type - Source"
    split_df = df_long["Category"].str.split(" - ", n=1, expand=True)
    if split_df.shape[1] == 1:
        split_df[1] = np.nan
    split_df.columns = ["Type", "Source"]
    df_long = pd.concat([df_long, split_df], axis=1)

    # Year/Month from label; fallback to year in filename
    ym = df_long["Month"].apply(_parse_period_label)
    df_long["Year"] = [y for (y, m) in ym]
    df_long["MonthNum"] = [m for (y, m) in ym]

    y_file = _parse_year_from_filename(file_path)
    # Avoid FutureWarning by making it numeric first, then filling
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
    if y_file is not None:
        df_long["Year"] = df_long["Year"].fillna(y_file).astype("Int64")

    # Keep valid rows only (no amount / no month -> drop)
    df_long = df_long.dropna(subset=["Amount", "MonthNum"]).copy()

    # Final tidy WITHOUT 'period'
    tidy = df_long.loc[
           :, ["Year", "MonthNum", "Type", "Source", "Amount"]
           ].rename(columns={"MonthNum": "Month"}) \
        .sort_values(["Year", "Month", "Type", "Source"], kind="stable")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(output_path, index=False)
    print(f"Saved tidy file: {output_path}")


# Read all *_tidy.csv, concat, write a single parquet.
def merge_tidy_csv_to_parquet(tidy_dir: Path, parquet_file: Path) -> None:
    tidy_files = sorted(glob(os.path.join(tidy_dir, "*_tidy.csv")))
    if not tidy_files:
        print(f"No tidy CSVs to merge in {tidy_dir}")
        return

    frames = [pd.read_csv(f) for f in tidy_files]
    merged = pd.concat(frames, ignore_index=True)

    # Write compressed parquet (needs pyarrow or fastparquet installed)
    merged.to_parquet(parquet_file, index=False)  # needs pyarrow or fastparquet
    print(f"\n✅  Merged {len(tidy_files)} tidy files → {parquet_file}")
    print(f"   Rows written: {len(merged):,}")


# Convert all raw CSVs to tidy CSVs, then merge to parquet.
def run_pipeline(root: Path) -> None:
    # Prepare input/output paths
    raw_dir = root / "data" / "raw"
    tidy_dir = root / "data" / "tidy"
    processed_dir = root / "data" / "processed"
    parquet_file = processed_dir / "all_years_data.parquet"

    os.makedirs(tidy_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSV files found in {raw_dir}")
        return

    for csv in csvs:
        out = tidy_dir / f"{csv.stem}_tidy.csv"
        process_csv(csv, out)

    merge_tidy_csv_to_parquet(tidy_dir, parquet_file)
