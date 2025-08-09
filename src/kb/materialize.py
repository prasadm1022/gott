import argparse
from pathlib import Path

import pandas as pd


def pct(a, b):
    if b == 0 or pd.isna(a) or pd.isna(b): return None
    return 100.0 * (a - b) / b


def load_parquet(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    # expected columns: Year, Month, Type in {'Income','Expense'}, Source, Amount
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df["Amount"] = df["Amount"].astype(float)
    return df


def year_summary(df: pd.DataFrame, year: int) -> str:
    d = df.query("Year == @year").copy()
    inc = d.query("Type == 'Income'")["Amount"].sum()
    exp = d.query("Type == 'Expense'")["Amount"].sum()
    sav = inc - exp
    sr = (sav / inc * 100.0) if inc else None

    # top 5 expense categories
    top = (d.query("Type == 'Expense'")
           .groupby("Source", as_index=False)["Amount"].sum()
           .sort_values("Amount", ascending=False).head(5))

    lines = [f"# Year {year} overview", f"- Total income: {inc:,.2f}", f"- Total expense: {exp:,.2f}",
             f"- Savings: {sav:,.2f}"]
    if sr is not None:
        lines.append(f"- Savings rate: {sr:.1f}%")
    lines.append("")
    lines.append("## Top expense categories")
    for _, r in top.iterrows():
        lines.append(f"- {r['Source']}: {r['Amount']:,.2f}")

    # simple MoM for total expense
    m = (d.query("Type == 'Expense'")
         .groupby(["Year", "Month"], as_index=False)["Amount"].sum()
         .sort_values(["Year", "Month"]))
    mom_lines = []
    prev = None
    for _, row in m.iterrows():
        cur = row["Amount"];
        cur_m = int(row["Month"])
        change = pct(cur, prev) if prev is not None else None
        if change is not None:
            mom_lines.append(f"- {year}-{cur_m:02d}: {cur:,.2f} (MoM {change:+.1f}%)")
        else:
            mom_lines.append(f"- {year}-{cur_m:02d}: {cur:,.2f}")
        prev = cur

    lines.append("")
    lines.append("## Monthly expense trend")
    lines.extend(mom_lines)

    return "\n".join(lines).strip() + "\n"


def category_summary(df: pd.DataFrame, source: str) -> str:
    d = df.query("Source == @source").copy()
    inc = d.query("Type == 'Income'")["Amount"].sum()
    exp = d.query("Type == 'Expense'")["Amount"].sum()

    lines = []
    lines.append(f"# Category: {source}")
    if exp:
        lines.append(f"- Lifetime expense total: {exp:,.2f}")
    if inc:
        lines.append(f"- Lifetime income total: {inc:,.2f}")

    y = (d.groupby(["Year", "Type"], as_index=False)["Amount"].sum()
         .sort_values(["Year", "Type"]))
    lines.append("")
    lines.append("## Yearly totals")
    for _, r in y.iterrows():
        lines.append(f"- {int(r['Year'])} {r['Type']}: {r['Amount']:,.2f}")

    # simple spike detection on monthly expense for this source
    m = (d.query("Type == 'Expense'")
         .groupby(["Year", "Month"], as_index=False)["Amount"].sum()
         .assign(ds=lambda x: pd.to_datetime(dict(year=x["Year"], month=x["Month"], day=1)))
         .sort_values("ds"))
    if len(m) >= 6:
        mean = m["Amount"].rolling(6, min_periods=6).mean()
        std = m["Amount"].rolling(6, min_periods=6).std()
        spikes = []
        for i in range(len(m)):
            if pd.notna(std.iloc[i]) and std.iloc[i] > 0:
                z = (m["Amount"].iloc[i] - mean.iloc[i]) / std.iloc[i]
                if z >= 2.5:
                    yv, mv, val = int(m["Year"].iloc[i]), int(m["Month"].iloc[i]), m["Amount"].iloc[i]
                    spikes.append(f"- Spike {yv}-{mv:02d}: {val:,.2f} (z≈{z:.1f})")
        if spikes:
            lines.append("")
            lines.append("## Notable spikes (6-month z≥2.5)")
            lines.extend(spikes)

    return "\n".join(lines).strip() + "\n"


def write_md(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def materialize(root: Path) -> None:
    parquet_file = root / "data" / "processed" / "all_years_data.parquet"
    kb_raw = root / "kb" / "raw"

    df = load_parquet(parquet_file)

    # per-year facts
    for year in sorted(df["Year"].unique()):
        md = year_summary(df, int(year))
        write_md(md, kb_raw / "facts" / f"{year}.md")

    # per-category narratives (top N by total expense to keep KB small)
    top_cats = (df.query("Type == 'Expense'")
                .groupby("Source", as_index=False)["Amount"].sum()
                .sort_values("Amount", ascending=False)
                .head(30)["Source"].tolist())
    for s in top_cats:
        md = category_summary(df, s)
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)
        write_md(md, kb_raw / "categories" / f"{safe}.md")

    print(f"[kb] Wrote summaries into {kb_raw}")
