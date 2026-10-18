import pandas as pd

PLACEHOLDERS = {"", "na", "n/a", "nan", "none", "null", "unknown", "?", "-", "--", "missing"}


def audit_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Combines table-level summary metrics and column-level audit into one clean DataFrame."""
    rows = []
    n_rows = len(df)

    # 1. Calculate table-level stats to attach or display
    total_duplicates = int(df.astype(str).duplicated().sum())

    for col in df.columns:
        s = df[col]

        # Safe check for nested lists or dicts
        if s.map(lambda v: isinstance(v, (list, dict))).any():
            s = s.astype(str).where(s.notna())

        missing = int(s.isna().sum())
        missing_pct = round(100 * missing / n_rows, 1) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=False))

        # Determine data quality flags
        flags = []
        if missing_pct >= 99.9:
            flags.append("EMPTY")
        elif missing_pct >= 30:
            flags.append("MOSTLY_MISSING")
        elif missing_pct > 0:
            flags.append("MISSING")

        if unique <= 1 and missing < n_rows:
            flags.append("CONSTANT")
        if unique == n_rows:
            flags.append("ID?")

        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "missing values": missing,
            "missing_%": missing_pct,
            "unique values": unique,
            "duplicated values": int(s.duplicated().sum()),
            "flags": ", ".join(flags) if flags else "OK"
        })

    audit_df = pd.DataFrame(rows).set_index("column")
    return audit_df