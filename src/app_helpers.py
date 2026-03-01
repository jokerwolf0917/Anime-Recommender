import numpy as np
import pandas as pd


NUMERIC_FEATURE_COLUMNS = ["Score", "Members", "Favorites", "Episodes"]
SYNOPSIS_CANDIDATES = ["synopsis", "sypnopsis"]


def _safe_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "unknown", "none", "nan", "null"}:
        return ""
    return text


def _to_float_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .replace(
            {
                "": np.nan,
                "Unknown": np.nan,
                "unknown": np.nan,
                "None": np.nan,
                "none": np.nan,
                "nan": np.nan,
                "null": np.nan,
                "-": np.nan,
            }
        )
    )
    return pd.to_numeric(cleaned, errors="coerce")


def pick_synopsis_column(df: pd.DataFrame) -> str | None:
    lower_map = {col.lower().strip(): col for col in df.columns}
    for candidate in SYNOPSIS_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    for col in df.columns:
        if "synops" in col.lower():
            return col
    return None


def merge_synopsis_robust(anime_df: pd.DataFrame, syn_df: pd.DataFrame) -> pd.DataFrame:
    merged = anime_df.copy()
    if "MAL_ID" not in merged.columns:
        merged["synopsis"] = ""
        return merged

    if syn_df is None or syn_df.empty or "MAL_ID" not in syn_df.columns:
        if "synopsis" in merged.columns:
            merged["synopsis"] = merged["synopsis"].fillna("")
        else:
            merged["synopsis"] = ""
        return merged

    synopsis_col = pick_synopsis_column(syn_df)
    keep_cols = ["MAL_ID"]
    rename_map = {}

    if synopsis_col is not None:
        keep_cols.append(synopsis_col)
        rename_map[synopsis_col] = "synopsis"

    if "Genres" in syn_df.columns:
        keep_cols.append("Genres")
        rename_map["Genres"] = "Genres_syn"

    syn_small = syn_df[keep_cols].rename(columns=rename_map)
    merged = merged.merge(syn_small, on="MAL_ID", how="left")

    if "Genres_syn" in merged.columns:
        if "Genres" in merged.columns:
            base_genres = merged["Genres"]
        else:
            base_genres = pd.Series([""] * len(merged), index=merged.index)
        merged["Genres"] = merged["Genres_syn"].combine_first(base_genres).fillna("")
        merged = merged.drop(columns=["Genres_syn"])
    elif "Genres" not in merged.columns:
        merged["Genres"] = ""

    if "synopsis" not in merged.columns:
        merged["synopsis"] = ""
    merged["synopsis"] = merged["synopsis"].fillna("")
    return merged


def combine_item_text(row: pd.Series, clean_text_func) -> str:
    fields: list[str] = []
    for col in ["Genres", "Type", "Studios", "Producers", "Source"]:
        value = _safe_text(row.get(col, ""))
        if value:
            fields.append(f"{col.lower()}={clean_text_func(value)}")

    name_value = _safe_text(row.get("Name", ""))
    if name_value:
        fields.append(f"name={clean_text_func(name_value)}")

    episodes = _to_float_series(pd.Series([row.get("Episodes", np.nan)])).iloc[0]
    if pd.isna(episodes):
        episodes = 0.0
    fields.append(f"episodes={float(episodes):.1f}")

    synopsis_raw = (
        row.get("synopsis", row.get("Synopsis", row.get("sypnopsis", "")))
    )
    synopsis_value = _safe_text(synopsis_raw)
    if synopsis_value:
        fields.append(clean_text_func(synopsis_value))

    return " ".join(part for part in fields if part)


def build_numeric_features(df: pd.DataFrame) -> np.ndarray:
    numeric_cols = [col for col in NUMERIC_FEATURE_COLUMNS if col in df.columns]
    if not numeric_cols:
        return np.zeros((len(df), 1), dtype=float)

    safe_numeric = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        safe_numeric[col] = _to_float_series(df[col])

    safe_numeric = safe_numeric.fillna(0.0)
    return safe_numeric.to_numpy(dtype=float)
