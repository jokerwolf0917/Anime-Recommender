from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def _resolve_csv_path(path: str | Path, filename: str) -> Path:
    """Resolve CSV path robustly across different working directories."""
    if path is None:
        return (DEFAULT_DATA_DIR / filename).resolve()

    p = Path(path)
    if p.exists():
        return p.resolve()

    # If a relative path was provided, try project-root based fallback.
    if not p.is_absolute():
        candidate = (PROJECT_ROOT / p).resolve()
        if candidate.exists():
            return candidate
        candidate = (DEFAULT_DATA_DIR / p.name).resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"CSV file not found: {path}")


def load_anime(path: str | Path = DEFAULT_DATA_DIR / "anime.csv"):
    return pd.read_csv(_resolve_csv_path(path, "anime.csv"))


def load_synopsis(path: str | Path = DEFAULT_DATA_DIR / "anime_with_synopsis.csv"):
    return pd.read_csv(_resolve_csv_path(path, "anime_with_synopsis.csv"))


def load_ratings(path: str | Path = DEFAULT_DATA_DIR / "rating_complete.csv"):
    return pd.read_csv(_resolve_csv_path(path, "rating_complete.csv"))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("source:", "")
    text = text.replace("url", "")
    return text.strip()
