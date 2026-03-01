import pandas as pd

def load_anime(path="data/anime.csv"):
    return pd.read_csv(path)

def load_synopsis(path="data/anime_with_synopsis.csv"):
    return pd.read_csv(path)

def load_ratings(path="data/rating_complete.csv"):
    return pd.read_csv(path)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("source:", "")
    text = text.replace("url", "")
    return text.strip()
