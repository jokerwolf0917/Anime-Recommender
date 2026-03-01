import time
import numpy as np
import pandas as pd
import streamlit as st

try:
    from .data_loader import load_anime, load_synopsis, load_ratings, clean_text
    from .collaborative_model import CFRecommender
    from .content_model import ContentRecommender
    from .app_helpers import combine_item_text, build_numeric_features, merge_synopsis_robust
except ImportError:
    from data_loader import load_anime, load_synopsis, load_ratings, clean_text
    from collaborative_model import CFRecommender
    from content_model import ContentRecommender
    from app_helpers import combine_item_text, build_numeric_features, merge_synopsis_robust


@st.cache_data(show_spinner=False)
def load_datasets(anime_path: str, synopsis_path: str, ratings_path: str):
    anime_df = load_anime(anime_path)
    syn_df = load_synopsis(synopsis_path)
    ratings_df = load_ratings(ratings_path)
    return anime_df, syn_df, ratings_df


def sample_ratings_for_cf(ratings_df: pd.DataFrame, enabled: bool, sample_size: int) -> pd.DataFrame:
    if not enabled:
        return ratings_df
    if sample_size <= 0 or sample_size >= len(ratings_df):
        return ratings_df
    return ratings_df.sample(n=sample_size, random_state=42)


def compute_user_profile_vector(
    user_id: int,
    ratings_df: pd.DataFrame,
    threshold: float,
    id2idx: dict,
    item_matrix: np.ndarray,
) -> np.ndarray:
    user_rows = ratings_df[ratings_df["user_id"] == user_id]
    liked_rows = user_rows[user_rows["rating"] >= threshold]
    liked_ids = liked_rows["anime_id"].tolist()
    if not liked_ids:
        return np.zeros(item_matrix.shape[1], dtype=float)

    valid_pairs = []
    for anime_id in liked_ids:
        if anime_id in id2idx:
            rating_series = liked_rows[liked_rows["anime_id"] == anime_id]["rating"]
            if not rating_series.empty:
                valid_pairs.append((id2idx[anime_id], float(rating_series.iloc[0])))

    if not valid_pairs:
        return np.zeros(item_matrix.shape[1], dtype=float)

    idxs = [idx for idx, _ in valid_pairs]
    weights = np.array([rating for _, rating in valid_pairs], dtype=float)
    weights = weights - weights.min() + 1.0
    user_vec = (item_matrix[idxs] * weights[:, None]).sum(axis=0) / (weights.sum() + 1e-8)
    norm = np.linalg.norm(user_vec) + 1e-8
    return user_vec / norm


def cosine_scores(user_vec: np.ndarray, item_matrix: np.ndarray) -> np.ndarray:
    if user_vec.ndim == 1:
        user_vec = user_vec.reshape(1, -1)
    sims = item_matrix @ user_vec.T
    return sims.ravel()


def format_reason(rec_row: pd.Series, liked_genre_counts: dict) -> str:
    genres_value = rec_row.get("Genres")
    rec_genres = set([g.strip().lower() for g in str(genres_value or "").split(",") if g.strip()])
    overlap = [(g, liked_genre_counts.get(g, 0)) for g in rec_genres]
    overlap.sort(key=lambda x: x[1], reverse=True)
    top = [g for g, count in overlap if count > 0][:2]
    if top:
        return f"Because you like: {', '.join(top)}"
    return "Content match to your profile"


def compute_liked_genre_counts(
    user_id: int,
    ratings_df: pd.DataFrame,
    anime_df: pd.DataFrame,
    threshold: float = 7.0,
):
    user_rows = ratings_df[(ratings_df["user_id"] == user_id) & (ratings_df["rating"] >= threshold)]
    genre_counts = {}
    for aid in user_rows["anime_id"].tolist():
        row = anime_df.loc[anime_df["MAL_ID"] == aid]
        if row.empty:
            continue
        genres = str(row.iloc[0].get("Genres") or "")
        for genre in genres.split(","):
            key = genre.strip().lower()
            if key:
                genre_counts[key] = genre_counts.get(key, 0) + 1
    return genre_counts


@st.cache_resource(show_spinner=True)
def fit_content_model(anime_df: pd.DataFrame) -> dict:
    prepared = anime_df.copy()
    if "synopsis" not in prepared.columns:
        prepared["synopsis"] = ""

    item_texts = prepared.apply(lambda row: combine_item_text(row, clean_text), axis=1).tolist()
    numeric = build_numeric_features(prepared)

    content = ContentRecommender(n_components=200)
    content.fit(item_texts, numeric)

    item_ids = prepared["MAL_ID"].tolist()
    id2idx = {aid: idx for idx, aid in enumerate(item_ids)}

    item_matrix = content.item_matrix.astype(np.float32)
    norms = np.linalg.norm(item_matrix, axis=1, keepdims=True) + 1e-8
    item_matrix = item_matrix / norms

    return {"content": content, "item_matrix": item_matrix, "id2idx": id2idx}


@st.cache_resource(show_spinner=True)
def fit_cf_model(ratings_df: pd.DataFrame) -> CFRecommender:
    required = {"user_id", "anime_id", "rating"}
    missing = required - set(ratings_df.columns)
    if missing:
        raise ValueError(f"Ratings data is missing required columns: {sorted(missing)}")

    safe = ratings_df[list(required)].copy()
    safe["rating"] = pd.to_numeric(safe["rating"], errors="coerce")
    safe = safe.dropna(subset=["user_id", "anime_id", "rating"])
    if safe.empty:
        raise ValueError("Ratings data is empty after removing invalid rows.")

    cf = CFRecommender(k=40)
    cf.fit(safe)
    return cf


def recommend_content_only(
    user_id: int,
    anime_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    model_cache: dict,
    top_n: int = 10,
    like_threshold: float = 7.0,
):
    id2idx = model_cache["id2idx"]
    item_matrix = model_cache["item_matrix"]

    seen_set = set(ratings_df[ratings_df["user_id"] == user_id]["anime_id"].tolist())

    user_vec = compute_user_profile_vector(user_id, ratings_df, like_threshold, id2idx, item_matrix)
    if not user_vec.any():
        pop_cols = [c for c in ["Members", "Score"] if c in anime_df.columns]
        ranked = anime_df.sort_values(by=pop_cols, ascending=False) if pop_cols else anime_df
        ranked = ranked[~ranked["MAL_ID"].isin(seen_set)]
        return ranked.head(top_n), "cold-start"

    sims = cosine_scores(user_vec, item_matrix)
    ranked = anime_df.copy()
    ranked["__sim"] = sims
    recs = ranked[~ranked["MAL_ID"].isin(seen_set)].sort_values("__sim", ascending=False).head(top_n)
    return recs, "content"


def recommend_hybrid(
    user_id: int,
    anime_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    model_cache: dict,
    cf_model: CFRecommender,
    candidate_size: int = 200,
    top_n: int = 10,
    like_threshold: float = 7.0,
):
    id2idx = model_cache["id2idx"]
    item_matrix = model_cache["item_matrix"]
    item_ids = anime_df["MAL_ID"].tolist()
    seen_set = set(ratings_df[ratings_df["user_id"] == user_id]["anime_id"].tolist())

    candidates = [aid for aid in item_ids if aid not in seen_set]
    scores = []
    for aid in candidates:
        try:
            est = cf_model.predict(user_id, aid)
        except Exception:
            est = 0.0
        scores.append((aid, est))

    scores.sort(key=lambda x: x[1], reverse=True)
    cf_candidates = [aid for aid, _ in scores[:candidate_size]]
    if not cf_candidates:
        return recommend_content_only(user_id, anime_df, ratings_df, model_cache, top_n, like_threshold)

    user_vec = compute_user_profile_vector(user_id, ratings_df, like_threshold, id2idx, item_matrix)
    if not user_vec.any():
        recs = anime_df[anime_df["MAL_ID"].isin(cf_candidates)].copy()
        pop_cols = [c for c in ["Members", "Score"] if c in recs.columns]
        recs = recs.sort_values(by=pop_cols, ascending=False) if pop_cols else recs
        return recs.head(top_n), "cf"

    sims = cosine_scores(user_vec, item_matrix)
    sid_sims = [(aid, sims[id2idx[aid]]) for aid in cf_candidates if aid in id2idx]
    sid_sims.sort(key=lambda x: x[1], reverse=True)
    top_ids = [aid for aid, _ in sid_sims[:top_n]]

    recs = anime_df[anime_df["MAL_ID"].isin(top_ids)].copy()
    recs["__order"] = recs["MAL_ID"].apply(lambda x: top_ids.index(x) if x in top_ids else 9999)
    recs = recs.sort_values("__order").drop(columns="__order", errors="ignore")
    return recs, "hybrid"


def show_error(message: str, exc: Exception):
    st.error(message)
    with st.expander("Technical details"):
        st.exception(exc)


st.set_page_config(page_title="Anime Recommender - Hybrid (Two-stage) Demo", layout="wide")

st.title("Anime Recommender - Hybrid (Two-stage) Demo")
st.caption("Place the Kaggle dataset under ./data and get Top-N recommendations.")

with st.sidebar:
    st.header("Configuration")
    anime_path = st.text_input("anime.csv path", value="data/anime.csv")
    synopsis_path = st.text_input("anime_with_synopsis.csv path", value="data/anime_with_synopsis.csv")
    ratings_path = st.text_input("rating_complete.csv path", value="data/rating_complete.csv")
    mode = st.radio("Mode", ["Hybrid", "Content-only"], index=0)
    like_threshold = st.slider("Like threshold (rating >=", 1, 10, 7)
    candidate_size = st.slider("CF candidate size (M)", 50, 500, 200, step=50)
    top_n = st.slider("Top-N", 5, 20, 10)
    fast_cf = st.checkbox("Fast demo CF training (sample ratings)", value=True)
    cf_sample_size = st.number_input("CF sample size (rows)", min_value=10000, value=200000, step=10000)
    build_btn = st.button("Build Indexes / Fit Models", type="primary")

data_ok = True
try:
    anime_df, syn_df, ratings_df = load_datasets(anime_path, synopsis_path, ratings_path)
except Exception as exc:
    data_ok = False
    show_error(
        "Could not load one or more dataset files. The app will run in toy-demo mode.",
        exc,
    )

if data_ok:
    try:
        anime_df = merge_synopsis_robust(anime_df, syn_df)
    except Exception as exc:
        data_ok = False
        show_error("Failed while merging anime metadata with synopsis data.", exc)

if data_ok:
    if build_btn:
        try:
            _ = fit_content_model(anime_df)
            st.success("Content model is ready (TF-IDF + SVD + numeric features).")
        except Exception as exc:
            show_error("Content model training failed.", exc)

    try:
        all_user_ids = sorted(ratings_df["user_id"].dropna().unique().tolist())
        default_uid = int(all_user_ids[0]) if all_user_ids else 1
        min_uid = int(min(all_user_ids)) if all_user_ids else 1
    except Exception:
        all_user_ids = [1]
        default_uid = 1
        min_uid = 1

    user_id = st.number_input("User ID", value=default_uid, min_value=min_uid, step=1)

    cf_train_df = sample_ratings_for_cf(ratings_df, fast_cf, int(cf_sample_size))
    if mode == "Hybrid":
        st.caption(
            f"CF training rows: {len(cf_train_df):,} / {len(ratings_df):,}"
            if fast_cf
            else f"CF training rows: {len(ratings_df):,} (full dataset)"
        )

    if mode == "Hybrid" and build_btn:
        try:
            _ = fit_cf_model(cf_train_df)
            st.success("Collaborative filtering model is ready (item-item KNN).")
        except Exception as exc:
            show_error("Collaborative filtering model training failed.", exc)

    if st.button("Recommend Top-N", type="primary"):
        try:
            t0 = time.time()
            cache = fit_content_model(anime_df)
            if mode == "Hybrid":
                cf = fit_cf_model(cf_train_df)
                recs, used = recommend_hybrid(
                    user_id,
                    anime_df,
                    ratings_df,
                    cache,
                    cf,
                    candidate_size=candidate_size,
                    top_n=top_n,
                    like_threshold=like_threshold,
                )
            else:
                recs, used = recommend_content_only(
                    user_id,
                    anime_df,
                    ratings_df,
                    cache,
                    top_n=top_n,
                    like_threshold=like_threshold,
                )
            dt_ms = (time.time() - t0) * 1000.0

            st.subheader(f"Top-{top_n} Recommendations")
            st.caption(f"Strategy: {used} | Latency: ~{dt_ms:.1f} ms")

            liked_genres = compute_liked_genre_counts(
                user_id, ratings_df, anime_df, threshold=like_threshold
            )

            display_cols = [
                c
                for c in ["MAL_ID", "Name", "Type", "Episodes", "Genres", "Score", "Members"]
                if c in recs.columns
            ]
            recs = recs.copy()
            recs["Reason"] = recs.apply(lambda r: format_reason(r, liked_genres), axis=1)
            st.dataframe(recs[display_cols + ["Reason"]].reset_index(drop=True), use_container_width=True)
        except Exception as exc:
            show_error("Failed to generate recommendations.", exc)
else:
    st.info(
        "Toy demo mode: using a tiny synthetic dataset. "
        "For full functionality, provide Kaggle dataset CSV files under ./data."
    )
    toy_anime = pd.DataFrame(
        {
            "MAL_ID": [1, 2, 3, 4, 5],
            "Name": ["A", "B", "C", "D", "E"],
            "Genres": ["Action, Adventure", "Romance", "Action, Sci-Fi", "Slice of Life", "Comedy"],
            "Type": ["TV", "TV", "Movie", "TV", "TV"],
            "Episodes": [12, 12, 1, 24, 12],
            "Score": [8.1, 7.9, 8.3, 7.0, 7.4],
            "Members": [100000, 85000, 120000, 50000, 60000],
            "synopsis": [
                "A hero fights monsters to save the city.",
                "Two students fall in love at school.",
                "A space battle between fleets.",
                "Daily life of a small town.",
                "A group of friends in funny situations.",
            ],
        }
    )
    toy_ratings = pd.DataFrame(
        {
            "user_id": [10, 10, 10, 20, 20],
            "anime_id": [1, 2, 5, 2, 3],
            "rating": [9, 8, 7, 8, 9],
        }
    )

    cache = fit_content_model(toy_anime)
    user_id = st.number_input("User ID (toy)", value=10, min_value=1, step=1)
    if st.button("Recommend (toy)"):
        recs, used = recommend_content_only(user_id, toy_anime, toy_ratings, cache)
        st.caption(f"Strategy: {used}")
        st.dataframe(recs[["MAL_ID", "Name", "Genres", "Score", "Members"]])
