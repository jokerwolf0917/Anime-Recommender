from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class ContentRecommender:
    def __init__(self, n_components=200):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.6)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.scaler = MinMaxScaler()
        self.item_matrix = None
        self.item_ids = None

    def fit(self, items, numeric_features):
        tfidf = self.vectorizer.fit_transform(items)
        reduced = self.svd.fit_transform(tfidf)
        scaled = self.scaler.fit_transform(numeric_features)
        self.item_matrix = np.hstack([reduced, scaled])

    def recommend(self, user_vector, top_n=10):
        sims = self.item_matrix @ user_vector
        top_idx = np.argsort(sims)[::-1][:top_n]
        return top_idx
