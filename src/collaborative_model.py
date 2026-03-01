from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

class CFRecommender:
    def __init__(self, k=40):
        self.sim_options = {'name': 'cosine', 'user_based': False, 'min_support': 3}
        self.algo = KNNBasic(k=k, sim_options=self.sim_options)

    def fit(self, ratings_df):
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(ratings_df[['user_id', 'anime_id', 'rating']], reader)
        trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
        self.algo.fit(trainset)

    def predict(self, user_id, item_id):
        return self.algo.predict(user_id, item_id).est
