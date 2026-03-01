class HybridRecommender:
    def __init__(self, cf_model, content_model, candidate_size=200):
        self.cf = cf_model
        self.content = content_model
        self.candidate_size = candidate_size

    def recommend(self, user_id, seen_items, user_vector, all_items):
        # Step 1: CF candidate generation (mock example)
        cf_candidates = [item for item in all_items if item not in seen_items]
        cf_scores = [(item, self.cf.predict(user_id, item)) for item in cf_candidates]
        cf_sorted = sorted(cf_scores, key=lambda x: x[1], reverse=True)[:self.candidate_size]

        # Step 2: Content-based re-ranking
        item_indices = [c[0] for c in cf_sorted]
        re_ranked = self.content.recommend(user_vector, top_n=len(item_indices))

        return re_ranked[:10]
