"""Step 5: cluster-aware hybrid recommender.

score(user, course) = w_content * content match
                    + w_cluster_pop * popularity of the course inside the learner's segment
                    + w_similar * enrolment by the most similar learners of the same segment
                    + w_rating * normalised course rating
All four components are scaled to 0-1 before they are combined.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from config import REC_WEIGHTS, LEVEL_SCORE

LEVELS = list(LEVEL_SCORE)


def _minmax_rows(a):
    mx = a.max(axis=1, keepdims=True)
    return np.divide(a, mx, out=np.zeros_like(a, dtype=float), where=mx > 0)


class HybridRecommender:
    def __init__(self, weights=None, n_neighbors=30):
        self.w = weights or REC_WEIGHTS
        self.n_neighbors = n_neighbors

    # ------------------------------------------------------------------ fit
    def fit(self, courses, txns, features, labels, X):
        self.courses = courses.reset_index(drop=True)
        self.users = features["UserID"].tolist()
        self.uidx = {u: i for i, u in enumerate(self.users)}
        self.cidx = {c: j for j, c in enumerate(self.courses["CourseID"])}
        self.labels = np.asarray(labels)
        nU, nC = len(self.users), len(self.courses)

        # binary user x course matrix
        R = np.zeros((nU, nC))
        for u, c in zip(txns["UserID"], txns["CourseID"]):
            if u in self.uidx and c in self.cidx:
                R[self.uidx[u], self.cidx[c]] = 1
        self.R = R

        # 1) content-based: learner's category share & level share vs course attributes
        cats = sorted(self.courses["CourseCategory"].unique())
        C_cat = np.array([[1.0 if c == k else 0 for k in cats] for c in self.courses["CourseCategory"]])
        C_lvl = np.array([[1.0 if c == k else 0 for k in LEVELS] for c in self.courses["CourseLevel"]])
        U_cat = R @ C_cat; U_lvl = R @ C_lvl
        U_cat = U_cat / np.maximum(U_cat.sum(1, keepdims=True), 1)
        U_lvl = U_lvl / np.maximum(U_lvl.sum(1, keepdims=True), 1)
        self.S_content = 0.7 * (U_cat @ C_cat.T) + 0.3 * (U_lvl @ C_lvl.T)

        # 2) popularity inside each segment
        self.n_clusters = int(self.labels.max()) + 1
        pop = np.zeros((self.n_clusters, nC))
        for c in range(self.n_clusters):
            pop[c] = R[self.labels == c].mean(axis=0)
        self.cluster_pop = _minmax_rows(pop)
        self.S_pop = self.cluster_pop[self.labels]

        # 3) similar learners: nearest neighbours (behavioural profile space X) inside the same segment
        sim = cosine_similarity(X)
        np.fill_diagonal(sim, 0)
        sim[self.labels[:, None] != self.labels[None, :]] = 0         # same segment only
        k = min(self.n_neighbors, nU - 1)
        thr = -np.partition(-sim, k - 1, axis=1)[:, k - 1][:, None]
        sim[sim < thr] = 0
        sim = np.clip(sim, 0, None)
        self.S_similar = _minmax_rows(sim @ R)

        # 4) rating-weighted relevance
        self.S_rating = (self.courses["CourseRating"].values / 5.0)[None, :].repeat(nU, axis=0)

        self.S = (self.w["content"] * self.S_content + self.w["cluster_pop"] * self.S_pop +
                  self.w["similar"] * self.S_similar + self.w["rating"] * self.S_rating)
        return self

    # ------------------------------------------------------------ recommend
    def recommend(self, user_id, n=5, level=None, category=None, scores=None, exclude_seen=True):
        i = self.uidx[user_id]
        s = (self.S if scores is None else scores)[i].copy()
        if exclude_seen:
            s[self.R[i] == 1] = -np.inf
        out = self.courses.copy()
        out["score"] = s
        out["why"] = [self._why(i, j) for j in range(len(out))]
        out = out[np.isfinite(out["score"])]
        if level:
            out = out[out["CourseLevel"].isin(level)]
        if category:
            out = out[out["CourseCategory"].isin(category)]
        return out.sort_values("score", ascending=False).head(n)

    def _why(self, i, j):
        parts = {"matches your interests": self.w["content"] * self.S_content[i, j],
                 "popular in your segment": self.w["cluster_pop"] * self.S_pop[i, j],
                 "liked by similar learners": self.w["similar"] * self.S_similar[i, j],
                 "highly rated": self.w["rating"] * self.S_rating[i, j]}
        top = sorted(parts, key=parts.get, reverse=True)[:2]
        return " + ".join(top)

    # variants used for ablation / baselines
    def component_scores(self):
        return {"Content only": self.S_content, "Segment popularity only": self.S_pop,
                "Similar learners only": self.S_similar, "Rating only": self.S_rating,
                "Hybrid (final)": self.S}