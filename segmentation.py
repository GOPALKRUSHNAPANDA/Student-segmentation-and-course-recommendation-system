"""Step 3-4: preprocessing, K-Means, hierarchical validation, cluster selection and profiling."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from config import RANDOM_STATE, K_RANGE

# log-transform skewed counts/money -> reduces noise from heavy users and sparse enrolments
LOG_COLS = ["total_courses", "n_categories", "avg_spend", "enrollment_frequency"]
NUM_COLS = ["avg_courses_per_category", "avg_rating_enrolled", "learning_depth_index", "paid_ratio", "Age"]
CAT_COLS = ["preferred_category", "preferred_level", "Gender"]


def build_matrix(features, cat_weight=0.5):
    """Scale numeric features and one-hot encode categoricals. Returns (X, feature_names)."""
    pre = ColumnTransformer([
        ("log", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("sc", StandardScaler())]), LOG_COLS),
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])
    X = pre.fit_transform(features)
    names = list(pre.get_feature_names_out())
    # down-weight one-hot columns so 12 category dummies do not dominate the distance
    idx = [i for i, n in enumerate(names) if n.startswith("cat__")]
    X[:, idx] *= cat_weight
    return X, names


def evaluate_k(X, k_range=K_RANGE):
    """Elbow (inertia) + Silhouette + Davies-Bouldin + Calinski-Harabasz for each k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
        rows.append(dict(k=k, inertia=km.inertia_,
                         silhouette=silhouette_score(X, km.labels_),
                         davies_bouldin=davies_bouldin_score(X, km.labels_),
                         calinski_harabasz=calinski_harabasz_score(X, km.labels_)))
    return pd.DataFrame(rows)


def fit_kmeans(X, k):
    return KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit(X)


def fit_hierarchical(X, k):
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)


def validate_hierarchical(X, km_labels, k):
    """Compare Ward hierarchical clustering with K-Means (agreement = Adjusted Rand Index)."""
    hc = fit_hierarchical(X, k)
    return dict(hier_silhouette=silhouette_score(X, hc.labels_),
                kmeans_silhouette=silhouette_score(X, km_labels),
                adjusted_rand_index=adjusted_rand_score(km_labels, hc.labels_)), hc.labels_


def intra_cluster_similarity(X, labels):
    """Mean cosine similarity of members to their cluster centroid (higher = more consistent)."""
    out = {}
    for c in np.unique(labels):
        m = X[labels == c]
        out[int(c)] = float(cosine_similarity(m, m.mean(axis=0, keepdims=True)).mean())
    return out


def profile_clusters(features, labels):
    """Average behaviour per cluster + dominant category / level."""
    f = features.copy(); f["cluster"] = labels
    prof = f.groupby("cluster").agg(
        learners=("UserID", "count"), avg_courses=("total_courses", "mean"),
        avg_categories=("n_categories", "mean"), avg_spend=("avg_spend", "mean"),
        avg_rating=("avg_rating_enrolled", "mean"), depth_index=("learning_depth_index", "mean"),
        paid_ratio=("paid_ratio", "mean"), avg_age=("Age", "mean"),
        enroll_freq=("enrollment_frequency", "mean"))
    prof["share_%"] = 100 * prof["learners"] / prof["learners"].sum()
    prof["top_category"] = f.groupby("cluster")["preferred_category"].agg(lambda s: s.value_counts().index[0])
    prof["top_level"] = f.groupby("cluster")["preferred_level"].agg(lambda s: s.value_counts().index[0])
    return prof.round(3)


def name_clusters(prof):
    """Give each cluster a business-friendly name using simple, explainable rules."""
    names = {}
    for c, r in prof.iterrows():
        if r["avg_courses"] >= 8:
            n = "Multi-Domain Explorers"
        elif r["paid_ratio"] >= 0.8:
            n = "Premium Career Investors"
        elif r["avg_courses"] >= 1.5:
            n = "Steady Multi-Course Learners"
        elif r["depth_index"] >= 0.65:
            n = "Advanced Free Specialists"
        elif r["depth_index"] <= 0.40:
            n = "Beginner Free Starters"
        else:
            n = "Casual Learners"
        names[c] = n
    seen = {}                                   # keep names unique
    for c, n in list(names.items()):
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[c] = f"{n} ({prof.loc[c, 'top_category']})"
    return names