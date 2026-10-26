"""Run the WHOLE analysis once: EDA -> features -> clustering -> recommender -> evaluation.
Saves every figure and table into the outputs/ folder (used by the paper and the dashboard).
Usage:  python run_pipeline.py
"""
import json
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

from config import FIG_DIR, TABLE_DIR, OUT_DIR, RANDOM_STATE
from data_prep import load_data, build_learner_features
from segmentation import (build_matrix, evaluate_k, fit_kmeans, validate_hierarchical,
                          intra_cluster_similarity, profile_clusters, name_clusters)
from recommender import HybridRecommender
from evaluation import run_recommendation_evaluation

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="viridis")
K_FINAL = 5
for d in (FIG_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)


def save(name):
    plt.tight_layout(); plt.savefig(FIG_DIR / f"{name}.png", dpi=150); plt.close()


# ------------------------------------------------------------------ 1. load
users, courses, txns = load_data()
feats = build_learner_features(users, courses, txns)
print(f"Learners: {len(users)} | Courses: {len(courses)} | Transactions: {len(txns)}")

# ------------------------------------------------------------------ 2. EDA
d = txns.merge(courses, on="CourseID")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(users["Age"], bins=20, ax=ax[0]); ax[0].set_title("Learner age distribution")
users["Gender"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax[1], ylabel=""); ax[1].set_title("Gender split")
save("01_age_gender")

plt.figure(figsize=(9, 5))
d["CourseCategory"].value_counts().sort_values().plot.barh(color="#2a788e")
plt.title("Enrolments by course category"); plt.xlabel("Enrolments"); save("02_category_enrolments")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
d["CourseLevel"].value_counts().plot.bar(ax=ax[0], color="#2a788e"); ax[0].set_title("Enrolments by level")
d["CourseType"].value_counts().plot.bar(ax=ax[1], color="#7ad151"); ax[1].set_title("Free vs paid enrolments")
sns.histplot(courses["CourseRating"], bins=15, ax=ax[2]); ax[2].set_title("Course rating distribution")
for a in ax: a.tick_params(axis="x", rotation=0)
save("03_level_type_rating")

plt.figure(figsize=(8, 4))
sns.countplot(x=feats["total_courses"].clip(upper=12), color="#2a788e")
plt.title("Courses per learner (12 = 12 or more)"); plt.xlabel("Courses enrolled"); save("04_courses_per_learner")

plt.figure(figsize=(9, 4))
txns.set_index("TransactionDate").resample("MS").size().plot(marker="o")
plt.title("Monthly enrolments in 2025"); plt.ylabel("Enrolments"); save("05_monthly_trend")

num = ["total_courses", "n_categories", "avg_courses_per_category", "enrollment_frequency", "avg_rating_enrolled",
       "avg_spend", "paid_ratio", "learning_depth_index", "Age"]
plt.figure(figsize=(9, 7)); sns.heatmap(feats[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation of learner features"); save("06_correlation")

# ------------------------------------------------------------------ 3. clustering
X, names = build_matrix(feats)
kdf = evaluate_k(X)
kdf.to_csv(TABLE_DIR / "k_selection.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(kdf.k, kdf.inertia, "o-"); ax[0].axvline(K_FINAL, ls="--", c="red"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
ax[1].plot(kdf.k, kdf.silhouette, "o-g"); ax[1].axvline(K_FINAL, ls="--", c="red"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
save("07_elbow_silhouette")

km = fit_kmeans(X, K_FINAL)
labels = km.labels_
hier_stats, hier_labels = validate_hierarchical(X, labels, K_FINAL)
icsim = intra_cluster_similarity(X, labels)

prof = profile_clusters(feats, labels)
seg_names = name_clusters(prof)
prof["segment"] = prof.index.map(seg_names)
prof["intra_cluster_similarity"] = prof.index.map(icsim).astype(float).round(3)
prof.to_csv(TABLE_DIR / "cluster_profiles.csv")
feats["cluster"] = labels
feats["segment"] = feats["cluster"].map(seg_names)
feats.to_csv(TABLE_DIR / "learner_segments.csv", index=False)
print(prof[["segment", "learners", "avg_courses", "avg_spend", "avg_rating", "depth_index", "paid_ratio"]].to_string())

# PCA scatter, dendrogram, profile heatmap
pcs = PCA(2, random_state=RANDOM_STATE).fit(X); P = pcs.transform(X)
plt.figure(figsize=(9, 6))
sns.scatterplot(x=P[:, 0], y=P[:, 1], hue=feats["segment"], s=14, palette="tab10", alpha=.7)
plt.title(f"Learner segments (PCA, {pcs.explained_variance_ratio_.sum():.0%} variance explained)")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(fontsize=8); save("08_pca_clusters")

samp = np.random.default_rng(RANDOM_STATE).choice(len(X), 250, replace=False)
plt.figure(figsize=(11, 4)); dendrogram(linkage(X[samp], "ward"), no_labels=True, color_threshold=0)
plt.title("Hierarchical clustering dendrogram (random sample of 250 learners)"); save("09_dendrogram")

hm = prof.set_index("segment")[["avg_courses", "avg_categories", "avg_spend", "avg_rating", "depth_index", "paid_ratio", "enroll_freq"]]
plt.figure(figsize=(10, 4.5)); sns.heatmap((hm - hm.mean()) / hm.std(), annot=hm.round(2), fmt="", cmap="RdYlGn", center=0, cbar_kws={"label": "z-score"})
plt.title("Segment profile (values shown, colour = relative to other segments)"); plt.ylabel(""); save("10_segment_heatmap")

# ------------------------------------------------------------------ 4. recommender + evaluation
rec = HybridRecommender().fit(courses, txns, feats, labels, X)
res, lift = run_recommendation_evaluation(users, courses, txns, K_FINAL)
res.to_csv(TABLE_DIR / "recommendation_evaluation.csv", index=False)
print(res[["Model", "HitRate@5", "Precision@5", "CategoryPrecision@5", "HitRate@10"]].to_string())

plt.figure(figsize=(9, 4.5))
r5 = res.set_index("Model")["HitRate@5"]; r5.plot.barh(color="#2a788e"); plt.xlabel("Hit rate @5")
plt.title("Recommendation quality: hold-out test (last enrolment)"); save("11_recommendation_eval")

# engagement index per segment (proxy for impact)
eng = prof.set_index("segment")["avg_courses"] / feats["total_courses"].mean()
eng.sort_values().plot.barh(color="#7ad151", figsize=(8, 4)); plt.axvline(1, c="k", ls="--")
plt.title("Engagement index by segment (1.0 = platform average)"); save("12_engagement_index")

metrics = dict(k_final=K_FINAL, silhouette_kmeans=float(hier_stats["kmeans_silhouette"]),
               silhouette_hierarchical=float(hier_stats["hier_silhouette"]),
               adjusted_rand_index=float(hier_stats["adjusted_rand_index"]),
               davies_bouldin=float(kdf.loc[kdf.k == K_FINAL, "davies_bouldin"].iloc[0]),
               mean_intra_cluster_similarity=float(np.mean(list(icsim.values()))),
               engagement_index=eng.round(2).to_dict(),
               **{k: float(v) for k, v in lift.items()})
json.dump(metrics, open(OUT_DIR / "metrics.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))
print("\nDone. See the outputs/ folder.")