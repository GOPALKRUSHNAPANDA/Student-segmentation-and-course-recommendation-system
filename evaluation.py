"""Step 6: evaluation - clustering quality + recommendation quality (leave-last-out)."""
import numpy as np
import pandas as pd
from data_prep import build_learner_features
from segmentation import build_matrix, fit_kmeans
from recommender import HybridRecommender
from config import RANDOM_STATE


def leave_last_out_split(txns, min_courses=2):
    """Hold out each learner's most recent enrolment (learners with >= min_courses only)."""
    t = txns.sort_values(["UserID", "TransactionDate", "TransactionID"])
    counts = t.groupby("UserID")["CourseID"].transform("size")
    last_idx = t[counts >= min_courses].groupby("UserID").tail(1).index
    test = t.loc[last_idx]
    train = t.drop(index=last_idx)
    return train, test


def precision_hit_at_k(scores, rec, test, k=5):
    """Precision@K, HitRate@K and category-precision@K for a given score matrix."""
    hits, prec, cat_prec = [], [], []
    cat_of = rec.courses["CourseCategory"].values
    for u, c in zip(test["UserID"], test["CourseID"]):
        i, j = rec.uidx[u], rec.cidx[c]
        s = scores[i].copy(); s[rec.R[i] == 1] = -np.inf
        top = np.argsort(-s)[:k]
        hit = int(j in top)
        hits.append(hit); prec.append(hit / k)
        cat_prec.append(np.mean(cat_of[top] == cat_of[j]))
    return dict(**{f"HitRate@{k}": np.mean(hits), f"Precision@{k}": np.mean(prec),
                   f"CategoryPrecision@{k}": np.mean(cat_prec)})


def run_recommendation_evaluation(users, courses, txns, k_final, ks=(3, 5, 10)):
    train, test = leave_last_out_split(txns)
    feats = build_learner_features(users, courses, train)
    X, _ = build_matrix(feats)
    km = fit_kmeans(X, k_final)
    rec = HybridRecommender().fit(courses, train, feats, km.labels_, X)

    comps = rec.component_scores()
    rng = np.random.default_rng(RANDOM_STATE)
    comps = {"Random": rng.random(rec.S.shape),
             "Global popularity (baseline)": np.tile(rec.R.mean(0), (len(rec.users), 1)), **comps}
    rows = []
    for name, sc in comps.items():
        r = {"Model": name}
        for k in ks:
            m = precision_hit_at_k(sc, rec, test, k)
            r.update({kk: round(v, 4) for kk, v in m.items()})
        rows.append(r)
    res = pd.DataFrame(rows)

    base = res.loc[res.Model == "Global popularity (baseline)", "HitRate@5"].iloc[0]
    hyb = res.loc[res.Model == "Hybrid (final)", "HitRate@5"].iloc[0]
    lift = dict(baseline_hit_rate=base, hybrid_hit_rate=hyb,
                relative_lift_pct=100 * (hyb - base) / base if base else np.nan,
                n_test_learners=len(test))
    return res, lift