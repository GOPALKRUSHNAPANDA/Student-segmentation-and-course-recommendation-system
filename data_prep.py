"""Step 1-2: load data, clean it, and engineer learner-level features."""
import numpy as np
import pandas as pd
from config import DATA_FILE, LEVEL_SCORE


def load_data(path=DATA_FILE):
    """Read the three sheets we need and run basic quality checks."""
    sheets = pd.read_excel(path, sheet_name=None)
    users, courses, txns = sheets["Users"], sheets["Courses"], sheets["Transactions"]

    txns["TransactionDate"] = pd.to_datetime(txns["TransactionDate"])
    users = users.drop_duplicates("UserID")
    courses = courses.drop_duplicates("CourseID")
    txns = txns.drop_duplicates(["UserID", "CourseID"])            # one enrolment per user-course
    txns = txns[txns["UserID"].isin(users["UserID"]) & txns["CourseID"].isin(courses["CourseID"])]
    return users, courses, txns


def build_learner_features(users, courses, txns):
    """Aggregate transactions to ONE ROW PER LEARNER and engineer all required features."""
    df = txns.merge(courses[["CourseID", "CourseCategory", "CourseLevel", "CourseType", "CourseRating"]],
                    on="CourseID", how="left")
    df["level_score"] = df["CourseLevel"].map(LEVEL_SCORE)

    g = df.groupby("UserID")

    # ---- Engagement features
    f = pd.DataFrame({"total_courses": g.size()})
    f["n_categories"] = g["CourseCategory"].nunique()                    # diversity score
    f["avg_courses_per_category"] = f["total_courses"] / f["n_categories"]
    span_months = ((g["TransactionDate"].max() - g["TransactionDate"].min()).dt.days / 30.0) + 1
    f["enrollment_frequency"] = f["total_courses"] / span_months         # courses per active month

    # ---- Preference features
    f["preferred_category"] = g["CourseCategory"].agg(lambda s: s.value_counts().index[0])
    f["preferred_level"] = g["CourseLevel"].agg(lambda s: s.value_counts().index[0])
    f["avg_rating_enrolled"] = g["CourseRating"].mean()

    # ---- Behavioural features
    f["avg_spend"] = g["Amount"].mean()
    f["total_spend"] = g["Amount"].sum()
    f["paid_ratio"] = g["Amount"].apply(lambda s: (s > 0).mean())
    lvl = df.pivot_table(index="UserID", columns="CourseLevel", values="CourseID",
                         aggfunc="count", fill_value=0).reindex(columns=list(LEVEL_SCORE), fill_value=0)
    f["n_beginner"], f["n_intermediate"], f["n_advanced"] = lvl["Beginner"], lvl["Intermediate"], lvl["Advanced"]
    # Learning-depth index: advanced vs beginner (smoothed so it is defined for everybody) -> 0..1
    f["learning_depth_index"] = (f["n_advanced"] + 0.5 * f["n_intermediate"]) / f["total_courses"]
    f["adv_to_beg_ratio"] = (f["n_advanced"] + 1) / (f["n_beginner"] + 1)

    f = f.reset_index().merge(users[["UserID", "Age", "Gender"]], on="UserID", how="left")
    return f


def category_share_matrix(txns, courses, user_ids):
    """Share of each learner's enrolments per category (rows sum to 1). Used by the recommender."""
    d = txns.merge(courses[["CourseID", "CourseCategory"]], on="CourseID")
    m = pd.crosstab(d["UserID"], d["CourseCategory"]).reindex(user_ids, fill_value=0)
    return m.div(m.sum(axis=1).replace(0, 1), axis=0)