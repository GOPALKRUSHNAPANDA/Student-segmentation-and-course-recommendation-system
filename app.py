"""EduPro - Learner Intelligence Platform (Streamlit dashboard).
Run with:  streamlit run app.py
"""
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from config import RANDOM_STATE, LEVEL_SCORE, OUT_DIR, TABLE_DIR
from data_prep import load_data, build_learner_features
from segmentation import build_matrix, fit_kmeans, profile_clusters, name_clusters, evaluate_k
from recommender import HybridRecommender

K_FINAL = 5
LEVEL_ORDER = list(LEVEL_SCORE)
st.set_page_config(page_title="EduPro Learner Intelligence", page_icon="🎓", layout="wide")

# ------------------------------------------------------------------ look & feel
SEG_STYLE = {   # colour, emoji, recommended action
    "Multi-Domain Explorers": ("#6C5CE7", "🧭", "Offer learning bundles and cross-domain pathways; reward with loyalty badges."),
    "Premium Career Investors": ("#E17055", "💎", "Promote certification tracks and career-outcome content; upsell advanced paid courses."),
    "Steady Multi-Course Learners": ("#00B894", "📈", "Send timely nudges to raise enrolment frequency; suggest the next course in their category."),
    "Advanced Free Specialists": ("#0984E3", "🎓", "Highlight higher-quality advanced content; convert to paid with expert-led courses."),
    "Beginner Free Starters": ("#FDCB6E", "🌱", "Guide with starter paths and a free-to-paid upgrade offer after the first course."),
}
LEVEL_COLOR = {"Beginner": "#00B894", "Intermediate": "#FDCB6E", "Advanced": "#E17055"}


def seg_info(name):
    base = name.split(" (")[0]
    return SEG_STYLE.get(base, ("#636E72", "👤", "Monitor and personalise gradually."))


st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}
.block-container {padding-top:1.4rem; padding-bottom:3rem; max-width:1300px;}
.hero {background:linear-gradient(120deg,#6C5CE7 0%,#0984E3 55%,#00CEC9 100%); color:white; padding:28px 34px;
       border-radius:20px; margin-bottom:22px; box-shadow:0 10px 30px rgba(108,92,231,.25);}
.hero h1 {color:white; margin:0; font-size:2rem; font-weight:800;}
.hero p {margin:6px 0 0 0; opacity:.92; font-size:1.02rem;}
.kpi {background:white; border-radius:16px; padding:18px 20px; box-shadow:0 4px 18px rgba(31,36,48,.07);
      border-top:4px solid var(--c); height:100%;}
.kpi .v {font-size:1.9rem; font-weight:800; color:#1F2430; line-height:1.1;}
.kpi .l {font-size:.82rem; color:#6B7280; text-transform:uppercase; letter-spacing:.06em; margin-top:4px;}
.segcard {background:white; border-radius:16px; padding:16px 18px; box-shadow:0 4px 18px rgba(31,36,48,.07);
          border-left:6px solid var(--c); margin-bottom:12px; height:100%;}
.segcard .t {font-weight:800; font-size:1.05rem; color:#1F2430;}
.segcard .m {color:#6B7280; font-size:.85rem; margin:3px 0 7px 0;}
.segcard .a {font-size:.86rem; color:#374151;}
.course {background:white; border-radius:16px; padding:16px 20px; margin-bottom:12px;
         box-shadow:0 4px 18px rgba(31,36,48,.07); border-left:6px solid var(--c);}
.course .n {font-weight:800; font-size:1.08rem; color:#1F2430;}
.course .meta {color:#6B7280; font-size:.85rem; margin:4px 0;}
.chip {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:700; color:white; margin-right:6px;}
.bar {height:8px; border-radius:6px; background:#EEF0F6; margin-top:8px; overflow:hidden;}
.bar > div {height:100%; border-radius:6px; background:linear-gradient(90deg,#6C5CE7,#00CEC9);}
.why {font-size:.8rem; color:#6C5CE7; font-weight:600; margin-top:6px;}
.path {display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:6px 0 4px 0;}
.step {background:white; border-radius:12px; padding:8px 14px; box-shadow:0 3px 12px rgba(31,36,48,.08);
       font-size:.88rem; font-weight:600; border-bottom:4px solid var(--c);}
.arrow {color:#9CA3AF; font-size:1.2rem;}
.profile {background:white; border-radius:18px; padding:20px 24px; box-shadow:0 4px 18px rgba(31,36,48,.07);
          border-left:8px solid var(--c); margin-bottom:16px;}
.profile .t {font-size:1.35rem; font-weight:800;}
[data-testid="stSidebar"] {background:#1F2430;}
[data-testid="stSidebar"] * {color:#E5E7EB !important;}
.stTabs [data-baseweb="tab"] {font-weight:600;}
</style>
""", unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(col, value, label, color="#6C5CE7"):
    col.markdown(f'<div class="kpi" style="--c:{color}"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


def style_fig(fig, h=420):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="sans-serif", size=13), margin=dict(l=10, r=10, t=50, b=10),
                      title_font=dict(size=16))
    return fig


# ------------------------------------------------------------------ cached pipeline
@st.cache_resource(show_spinner="Building learner intelligence models...")
def load_everything():
    users, courses, txns = load_data()
    feats = build_learner_features(users, courses, txns)
    X, _ = build_matrix(feats)
    labels = fit_kmeans(X, K_FINAL).labels_
    prof = profile_clusters(feats, labels)
    names = name_clusters(prof)
    prof["segment"] = prof.index.map(names)
    feats["cluster"] = labels
    feats["segment"] = feats["cluster"].map(names)
    rec = HybridRecommender().fit(courses, txns, feats, labels, X)
    P = PCA(2, random_state=RANDOM_STATE).fit_transform(X)
    feats["PC1"], feats["PC2"] = P[:, 0], P[:, 1]
    return users, courses, txns, feats, prof, rec


@st.cache_data(show_spinner="Computing k-selection curves...")
def k_table(_feats):
    X, _ = build_matrix(_feats)
    return evaluate_k(X)


users, courses, txns, feats, prof, rec = load_everything()
COLOR_MAP = {s: seg_info(s)[0] for s in feats.segment.unique()}

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("## 🎓 EduPro")
st.sidebar.caption("Learner Intelligence Platform")
page = st.sidebar.radio("Navigate", ["🏠 Overview", "🔍 Learner Explorer", "🎯 Recommendations",
                                     "⚖️ Segment Comparison", "📊 Model Insights"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{len(users):,}** learners  \n**{len(courses)}** courses  \n**{len(txns):,}** enrolments")
st.sidebar.markdown("---")
st.sidebar.caption("K-Means • Ward hierarchical validation • Hybrid recommender")

# ================================================================== OVERVIEW
if page == "🏠 Overview":
    hero("Learner Segmentation Dashboard",
         "Five data-driven learner segments that power personalised course recommendations for EduPro.")
    c = st.columns(5)
    kpi(c[0], f"{len(feats):,}", "Total learners", "#6C5CE7")
    kpi(c[1], K_FINAL, "Learner segments", "#0984E3")
    kpi(c[2], f"{feats.total_courses.mean():.2f}", "Avg courses / learner", "#00B894")
    kpi(c[3], f"{(txns.Amount > 0).mean()*100:.0f}%", "Paid enrolments", "#E17055")
    kpi(c[4], f"{txns.Amount.sum():,.0f}", "Total revenue", "#FDCB6E")
    st.write("")

    l, r = st.columns([3, 2])
    fig = px.scatter(feats, x="PC1", y="PC2", color="segment", color_discrete_map=COLOR_MAP,
                     hover_data=["UserID", "total_courses", "avg_spend"], opacity=.75,
                     title="Learner map (PCA projection of behaviour)")
    fig.update_traces(marker=dict(size=6)); fig.update_layout(legend_title_text="")
    l.plotly_chart(style_fig(fig, 430), width="stretch")
    sz = prof.set_index("segment")["learners"].reset_index()
    fig = px.pie(sz, names="segment", values="learners", hole=.55, color="segment", color_discrete_map=COLOR_MAP,
                 title="Segment size")
    fig.update_layout(showlegend=False); fig.update_traces(textinfo="percent")
    r.plotly_chart(style_fig(fig, 430), width="stretch")

    st.subheader("Meet the segments")
    cols = st.columns(len(prof))
    for col, (_, p) in zip(cols, prof.sort_values("learners", ascending=False).iterrows()):
        color, emoji, action = seg_info(p.segment)
        col.markdown(f'<div class="segcard" style="--c:{color}"><div class="t">{emoji} {p.segment}</div>'
                     f'<div class="m">{int(p.learners):,} learners • {p["share_%"]:.0f}% • avg {p.avg_courses:.1f} courses</div>'
                     f'<div class="a"><b>Action:</b> {action}</div></div>', unsafe_allow_html=True)

    t1, t2 = st.tabs(["📋 Segment profile table", "🔥 Category interest by segment"])
    with t1:
        show = prof.set_index("segment")[["learners", "share_%", "avg_courses", "avg_categories", "avg_spend",
                                          "avg_rating", "depth_index", "paid_ratio", "top_category", "top_level"]]
        st.dataframe(show.style.format(precision=2).background_gradient(cmap="Purples", subset=["avg_courses", "avg_spend"]),
                     width="stretch")
    with t2:
        ct = pd.crosstab(feats.segment, feats.preferred_category, normalize="index") * 100
        fig = px.imshow(ct.round(1), text_auto=".0f", aspect="auto", color_continuous_scale="Purples",
                        title="% of each segment's learners by preferred category")
        st.plotly_chart(style_fig(fig, 380), width="stretch")

    a, b = st.columns(2)
    d = txns.merge(courses, on="CourseID")
    top = d.groupby("CourseName").size().sort_values().tail(10).reset_index(name="enrolments")
    a.plotly_chart(style_fig(px.bar(top, x="enrolments", y="CourseName", orientation="h", title="Top 10 most popular courses",
                                    color_discrete_sequence=["#6C5CE7"]), 400), width="stretch")
    mt = txns.set_index("TransactionDate").resample("MS").size().reset_index(name="enrolments")
    b.plotly_chart(style_fig(px.area(mt, x="TransactionDate", y="enrolments", title="Monthly enrolments (2025)",
                                     color_discrete_sequence=["#00B894"]), 400), width="stretch")

# ================================================================== LEARNER EXPLORER
elif page == "🔍 Learner Explorer":
    hero("Learner Profile Explorer", "Filter learners, open any profile and see their full learning behaviour.")
    f1, f2, f3, f4 = st.columns(4)
    seg_f = f1.multiselect("Segment", sorted(feats.segment.unique()))
    gen_f = f2.multiselect("Gender", sorted(feats.Gender.unique()))
    cat_f = f3.multiselect("Preferred category", sorted(feats.preferred_category.unique()))
    age_f = f4.slider("Age range", int(feats.Age.min()), int(feats.Age.max()), (int(feats.Age.min()), int(feats.Age.max())))
    d = feats[feats.Age.between(*age_f)]
    if seg_f: d = d[d.segment.isin(seg_f)]
    if gen_f: d = d[d.Gender.isin(gen_f)]
    if cat_f: d = d[d.preferred_category.isin(cat_f)]
    st.markdown(f"**{len(d):,}** learners match your filters")
    cols = ["UserID", "segment", "Age", "Gender", "total_courses", "n_categories", "preferred_category",
            "preferred_level", "avg_rating_enrolled", "avg_spend", "learning_depth_index"]
    st.dataframe(d[cols].round(2), width="stretch", height=280)
    st.download_button("⬇️ Download filtered learners (CSV)", d[cols].to_csv(index=False), "filtered_learners.csv")

    if len(d):
        uid = st.selectbox("Open a learner profile", d.UserID.tolist())
        r = feats[feats.UserID == uid].iloc[0]
        color, emoji, _ = seg_info(r.segment)
        st.markdown(f'<div class="profile" style="--c:{color}"><div class="t">{emoji} {uid} &nbsp;•&nbsp; {r.segment}</div>'
                    f'<div style="color:#6B7280">Age {int(r.Age)} • {r.Gender} • prefers <b>{r.preferred_category}</b> '
                    f'at <b>{r.preferred_level}</b> level</div></div>', unsafe_allow_html=True)
        k = st.columns(5)
        kpi(k[0], int(r.total_courses), "Courses enrolled", color)
        kpi(k[1], int(r.n_categories), "Categories explored", color)
        kpi(k[2], f"{r.avg_rating_enrolled:.2f}", "Avg rating enrolled", color)
        kpi(k[3], f"{r.avg_spend:.0f}", "Avg spend", color)
        kpi(k[4], f"{r.learning_depth_index:.2f}", "Depth index", color)
        st.write("")
        hist = txns[txns.UserID == uid].merge(courses, on="CourseID").sort_values("TransactionDate")
        a, b = st.columns([1, 1])
        mix = hist.groupby("CourseCategory").size().reset_index(name="courses")
        a.plotly_chart(style_fig(px.pie(mix, names="CourseCategory", values="courses", hole=.5,
                                        title="Categories this learner explored"), 340), width="stretch")
        lv = hist.groupby("CourseLevel").size().reindex(LEVEL_ORDER, fill_value=0).reset_index(name="courses")
        b.plotly_chart(style_fig(px.bar(lv, x="CourseLevel", y="courses", color="CourseLevel", color_discrete_map=LEVEL_COLOR,
                                        title="Level mix"), 340), width="stretch")
        st.markdown("**Enrolment history**")
        st.dataframe(hist[["TransactionDate", "CourseName", "CourseCategory", "CourseLevel", "CourseType",
                           "CourseRating", "Amount"]], width="stretch")

# ================================================================== RECOMMENDATIONS
elif page == "🎯 Recommendations":
    hero("Personalised Course Recommendations", "Cluster-aware hybrid engine: content match + segment popularity + similar learners + ratings.")
    sel1, sel2 = st.columns([1, 2])
    seg_pick = sel1.selectbox("Segment (optional filter)", ["All"] + sorted(feats.segment.unique()))
    pool = feats if seg_pick == "All" else feats[feats.segment == seg_pick]
    ids = pool.UserID.tolist()
    uid = sel2.selectbox("Select a learner", ids, index=min(2, len(ids) - 1))
    r = feats[feats.UserID == uid].iloc[0]
    color, emoji, action = seg_info(r.segment)
    st.markdown(f'<div class="profile" style="--c:{color}"><div class="t">{emoji} {r.segment}</div>'
                f'<div style="color:#374151">{int(r.total_courses)} course(s) so far • prefers <b>{r.preferred_category}</b> '
                f'({r.preferred_level}) • Age {int(r.Age)}, {r.Gender}</div>'
                f'<div style="margin-top:6px;color:#6C5CE7;font-weight:600">💡 Strategy: {action}</div></div>',
                unsafe_allow_html=True)

    a, b, c = st.columns([2, 2, 1])
    lvl = a.multiselect("Filter by level", LEVEL_ORDER)
    cat = b.multiselect("Filter by category", sorted(courses.CourseCategory.unique()))
    n = c.slider("How many", 3, 10, 5)
    recs = rec.recommend(uid, n=n, level=lvl or None, category=cat or None)

    if recs.empty:
        st.warning("No courses match these filters.")
    else:
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Top recommendations")
            top_score = recs.score.max()
            for i, (_, x) in enumerate(recs.iterrows(), 1):
                lc = LEVEL_COLOR[x.CourseLevel]
                pct = 100 * x.score / top_score if top_score else 0
                st.markdown(
                    f'<div class="course" style="--c:{lc}"><div class="n">{i}. {x.CourseName}</div>'
                    f'<div class="meta"><span class="chip" style="background:{lc}">{x.CourseLevel}</span>'
                    f'<span class="chip" style="background:#6B7280">{x.CourseCategory}</span>'
                    f'<span class="chip" style="background:#636E72">{x.CourseType}</span> ⭐ {x.CourseRating:.1f} '
                    f'• {x.CourseDuration:.0f} h • {x.CoursePrice:,.0f}</div>'
                    f'<div class="bar"><div style="width:{pct:.0f}%"></div></div>'
                    f'<div class="why">Why recommended: {x.why}</div></div>', unsafe_allow_html=True)
        with right:
            st.subheader("Learning path")
            path = recs.assign(o=recs.CourseLevel.map({l: i for i, l in enumerate(LEVEL_ORDER)})).sort_values(["o", "score"], ascending=[True, False])
            html = '<div class="path">'
            for j, (_, x) in enumerate(path.iterrows()):
                html += f'<div class="step" style="--c:{LEVEL_COLOR[x.CourseLevel]}">{x.CourseName}<br><small style="color:#6B7280">{x.CourseLevel}</small></div>'
                if j < len(path) - 1: html += '<span class="arrow">➜</span>'
            st.markdown(html + '</div>', unsafe_allow_html=True)
            fig = px.bar(recs.sort_values("score"), x="score", y="CourseName", color="CourseLevel", orientation="h",
                         color_discrete_map=LEVEL_COLOR, title="Match score")
            fig.update_layout(showlegend=False, yaxis_title="")
            st.plotly_chart(style_fig(fig, 340), width="stretch")
        st.download_button("⬇️ Download these recommendations (CSV)",
                           recs[["CourseID", "CourseName", "CourseCategory", "CourseLevel", "CourseRating", "score", "why"]].to_csv(index=False),
                           f"recommendations_{uid}.csv")

    with st.expander("📚 Courses this learner already took"):
        h = txns[txns.UserID == uid].merge(courses, on="CourseID")
        st.dataframe(h[["CourseName", "CourseCategory", "CourseLevel", "CourseRating"]], width="stretch")

# ================================================================== SEGMENT COMPARISON
elif page == "⚖️ Segment Comparison":
    hero("Segment Comparison", "Compare behaviour, spending and interests across learner segments side by side.")
    segs = sorted(feats.segment.unique())
    chosen = st.multiselect("Choose segments to compare", segs, default=segs)
    metrics = ["avg_courses", "avg_categories", "avg_spend", "avg_rating", "depth_index", "paid_ratio", "enroll_freq"]
    p = prof.set_index("segment")
    if chosen:
        norm = (p[metrics] - p[metrics].min()) / (p[metrics].max() - p[metrics].min())
        l, r = st.columns([1, 1])
        radar = go.Figure()
        for s in chosen:
            radar.add_trace(go.Scatterpolar(r=norm.loc[s].tolist() + [norm.loc[s].iloc[0]], theta=metrics + [metrics[0]],
                                            fill="toself", name=s, line_color=COLOR_MAP[s], opacity=.65))
        radar.update_layout(title="Behaviour fingerprint (0 = lowest, 1 = highest)", polar=dict(radialaxis=dict(range=[0, 1])))
        l.plotly_chart(style_fig(radar, 480), width="stretch")
        zs = ((p[metrics] - p[metrics].mean()) / p[metrics].std()).loc[chosen]
        fig = px.imshow(zs.round(2), text_auto=True, aspect="auto", color_continuous_scale="RdYlGn",
                        title="Relative strength (z-score)")
        r.plotly_chart(style_fig(fig, 480), width="stretch")

        m1, m2 = st.columns(2)
        m1.plotly_chart(style_fig(px.bar(p.loc[chosen].reset_index(), x="segment", y="avg_spend", color="segment",
                                         color_discrete_map=COLOR_MAP, title="Average spend per enrolment")
                                  .update_layout(showlegend=False), 360), width="stretch")
        m2.plotly_chart(style_fig(px.bar(p.loc[chosen].reset_index(), x="segment", y="avg_courses", color="segment",
                                         color_discrete_map=COLOR_MAP, title="Average courses per learner")
                                  .update_layout(showlegend=False), 360), width="stretch")

        mix = feats[feats.segment.isin(chosen)].groupby(["segment", "preferred_category"]).size().reset_index(name="learners")
        st.plotly_chart(style_fig(px.bar(mix, x="preferred_category", y="learners", color="segment", barmode="group",
                                         color_discrete_map=COLOR_MAP, title="Preferred category by segment"), 400), width="stretch")
        lv = feats[feats.segment.isin(chosen)].groupby(["segment", "preferred_level"]).size().reset_index(name="learners")
        st.plotly_chart(style_fig(px.bar(lv, x="segment", y="learners", color="preferred_level", barmode="stack",
                                         color_discrete_map=LEVEL_COLOR, title="Preferred level by segment"), 360), width="stretch")
        st.subheader("Recommended business action")
        cc = st.columns(min(3, len(chosen)))
        for i, s in enumerate(chosen):
            color, emoji, action = seg_info(s)
            cc[i % len(cc)].markdown(f'<div class="segcard" style="--c:{color}"><div class="t">{emoji} {s}</div>'
                                     f'<div class="a">{action}</div></div>', unsafe_allow_html=True)

# ================================================================== MODEL INSIGHTS
else:
    hero("Model Insights & Evaluation", "How good are the segments and the recommender? Validation metrics at a glance.")
    mpath = OUT_DIR / "metrics.json"
    if mpath.exists():
        m = json.load(open(mpath))
        k = st.columns(5)
        kpi(k[0], f"{m['silhouette_kmeans']:.3f}", "Silhouette (K-Means)", "#6C5CE7")
        kpi(k[1], f"{m['adjusted_rand_index']:.2f}", "K-Means vs Hierarchical (ARI)", "#0984E3")
        kpi(k[2], f"{m['davies_bouldin']:.2f}", "Davies-Bouldin (lower=better)", "#00B894")
        kpi(k[3], f"{m['mean_intra_cluster_similarity']:.2f}", "Intra-cluster similarity", "#E17055")
        kpi(k[4], f"{m['relative_lift_pct']:+.1f}%", "Hit-rate lift vs popularity", "#FDCB6E")
        st.write("")
    else:
        st.info("Run `python run_pipeline.py` once to generate the evaluation metrics.")

    t1, t2, t3 = st.tabs(["🔢 Choosing k", "🎯 Recommendation quality", "⚡ Engagement lift"])
    with t1:
        kdf = k_table(feats)
        a, b = st.columns(2)
        fig = px.line(kdf, x="k", y="inertia", markers=True, title="Elbow method (inertia)")
        fig.add_vline(x=K_FINAL, line_dash="dash", line_color="#E17055")
        a.plotly_chart(style_fig(fig, 360), width="stretch")
        fig = px.line(kdf, x="k", y="silhouette", markers=True, title="Silhouette score")
        fig.add_vline(x=K_FINAL, line_dash="dash", line_color="#E17055")
        b.plotly_chart(style_fig(fig, 360), width="stretch")
        st.caption("k = 5 chosen: it keeps a solid silhouette while giving five clearly different, actionable segments.")
    with t2:
        ep = TABLE_DIR / "recommendation_evaluation.csv"
        if ep.exists():
            ev = pd.read_csv(ep)
            fig = px.bar(ev, x="HitRate@5", y="Model", orientation="h", color="Model", title="Hit rate @5 (hold-out last enrolment)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, 380), width="stretch")
            st.dataframe(ev, width="stretch")
            st.caption("Learners' course choices are close to random inside the dataset, so lifts are modest - see the paper's limitations.")
        else:
            st.info("Run `python run_pipeline.py` to create this table.")
    with t3:
        if mpath.exists():
            eng = pd.Series(m["engagement_index"]).reset_index(); eng.columns = ["segment", "index"]
            fig = px.bar(eng.sort_values("index"), x="index", y="segment", orientation="h", color="segment",
                         color_discrete_map=COLOR_MAP, title="Engagement index (1.0 = platform average)")
            fig.add_vline(x=1, line_dash="dash"); fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, 360), width="stretch")
            st.caption("Engagement index = average courses per learner in the segment divided by the platform average.")