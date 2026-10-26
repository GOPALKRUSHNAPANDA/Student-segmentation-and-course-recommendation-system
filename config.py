"""Central configuration for the EduPro project (edit values here, not in the code)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "EduPro_Online_Platform.xlsx"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

RANDOM_STATE = 42
K_RANGE = range(2, 11)          # candidate numbers of clusters
LEVEL_SCORE = {"Beginner": 0.0, "Intermediate": 0.5, "Advanced": 1.0}

# Hybrid recommender weights (must sum to 1.0)
REC_WEIGHTS = {
    "content": 0.30,     # match with learner's category / level preference
    "cluster_pop": 0.30, # popularity of the course inside the learner's segment
    "similar": 0.25,     # enrolled by the most similar learners (collaborative)
    "rating": 0.15,      # rating-weighted relevance
}
TOP_N = 5