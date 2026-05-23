"""
================================================================================
 CONFIGURATION — Alpha Defect Detection in Hot Rolling Mills
================================================================================
Central configuration for paths, constants, and reproducibility.
"""
import os
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = r"d:\Tata_steel"
DATA_DIR       = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")
MODEL_DIR      = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR       = os.path.join(OUTPUT_DIR, "plots")
SUBMISSION_DIR = os.path.join(OUTPUT_DIR, "submissions")

TRAIN_PATH     = os.path.join(DATA_DIR, "train.csv")
TEST_PATH      = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB     = os.path.join(DATA_DIR, "sample_submission.csv")

for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, SUBMISSION_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_FOLDS     = 5

# ── Feature columns ─────────────────────────────────────────────────────────
ID_COL     = "CoilID"
TARGET_COL = "Y"
FEATURE_COLS = [f"X{i}" for i in range(1, 50)]  # X1 .. X49

# ── Threshold search ─────────────────────────────────────────────────────────
THRESHOLD_SEARCH_RANGE = (0.01, 0.99)
THRESHOLD_SEARCH_STEPS = 1000

# ── Business constraints ─────────────────────────────────────────────────────
MIN_RECALL    = 1.0      # FALSE NEGATIVES ARE UNACCEPTABLE
MIN_PRECISION = 0.90     # Minimize false positives
