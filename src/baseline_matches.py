from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

# ============================================================
# CONFIG
# ============================================================

POISSON_PATH = Path("data/processed/poisson_walk_forward_predictions.csv")
DIXON_COLES_PATH = Path(
    "data/processed/dixon_coles_walk_forward_predictions.csv"
)

# Load predictions
poisson = pd.read_csv(POISSON_PATH)
dixon_coles = pd.read_csv(DIXON_COLES_PATH)

# Column name standardization safeguard
rename_map = {
    "HomeProb": "HomeProbability",
    "DrawProb": "DrawProbability",
    "AwayProb": "AwayProbability",
    "Actual": "ActualResult",
}
poisson = poisson.rename(columns=rename_map)
dixon_coles = dixon_coles.rename(columns=rename_map)

# Datetime parsing
poisson["Date"] = pd.to_datetime(poisson["Date"])
dixon_coles["Date"] = pd.to_datetime(dixon_coles["Date"])

KEY_COLUMNS = ["Season", "Date", "HomeTeam", "AwayTeam"]

# Inner Join for strict head-to-head match alignment
merged = poisson.merge(
    dixon_coles, on=KEY_COLUMNS, suffixes=("_Poisson", "_DixonColes")
)

print("=" * 70)
print("FAIR HEAD-TO-HEAD BASELINE COMPARISON")
print("=" * 70)
print(f"Poisson total matches:     {len(poisson)}")
print(f"Dixon-Coles total matches: {len(dixon_coles)}")
print(f"Overlapping match sample:  {len(merged)}")


# ============================================================
# METRICS HELPER: RANKED PROBABILITY SCORE (RPS)
# ============================================================


def calculate_rps(y_true_str: pd.Series, probs_hda: np.ndarray) -> float:
    """Calculates RPS where probs_hda matrix is ordered [Home, Draw, Away]."""
    res_map = {"H": 0, "D": 1, "A": 2}
    y_num = y_true_str.map(res_map).to_numpy()

    rps_list = []
    for i in range(len(y_num)):
        obs = y_num[i]
        p = probs_hda[i]  # [P_Home, P_Draw, P_Away]
        p_cum = np.cumsum(p)
        e_cum = np.array([1 if obs <= 0 else 0, 1 if obs <= 1 else 0])
        rps_list.append(0.5 * np.sum((p_cum[:2] - e_cum) ** 2))

    return np.mean(rps_list)


# Actual outcomes
y_true = merged["ActualResult_Poisson"]
res_map = {"H": 0, "D": 1, "A": 2}
y_true_num = y_true.map(res_map).to_numpy()

# One-hot encoding for Brier Score
y_one_hot = np.zeros((len(y_true_num), 3))
y_one_hot[np.arange(len(y_true_num)), y_true_num] = 1


# ============================================================
# 1. POISSON MODEL EVALUATION
# ============================================================

poisson_probs_adh = merged[
    [
        "AwayProbability_Poisson",
        "DrawProbability_Poisson",
        "HomeProbability_Poisson",
    ]
].to_numpy()

poisson_probs_hda = merged[
    [
        "HomeProbability_Poisson",
        "DrawProbability_Poisson",
        "AwayProbability_Poisson",
    ]
].to_numpy()

poisson_pred_num = poisson_probs_hda.argmax(axis=1)

poisson_acc = accuracy_score(y_true_num, poisson_pred_num)
poisson_loss = log_loss(y_true, poisson_probs_adh, labels=["A", "D", "H"])
poisson_brier = np.mean(np.sum((poisson_probs_hda - y_one_hot) ** 2, axis=1))
poisson_rps = calculate_rps(y_true, poisson_probs_hda)


# ============================================================
# 2. DIXON-COLES MODEL EVALUATION
# ============================================================

dc_probs_adh = merged[
    [
        "AwayProbability_DixonColes",
        "DrawProbability_DixonColes",
        "HomeProbability_DixonColes",
    ]
].to_numpy()

dc_probs_hda = merged[
    [
        "HomeProbability_DixonColes",
        "DrawProbability_DixonColes",
        "AwayProbability_DixonColes",
    ]
].to_numpy()

dc_pred_num = dc_probs_hda.argmax(axis=1)

dc_acc = accuracy_score(y_true_num, dc_pred_num)
dc_loss = log_loss(y_true, dc_probs_adh, labels=["A", "D", "H"])
dc_brier = np.mean(np.sum((dc_probs_hda - y_one_hot) ** 2, axis=1))
dc_rps = calculate_rps(y_true, dc_probs_hda)


# ============================================================
# 3. BET365 MARKET BENCHMARK
# ============================================================

if "B365_HomeProb_Poisson" in merged.columns:
    b365_probs_adh = merged[
        [
            "B365_AwayProb_Poisson",
            "B365_DrawProb_Poisson",
            "B365_HomeProb_Poisson",
        ]
    ].to_numpy()

    b365_probs_hda = merged[
        [
            "B365_HomeProb_Poisson",
            "B365_DrawProb_Poisson",
            "B365_AwayProb_Poisson",
        ]
    ].to_numpy()

    b365_pred_num = b365_probs_hda.argmax(axis=1)

    b365_acc = accuracy_score(y_true_num, b365_pred_num)
    b365_loss = log_loss(y_true, b365_probs_adh, labels=["A", "D", "H"])
    b365_brier = np.mean(np.sum((b365_probs_hda - y_one_hot) ** 2, axis=1))
    b365_rps = calculate_rps(y_true, b365_probs_hda)
else:
    b365_acc, b365_loss, b365_brier, b365_rps = (
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    )


# ============================================================
# FINAL COMPARISON TABLE
# ============================================================

results = pd.DataFrame(
    {
        "Model": ["V1 Rolling Poisson", "V2 Dixon-Coles", "Bet365 Benchmark"],
        "Matches": [len(merged), len(merged), len(merged)],
        "Accuracy": [poisson_acc, dc_acc, b365_acc],
        "LogLoss": [poisson_loss, dc_loss, b365_loss],
        "BrierScore": [poisson_brier, dc_brier, b365_brier],
        "RPS": [poisson_rps, dc_rps, b365_rps],
    }
)

print("\n" + results.to_string(index=False, float_format="{:.4f}".format))