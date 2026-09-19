from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path("data/processed/epl_features.csv")
OUTPUT_PATH = Path("data/processed/poisson_walk_forward_predictions.csv")

MAX_GOALS = 8

TEST_SEASONS = [
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]

FEATURE_COLS = [
    "HomeXGLast5",
    "AwayXGALast5",
    "AwayXGLast5",
    "HomeXGALast5",
    "EloDifference",
]


# ============================================================
# HELPER METRICS
# ============================================================


def calculate_rps(y_true_numeric: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculates Ranked Probability Score (RPS) for 3-way outcomes (Home, Draw, Away)."""
    n_samples = len(y_true_numeric)
    rps_list = []

    for i in range(n_samples):
        obs = y_true_numeric[i]
        probs = y_prob[i]

        p_cum = np.cumsum(probs)
        e_cum = np.array([1 if obs <= 0 else 0, 1 if obs <= 1 else 0])

        rps = 0.5 * np.sum((p_cum[:2] - e_cum) ** 2)
        rps_list.append(rps)

    return np.mean(rps_list)


# ============================================================
# PURE POISSON MODEL FOR WALK-FORWARD FITTING
# ============================================================


class SimplePoissonModel:

    def __init__(self, max_goals: int = 8):
        self.max_goals = max_goals
        self.home_cols = ["HomeXGLast5", "AwayXGALast5", "EloDifference"]
        self.away_cols = ["AwayXGLast5", "HomeXGALast5", "EloDifference"]

        self.home_regressor = PoissonRegressor(alpha=1e-3, max_iter=300)
        self.away_regressor = PoissonRegressor(alpha=1e-3, max_iter=300)

    def fit(self, train_df: pd.DataFrame):
        X_home = train_df[self.home_cols]
        y_home = train_df["FTHG"]
        X_away = train_df[self.away_cols]
        y_away = train_df["FTAG"]

        self.home_regressor.fit(X_home, y_home)
        self.away_regressor.fit(X_away, y_away)
        return self

    def predict_match(self, row: pd.Series) -> tuple[float, float, np.ndarray]:
        X_home = row[self.home_cols].to_frame().T
        X_away = row[self.away_cols].to_frame().T

        lh = float(np.clip(self.home_regressor.predict(X_home)[0], 0.05, 6.0))
        la = float(np.clip(self.away_regressor.predict(X_away)[0], 0.05, 6.0))

        home_win, draw, away_win = 0.0, 0.0, 0.0
        for h in range(self.max_goals + 1):
            for a in range(self.max_goals + 1):
                prob = poisson.pmf(h, lh) * poisson.pmf(a, la)
                if h > a:
                    home_win += prob
                elif h == a:
                    draw += prob
                else:
                    away_win += prob

        total = home_win + draw + away_win
        probs = np.array([home_win / total, draw / total, away_win / total])

        return lh, la, probs


# ============================================================
# MAIN WALK-FORWARD EXECUTION
# ============================================================


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Filter complete rows
    required_cols = FEATURE_COLS + ["FTHG", "FTAG", "Result"]
    valid_df = df.dropna(subset=required_cols).copy().reset_index(drop=True)

    result_map = {"H": 0, "D": 1, "A": 2}
    valid_df["ResultNumeric"] = valid_df["Result"].map(result_map)

    all_predictions = []

    print("\nRunning expanding-window Walk-Forward Poisson evaluation...\n")

    for season in TEST_SEASONS:
        # Expanding window: Train on ALL data prior to current test season
        train_df = valid_df[valid_df["Season"] < season].copy()
        test_df = valid_df[valid_df["Season"] == season].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            print(f"Skipping {season} (insufficient training/testing data)")
            continue

        print(
            f"Testing {season}: Trained on {len(train_df)} matches | Testing on {len(test_df)} matches"
        )

        # Fit model strictly on historical expanding window
        model = SimplePoissonModel(max_goals=MAX_GOALS)
        model.fit(train_df)

        for _, row in test_df.iterrows():
            lh, la, probs = model.predict_match(row)

            # Bet365 Implied Probabilities Benchmark
            b365_h = row.get("B365H", np.nan)
            b365_d = row.get("B365D", np.nan)
            b365_a = row.get("B365A", np.nan)

            if pd.notna(b365_h) and pd.notna(b365_d) and pd.notna(b365_a):
                raw_h, raw_d, raw_a = 1 / b365_h, 1 / b365_d, 1 / b365_a
                margin = raw_h + raw_d + raw_a
                b365_probs = [raw_h / margin, raw_d / margin, raw_a / margin]
            else:
                b365_probs = [np.nan, np.nan, np.nan]

            all_predictions.append(
                {
                    "Season": season,
                    "Date": row["Date"],
                    "HomeTeam": row["HomeTeam"],
                    "AwayTeam": row["AwayTeam"],
                    "ActualResult": row["Result"],
                    "ActualNumeric": row["ResultNumeric"],
                    "ExpectedHomeGoals": lh,
                    "ExpectedAwayGoals": la,
                    "HomeProb": probs[0],
                    "DrawProb": probs[1],
                    "AwayProb": probs[2],
                    "B365_HomeProb": b365_probs[0],
                    "B365_DrawProb": b365_probs[1],
                    "B365_AwayProb": b365_probs[2],
                }
            )

    pred_df = pd.DataFrame(all_predictions)

    # Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(OUTPUT_PATH, index=False)

    # Global Performance Metrics
    y_true = pred_df["ActualNumeric"].values
    y_probs = pred_df[["HomeProb", "DrawProb", "AwayProb"]].values
    y_pred = y_probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    loss = log_loss(y_true, y_probs, labels=[0, 1, 2])
    rps = calculate_rps(y_true, y_probs)

    # Bookmaker Market Benchmark Metrics
    b365_mask = pred_df["B365_HomeProb"].notna()
    if b365_mask.sum() > 0:
        b365_true = pred_df.loc[b365_mask, "ActualNumeric"].values
        b365_probs_mat = pred_df.loc[
            b365_mask, ["B365_HomeProb", "B365_DrawProb", "B365_AwayProb"]
        ].values
        b365_rps = calculate_rps(b365_true, b365_probs_mat)
        b365_loss = log_loss(b365_true, b365_probs_mat, labels=[0, 1, 2])
    else:
        b365_rps, b365_loss = np.nan, np.nan

    print("\n" + "=" * 60)
    print("WALK-FORWARD POISSON MODEL OVERALL RESULTS")
    print("=" * 60)
    print(f"Matches evaluated: {len(pred_df)}")
    print(f"Model Accuracy:    {acc:.4f}")
    print(f"Model Log Loss:    {loss:.4f}  |  B365 Log Loss: {b365_loss:.4f}")
    print(f"Model RPS:         {rps:.4f}  |  B365 RPS:      {b365_rps:.4f}")

    # Season Breakdown Table
    season_summary = []
    for season, grp in pred_df.groupby("Season", sort=True):
        s_true = grp["ActualNumeric"].values
        s_probs = grp[["HomeProb", "DrawProb", "AwayProb"]].values
        s_pred = s_probs.argmax(axis=1)

        s_acc = accuracy_score(s_true, s_pred)
        s_loss = log_loss(s_true, s_probs, labels=[0, 1, 2])
        s_rps = calculate_rps(s_true, s_probs)

        # Bet365 Season RPS
        s_b365_mask = grp["B365_HomeProb"].notna()
        if s_b365_mask.sum() > 0:
            s_b365_rps = calculate_rps(
                grp.loc[s_b365_mask, "ActualNumeric"].values,
                grp.loc[
                    s_b365_mask,
                    ["B365_HomeProb", "B365_DrawProb", "B365_AwayProb"],
                ].values,
            )
        else:
            s_b365_rps = np.nan

        season_summary.append(
            {
                "Season": season,
                "Matches": len(grp),
                "Accuracy": s_acc,
                "LogLoss": s_loss,
                "Model_RPS": s_rps,
                "B365_RPS": s_b365_rps,
            }
        )

    print("\nRESULTS BY SEASON")
    s_df = pd.DataFrame(season_summary)
    print(
        s_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "LogLoss": "{:.4f}".format,
                "Model_RPS": "{:.4f}".format,
                "B365_RPS": "{:.4f}".format,
            },
        )
    )

    print(f"\nSaved out-of-sample predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()