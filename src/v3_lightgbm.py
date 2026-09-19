from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/processed/epl_features.csv")
OUTPUT_PATH = Path("data/processed/lightgbm_walk_forward_predictions.csv")

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

# TOP 15 CORE PRE-MATCH SIGNALS (Prunes out noise)
PREFERRED_FEATURES = [
    "EloDifference",
    "HomeElo",
    "AwayElo",
    "HomeXGLast5",
    "AwayXGLast5",
    "HomeXGALast5",
    "AwayXGALast5",
    "HomeXGAtHomeLast5",
    "AwayXGAwayLast5",
    "HomeGoalsForLast5",
    "AwayGoalsForLast5",
    "HomeGoalsAgainstLast5",
    "AwayGoalsAgainstLast5",
    "RestDaysDiff",
    "HomeRestDays",
    "AwayRestDays",
    "HomeFormLast5",
    "AwayFormLast5",
]

# Regularized LightGBM Base Parameters
LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "boosting_type": "gbdt",
    "n_estimators": 80,  # Lowered tree count to prevent overfitting
    "learning_rate": 0.02,
    "max_depth": 3,  # Shallow trees for better generalization
    "num_leaves": 7,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}


# ============================================================
# METRICS HELPER: RANKED PROBABILITY SCORE (RPS)
# ============================================================


def calculate_rps(y_true_numeric: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculates Ranked Probability Score (RPS) for 3-way outcomes.

    y_true_numeric: 0 = Home, 1 = Draw, 2 = Away
    y_prob: Matrix [[P_Home, P_Draw, P_Away], ...]
    """
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
# MAIN EXECUTION
# ============================================================


def main():
    print("Loading engineered features dataset...")
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    # Target mapping
    result_map = {"H": 0, "D": 1, "A": 2}
    df["ResultNumeric"] = df["Result"].map(result_map)

    # Select top available pre-match features
    feature_cols = [c for c in PREFERRED_FEATURES if c in df.columns]

    # Fallback if specific names differ
    if len(feature_cols) < 5:
        non_feature_cols = [
            "Date",
            "Season",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "HTHG",
            "HTAG",
            "Result",
            "ResultNumeric",
            "MatchID",
            "HomeXG",
            "AwayXG",
            "HS",
            "AS",
            "HST",
            "AST",
            "HC",
            "AC",
            "HY",
            "AY",
            "HR",
            "AR",
            "B365H",
            "B365D",
            "B365A",
        ]
        feature_cols = [c for c in df.columns if c not in non_feature_cols]

    print(
        f"Selected {len(feature_cols)} pruned pre-match features for model training."
    )

    valid_df = df.dropna(subset=feature_cols + ["ResultNumeric"]).reset_index(
        drop=True
    )
    all_predictions = []

    print(
        "\nRunning expanding-window Walk-Forward LightGBM with Platt Calibration...\n"
    )

    for season in TEST_SEASONS:
        train_df = valid_df[valid_df["Season"] < season].copy()
        test_df = valid_df[valid_df["Season"] == season].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        print(
            f"Testing {season}: Trained on {len(train_df)} matches | Testing on {len(test_df)} matches"
        )

        X_train = train_df[feature_cols]
        y_train = train_df["ResultNumeric"]
        X_test = test_df[feature_cols]

        # Base LightGBM model
        base_lgbm = LGBMClassifier(**LGBM_PARAMS)

        # Wrap in 3-Fold Inner Calibration (Platt Scaling) on training fold
        calibrated_model = CalibratedClassifierCV(
            estimator=base_lgbm, method="sigmoid", cv=3
        )
        calibrated_model.fit(X_train, y_train)

        # Predict calibrated probabilities matrix: [[P_Home, P_Draw, P_Away], ...]
        probs = calibrated_model.predict_proba(X_test)

        for idx, (_, row) in enumerate(test_df.iterrows()):
            p_h, p_d, p_a = probs[idx][0], probs[idx][1], probs[idx][2]

            # Bet365 Implied Odds Benchmark
            b365_h, b365_d, b365_a = (
                row.get("B365H", np.nan),
                row.get("B365D", np.nan),
                row.get("B365A", np.nan),
            )
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
                    "HomeProbability": p_h,
                    "DrawProbability": p_d,
                    "AwayProbability": p_a,
                    "PredictedResult": ["H", "D", "A"][np.argmax([p_h, p_d, p_a])],
                    "B365_HomeProb": b365_probs[0],
                    "B365_DrawProb": b365_probs[1],
                    "B365_AwayProb": b365_probs[2],
                }
            )

    pred_df = pd.DataFrame(all_predictions)

    # Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(OUTPUT_PATH, index=False)

    # Evaluation
    y_true = pred_df["ActualNumeric"].values
    y_probs = pred_df[
        ["HomeProbability", "DrawProbability", "AwayProbability"]
    ].values
    y_pred = y_probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    loss = log_loss(y_true, y_probs, labels=[0, 1, 2])
    rps = calculate_rps(y_true, y_probs)

    b365_mask = pred_df["B365_HomeProb"].notna()
    b365_true = pred_df.loc[b365_mask, "ActualNumeric"].values
    b365_probs_mat = pred_df.loc[
        b365_mask, ["B365_HomeProb", "B365_DrawProb", "B365_AwayProb"]
    ].values
    b365_rps = calculate_rps(b365_true, b365_probs_mat)
    b365_loss = log_loss(b365_true, b365_probs_mat, labels=[0, 1, 2])

    print("\n" + "=" * 60)
    print("WALK-FORWARD CALIBRATED LIGHTGBM OVERALL RESULTS")
    print("=" * 60)
    print(f"Matches evaluated:   {len(pred_df)}")
    print(f"Calibrated Accuracy: {acc:.4f}")
    print(f"Calibrated Log Loss: {loss:.4f}  |  B365 Log Loss: {b365_loss:.4f}")
    print(f"Calibrated RPS:      {rps:.4f}  |  B365 RPS:      {b365_rps:.4f}")

    # Season Breakdown
    season_summary = []
    for season, grp in pred_df.groupby("Season", sort=True):
        s_true = grp["ActualNumeric"].values
        s_probs = grp[
            ["HomeProbability", "DrawProbability", "AwayProbability"]
        ].values
        s_pred = s_probs.argmax(axis=1)

        s_acc = accuracy_score(s_true, s_pred)
        s_loss = log_loss(s_true, s_probs, labels=[0, 1, 2])
        s_rps = calculate_rps(s_true, s_probs)

        s_b365_mask = grp["B365_HomeProb"].notna()
        s_b365_rps = calculate_rps(
            grp.loc[s_b365_mask, "ActualNumeric"].values,
            grp.loc[
                s_b365_mask,
                ["B365_HomeProb", "B365_DrawProb", "B365_AwayProb"],
            ].values,
        )

        season_summary.append(
            {
                "Season": season,
                "Matches": len(grp),
                "Accuracy": s_acc,
                "LogLoss": s_loss,
                "LGBM_RPS": s_rps,
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
                "LGBM_RPS": "{:.4f}".format,
                "B365_RPS": "{:.4f}".format,
            },
        )
    )


if __name__ == "__main__":
    main()