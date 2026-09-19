from pathlib import Path
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss

# =========================================================
# CONFIG
# =========================================================

DATA_PATH = Path("data/processed/epl_features.csv")
OUTPUT_PATH = Path("data/processed/lightgbm_v2_walk_forward_predictions.csv")
IMPORTANCE_OUTPUT_PATH = Path(
    "data/processed/lightgbm_v2_feature_importance.csv"
)

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

# =========================================================
# V2 FEATURES
# =========================================================

FEATURE_COLUMNS = [
    # ELO
    "HomeElo",
    "AwayElo",
    "EloDifference",
    # RECENT GOALS
    "HomeGoalsForLast3",
    "HomeGoalsAgainstLast3",
    "AwayGoalsForLast3",
    "AwayGoalsAgainstLast3",
    "HomeGoalsForLast5",
    "HomeGoalsAgainstLast5",
    "AwayGoalsForLast5",
    "AwayGoalsAgainstLast5",
    "HomeGoalsForLast10",
    "HomeGoalsAgainstLast10",
    "AwayGoalsForLast10",
    "AwayGoalsAgainstLast10",
    # RECENT xG
    "HomeXGLast3",
    "HomeXGALast3",
    "AwayXGLast3",
    "AwayXGALast3",
    "HomeXGLast5",
    "HomeXGALast5",
    "AwayXGLast5",
    "AwayXGALast5",
    "HomeXGLast10",
    "HomeXGALast10",
    "AwayXGLast10",
    "AwayXGALast10",
    # GOAL DIFFERENCE
    "HomeGoalDifferenceLast3",
    "AwayGoalDifferenceLast3",
    "HomeGoalDifferenceLast5",
    "AwayGoalDifferenceLast5",
    "HomeGoalDifferenceLast10",
    "AwayGoalDifferenceLast10",
    # xG DIFFERENCE
    "HomeXGDifferenceLast3",
    "AwayXGDifferenceLast3",
    "HomeXGDifferenceLast5",
    "AwayXGDifferenceLast5",
    "HomeXGDifferenceLast10",
    "AwayXGDifferenceLast10",
    # HOME-SPECIFIC
    "HomeGoalsForAtHome",
    "HomeGoalsAgainstAtHome",
    "HomeXGAtHome",
    "HomeXGAAtHome",
    # AWAY-SPECIFIC
    "AwayGoalsForAway",
    "AwayGoalsAgainstAway",
    "AwayXGAway",
    "AwayXGAgainstAway",
    # FORM
    "HomeForm",
    "AwayForm",
    # REST
    "HomeRestDays",
    "AwayRestDays",
    # MATCHUP FEATURES
    "HomeAttackVsAwayDefense",
    "AwayAttackVsHomeDefense",
    "HomeXGAttackVsAwayXGDefense",
    "AwayXGAttackVsHomeXGDefense",
    "HomeAttackVsAwayDefense_HA",
    "AwayAttackVsHomeDefense_AH",
    "HomeXGAttackVsAwayXGDefense_HA",
    "AwayXGAttackVsHomeXGDefense_AH",
    # RELATIVE FEATURES
    "FormDifference",
    "RestDaysDifference",
    "RecentGoalDifferenceMatchup",
    "RecentXGDifferenceMatchup",
    # RATIOS
    "HomeXGAttackDefenseRatio",
    "AwayXGAttackDefenseRatio",
    "HomeGoalAttackDefenseRatio",
    "AwayGoalAttackDefenseRatio",
]


# =========================================================
# HELPER METRICS
# =========================================================


def calculate_brier_score(y_true, probabilities_adh):
    """Multiclass Brier score expecting probabilities aligned [A, D, H]."""
    y_encoded = (
        pd.get_dummies(y_true)
        .reindex(columns=["A", "D", "H"], fill_value=0)
        .to_numpy()
    )
    return np.mean(np.sum((probabilities_adh - y_encoded) ** 2, axis=1))


def calculate_rps(y_true, probabilities_adh):
    """Calculates Ranked Probability Score (RPS) expecting probabilities aligned [A, D, H]."""
    res_map = {"A": 2, "D": 1, "H": 0}
    y_num = np.array([res_map[r] for r in y_true])

    # Convert [A, D, H] to [H, D, A] for standard cumulative sum calculation
    probs_hda = probabilities_adh[:, [2, 1, 0]]

    rps_list = []
    for i in range(len(y_num)):
        obs = y_num[i]
        p_cum = np.cumsum(probs_hda[i])
        e_cum = np.array([1 if obs <= 0 else 0, 1 if obs <= 1 else 0])
        rps_list.append(0.5 * np.sum((p_cum[:2] - e_cum) ** 2))

    return np.mean(rps_list)


# =========================================================
# MAIN EXECUTION
# =========================================================


def main():
    print("Loading engineered features dataset...")
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    print(f"Total matches: {len(df)}")

    # Verify feature presence
    available_features = [f for f in FEATURE_COLUMNS if f in df.columns]
    missing = [f for f in FEATURE_COLUMNS if f not in df.columns]

    if missing:
        print(f"\nWARNING: Missing {len(missing)} specified features:")
        for feat in missing[:5]:
            print(f"  - {feat}")

    print(f"Using {len(available_features)} verified pre-match features.")

    predictions = []
    feature_importances = []

    print("\n" + "=" * 60)
    print("RUNNING WALK-FORWARD LIGHTGBM V2 (CALIBRATED)")
    print("=" * 60)

    for test_season in TEST_SEASONS:
        train_df = df[df["Season"] < test_season].copy()
        test_df = df[df["Season"] == test_season].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        print(
            f"\nTesting {test_season}: Trained on {len(train_df)} matches | Testing on {len(test_df)} matches"
        )

        X_train = train_df[available_features]
        y_train = train_df["Result"]

        X_test = test_df[available_features]
        y_test = test_df["Result"]

        # Base Regularized LightGBM Model
        base_lgbm = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=100,
            learning_rate=0.02,
            num_leaves=15,
            max_depth=4,
            min_child_samples=25,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbosity=-1,
        )

        # Wrap in 3-Fold Inner Cross-Validation Platt Calibration
        calibrated_model = CalibratedClassifierCV(
            estimator=base_lgbm, method="sigmoid", cv=3
        )
        calibrated_model.fit(X_train, y_train)

        # Extract average gain importance across calibrated inner estimators
        fold_importances = []
        for calibrated_classifier in calibrated_model.calibrated_classifiers_:
            base_estimator = calibrated_classifier.estimator
            imp = base_estimator.booster_.feature_importance(
                importance_type="gain"
            )
            fold_importances.append(imp)

        avg_fold_importance = np.mean(fold_importances, axis=0)
        importance_series = pd.Series(
            avg_fold_importance, index=available_features
        )
        imp_sum = importance_series.sum()
        if imp_sum > 0:
            importance_series = importance_series / imp_sum

        feature_importances.append(importance_series)

        # Predict calibrated probabilities
        probabilities = calibrated_model.predict_proba(X_test)
        class_order = list(calibrated_model.classes_)

        prob_df = pd.DataFrame(
            probabilities, columns=class_order, index=test_df.index
        )
        prob_df = prob_df[["A", "D", "H"]]
        predicted_results = prob_df.idxmax(axis=1)

        # Save predictions & extract Bet365 odds
        for idx in test_df.index:
            row = test_df.loc[idx]

            b365_h = row.get("B365H", np.nan)
            b365_d = row.get("B365D", np.nan)
            b365_a = row.get("B365A", np.nan)

            if pd.notna(b365_h) and pd.notna(b365_d) and pd.notna(b365_a):
                raw_h, raw_d, raw_a = 1 / b365_h, 1 / b365_d, 1 / b365_a
                margin = raw_h + raw_d + raw_a
                b365_probs = [raw_a / margin, raw_d / margin, raw_h / margin]
            else:
                b365_probs = [np.nan, np.nan, np.nan]

            predictions.append(
                {
                    "Season": test_season,
                    "Date": row["Date"],
                    "HomeTeam": row["HomeTeam"],
                    "AwayTeam": row["AwayTeam"],
                    "ActualResult": row["Result"],
                    "AwayProbability": prob_df.loc[idx, "A"],
                    "DrawProbability": prob_df.loc[idx, "D"],
                    "HomeProbability": prob_df.loc[idx, "H"],
                    "PredictedResult": predicted_results.loc[idx],
                    "B365_AwayProb": b365_probs[0],
                    "B365_DrawProb": b365_probs[1],
                    "B365_HomeProb": b365_probs[2],
                }
            )

        # Season Metrics
        s_probs = prob_df[["A", "D", "H"]].to_numpy()
        s_acc = accuracy_score(y_test, predicted_results)
        s_loss = log_loss(y_test, s_probs, labels=["A", "D", "H"])
        s_brier = calculate_brier_score(y_test, s_probs)
        s_rps = calculate_rps(y_test, s_probs)

        print(
            f"Accuracy: {s_acc:.4f} | LogLoss: {s_loss:.4f} | RPS: {s_rps:.4f} | Brier: {s_brier:.4f}"
        )

    # Predictions DataFrame
    predictions_df = pd.DataFrame(predictions)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(OUTPUT_PATH, index=False)

    # Overall Metrics
    y_true = predictions_df["ActualResult"]
    y_pred = predictions_df["PredictedResult"]
    all_probs = predictions_df[
        ["AwayProbability", "DrawProbability", "HomeProbability"]
    ].to_numpy()

    acc = accuracy_score(y_true, y_pred)
    loss = log_loss(y_true, all_probs, labels=["A", "D", "H"])
    brier = calculate_brier_score(y_true, all_probs)
    rps = calculate_rps(y_true, all_probs)

    # Bet365 Benchmarks
    b365_mask = predictions_df["B365_HomeProb"].notna()
    if b365_mask.sum() > 0:
        b365_probs = predictions_df.loc[
            b365_mask, ["B365_AwayProb", "B365_DrawProb", "B365_HomeProb"]
        ].to_numpy()
        b365_true = y_true[b365_mask]
        b365_loss = log_loss(b365_true, b365_probs, labels=["A", "D", "H"])
        b365_rps = calculate_rps(b365_true, b365_probs)
    else:
        b365_loss, b365_rps = np.nan, np.nan

    print("\n" + "=" * 60)
    print("WALK-FORWARD LIGHTGBM V2 OVERALL RESULTS")
    print("=" * 60)
    print(f"Matches evaluated:   {len(predictions_df)}")
    print(f"Calibrated Accuracy: {acc:.4f}")
    print(f"Calibrated Log Loss: {loss:.4f}  |  B365 Log Loss: {b365_loss:.4f}")
    print(f"Calibrated RPS:      {rps:.4f}  |  B365 RPS:      {b365_rps:.4f}")
    print(f"Brier Score:         {brier:.4f}")

    # Feature Importance Export
    importance_df = pd.concat(feature_importances, axis=1)
    avg_importance = (
        importance_df.mean(axis=1).sort_values(ascending=False).rename("Importance")
    )

    print("\n" + "=" * 60)
    print("TOP 15 FEATURES BY NORMALIZED GAIN IMPORTANCE")
    print("=" * 60)
    print(avg_importance.head(15).to_string())

    IMPORTANCE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    avg_importance.reset_index().rename(columns={"index": "Feature"}).to_csv(
        IMPORTANCE_OUTPUT_PATH, index=False
    )

    print(f"\nSaved predictions to: {OUTPUT_PATH}")
    print(f"Saved feature importance to: {IMPORTANCE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()