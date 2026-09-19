from pathlib import Path
import numpy as np
import pandas as pd
import scipy.optimize as opt
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/processed/epl_features.csv")
OUTPUT_PATH = Path("data/processed/calibration_experiment_predictions.csv")

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

# Top pre-match feature signals
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

LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "boosting_type": "gbdt",
    "n_estimators": 100,
    "learning_rate": 0.02,
    "max_depth": 4,
    "num_leaves": 15,
    "min_child_samples": 25,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}


# ============================================================
# TEMPERATURE SCALER CLASS
# ============================================================


class TemperatureScaler:
    """Optimizes a single temperature scalar T > 0 on out-of-fold logits via NLL minimization."""

    def __init__(self):
        self.temp = 1.0

    def _nll(self, T: float, logits: np.ndarray, y_true: np.ndarray) -> float:
        scaled_logits = logits / T
        # Numerically stable Softmax
        exp_l = np.exp(
            scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)
        )
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        nll = -np.mean(np.log(probs[np.arange(len(y_true)), y_true] + 1e-12))
        return float(nll)

    def fit(self, oof_logits: np.ndarray, y_true: np.ndarray):
        res = opt.minimize(
            self._nll,
            x0=[1.0],
            args=(oof_logits, y_true),
            bounds=[(0.05, 10.0)],
            method="L-BFGS-B",
        )
        self.temp = float(res.x[0])
        return self

    def transform(self, test_logits: np.ndarray) -> np.ndarray:
        scaled_logits = test_logits / self.temp
        exp_l = np.exp(
            scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)
        )
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)


# ============================================================
# HELPER METRICS
# ============================================================


def calculate_rps(y_true_numeric: np.ndarray, y_prob_hda: np.ndarray) -> float:
    """Calculates Ranked Probability Score (RPS) for 3-way outcomes [Home, Draw, Away]."""
    n_samples = len(y_true_numeric)
    rps_list = []

    for i in range(n_samples):
        obs = y_true_numeric[i]
        probs = y_prob_hda[i]

        p_cum = np.cumsum(probs)
        e_cum = np.array([1 if obs <= 0 else 0, 1 if obs <= 1 else 0])

        rps = 0.5 * np.sum((p_cum[:2] - e_cum) ** 2)
        rps_list.append(rps)

    return float(np.mean(rps_list))


def get_oof_logits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    lgbm_params: dict,
    n_splits: int = 3,
) -> np.ndarray:
    """Generates out-of-fold (OOF) raw logits using inner K-Fold cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_logits = np.zeros((len(X_train), 3))

    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va = X_train.iloc[val_idx]

        m = LGBMClassifier(**lgbm_params)
        m.fit(X_tr, y_tr)
        oof_logits[val_idx] = m.predict_proba(X_va, raw_score=True)

    return oof_logits


# ============================================================
# MAIN EXECUTION
# ============================================================


def main():
    print("Loading engineered features dataset...")
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    # Encode target result ("H": 0, "D": 1, "A": 2)
    result_map = {"H": 0, "D": 1, "A": 2}
    df["ResultNumeric"] = df["Result"].map(result_map)

    # Select available feature subset
    feature_cols = [c for c in PREFERRED_FEATURES if c in df.columns]
    if len(feature_cols) < 5:
        non_feat = [
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
        feature_cols = [c for c in df.columns if c not in non_feat]

    print(f"Selected {len(feature_cols)} pre-match features.")

    valid_df = df.dropna(subset=feature_cols + ["ResultNumeric"]).reset_index(
        drop=True
    )
    all_predictions = []

    print(
        "\nRunning expanding-window Walk-Forward Calibration Experiments...\n"
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
        y_test = test_df["ResultNumeric"].values

        # 1. RAW LIGHTGBM
        base_lgbm = LGBMClassifier(**LGBM_PARAMS)
        base_lgbm.fit(X_train, y_train)

        raw_probs_test = base_lgbm.predict_proba(X_test)
        raw_logits_test = base_lgbm.predict_proba(X_test, raw_score=True)

        # 2. TEMPERATURE SCALING
        oof_logits = get_oof_logits(
            X_train, y_train, LGBM_PARAMS, n_splits=3
        )
        temp_scaler = TemperatureScaler().fit(oof_logits, y_train.values)
        temp_probs_test = temp_scaler.transform(raw_logits_test)

        # 3. PLATT SCALING (LOGISTIC)
        platt_model = CalibratedClassifierCV(
            estimator=LGBMClassifier(**LGBM_PARAMS), method="sigmoid", cv=3
        )
        platt_model.fit(X_train, y_train)
        platt_probs_test = platt_model.predict_proba(X_test)

        # 4. ISOTONIC SCALING
        iso_model = CalibratedClassifierCV(
            estimator=LGBMClassifier(**LGBM_PARAMS), method="isotonic", cv=3
        )
        iso_model.fit(X_train, y_train)
        iso_probs_test = iso_model.predict_proba(X_test)

        for idx, (_, row) in enumerate(test_df.iterrows()):
            b365_h, b365_d, b365_a = (
                row.get("B365H", np.nan),
                row.get("B365D", np.nan),
                row.get("B365A", np.nan),
            )
            if pd.notna(b365_h) and pd.notna(b365_d) and pd.notna(b365_a):
                raw_h, raw_d, raw_a = 1 / b365_h, 1 / b365_d, 1 / b365_a
                margin = raw_h + raw_d + raw_a
                b365_p = [raw_h / margin, raw_d / margin, raw_a / margin]
            else:
                b365_p = [np.nan, np.nan, np.nan]

            all_predictions.append(
                {
                    "Season": season,
                    "Date": row["Date"],
                    "HomeTeam": row["HomeTeam"],
                    "AwayTeam": row["AwayTeam"],
                    "ActualNumeric": y_test[idx],
                    # Raw
                    "Raw_H": raw_probs_test[idx, 0],
                    "Raw_D": raw_probs_test[idx, 1],
                    "Raw_A": raw_probs_test[idx, 2],
                    # Temp
                    "Temp_H": temp_probs_test[idx, 0],
                    "Temp_D": temp_probs_test[idx, 1],
                    "Temp_A": temp_probs_test[idx, 2],
                    "Learned_Temperature": temp_scaler.temp,
                    # Platt
                    "Platt_H": platt_probs_test[idx, 0],
                    "Platt_D": platt_probs_test[idx, 1],
                    "Platt_A": platt_probs_test[idx, 2],
                    # Isotonic
                    "Iso_H": iso_probs_test[idx, 0],
                    "Iso_D": iso_probs_test[idx, 1],
                    "Iso_A": iso_probs_test[idx, 2],
                    # Bet365
                    "B365_H": b365_p[0],
                    "B365_D": b365_p[1],
                    "B365_A": b365_p[2],
                }
            )

    pred_df = pd.DataFrame(all_predictions)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(OUTPUT_PATH, index=False)

    # ============================================================
    # GLOBAL EVALUATION SUMMARY
    # ============================================================

    y_true = pred_df["ActualNumeric"].values

    methods = {
        "Raw LightGBM": ["Raw_H", "Raw_D", "Raw_A"],
        "Temperature Scaling": ["Temp_H", "Temp_D", "Temp_A"],
        "Platt (Sigmoid) Scaling": ["Platt_H", "Platt_D", "Platt_A"],
        "Isotonic Calibration": ["Iso_H", "Iso_D", "Iso_A"],
    }

    summary = []
    for name, cols in methods.items():
        probs = pred_df[cols].values
        preds = probs.argmax(axis=1)

        acc = accuracy_score(y_true, preds)
        loss = log_loss(y_true, probs, labels=[0, 1, 2])
        rps = calculate_rps(y_true, probs)

        # Multiclass Brier
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(y_true)), y_true] = 1
        brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))

        summary.append(
            {
                "Method": name,
                "Accuracy": acc,
                "LogLoss": loss,
                "BrierScore": brier,
                "RPS": rps,
            }
        )

    # Bet365 Market
    b365_mask = pred_df["B365_H"].notna()
    if b365_mask.sum() > 0:
        b365_true = y_true[b365_mask]
        b365_p = pred_df.loc[
            b365_mask, ["B365_H", "B365_D", "B365_A"]
        ].values
        b365_acc = accuracy_score(b365_true, b365_p.argmax(axis=1))
        b365_loss = log_loss(b365_true, b365_p, labels=[0, 1, 2])
        b365_rps = calculate_rps(b365_true, b365_p)
        one_hot = np.zeros_like(b365_p)
        one_hot[np.arange(len(b365_true)), b365_true] = 1
        b365_brier = np.mean(np.sum((b365_p - one_hot) ** 2, axis=1))

        summary.append(
            {
                "Method": "Bet365 Market Benchmark",
                "Accuracy": b365_acc,
                "LogLoss": b365_loss,
                "BrierScore": b365_brier,
                "RPS": b365_rps,
            }
        )

    summary_df = pd.DataFrame(summary)

    print("\n" + "=" * 70)
    print("PROBABILITY CALIBRATION BENCHMARK RESULTS")
    print("=" * 70)
    print(
        summary_df.to_string(
            index=False, float_format="{:.4f}".format
        )
    )

    # Print learned temperatures per season
    temp_by_season = (
        pred_df.groupby("Season")["Learned_Temperature"].first().reset_index()
    )
    print("\nLEARNED TEMPERATURE (T) PER WALK-FORWARD FOLD")
    print(
        temp_by_season.to_string(
            index=False, float_format="{:.3f}".format
        )
    )


if __name__ == "__main__":
    main()