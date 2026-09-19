from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss

# ============================================================
# CONFIGURATION
# ============================================================

FILES = {
    "Poisson": Path("data/processed/poisson_walk_forward_predictions.csv"),
    "Elo": Path("data/processed/elo_dynamic_regression_10.csv"),
    "LGBM": Path("data/processed/lightgbm_walk_forward_predictions.csv"),
}

OUTPUT_PATH = Path("data/processed/ensemble_walk_forward_predictions.csv")

# ============================================================
# METRICS HELPER
# ============================================================


def calculate_rps(y_true_str: pd.Series, probs_adh: np.ndarray) -> float:
    """Calculates Ranked Probability Score (RPS) expecting [A, D, H] order."""
    res_map = {"A": 2, "D": 1, "H": 0}
    y_num = np.array([res_map[r] for r in y_true_str])
    probs_hda = probs_adh[:, [2, 1, 0]]  # Reorder to [H, D, A]

    rps_list = []
    for i in range(len(y_num)):
        obs = y_num[i]
        p_cum = np.cumsum(probs_hda[i])
        e_cum = np.array([1 if obs <= 0 else 0, 1 if obs <= 1 else 0])
        rps_list.append(0.5 * np.sum((p_cum[:2] - e_cum) ** 2))

    return float(np.mean(rps_list))


# ============================================================
# WEIGHT OPTIMIZER (LOG LOSS MINIMIZATION)
# ============================================================


def optimize_weights(
    probs_list: list[np.ndarray], y_true: pd.Series
) -> np.ndarray:
    """Finds optimal weights (w1, w2, w3) that minimize Log Loss subject to sum(w) = 1 and w >= 0."""
    n_models = len(probs_list)

    # Target encoding
    res_map = {"A": 0, "D": 1, "H": 2}
    y_num = np.array([res_map[r] for r in y_true])

    def loss_func(weights):
        # Normalize weights so they sum to 1.0
        w = weights / np.sum(weights)
        blended_p = sum(w[i] * probs_list[i] for i in range(n_models))
        # Clip for numerical stability
        blended_p = np.clip(blended_p, 1e-12, 1.0 - 1e-12)
        return log_loss(y_num, blended_p, labels=[0, 1, 2])

    init_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]

    res = minimize(loss_func, x0=init_weights, bounds=bounds, method="L-BFGS-B")
    optimal_w = res.x / np.sum(res.x)
    return optimal_w


# ============================================================
# MAIN EXECUTION
# ============================================================


def main():
    print("Loading out-of-sample predictions from all 3 model tiers...")

    # Load and standardize column names
    rename_map = {
        "AwayProb": "AwayProbability",
        "DrawProb": "DrawProbability",
        "HomeProb": "HomeProbability",
    }

    p_df = pd.read_csv(FILES["Poisson"]).rename(columns=rename_map)
    e_df = pd.read_csv(FILES["Elo"]).rename(columns=rename_map)
    l_df = pd.read_csv(FILES["LGBM"]).rename(columns=rename_map)

    merge_keys = ["Season", "Date", "HomeTeam", "AwayTeam"]

    # Select necessary columns and add suffixes for clean merging
    p_sub = p_df[
        merge_keys
        + ["ActualResult", "AwayProbability", "DrawProbability", "HomeProbability"]
    ].rename(
        columns={
            "AwayProbability": "P_A_pois",
            "DrawProbability": "P_D_pois",
            "HomeProbability": "P_H_pois",
        }
    )

    e_sub = e_df[
        merge_keys
        + ["AwayProbability", "DrawProbability", "HomeProbability"]
    ].rename(
        columns={
            "AwayProbability": "P_A_elo",
            "DrawProbability": "P_D_elo",
            "HomeProbability": "P_H_elo",
        }
    )

    l_sub = l_df[
        merge_keys
        + ["AwayProbability", "DrawProbability", "HomeProbability"]
    ].rename(
        columns={
            "AwayProbability": "P_A_lgbm",
            "DrawProbability": "P_D_lgbm",
            "HomeProbability": "P_H_lgbm",
        }
    )

    # Align DataFrames on common matches
    merged = p_sub.merge(e_sub, on=merge_keys, how="inner").merge(
        l_sub, on=merge_keys, how="inner"
    )

    print(
        f"Successfully aligned {len(merged)} matching fixtures across all 3 models."
    )

    # Extract probability matrices [A, D, H]
    p_mat = merged[["P_A_pois", "P_D_pois", "P_H_pois"]].to_numpy()
    e_mat = merged[["P_A_elo", "P_D_elo", "P_H_elo"]].to_numpy()
    l_mat = merged[["P_A_lgbm", "P_D_lgbm", "P_H_lgbm"]].to_numpy()

    y_true = merged["ActualResult"]

    print("\nFinding optimal blending weights via Log Loss optimization...")
    weights = optimize_weights([p_mat, e_mat, l_mat], y_true)

    w_poisson, w_elo, w_lgbm = weights[0], weights[1], weights[2]

    print("\n" + "=" * 60)
    print("OPTIMAL ENSEMBLE WEIGHTS")
    print("=" * 60)
    print(f"  V1 Rolling Poisson Weight:  {w_poisson:.1%}")
    print(f"  Dynamic Elo Weight:         {w_elo:.1%}")
    print(f"  Calibrated LightGBM Weight: {w_lgbm:.1%}")

    # Generate blended probabilities
    ensemble_mat = (w_poisson * p_mat) + (w_elo * e_mat) + (w_lgbm * l_mat)

    # Save ensemble predictions
    ens_df = merged[
        merge_keys + ["ActualResult"]
    ].copy()
    ens_df["AwayProbability"] = ensemble_mat[:, 0]
    ens_df["DrawProbability"] = ensemble_mat[:, 1]
    ens_df["HomeProbability"] = ensemble_mat[:, 2]

    res_labels = ["A", "D", "H"]
    ens_df["PredictedResult"] = [
        res_labels[i] for i in np.argmax(ensemble_mat, axis=1)
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ens_df.to_csv(OUTPUT_PATH, index=False)

    # ============================================================
    # EVALUATION COMPARISON TABLE
    # ============================================================

    models_eval = {
        "V1 Rolling Poisson": p_mat,
        "Dynamic Elo (10% Reg)": e_mat,
        "Calibrated LightGBM": l_mat,
        "Ensemble Model": ensemble_mat,
    }

    summary = []
    for name, mat in models_eval.items():
        preds = [res_labels[i] for i in np.argmax(mat, axis=1)]
        acc = accuracy_score(y_true, preds)
        loss = log_loss(y_true, mat, labels=["A", "D", "H"])
        rps = calculate_rps(y_true, mat)

        summary.append(
            {
                "Model": name,
                "Accuracy": acc,
                "LogLoss": loss,
                "RPS": rps,
            }
        )

    summary_df = pd.DataFrame(summary)

    print("\n" + "=" * 60)
    print(
        f"OUT-OF-SAMPLE BENCHMARK COMPARISON ({len(merged):,} MATCHES)"
    )
    print("=" * 60)
    print(summary_df.to_string(index=False, float_format="{:.4f}".format))
    print(f"\nSaved ensemble predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()