from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/processed/epl_features.csv")
OUTPUT_DIR = Path("data/processed")

INITIAL_ELO = 1500  # Average league rating
PROMOTED_ELO = 1420  # Discounted rating for newly promoted teams
HOME_ADVANTAGE = 60  # Elo points added to Home team
K_FACTOR = 20  # Base update weight

DRAW_BASE = 0.26  # Peak draw probability when teams are equal
DRAW_SIGMA = 320.0  # Gaussian decay factor for rating gap

REGRESSION_RATES = [
    0.00,
    0.10,
    0.20,
    0.30,
]

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


# ============================================================
# DYNAMIC ELO PROBABILITIES
# ============================================================


def expected_home_win_probability(home_elo: float, away_elo: float) -> float:
    """Standard 2-way expected home win probability given home advantage."""
    adjusted_home_elo = home_elo + HOME_ADVANTAGE
    return 1.0 / (1.0 + 10.0 ** ((away_elo - adjusted_home_elo) / 400.0))


def elo_probabilities(home_elo: float, away_elo: float) -> np.ndarray:
    """Generates 3-way [Away, Draw, Home] probabilities using dynamic draw scaling."""
    adj_elo_diff = (home_elo + HOME_ADVANTAGE) - away_elo

    # Dynamic Draw Probability shrinks as rating gap widens
    draw_prob = DRAW_BASE * np.exp(-((abs(adj_elo_diff) / DRAW_SIGMA) ** 2))
    draw_prob = float(np.clip(draw_prob, 0.10, 0.30))

    # Raw expected 2-way probabilities
    expected_home = expected_home_win_probability(home_elo, away_elo)
    expected_away = 1.0 - expected_home

    # Scale win probabilities around dynamic draw likelihood
    remaining = 1.0 - draw_prob
    home_prob = remaining * expected_home
    away_prob = remaining * expected_away

    return np.array([away_prob, draw_prob, home_prob])


# ============================================================
# GOAL MARGIN MULTIPLIER & ELO UPDATES
# ============================================================


def margin_multiplier(
    home_goals: float, away_goals: float, home_elo: float, away_elo: float
) -> float:
    """FiveThirtyEight logarithmic margin of victory multiplier."""
    margin = abs(home_goals - away_goals)
    if margin == 0:
        return 1.0

    base_multiplier = np.log(margin + 1.0)
    elo_diff = abs(home_elo - away_elo)
    strength_adjustment = 2.2 / (2.2 + 0.001 * elo_diff)

    return base_multiplier * strength_adjustment


def update_elo(
    home_elo: float,
    away_elo: float,
    result: str,
    home_goals: float,
    away_goals: float,
) -> tuple[float, float]:
    """Updates Home and Away Elo ratings post-match."""
    expected_home = expected_home_win_probability(home_elo, away_elo)

    if result == "H":
        actual_home = 1.0
    elif result == "D":
        actual_home = 0.5
    else:
        actual_home = 0.0

    mult = margin_multiplier(home_goals, away_goals, home_elo, away_elo)
    change = K_FACTOR * mult * (actual_home - expected_home)

    return home_elo + change, away_elo - change


def regress_ratings(elo: dict[str, float], regression_rate: float):
    """Regresses team ratings back toward 1500 league mean during summer off-season."""
    for team in elo:
        elo[team] = INITIAL_ELO + (elo[team] - INITIAL_ELO) * (
            1.0 - regression_rate
        )


# ============================================================
# EVALUATION & METRICS
# ============================================================


def calculate_rps(y_true_str: pd.Series, probs_adh: np.ndarray) -> float:
    """Calculates Ranked Probability Score (RPS) for 3-way outcomes."""
    res_map = {"A": 2, "D": 1, "H": 0}
    y_num = np.array([res_map[r] for r in y_true_str])

    # Reorder [Away, Draw, Home] to [Home, Draw, Away]
    probs_hda = probs_adh[:, [2, 1, 0]]

    rps_list = []
    for i in range(len(y_num)):
        obs = y_num[i]
        p_cum = np.cumsum(probs_hda[i])
        e_cum = np.array([1 if obs <= 0 else 0, 1 if obs <= 1 else 0])
        rps_list.append(0.5 * np.sum((p_cum[:2] - e_cum) ** 2))

    return float(np.mean(rps_list))


def calculate_metrics(
    predictions: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """Computes Accuracy, Log Loss, Brier Score, and RPS."""
    y_true = predictions["ActualResult"]
    probs = predictions[
        ["AwayProbability", "DrawProbability", "HomeProbability"]
    ].to_numpy()
    y_pred = predictions["PredictedResult"]

    acc = accuracy_score(y_true, y_pred)
    loss = log_loss(y_true, probs, labels=["A", "D", "H"])

    # Multiclass Brier Score
    actual_one_hot = np.zeros_like(probs)
    label_to_index = {"A": 0, "D": 1, "H": 2}
    for i, result in enumerate(y_true):
        actual_one_hot[i, label_to_index[result]] = 1
    brier = np.mean(np.sum((probs - actual_one_hot) ** 2, axis=1))

    # Ranked Probability Score (RPS)
    rps = calculate_rps(y_true, probs)

    return float(acc), float(loss), float(brier), float(rps)


# ============================================================
# WALK-FORWARD TESTING LOOP
# ============================================================


def evaluate_regression_rate(
    df: pd.DataFrame, regression_rate: float
) -> pd.DataFrame:
    elo = {}
    all_predictions = []

    season_order = sorted(df["Season"].unique())
    first_test_season = TEST_SEASONS[0]

    # Pre-train Elo on historical seasons prior to test window
    historical_seasons = [
        s for s in season_order if season_order.index(s) < season_order.index(first_test_season)
    ]

    for season in historical_seasons:
        season_df = df[df["Season"] == season].sort_values("Date")
        for _, row in season_df.iterrows():
            h_team, a_team = row["HomeTeam"], row["AwayTeam"]
            if h_team not in elo:
                elo[h_team] = PROMOTED_ELO
            if a_team not in elo:
                elo[a_team] = PROMOTED_ELO

            new_h, new_a = update_elo(
                elo[h_team], elo[a_team], row["Result"], row["FTHG"], row["FTAG"]
            )
            elo[h_team], elo[a_team] = new_h, new_a

    # Expanding window walk-forward evaluation
    for test_index, season in enumerate(TEST_SEASONS):
        print(f"    Evaluating {season}...")

        # Apply off-season regression between test seasons
        if test_index > 0 and regression_rate > 0.0:
            regress_ratings(elo, regression_rate)

        test_df = df[df["Season"] == season].sort_values("Date")

        for _, row in test_df.iterrows():
            h_team, a_team = row["HomeTeam"], row["AwayTeam"]

            # Initialize newly promoted teams at PROMOTED_ELO
            if h_team not in elo:
                elo[h_team] = PROMOTED_ELO
            if a_team not in elo:
                elo[a_team] = PROMOTED_ELO

            # 1. Capture Pre-Match Ratings & Probabilities
            h_elo, a_elo = elo[h_team], elo[a_team]
            probs = elo_probabilities(h_elo, a_elo)
            pred_res = ["A", "D", "H"][np.argmax(probs)]

            all_predictions.append(
                {
                    "Season": row["Season"],
                    "Date": row["Date"],
                    "HomeTeam": h_team,
                    "AwayTeam": a_team,
                    "ActualResult": row["Result"],
                    "HomeElo": h_elo,
                    "AwayElo": a_elo,
                    "EloDifference": h_elo - a_elo,
                    "AwayProbability": probs[0],
                    "DrawProbability": probs[1],
                    "HomeProbability": probs[2],
                    "PredictedResult": pred_res,
                    "RegressionRate": regression_rate,
                }
            )

            # 2. Update Ratings Post-Match
            new_h, new_a = update_elo(
                h_elo, a_elo, row["Result"], row["FTHG"], row["FTAG"]
            )
            elo[h_team], elo[a_team] = new_h, new_a

    return pd.DataFrame(all_predictions)


# ============================================================
# MAIN EXECUTION
# ============================================================


def main():
    print("=" * 70)
    print("DYNAMIC ELO MODEL: SEASONAL REGRESSION EXPERIMENT")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    results = []

    for regression_rate in REGRESSION_RATES:
        print(f"\n==============================")
        print(f"REGRESSION RATE: {regression_rate:.0%}")
        print("==============================")

        predictions = evaluate_regression_rate(df, regression_rate)
        acc, loss, brier, rps = calculate_metrics(predictions)

        results.append(
            {
                "RegressionRate": f"{regression_rate:.0%}",
                "Matches": len(predictions),
                "Accuracy": acc,
                "LogLoss": loss,
                "BrierScore": brier,
                "RPS": rps,
            }
        )

        output_path = (
            OUTPUT_DIR / f"elo_dynamic_regression_{int(regression_rate * 100)}.csv"
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)

        print(
            f"Accuracy: {acc:.4f} | LogLoss: {loss:.4f} | RPS: {rps:.4f} | Brier: {brier:.4f}"
        )

    # Summary
    results_df = pd.DataFrame(results)
    comparison_path = OUTPUT_DIR / "elo_dynamic_regression_comparison.csv"
    results_df.to_csv(comparison_path, index=False)

    print("\n==============================")
    print("SEASONAL REGRESSION SUMMARY")
    print("==============================")
    print(results_df.to_string(index=False, float_format="{:.4f}".format))
    print(f"\nSaved comparison report to: {comparison_path}")


if __name__ == "__main__":
    main()