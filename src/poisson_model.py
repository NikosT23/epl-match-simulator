from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/processed/epl_features.csv")
OUTPUT_FILE = Path("data/processed/poisson_predictions.csv")

MAX_GOALS = 8


# ============================================================
# METRICS HELPER: RANKED PROBABILITY SCORE (RPS)
# ============================================================


def calculate_rps(y_true_numeric: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculates Ranked Probability Score (RPS) for 3-way outcomes.

    y_true_numeric: Array of actual outcomes (0 = Home, 1 = Draw, 2 = Away)
    y_prob: Matrix of predicted probabilities [[P_home, P_draw, P_away], ...]
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
# PURE INDEPENDENT POISSON MODEL CLASS (V1)
# ============================================================


class SimplePoissonModel:
    """Pure Independent Poisson Model for V1 Baseline.

    Fits lambda_home and lambda_away independently using basic xG features.
    """

    def __init__(self, max_goals: int = 8):
        self.max_goals = max_goals

        self.home_feature_cols = [
            "HomeXGLast5",
            "AwayXGALast5",
            "EloDifference",
        ]

        self.away_feature_cols = [
            "AwayXGLast5",
            "HomeXGALast5",
            "EloDifference",
        ]

        self.home_regressor = PoissonRegressor(alpha=1e-3, max_iter=300)
        self.away_regressor = PoissonRegressor(alpha=1e-3, max_iter=300)

    def fit(self, train_df: pd.DataFrame):
        """Fit simple Poisson GLM regressions for Home and Away expected goals."""
        X_home = train_df[self.home_feature_cols]
        y_home = train_df["FTHG"]

        X_away = train_df[self.away_feature_cols]
        y_away = train_df["FTAG"]

        self.home_regressor.fit(X_home, y_home)
        self.away_regressor.fit(X_away, y_away)
        return self

    def predict_lambdas(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict expected goal parameters (lambda_home, lambda_away)."""
        X_home = df[self.home_feature_cols]
        X_away = df[self.away_feature_cols]

        lambda_home = np.clip(self.home_regressor.predict(X_home), 0.05, 6.0)
        lambda_away = np.clip(self.away_regressor.predict(X_away), 0.05, 6.0)

        return lambda_home, lambda_away

    def calculate_match_probabilities(
        self, lambda_h: float, lambda_a: float
    ) -> np.ndarray:
        """Pure Independent Poisson 3-way probability calculation."""
        home_win, draw, away_win = 0.0, 0.0, 0.0

        for h in range(self.max_goals + 1):
            for a in range(self.max_goals + 1):
                prob = poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)

                if h > a:
                    home_win += prob
                elif h == a:
                    draw += prob
                else:
                    away_win += prob

        total_prob = home_win + draw + away_win
        return np.array(
            [
                home_win / total_prob,
                draw / total_prob,
                away_win / total_prob,
            ]
        )

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Generate match probabilities matrix [[P_home, P_draw, P_away], ...]."""
        lambda_home, lambda_away = self.predict_lambdas(df)
        probs = []

        for lh, la in zip(lambda_home, lambda_away):
            match_probs = self.calculate_match_probabilities(lh, la)
            probs.append(match_probs)

        return np.array(probs)


# ============================================================
# MAIN EXECUTION & EVALUATION
# ============================================================


def main():
    print("Loading engineered features dataset...")
    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    required_cols = [
        "HomeXGLast5",
        "AwayXGALast5",
        "AwayXGLast5",
        "HomeXGALast5",
        "EloDifference",
        "FTHG",
        "FTAG",
        "Result",
    ]

    model_df = df.dropna(subset=required_cols).copy().reset_index(drop=True)
    print(f"Loaded matches:    {len(df)}")
    print(f"Evaluated matches: {len(model_df)}")

    # Initialize and fit Pure Simple Poisson model
    model = SimplePoissonModel(max_goals=MAX_GOALS)
    model.fit(model_df)

    # Predict probabilities and expected goals
    probabilities = model.predict_proba(model_df)
    lambda_h, lambda_a = model.predict_lambdas(model_df)

    # Target Encoding
    result_mapping = {"H": 0, "D": 1, "A": 2}
    y_true = model_df["Result"].map(result_mapping).values
    y_pred = probabilities.argmax(axis=1)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    loss = log_loss(y_true, probabilities, labels=[0, 1, 2])
    rps = calculate_rps(y_true, probabilities)

    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), y_true] = 1
    brier = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))

    print("\n==============================")
    print("V1 PURE POISSON MODEL RESULTS")
    print("==============================")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Log Loss:    {loss:.4f}")
    print(f"RPS:         {rps:.4f}")
    print(f"Brier Score: {brier:.4f}")

    # Build predictions dataframe
    pred_df = pd.DataFrame(
        {
            "Date": model_df["Date"],
            "Season": model_df["Season"],
            "HomeTeam": model_df["HomeTeam"],
            "AwayTeam": model_df["AwayTeam"],
            "Actual": model_df["Result"],
            "ActualNumeric": y_true,
            "HomeProb": probabilities[:, 0],
            "DrawProb": probabilities[:, 1],
            "AwayProb": probabilities[:, 2],
            "ExpectedHomeGoals": lambda_h,
            "ExpectedAwayGoals": lambda_a,
        }
    )

    # Save output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved V1 predictions to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()