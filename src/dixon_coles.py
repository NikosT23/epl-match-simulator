import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/processed/epl_matches_with_xg.csv")
OUTPUT_FILE = Path("data/processed/dixon_coles_predictions.csv")

MAX_GOALS = 8


# ============================================================
# METRICS HELPER
# ============================================================


def calculate_rps(y_true_numeric: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculates Ranked Probability Score (RPS) for 3-way outcomes."""
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
# DIXON-COLES CLASS WITH TIME DECAY
# ============================================================


class DixonColesModel:
    """Dixon-Coles Bivariate Poisson Model with Time-Decay Weighting."""

    def __init__(
        self,
        xi: float = 0.0018,
        use_xg: bool = True,
        max_goals: int = 8,
    ):
        self.xi = xi  # Time-decay rate per day
        self.use_xg = use_xg
        self.max_goals = max_goals

        self.teams = []
        self.team_to_idx = {}
        self.attack = None
        self.defence = None
        self.home_advantage = 0.20
        self.intercept = 0.10
        self.rho = -0.05

    def _tau(
        self,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        lambda_h: np.ndarray,
        lambda_a: np.ndarray,
        rho: float,
    ) -> np.ndarray:
        """Dixon-Coles tau factor adjustment for low-scoring match outcomes."""
        correction = np.ones_like(lambda_h)

        mask_00 = (home_goals == 0) & (away_goals == 0)
        mask_01 = (home_goals == 0) & (away_goals == 1)
        mask_10 = (home_goals == 1) & (away_goals == 0)
        mask_11 = (home_goals == 1) & (away_goals == 1)

        correction[mask_00] = 1.0 - (lambda_h[mask_00] * lambda_a[mask_00] * rho)
        correction[mask_01] = 1.0 + (lambda_h[mask_01] * rho)
        correction[mask_10] = 1.0 + (lambda_a[mask_10] * rho)
        correction[mask_11] = 1.0 - rho

        return correction

    def _unpack_parameters(self, params: np.ndarray, n_teams: int):
        attack_free = params[: n_teams - 1]
        defence_free = params[n_teams - 1 : 2 * (n_teams - 1)]

        home_advantage = params[-3]
        intercept = params[-2]
        rho = params[-1]

        attack = np.append(attack_free, -np.sum(attack_free))
        defence = np.append(defence_free, -np.sum(defence_free))

        return attack, defence, home_advantage, intercept, rho

    def fit(self, df: pd.DataFrame):
        """Fit Dixon-Coles parameters using Maximum Likelihood Estimation."""
        self.teams = sorted(set(df["HomeTeam"]).union(set(df["AwayTeam"])))
        self.team_to_idx = {t: i for i, t in enumerate(self.teams)}
        n_teams = len(self.teams)

        home_idx = df["HomeTeam"].map(self.team_to_idx).to_numpy(dtype=np.int32)
        away_idx = df["AwayTeam"].map(self.team_to_idx).to_numpy(dtype=np.int32)

        # Use xG or actual goals based on config
        if self.use_xg and "HomeXG" in df.columns:
            home_g = df["HomeXG"].to_numpy(dtype=np.float64)
            away_g = df["AwayXG"].to_numpy(dtype=np.float64)
        else:
            home_g = df["FTHG"].to_numpy(dtype=np.float64)
            away_g = df["FTAG"].to_numpy(dtype=np.float64)

        # Time decay weights
        max_date = df["Date"].max()
        days_diff = (max_date - df["Date"]).dt.days.to_numpy(dtype=np.float64)
        weights = np.exp(-self.xi * days_diff)

        def nll(params):
            att, deff, ha, inter, r = self._unpack_parameters(params, n_teams)

            lambda_h = np.exp(inter + ha + att[home_idx] + deff[away_idx])
            lambda_a = np.exp(inter + att[away_idx] + deff[home_idx])

            lambda_h = np.clip(lambda_h, 1e-6, 15.0)
            lambda_a = np.clip(lambda_a, 1e-6, 15.0)

            # Weighted Poisson log-likelihood
            log_p_h = -lambda_h + home_g * np.log(lambda_h) - gammaln(home_g + 1)
            log_p_a = -lambda_a + away_g * np.log(lambda_a) - gammaln(away_g + 1)

            corr = self._tau(
                np.round(home_g), np.round(away_g), lambda_h, lambda_a, r
            )
            if np.any(corr <= 0):
                return 1e10

            log_lik = weights * (log_p_h + log_p_a + np.log(corr))
            return -np.sum(log_lik)

        # Initial parameters & bounds
        init_params = np.zeros(2 * (n_teams - 1) + 3)
        init_params[-3] = 0.20
        init_params[-2] = 0.10
        init_params[-1] = -0.05

        bounds = [(-3.0, 3.0)] * (2 * (n_teams - 1))
        bounds.extend([(-1.0, 1.0), (-2.0, 1.0), (-0.5, 0.5)])

        res = minimize(
            nll,
            init_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        (
            self.attack,
            self.defence,
            self.home_advantage,
            self.intercept,
            self.rho,
        ) = self._unpack_parameters(res.x, n_teams)

        return self

    def predict_match(
        self, home_team: str, away_team: str
    ) -> tuple[float, float, float]:
        """Generate match probabilities for a given home and away team."""
        # Fallback for teams not in training set
        h_idx = self.team_to_idx.get(home_team, None)
        a_idx = self.team_to_idx.get(away_team, None)

        att_h = self.attack[h_idx] if h_idx is not None else 0.0
        def_h = self.defence[h_idx] if h_idx is not None else 0.0
        att_a = self.attack[a_idx] if a_idx is not None else 0.0
        def_a = self.defence[a_idx] if a_idx is not None else 0.0

        lh = math.exp(self.intercept + self.home_advantage + att_h + def_a)
        la = math.exp(self.intercept + att_a + def_h)

        home_win, draw, away_win = 0.0, 0.0, 0.0

        for hg in range(self.max_goals + 1):
            p_h = (math.exp(-lh) * (lh**hg)) / math.factorial(hg)
            for ag in range(self.max_goals + 1):
                p_a = (math.exp(-la) * (la**ag)) / math.factorial(ag)
                prob = p_h * p_a

                # Tau correction
                corr = 1.0
                if hg == 0 and ag == 0:
                    corr = 1.0 - (lh * la * self.rho)
                elif hg == 0 and ag == 1:
                    corr = 1.0 + (lh * self.rho)
                elif hg == 1 and ag == 0:
                    corr = 1.0 + (la * self.rho)
                elif hg == 1 and ag == 1:
                    corr = 1.0 - self.rho

                prob *= corr

                if hg > ag:
                    home_win += prob
                elif hg == ag:
                    draw += prob
                else:
                    away_win += prob

        total = home_win + draw + away_win
        return home_win / total, draw / total, away_win / total

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probabilities matrix for a DataFrame of fixtures."""
        probs = []
        for _, row in df.iterrows():
            hw, d, aw = self.predict_match(row["HomeTeam"], row["AwayTeam"])
            probs.append([hw, d, aw])
        return np.array(probs)


# ============================================================
# MAIN EXECUTION
# ============================================================


def main():
    print("Loading match dataset...")
    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = (
        df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    print(f"Loaded matches: {len(df)}")

    # Fit Dixon-Coles model
    dc_model = DixonColesModel(xi=0.0018, use_xg=True, max_goals=MAX_GOALS)
    dc_model.fit(df)

    print("\n==============================")
    print("DIXON-COLES FITTED PARAMETERS")
    print("==============================")
    print(f"Home advantage: {dc_model.home_advantage:.4f}")
    print(f"Intercept:      {dc_model.intercept:.4f}")
    print(f"Rho:            {dc_model.rho:.4f}")

    # Generate predictions
    probabilities = dc_model.predict_proba(df)

    result_map = {"H": 0, "D": 1, "A": 2}
    y_true = (
        np.where(
            df["FTHG"] > df["FTAG"],
            "H",
            np.where(df["FTHG"] < df["FTAG"], "A", "D"),
        )
    )
    y_true_numeric = np.array([result_map[r] for r in y_true])
    y_pred = probabilities.argmax(axis=1)

    acc = accuracy_score(y_true_numeric, y_pred)
    loss = log_loss(y_true_numeric, probabilities, labels=[0, 1, 2])
    rps = calculate_rps(y_true_numeric, probabilities)

    print("\n==============================")
    print("DIXON-COLES IN-SAMPLE RESULTS")
    print("==============================")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {loss:.4f}")
    print(f"RPS:      {rps:.4f}")

    # Save output predictions
    pred_df = pd.DataFrame(
        {
            "Date": df["Date"],
            "HomeTeam": df["HomeTeam"],
            "AwayTeam": df["AwayTeam"],
            "Actual": y_true,
            "HomeProb": probabilities[:, 0],
            "DrawProb": probabilities[:, 1],
            "AwayProb": probabilities[:, 2],
        }
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved Dixon-Coles predictions to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()