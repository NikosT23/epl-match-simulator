from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.metrics import accuracy_score, log_loss

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/processed/epl_matches_with_xg.csv")
OUTPUT_FILE = Path("data/processed/dixon_coles_walk_forward_predictions.csv")

MAX_GOALS = 8
XI = 0.0018  # Standard time-decay rate (~0.0018 per day)
USE_XG = True  # Fit Dixon-Coles on xG instead of raw goals


# ============================================================
# METRICS HELPER
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
# DIXON-COLES MODEL
# ============================================================


class DixonColesModel:

    def __init__(
        self,
        teams: list[str],
        xi: float = 0.0018,
        use_xg: bool = True,
        max_goals: int = 8,
    ):
        self.teams = sorted(teams)
        self.team_to_idx = {team: i for i, team in enumerate(self.teams)}
        self.n_teams = len(self.teams)
        self.xi = xi
        self.use_xg = use_xg
        self.max_goals = max_goals

        self.attack = None
        self.defence = None
        self.home_advantage = None
        self.intercept = None
        self.rho = None

    def unpack_parameters(self, params: np.ndarray):
        n = self.n_teams
        attack_free = params[: n - 1]
        defence_free = params[n - 1 : 2 * (n - 1)]

        home_advantage = params[-3]
        intercept = params[-2]
        rho = params[-1]

        attack = np.append(attack_free, -np.sum(attack_free))
        defence = np.append(defence_free, -np.sum(defence_free))

        return attack, defence, home_advantage, intercept, rho

    def negative_log_likelihood(
        self,
        params: np.ndarray,
        home_idx: np.ndarray,
        away_idx: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        attack, defence, home_advantage, intercept, rho = self.unpack_parameters(
            params
        )

        lambda_home = np.exp(
            intercept + home_advantage + attack[home_idx] + defence[away_idx]
        )
        lambda_away = np.exp(
            intercept + attack[away_idx] + defence[home_idx]
        )

        lambda_home = np.clip(lambda_home, 1e-6, 15.0)
        lambda_away = np.clip(lambda_away, 1e-6, 15.0)

        log_prob_home = (
            -lambda_home
            + home_goals * np.log(lambda_home)
            - gammaln(home_goals + 1)
        )
        log_prob_away = (
            -lambda_away
            + away_goals * np.log(lambda_away)
            - gammaln(away_goals + 1)
        )

        # Dixon-Coles tau adjustment
        correction = np.ones_like(lambda_home)
        mask_00 = (home_goals == 0) & (away_goals == 0)
        mask_01 = (home_goals == 0) & (away_goals == 1)
        mask_10 = (home_goals == 1) & (away_goals == 0)
        mask_11 = (home_goals == 1) & (away_goals == 1)

        correction[mask_00] = 1.0 - (lambda_home[mask_00] * lambda_away[mask_00] * rho)
        correction[mask_01] = 1.0 + (lambda_home[mask_01] * rho)
        correction[mask_10] = 1.0 + (lambda_away[mask_10] * rho)
        correction[mask_11] = 1.0 - rho

        if np.any(correction <= 0):
            return 1e10

        # Weighted Log Likelihood (Time-Decay)
        log_likelihood = weights * (
            log_prob_home + log_prob_away + np.log(correction)
        )
        return -np.sum(log_likelihood)

    def fit(self, matches: pd.DataFrame):
        home_idx = matches["HomeTeam"].map(self.team_to_idx).to_numpy(dtype=np.int32)
        away_idx = matches["AwayTeam"].map(self.team_to_idx).to_numpy(dtype=np.int32)

        if self.use_xg and "HomeXG" in matches.columns:
            home_goals = matches["HomeXG"].to_numpy(dtype=np.float64)
            away_goals = matches["AwayXG"].to_numpy(dtype=np.float64)
        else:
            home_goals = matches["FTHG"].to_numpy(dtype=np.float64)
            away_goals = matches["FTAG"].to_numpy(dtype=np.float64)

        # Time-decay weights relative to the latest match date in the training set
        max_date = matches["Date"].max()
        days_diff = (max_date - matches["Date"]).dt.days.to_numpy(dtype=np.float64)
        weights = np.exp(-self.xi * days_diff)

        n = self.n_teams
        initial_params = np.zeros(2 * (n - 1) + 3)
        initial_params[-3] = 0.20
        initial_params[-2] = 0.10
        initial_params[-1] = -0.05

        bounds = [(-3.0, 3.0)] * (2 * (n - 1))
        bounds.extend([(-1.0, 1.0), (-2.0, 1.0), (-0.5, 0.5)])

        result = minimize(
            self.negative_log_likelihood,
            initial_params,
            args=(home_idx, away_idx, home_goals, away_goals, weights),
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
        ) = self.unpack_parameters(result.x)

        return result

    def predict_match(
        self, home_team: str, away_team: str
    ) -> tuple[float, float, float]:
        h = self.team_to_idx[home_team]
        a = self.team_to_idx[away_team]

        lambda_home = np.exp(
            self.intercept + self.home_advantage + self.attack[h] + self.defence[a]
        )
        lambda_away = np.exp(
            self.intercept + self.attack[a] + self.defence[h]
        )

        lambda_home = min(lambda_home, 15.0)
        lambda_away = min(lambda_away, 15.0)

        home_win, draw, away_win = 0.0, 0.0, 0.0

        for hg in range(self.max_goals + 1):
            home_p = (
                np.exp(-lambda_home)
                * (lambda_home**hg)
                / np.exp(gammaln(hg + 1))
            )
            for ag in range(self.max_goals + 1):
                away_p = (
                    np.exp(-lambda_away)
                    * (lambda_away**ag)
                    / np.exp(gammaln(ag + 1))
                )
                prob = home_p * away_p

                correction = 1.0
                if hg == 0 and ag == 0:
                    correction = 1.0 - (lambda_home * lambda_away * self.rho)
                elif hg == 0 and ag == 1:
                    correction = 1.0 + (lambda_home * self.rho)
                elif hg == 1 and ag == 0:
                    correction = 1.0 + (lambda_away * self.rho)
                elif hg == 1 and ag == 1:
                    correction = 1.0 - self.rho

                prob *= correction

                if hg > ag:
                    home_win += prob
                elif hg == ag:
                    draw += prob
                else:
                    away_win += prob

        total = home_win + draw + away_win
        return home_win / total, draw / total, away_win / total


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = (
    df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    .sort_values("Date")
    .reset_index(drop=True)
)

print(f"Loaded matches: {len(df)}")
print(f"Seasons:        {df['Season'].nunique()}")


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

seasons = sorted(df["Season"].unique())
all_predictions = []

print("\n==============================")
print("WALK-FORWARD VALIDATION")
print("==============================\n")

teams = sorted(set(df["HomeTeam"]).union(set(df["AwayTeam"])))

for i in range(1, len(seasons)):
    train_seasons = seasons[:i]
    test_season = seasons[i]

    train = df[df["Season"].isin(train_seasons)].copy()
    test = df[df["Season"] == test_season].copy()

    print(f"Testing {test_season}: Trained on {len(train)} matches | Testing on {len(test)} matches")

    model = DixonColesModel(teams=teams, xi=XI, use_xg=USE_XG, max_goals=MAX_GOALS)
    model.fit(train)

    for _, row in test.iterrows():
        h_prob, d_prob, a_prob = model.predict_match(
            row["HomeTeam"], row["AwayTeam"]
        )

        probs = {"H": h_prob, "D": d_prob, "A": a_prob}
        predicted_result = max(probs, key=probs.get)

        actual_result = (
            "H"
            if row["FTHG"] > row["FTAG"]
            else ("D" if row["FTHG"] == row["FTAG"] else "A")
        )

        # Bet365 Implied Odds Benchmark
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
                "Season": test_season,
                "Date": row["Date"],
                "HomeTeam": row["HomeTeam"],
                "AwayTeam": row["AwayTeam"],
                "ActualResult": actual_result,
                "HomeProbability": h_prob,
                "DrawProbability": d_prob,
                "AwayProbability": a_prob,
                "PredictedResult": predicted_result,
                "B365_HomeProb": b365_probs[0],
                "B365_DrawProb": b365_probs[1],
                "B365_AwayProb": b365_probs[2],
            }
        )


# ============================================================
# RESULTS & METRICS
# ============================================================

predictions = pd.DataFrame(all_predictions)

result_map = {"H": 0, "D": 1, "A": 2}
y_true_numeric = predictions["ActualResult"].map(result_map).to_numpy()
y_pred_numeric = predictions["PredictedResult"].map(result_map).to_numpy()

probabilities = predictions[
    ["HomeProbability", "DrawProbability", "AwayProbability"]
].to_numpy()

accuracy = accuracy_score(y_true_numeric, y_pred_numeric)
logloss = log_loss(y_true_numeric, probabilities, labels=[0, 1, 2])
rps = calculate_rps(y_true_numeric, probabilities)

# Bet365 Benchmark Comparison
b365_mask = predictions["B365_HomeProb"].notna()
b365_true = y_true_numeric[b365_mask]
b365_probs_mat = predictions.loc[
    b365_mask, ["B365_HomeProb", "B365_DrawProb", "B365_AwayProb"]
].to_numpy()

b365_rps = calculate_rps(b365_true, b365_probs_mat)
b365_loss = log_loss(b365_true, b365_probs_mat, labels=[0, 1, 2])

print("\n==============================")
print("WALK-FORWARD DIXON-COLES RESULTS")
print("==============================")
print(f"Matches evaluated: {len(predictions)}")
print(f"DC Model Accuracy: {accuracy:.4f}")
print(f"DC Model Log Loss: {logloss:.4f}  |  B365 Log Loss: {b365_loss:.4f}")
print(f"DC Model RPS:      {rps:.4f}  |  B365 RPS:      {b365_rps:.4f}")


# ============================================================
# RESULTS BY SEASON
# ============================================================

print("\nRESULTS BY SEASON")

season_results = []
for season, group in predictions.groupby("Season"):
    s_y_true = group["ActualResult"].map(result_map).to_numpy()
    s_probs = group[["HomeProbability", "DrawProbability", "AwayProbability"]].to_numpy()
    s_y_pred = group["PredictedResult"].map(result_map).to_numpy()

    s_accuracy = accuracy_score(s_y_true, s_y_pred)
    s_logloss = log_loss(s_y_true, s_probs, labels=[0, 1, 2])
    s_rps = calculate_rps(s_y_true, s_probs)

    s_b365_mask = group["B365_HomeProb"].notna()
    if s_b365_mask.sum() > 0:
        s_b365_rps = calculate_rps(
            s_y_true[s_b365_mask],
            group.loc[s_b365_mask, ["B365_HomeProb", "B365_DrawProb", "B365_AwayProb"]].to_numpy(),
        )
    else:
        s_b365_rps = np.nan

    season_results.append(
        {
            "Season": season,
            "Matches": len(group),
            "Accuracy": s_accuracy,
            "LogLoss": s_logloss,
            "DC_RPS": s_rps,
            "B365_RPS": s_b365_rps,
        }
    )

season_results_df = pd.DataFrame(season_results)
print(
    season_results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "LogLoss": "{:.4f}".format,
            "DC_RPS": "{:.4f}".format,
            "B365_RPS": "{:.4f}".format,
        },
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(OUTPUT_FILE, index=False)

print("\n==============================")
print("PREDICTIONS SAVED")
print("==============================")
print(f"Output: {OUTPUT_FILE}")