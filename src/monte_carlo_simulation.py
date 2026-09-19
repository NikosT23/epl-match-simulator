from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

PREDICTION_FILE = Path("data/processed/lightgbm_walk_forward_predictions.csv")
TARGET_SEASON = "2025-26"
NUM_SIMULATIONS = 10000
RANDOM_SEED = 42

# Mode Options:
#  - "FULL_SEASON": Simulates all 380 matches from GW1 using predicted probabilities
#  - "MID_SEASON": Treats matches before CUTOFF_DATE as played, simulates the rest
SIMULATION_MODE = "FULL_SEASON"
CUTOFF_DATE = "2026-01-01"  # Only used if SIMULATION_MODE == "MID_SEASON"


# ============================================================
# HELPER: STANDINGS GENERATOR
# ============================================================

def build_current_table(played_matches: pd.DataFrame, teams: list) -> pd.Series:
    """Builds current points baseline from completed matches."""
    points = {team: 0 for team in teams}

    for _, row in played_matches.iterrows():
        res = row.get("ActualResult", np.nan)
        if pd.isna(res):
            continue

        home, away = row["HomeTeam"], row["AwayTeam"]
        if res == "H":
            points[home] += 3
        elif res == "D":
            points[home] += 1
            points[away] += 1
        elif res == "A":
            points[away] += 3

    return pd.Series(points)


# ============================================================
# VECTORIZED MONTE CARLO ENGINE
# ============================================================

def run_monte_carlo(
    teams: list,
    base_points: pd.Series,
    remaining_fixtures: pd.DataFrame,
    n_sims: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Runs N vectorized season simulations across remaining fixtures."""
    np.random.seed(seed)

    n_teams = len(teams)
    n_fixtures = len(remaining_fixtures)
    team_to_idx = {team: i for i, team in enumerate(teams)}

    base_pts_vec = np.array([base_points[team] for team in teams], dtype=float)

    if n_fixtures == 0:
        print("\nWARNING: 0 remaining fixtures to simulate!")
        final_points_matrix = np.tile(base_pts_vec, (n_sims, 1))
    else:
        home_indices = np.array(
            [team_to_idx[t] for t in remaining_fixtures["HomeTeam"]]
        )
        away_indices = np.array(
            [team_to_idx[t] for t in remaining_fixtures["AwayTeam"]]
        )

        p_away = remaining_fixtures["AwayProbability"].to_numpy()
        p_draw = remaining_fixtures["DrawProbability"].to_numpy()

        cum_p_away = p_away
        cum_p_draw = p_away + p_draw

        # Uniform random sampling across (N_sims, N_fixtures)
        rand_matrix = np.random.uniform(0.0, 1.0, size=(n_sims, n_fixtures))

        is_away_win = rand_matrix < cum_p_away
        is_draw = (rand_matrix >= cum_p_away) & (rand_matrix < cum_p_draw)
        is_home_win = rand_matrix >= cum_p_draw

        home_pts_mat = (is_home_win * 3) + (is_draw * 1)
        away_pts_mat = (is_away_win * 3) + (is_draw * 1)

        simulated_points_delta = np.zeros((n_sims, n_teams), dtype=float)

        for f_idx in range(n_fixtures):
            h_i = home_indices[f_idx]
            a_i = away_indices[f_idx]
            simulated_points_delta[:, h_i] += home_pts_mat[:, f_idx]
            simulated_points_delta[:, a_i] += away_pts_mat[:, f_idx]

        final_points_matrix = base_pts_vec + simulated_points_delta

    # Rank calculation (Inverted argsort: 1 = 1st place)
    ranks = n_teams - np.argsort(np.argsort(final_points_matrix, axis=1), axis=1)

    summary = []
    for i, team in enumerate(teams):
        team_pts = final_points_matrix[:, i]
        team_ranks = ranks[:, i]

        summary.append(
            {
                "Team": team,
                "BaselinePts": int(base_points[team]),
                "xPts": np.mean(team_pts),
                "StdPts": np.std(team_pts),
                "Pts_10th": np.percentile(team_pts, 10),
                "Pts_50th": np.percentile(team_pts, 50),
                "Pts_90th": np.percentile(team_pts, 90),
                "TitleProb_%": np.mean(team_ranks == 1) * 100.0,
                "Top4Prob_%": np.mean(team_ranks <= 4) * 100.0,
                "RelegationProb_%": np.mean(team_ranks >= 18) * 100.0,
                "AvgRank": np.mean(team_ranks),
            }
        )

    return (
        pd.DataFrame(summary)
        .sort_values("xPts", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 70)
    print(f"MONTE CARLO SEASON SIMULATOR ({NUM_SIMULATIONS:,} RUNS)")
    print("=" * 70)
    print(f"Prediction File: {PREDICTION_FILE}")
    print(f"Simulation Mode: {SIMULATION_MODE}")

    df = pd.read_csv(PREDICTION_FILE)

    rename_map = {
        "HomeProb": "HomeProbability",
        "DrawProb": "DrawProbability",
        "AwayProb": "AwayProbability",
        "Actual": "ActualResult",
    }
    df = df.rename(columns=rename_map)

    season_df = df[df["Season"] == TARGET_SEASON].copy()
    season_df["Date"] = pd.to_datetime(season_df["Date"], errors="coerce")

    if len(season_df) == 0:
        print(f"ERROR: No matches found for Season {TARGET_SEASON}")
        return

    teams = sorted(
        list(set(season_df["HomeTeam"]).union(set(season_df["AwayTeam"])))
    )

    if SIMULATION_MODE == "FULL_SEASON":
        played_matches = pd.DataFrame(columns=season_df.columns)
        remaining_fixtures = season_df.copy()
    else: # MID_SEASON
        cutoff = pd.to_datetime(CUTOFF_DATE)
        played_matches = season_df[season_df["Date"] < cutoff].copy()
        remaining_fixtures = season_df[season_df["Date"] >= cutoff].copy()

    print(f"Teams: {len(teams)} | Played: {len(played_matches)} | Simulating: {len(remaining_fixtures)}")

    base_points = build_current_table(played_matches, teams)

    sim_results = run_monte_carlo(
        teams=teams,
        base_points=base_points,
        remaining_fixtures=remaining_fixtures,
        n_sims=NUM_SIMULATIONS,
        seed=RANDOM_SEED,
    )

    print("\n" + "=" * 85)
    print(f"MONTE CARLO SIMULATION PROJECTIONS ({TARGET_SEASON} - {SIMULATION_MODE})")
    print("=" * 85)

    formatters = {
        "xPts": "{:.1f}".format,
        "StdPts": "{:.1f}".format,
        "Pts_10th": "{:.0f}".format,
        "Pts_50th": "{:.0f}".format,
        "Pts_90th": "{:.0f}".format,
        "TitleProb_%": "{:.1f}%".format,
        "Top4Prob_%": "{:.1f}%".format,
        "RelegationProb_%": "{:.1f}%".format,
        "AvgRank": "{:.1f}".format,
    }

    print(sim_results.to_string(index=False, formatters=formatters))


if __name__ == "__main__":
    main()