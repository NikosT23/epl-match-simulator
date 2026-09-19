from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

FILES = {
    "Poisson": Path("data/processed/poisson_walk_forward_predictions.csv"),
    "Dynamic_Elo": Path("data/processed/elo_dynamic_regression_10.csv"),
    "Calibrated_LGBM": Path("data/processed/lightgbm_walk_forward_predictions.csv"),
}

TARGET_SEASON = "2025-26"
NUM_SIMULATIONS = 10000
RANDOM_SEED = 42

def run_simulation(pred_file: Path) -> pd.Series:
    df = pd.read_csv(pred_file)
    rename_map = {"HomeProb": "HomeProbability", "DrawProb": "DrawProbability", "AwayProb": "AwayProbability"}
    df = df.rename(columns=rename_map)
    
    season_df = df[df["Season"] == TARGET_SEASON].copy()
    teams = sorted(list(set(season_df["HomeTeam"]).union(set(season_df["AwayTeam"]))))
    n_teams, n_fixtures = len(teams), len(season_df)
    team_to_idx = {team: i for i, team in enumerate(teams)}
    
    home_indices = np.array([team_to_idx[t] for t in season_df["HomeTeam"]])
    away_indices = np.array([team_to_idx[t] for t in season_df["AwayTeam"]])
    
    p_away = season_df["AwayProbability"].to_numpy()
    p_draw = season_df["DrawProbability"].to_numpy()
    cum_p_away, cum_p_draw = p_away, p_away + p_draw
    
    np.random.seed(RANDOM_SEED)
    rand_mat = np.random.uniform(0.0, 1.0, size=(NUM_SIMULATIONS, n_fixtures))
    
    is_away = rand_mat < cum_p_away
    is_draw = (rand_mat >= cum_p_away) & (rand_mat < cum_p_draw)
    is_home = rand_mat >= cum_p_draw
    
    home_pts = (is_home * 3) + (is_draw * 1)
    away_pts = (is_away * 3) + (is_draw * 1)
    
    sim_pts = np.zeros((NUM_SIMULATIONS, n_teams), dtype=float)
    for f in range(n_fixtures):
        sim_pts[:, home_indices[f]] += home_pts[:, f]
        sim_pts[:, away_indices[f]] += away_pts[:, f]
        
    xpts = np.mean(sim_pts, axis=0)
    return pd.Series(xpts, index=teams)

def main():
    results = {}
    for name, path in FILES.items():
        if path.exists():
            results[name] = run_simulation(path)
            
    comp_df = pd.DataFrame(results)
    print("=" * 70)
    print("MONTE CARLO PROJECTIONS COMPARISON (2025-26)")
    print("=" * 70)
    print(comp_df.to_string(float_format="{:.1f}".format))

if __name__ == "__main__":
    main()