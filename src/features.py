from pathlib import Path
import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

INPUT_PATH = Path("data/processed/epl_matches_with_xg.csv")
OUTPUT_PATH = Path("data/processed/epl_features.csv")

INITIAL_ELO = 1500
HOME_ADVANTAGE = 60
K_FACTOR = 20
ELO_REGRESSION_FACTOR = 0.25  # 25% pull to mean between seasons

ROLLING_WINDOWS = [3, 5, 10]
VENUE_WINDOWS = [5, 10]


# =========================================================
# HELPER FUNCTIONS
# =========================================================


def expected_result(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(
    home_elo: float, away_elo: float, result: str
) -> tuple[float, float]:
    expected_home = expected_result(home_elo + HOME_ADVANTAGE, away_elo)
    actual_home = 1.0 if result == "H" else (0.5 if result == "D" else 0.0)

    change = K_FACTOR * (actual_home - expected_home)
    return home_elo + change, away_elo - change


def safe_mean(values: list) -> float:
    if not values:
        return np.nan
    valid = [v for v in values if pd.notna(v)]
    return np.mean(valid) if valid else np.nan


def safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


# =========================================================
# MAIN PIPELINE
# =========================================================


def main():
    print("Loading merged match data...")
    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    # Create target result
    df["Result"] = np.where(
        df["FTHG"] > df["FTAG"],
        "H",
        np.where(df["FTHG"] < df["FTAG"], "A", "D"),
    )

    # Chronological sort
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"Matches loaded: {len(df)}")

    histories = {}
    elo_ratings = {}
    last_season_seen = {}
    feature_rows = []
    current_season = None

    def initialize_team(team: str, season: str):
        if team not in histories:
            histories[team] = {
                "goals_for": [],
                "goals_against": [],
                "xg_for": [],
                "xg_against": [],
                "home_goals_for": [],
                "home_goals_against": [],
                "home_xg_for": [],
                "home_xg_against": [],
                "away_goals_for": [],
                "away_goals_against": [],
                "away_xg_for": [],
                "away_xg_against": [],
                "form": [],
                "last_date": None,
            }

        # Initialize or reset Elo for new / returning teams
        if team not in elo_ratings:
            elo_ratings[team] = INITIAL_ELO
        elif (
            last_season_seen.get(team)
            and last_season_seen[team] != season
            and int(season[:4]) - int(last_season_seen[team][:4]) > 1
        ):
            # Team was out of EPL for >1 year: reset to baseline
            elo_ratings[team] = INITIAL_ELO

        last_season_seen[team] = season

    # Process match by match
    for i, row in df.iterrows():
        match_season = row["Season"]
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        # Season transition: Regress Elo ratings to the mean
        if current_season is not None and match_season != current_season:
            for team in elo_ratings:
                elo_ratings[team] = (
                    1 - ELO_REGRESSION_FACTOR
                ) * elo_ratings[
                    team
                ] + ELO_REGRESSION_FACTOR * INITIAL_ELO

        current_season = match_season

        initialize_team(home_team, match_season)
        initialize_team(away_team, match_season)

        home_history = histories[home_team]
        away_history = histories[away_team]

        home_elo = elo_ratings[home_team]
        away_elo = elo_ratings[away_team]

        features = {
            "HomeElo": home_elo,
            "AwayElo": away_elo,
            "EloDifference": home_elo - away_elo,
        }

        # 1. Overall Rolling Features (3, 5, 10 matches)
        for w in ROLLING_WINDOWS:
            h_gf = safe_mean(home_history["goals_for"][-w:])
            h_ga = safe_mean(home_history["goals_against"][-w:])
            a_gf = safe_mean(away_history["goals_for"][-w:])
            a_ga = safe_mean(away_history["goals_against"][-w:])

            h_xgf = safe_mean(home_history["xg_for"][-w:])
            h_xga = safe_mean(home_history["xg_against"][-w:])
            a_xgf = safe_mean(away_history["xg_for"][-w:])
            a_xga = safe_mean(away_history["xg_against"][-w:])

            features[f"HomeGoalsForLast{w}"] = h_gf
            features[f"HomeGoalsAgainstLast{w}"] = h_ga
            features[f"AwayGoalsForLast{w}"] = a_gf
            features[f"AwayGoalsAgainstLast{w}"] = a_ga

            features[f"HomeXGLast{w}"] = h_xgf
            features[f"HomeXGALast{w}"] = h_xga
            features[f"AwayXGLast{w}"] = a_xgf
            features[f"AwayXGALast{w}"] = a_xga

            features[f"HomeGoalDifferenceLast{w}"] = (
                (h_gf - h_ga) if pd.notna(h_gf) and pd.notna(h_ga) else np.nan
            )
            features[f"AwayGoalDifferenceLast{w}"] = (
                (a_gf - a_ga) if pd.notna(a_gf) and pd.notna(a_ga) else np.nan
            )
            features[f"HomeXGDifferenceLast{w}"] = (
                (h_xgf - h_xga)
                if pd.notna(h_xgf) and pd.notna(h_xga)
                else np.nan
            )
            features[f"AwayXGDifferenceLast{w}"] = (
                (a_xgf - a_xga)
                if pd.notna(a_xgf) and pd.notna(a_xga)
                else np.nan
            )

        # 2. Windowed Venue-Specific Features (Last 5 & 10 Venue Matches)
        for vw in VENUE_WINDOWS:
            features[f"HomeGoalsForAtHomeLast{vw}"] = safe_mean(
                home_history["home_goals_for"][-vw:]
            )
            features[f"HomeGoalsAgainstAtHomeLast{vw}"] = safe_mean(
                home_history["home_goals_against"][-vw:]
            )
            features[f"HomeXGAtHomeLast{vw}"] = safe_mean(
                home_history["home_xg_for"][-vw:]
            )
            features[f"HomeXGAAtHomeLast{vw}"] = safe_mean(
                home_history["home_xg_against"][-vw:]
            )

            features[f"AwayGoalsForAwayLast{vw}"] = safe_mean(
                away_history["away_goals_for"][-vw:]
            )
            features[f"AwayGoalsAgainstAwayLast{vw}"] = safe_mean(
                away_history["away_goals_against"][-vw:]
            )
            features[f"AwayXGAwayLast{vw}"] = safe_mean(
                away_history["away_xg_for"][-vw:]
            )
            features[f"AwayXGAgainstAwayLast{vw}"] = safe_mean(
                away_history["away_xg_against"][-vw:]
            )

        # 3. Form Features
        features["HomeForm"] = safe_mean(home_history["form"][-5:])
        features["AwayForm"] = safe_mean(away_history["form"][-5:])
        features["FormDifference"] = features["HomeForm"] - features["AwayForm"]

        # 4. Rest Days (Capped at 21 days)
        curr_date = row["Date"]
        h_rest = (
            (curr_date - home_history["last_date"]).days
            if home_history["last_date"]
            else np.nan
        )
        a_rest = (
            (curr_date - away_history["last_date"]).days
            if away_history["last_date"]
            else np.nan
        )

        features["HomeRestDays"] = (
            min(h_rest, 21) if pd.notna(h_rest) else np.nan
        )
        features["AwayRestDays"] = (
            min(a_rest, 21) if pd.notna(a_rest) else np.nan
        )
        features["RestDaysDifference"] = (
            features["HomeRestDays"] - features["AwayRestDays"]
        )

        # 5. Direct Matchup Features (5-Match Window)
        features["HomeAttackVsAwayDefense"] = (
            features["HomeGoalsForLast5"] - features["AwayGoalsAgainstLast5"]
        )
        features["AwayAttackVsHomeDefense"] = (
            features["AwayGoalsForLast5"] - features["HomeGoalsAgainstLast5"]
        )
        features["HomeXGAttackVsAwayXGDefense"] = (
            features["HomeXGLast5"] - features["AwayXGALast5"]
        )
        features["AwayXGAttackVsHomeXGDefense"] = (
            features["AwayXGLast5"] - features["HomeXGALast5"]
        )

        features["RecentGoalDifferenceMatchup"] = (
            features["HomeGoalDifferenceLast5"]
            - features["AwayGoalDifferenceLast5"]
        )
        features["RecentXGDifferenceMatchup"] = (
            features["HomeXGDifferenceLast5"] - features["AwayXGDifferenceLast5"]
        )

        # 6. Ratios
        features["HomeXGAttackDefenseRatio"] = safe_ratio(
            features["HomeXGLast5"], features["AwayXGALast5"]
        )
        features["AwayXGAttackDefenseRatio"] = safe_ratio(
            features["AwayXGLast5"], features["HomeXGALast5"]
        )

        feature_rows.append(features)

        # POST-MATCH STATE UPDATES (ZERO LEAKAGE)
        h_g, a_g = row["FTHG"], row["FTAG"]
        h_xg, a_xg = row["HomeXG"], row["AwayXG"]
        res = row["Result"]

        home_history["goals_for"].append(h_g)
        home_history["goals_against"].append(a_g)
        away_history["goals_for"].append(a_g)
        away_history["goals_against"].append(h_g)

        home_history["xg_for"].append(h_xg)
        home_history["xg_against"].append(a_xg)
        away_history["xg_for"].append(a_xg)
        away_history["xg_against"].append(h_xg)

        home_history["home_goals_for"].append(h_g)
        home_history["home_goals_against"].append(a_g)
        home_history["home_xg_for"].append(h_xg)
        home_history["home_xg_against"].append(a_xg)

        away_history["away_goals_for"].append(a_g)
        away_history["away_goals_against"].append(h_g)
        away_history["away_xg_for"].append(a_xg)
        away_history["away_xg_against"].append(h_xg)

        home_history["form"].append(3 if res == "H" else (1 if res == "D" else 0))
        away_history["form"].append(3 if res == "A" else (1 if res == "D" else 0))

        home_history["last_date"] = curr_date
        away_history["last_date"] = curr_date

        new_h_elo, new_a_elo = update_elo(home_elo, away_elo, res)
        elo_ratings[home_team] = new_h_elo
        elo_ratings[away_team] = new_a_elo

    # Build final DataFrame
    features_df = pd.DataFrame(feature_rows)
    output_df = pd.concat(
        [df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"Matches:  {len(output_df)}")
    print(f"Columns:  {len(output_df.columns)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()