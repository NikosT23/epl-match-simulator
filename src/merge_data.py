from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

FOOTBALL_DATA_FILE = PROCESSED_DIR / "epl_matches_2014_2026.csv"
UNDERSTAT_FILE = RAW_DIR / "understat_xg_2014_2026.csv"
OUTPUT_FILE = PROCESSED_DIR / "epl_matches_with_xg.csv"

# 1. Load datasets
football = pd.read_csv(FOOTBALL_DATA_FILE)
understat = pd.read_csv(UNDERSTAT_FILE)

print(f"Football-Data rows: {len(football)}")
print(f"Understat rows:     {len(understat)}")

# 2. Complete Team Name Mapping (All 35 EPL teams 2014–2026)
TEAM_MAPPING = {
    # Football-Data short names -> Canonical names
    "QPR": "Queens Park Rangers",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Spurs": "Tottenham",
    "Wolves": "Wolverhampton Wanderers",
    "West Brom": "West Bromwich Albion",
    "Sheffield Utd": "Sheffield United",
    "Brighton & Hove Albion": "Brighton and Hove Albion",
    "Brighton": "Brighton and Hove Albion",
    "Leicester City": "Leicester City",
    "Leicester": "Leicester City",
    "West Ham": "West Ham United",
    "Bournemouth": "AFC Bournemouth",
    "Cardiff": "Cardiff City",
    "Huddersfield": "Huddersfield Town",
    "Hull": "Hull City",
    "Ipswich": "Ipswich Town",
    "Leeds": "Leeds United",
    "Luton": "Luton Town",
    "Norwich": "Norwich City",
    "Stoke": "Stoke City",
    "Swansea": "Swansea City",
}


def normalize_team(team):
    team = str(team).strip()
    return TEAM_MAPPING.get(team, team)


football["HomeTeamNorm"] = football["HomeTeam"].apply(normalize_team)
football["AwayTeamNorm"] = football["AwayTeam"].apply(normalize_team)
understat["HomeTeamNorm"] = understat["HomeTeam"].apply(normalize_team)
understat["AwayTeamNorm"] = understat["AwayTeam"].apply(normalize_team)

# Verify canonical team sets match
fd_teams = set(football["HomeTeamNorm"])
us_teams = set(understat["HomeTeamNorm"])
diff = fd_teams.symmetric_difference(us_teams)

if diff:
    print(f"WARNING: Unmapped team names remaining: {diff}")
else:
    print("All Premier League team names match across both datasets.")

# Parse datetimes
football["Date_dt"] = pd.to_datetime(football["Date"], format="%Y-%m-%d")
understat["DateTime_dt"] = pd.to_datetime(understat["DateTime"])

# 3. Match each Understat row to Football-Data by (Home, Away) and Nearest Date
merged_rows = []

for idx, u_row in understat.iterrows():
    candidates = football[
        (football["HomeTeamNorm"] == u_row["HomeTeamNorm"])
        & (football["AwayTeamNorm"] == u_row["AwayTeamNorm"])
    ].copy()

    if candidates.empty:
        print(
            f"ERROR: No match found for {u_row['HomeTeamNorm']} vs {u_row['AwayTeamNorm']}"
        )
        continue

    candidates["DateDiff"] = (
        candidates["Date_dt"] - u_row["DateTime_dt"]
    ).abs()
    best_match = candidates.sort_values("DateDiff").iloc[0]

    merged_rows.append(
        {
            "FootballIndex": best_match.name,
            "MatchID": u_row["MatchID"],
            "HomeXG": u_row["HomeXG"],
            "AwayXG": u_row["AwayXG"],
            "DateDiffDays": best_match["DateDiff"].days,
        }
    )

match_df = pd.DataFrame(merged_rows)

# Validate exact 1-to-1 match count
assert (
    not match_df["FootballIndex"].duplicated().any()
), "Duplicate matches assigned!"
assert len(match_df) == len(
    football
), f"Expected {len(football)} matches, got {len(match_df)}"

# Join xG columns onto Football-Data DataFrame
football["MatchID"] = match_df.set_index("FootballIndex")["MatchID"]
football["HomeXG"] = match_df.set_index("FootballIndex")["HomeXG"]
football["AwayXG"] = match_df.set_index("FootballIndex")["AwayXG"]

# Clean up temp columns
merged = football.drop(
    columns=["HomeTeamNorm", "AwayTeamNorm", "Date_dt"], errors="ignore"
)

# 4. Save final master dataset
merged.to_csv(OUTPUT_FILE, index=False)

print("\n==============================")
print("SUCCESSFUL MERGE RESULTS")
print("==============================")
print(f"Master Dataset Rows:   {len(merged)}")
print(f"Matches with valid xG: {merged['HomeXG'].notna().sum()}")
print(
    f"Max Date Difference:   {match_df['DateDiffDays'].max()} days (timezones/rescheduling handled)"
)
print(f"Saved to:              {OUTPUT_FILE}")