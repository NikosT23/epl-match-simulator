from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Standardized mapping: Football-Data.co.uk -> Official / Understat Name
TEAM_NAME_MAP = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",
    "Spurs": "Tottenham",
    "Wolves": "Wolverhampton Wanderers",
    "West Brom": "West Bromwich Albion",
    "Brighton": "Brighton and Hove Albion",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Newcastle": "Newcastle United",
    "Norwich": "Norwich City",
    "Luton": "Luton Town",
    "Huddersfield": "Huddersfield Town",
    "Cardiff": "Cardiff City",
}

# Expanded column list retaining rich match statistics for ML feature engineering
TARGET_COLUMNS = [
    "Date",
    "Season",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "HTHG",
    "HTAG",
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

files = sorted(RAW_DATA_DIR.glob("EPL_*.csv"))
dfs = []

for file in files:
    season = file.stem.replace("EPL_", "")
    df = pd.read_csv(file)

    if "Season" not in df.columns:
        df["Season"] = season

    df = df.dropna(how="all")

    cols_to_keep = [col for col in TARGET_COLUMNS if col in df.columns]
    df = df[cols_to_keep]

    dfs.append(df)

matches = pd.concat(dfs, ignore_index=True)

# Remove unplayed / postponed matches
matches = matches.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])

# Map team names to standardized names
matches["HomeTeam"] = matches["HomeTeam"].replace(TEAM_NAME_MAP)
matches["AwayTeam"] = matches["AwayTeam"].replace(TEAM_NAME_MAP)

# Convert goals to integers
matches["FTHG"] = matches["FTHG"].astype(int)
matches["FTAG"] = matches["FTAG"].astype(int)

# Numeric conversions for betting odds
for col in ["B365H", "B365D", "B365A"]:
    if col in matches.columns:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")

# Fix: Explicit string format parsing to avoid day/month swapping
matches["Date"] = pd.to_datetime(matches["Date"], format="%Y-%m-%d")

# Deterministic sorting
matches = matches.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)

output_path = PROCESSED_DATA_DIR / "epl_matches_2014_2026.csv"
matches.to_csv(output_path, index=False)

print(f"Total matches: {len(matches)}")
print(f"Seasons: {matches['Season'].nunique()}")
print("\nMatches per season:")
print(matches.groupby("Season").size())