import time
from pathlib import Path
import pandas as pd
from understatapi import UnderstatClient

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Standardized mapping to align Understat names with our Football-Data dataset
UNDERSTAT_TEAM_MAP = {
    "Brighton": "Brighton and Hove Albion",
    "Cardiff": "Cardiff City",
    "Huddersfield": "Huddersfield Town",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Luton": "Luton Town",
    "Newcastle United": "Newcastle United",
    "Norwich": "Norwich City",
    "Spurs": "Tottenham",
    "West Bromwich Albion": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
}

SEASONS = range(2014, 2026)
all_matches = {}

with UnderstatClient() as understat:
    for season in SEASONS:
        print(f"Fetching EPL season {season}/{str(season + 1)[-2:]}...")
        try:
            matches = understat.league(league="EPL").get_match_data(
                season=str(season)
            )

            for match in matches:
                # Keep completed matches only
                if not match.get("isResult"):
                    continue

                match_id = match["id"]
                all_matches[match_id] = match

            time.sleep(0.5)

        except Exception as e:
            print(f"ERROR downloading season {season}: {e}")

rows = []
for match in all_matches.values():
    rows.append(
        {
            "MatchID": match["id"],
            "DateTime": match["datetime"],
            "HomeTeam": match["h"]["title"],
            "AwayTeam": match["a"]["title"],
            "HomeGoals": int(match["goals"]["h"]),
            "AwayGoals": int(match["goals"]["a"]),
            "HomeXG": float(match["xG"]["h"]),
            "AwayXG": float(match["xG"]["a"]),
        }
    )

df = pd.DataFrame(rows)

# Standardize date format to YYYY-MM-DD
df["DateTime"] = pd.to_datetime(df["DateTime"])
df["Date"] = df["DateTime"].dt.strftime("%Y-%m-%d")

# Align team names to canonical names
df["HomeTeam"] = df["HomeTeam"].replace(UNDERSTAT_TEAM_MAP)
df["AwayTeam"] = df["AwayTeam"].replace(UNDERSTAT_TEAM_MAP)

df = (
    df.drop_duplicates(subset="MatchID")
    .sort_values("DateTime")
    .reset_index(drop=True)
)

output_path = OUTPUT_DIR / "understat_xg_2014_2026.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved {len(df)} unique Understat matches to {output_path}")