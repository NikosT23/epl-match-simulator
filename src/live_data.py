from pathlib import Path
import os
import time
import pandas as pd
import requests

# ============================================================
# CONFIGURATION
# ============================================================

API_TOKEN = os.getenv("FOOTBALL_DATA_API_TOKEN")
if not API_TOKEN:
    raise ValueError(
        "FOOTBALL_DATA_API_TOKEN environment variable is missing. "
        "Set it in your local environment, .env file, or Render settings."
    )
BASE_URL = "https://api.football-data.org/v4/competitions/PL/matches"

DATA_DIR = Path("data/processed")
OUTPUT_CSV = DATA_DIR / "epl_2026_fixtures.csv"


# ============================================================
# RATE-LIMIT AWARE HTTP CLIENT
# ============================================================

def fetch_with_rate_limiting(url: str, headers: dict, params: dict, max_retries: int = 3) -> dict:
    """
    Executes GET request while inspecting football-data.org rate limit headers:
      - X-Requests-Available-Minute: Remaining calls in current 60s window
      - X-RequestCounter-Reset: Seconds remaining until reset
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=12)

            # Inspect rate-limit headers
            requests_remaining = response.headers.get("X-Requests-Available-Minute")
            reset_seconds = response.headers.get("X-RequestCounter-Reset")

            if requests_remaining is not None and reset_seconds is not None:
                rem = int(requests_remaining)
                rst = int(reset_seconds)

                # Proactive throttling: pause if approaching rate limit boundary
                if rem <= 1 and rst > 0:
                    print(f"[Rate Limiter] Proactive throttle: {rem} requests left in window. Pausing for {rst + 1}s...")
                    time.sleep(rst + 1)

            if response.status_code == 200:
                return response.json()

            elif response.status_code == 429:
                # Reactive handling if 429 limit hit
                wait_time = int(response.headers.get("Retry-After", reset_seconds if reset_seconds else 60))
                print(f"[Rate Limiter] HTTP 429 hit. Waiting {wait_time + 1}s before retry (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time + 1)

            else:
                response.raise_for_status()

        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            print(f"[API Warning] Request failed ({e}). Retrying in 3s...")
            time.sleep(3)

    raise RuntimeError("Exhausted retries due to persistent API rate limiting.")


# ============================================================
# LIVE DATA MODULE
# ============================================================

def fetch_2026_season_data() -> pd.DataFrame:
    """
    Fetches completed results and upcoming fixtures for the active 2026-27 EPL season.
    Falls back to local cached CSV if the network/API request fails.
    """
    headers = {"X-Auth-Token": API_TOKEN}
    params = {"season": 2026}

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        data = fetch_with_rate_limiting(BASE_URL, headers=headers, params=params)

        matches = []
        for match in data.get("matches", []):
            status = match.get("status")  # "FINISHED", "TIMED", "SCHEDULED", "IN_PLAY"

            fthg = match["score"]["fullTime"]["home"] if status == "FINISHED" else None
            ftag = match["score"]["fullTime"]["away"] if status == "FINISHED" else None

            winner = match["score"].get("winner")
            if winner == "HOME_TEAM":
                result = "H"
            elif winner == "AWAY_TEAM":
                result = "A"
            elif status == "FINISHED":
                result = "D"
            else:
                result = None

            matches.append({
                "MatchID": match["id"],
                "Matchday": match["matchday"],
                "Date": match["utcDate"],
                "HomeTeam": match["homeTeam"]["name"],
                "AwayTeam": match["awayTeam"]["name"],
                "Status": status,
                "FTHG": fthg,
                "FTAG": ftag,
                "Result": result,
            })

        df = pd.DataFrame(matches)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)

        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Successfully fetched & cached {len(df)} fixtures to {OUTPUT_CSV}")
        return df

    except Exception as e:
        print(f"API Fetch Error: {e}")
        if OUTPUT_CSV.exists():
            print(f"Loading local cached data from {OUTPUT_CSV}...")
            df = pd.read_csv(OUTPUT_CSV)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            return df
        else:
            raise FileNotFoundError(f"Could not connect to API and no local cache exists at {OUTPUT_CSV}")


if __name__ == "__main__":
    df_fixtures = fetch_2026_season_data()
    played = df_fixtures[df_fixtures["Status"] == "FINISHED"]
    unplayed = df_fixtures[df_fixtures["Status"] != "FINISHED"]

    print("\n2026-27 Season Status Summary:")
    print(f"  Total Fixtures:     {len(df_fixtures)}")
    print(f"  Matches Completed:  {len(played)}")
    print(f"  Fixtures Remaining: {len(unplayed)}")