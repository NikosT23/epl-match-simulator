from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="EPL 2026–27 Live Forecast Engine",
    page_icon="⚽",
    layout="wide",
)

DATA_DIR = Path("data/processed")
STANDINGS_CSV = DATA_DIR / "live_standings_projections.csv"
FIXTURES_CSV = DATA_DIR / "live_upcoming_predictions.csv"


@st.cache_data
def load_data():
    if not STANDINGS_CSV.exists() or not FIXTURES_CSV.exists():
        return None, None
    standings = pd.read_csv(STANDINGS_CSV)
    fixtures = pd.read_csv(FIXTURES_CSV)
    fixtures["Date"] = pd.to_datetime(fixtures["Date"], errors="coerce")
    return standings, fixtures


st.title("⚽ Premier League 2026–27 Live Prediction Dashboard")
st.caption(
    "Driven by a 63.3% Poisson / 35.2% Dynamic Elo / 1.5% Calibrated LightGBM Ensemble with 10,000 Monte Carlo Simulations."
)

standings_df, fixtures_df = load_data()

if standings_df is None or fixtures_df is None:
    st.warning("⚠️ Projection data loading... Stand by for initial sync.")
    st.stop()

# Layout
left_col, right_col = st.columns([1.6, 1.0], gap="large")

with left_col:
    st.subheader("📊 Projected Standings Board")
    st.dataframe(
        standings_df,
        column_config={
            "Team": "Team",
            "BaselinePts": "Current Pts",
            "xPts": st.column_config.NumberColumn(
                "Expected Pts", format="%.1f"
            ),
            "TitleProb_%": st.column_config.ProgressColumn(
                "Title Odds", format="%.1f%%", min_value=0, max_value=100
            ),
            "Top4Prob_%": st.column_config.ProgressColumn(
                "Top 4 Odds", format="%.1f%%", min_value=0, max_value=100
            ),
            "RelegationProb_%": st.column_config.ProgressColumn(
                "Relegation Odds", format="%.1f%%", min_value=0, max_value=100
            ),
        },
        use_container_width=True,
        hide_index=True,
    )

with right_col:
    st.subheader("🔮 Upcoming Gameweek Forecast")
    if len(fixtures_df) > 0:
        available_gws = sorted(fixtures_df["Matchday"].unique())
        selected_gw = st.selectbox("Select Gameweek:", available_gws)
        gw_matches = fixtures_df[
            fixtures_df["Matchday"] == selected_gw
        ].sort_values("Date")

        for _, match in gw_matches.iterrows():
            with st.expander(
                f"⚽ {match['HomeTeam']} vs {match['AwayTeam']}", expanded=True
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Home Win", f"{match['HomeProbability']*100:.1f}%")
                c2.metric("Draw", f"{match['DrawProbability']*100:.1f}%")
                c3.metric("Away Win", f"{match['AwayProbability']*100:.1f}%")