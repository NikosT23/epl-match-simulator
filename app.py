from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="EPL 2026–27 Live Forecast Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data/processed")
STANDINGS_CSV = DATA_DIR / "live_standings_projections.csv"
FIXTURES_CSV = DATA_DIR / "live_upcoming_predictions.csv"


@st.cache_data(ttl=300)
def load_live_data():
    if not STANDINGS_CSV.exists() or not FIXTURES_CSV.exists():
        return None, None

    standings = pd.read_csv(STANDINGS_CSV)
    fixtures = pd.read_csv(FIXTURES_CSV)
    fixtures["Date"] = pd.to_datetime(fixtures["Date"], errors="coerce")

    return standings, fixtures


st.title("⚽ Premier League 2026–27 Live Prediction Dashboard")
st.markdown(
    "Driven by a **63.3% Poisson / 35.2% Dynamic Elo / 1.5% Calibrated LightGBM Ensemble** "
    "with **10,000 Monte Carlo Season Simulations**."
)

standings_df, fixtures_df = load_live_data()

if standings_df is None or fixtures_df is None:
    st.warning(
        "⚠️ Live data files not found in `data/processed/`. "
        "Please run `python src/run_weekly_pipeline.py` to generate initial predictions."
    )
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric(label="Out-of-Sample Accuracy", value="54.12%", delta="+0.55% vs Single")
m2.metric(label="Out-of-Sample Log Loss", value="0.9763", delta="-0.0017 vs Poisson")
m3.metric(label="Out-of-Sample RPS", value="0.2004", delta="Bookmaker Standard: 0.1956")
m4.metric(label="Monte Carlo Iterations", value="10,000", delta="Vectorized NumPy")

st.divider()

left_col, right_col = st.columns([1.6, 1.0], gap="large")

with left_col:
    st.subheader("📊 Projected Probabilistic Standings Board")

    st.dataframe(
        standings_df,
        column_config={
            "Crest": st.column_config.ImageColumn("", width="small"),
            "Team": st.column_config.TextColumn("Team", width="medium"),
            "BaselinePts": st.column_config.NumberColumn("Current Pts", format="%d"),
            "xPts": st.column_config.NumberColumn("Expected Pts", format="%.1f"),
            "StdPts": st.column_config.NumberColumn("Std Dev", format="%.1f"),
            "TitleProb_%": st.column_config.ProgressColumn(
                "Title Odds", format="%.1f%%", min_value=0.0, max_value=100.0
            ),
            "Top4Prob_%": st.column_config.ProgressColumn(
                "Top 4 Odds", format="%.1f%%", min_value=0.0, max_value=100.0
            ),
            "RelegationProb_%": st.column_config.ProgressColumn(
                "Relegation Odds", format="%.1f%%", min_value=0.0, max_value=100.0
            ),
            "AvgRank": st.column_config.NumberColumn("Avg Rank", format="%.1f"),
        },
        column_order=[
            "Crest",
            "Team",
            "BaselinePts",
            "xPts",
            "StdPts",
            "TitleProb_%",
            "Top4Prob_%",
            "RelegationProb_%",
            "AvgRank",
        ],
        use_container_width=True,
        hide_index=True,
        height=740,
    )

with right_col:
    st.subheader("🔮 Upcoming Gameweek Forecast")

    if len(fixtures_df) > 0:
        available_gw = sorted(fixtures_df["Matchday"].unique())
        selected_gw = st.selectbox("Select Gameweek:", available_gw, index=0, key="gw_selector")

        gw_matches = fixtures_df[fixtures_df["Matchday"] == selected_gw].sort_values("Date")
        st.write(f"**Gameweek {selected_gw} Matches ({len(gw_matches)})**")

        for _, match in gw_matches.iterrows():
            home = match["HomeTeam"]
            away = match["AwayTeam"]
            p_home = match["HomeProbability"] * 100.0
            p_draw = match["DrawProbability"] * 100.0
            p_away = match["AwayProbability"] * 100.0

            match_date = (
                match["Date"].strftime("%a %b %d, %H:%M UTC")
                if pd.notna(match["Date"])
                else ""
            )

            with st.expander(f"⚽ {home} vs {away}", expanded=True):
                if match_date:
                    st.caption(f"📅 {match_date}")

                col_h, col_d, col_a = st.columns(3)

                with col_h:
                    if "HomeCrest" in match and pd.notna(match["HomeCrest"]):
                        st.image(match["HomeCrest"], width=30)
                    st.metric(f"Home ({home[:3].upper()})", f"{p_home:.1f}%")

                with col_d:
                    st.metric("Draw", f"{p_draw:.1f}%")

                with col_a:
                    if "AwayCrest" in match and pd.notna(match["AwayCrest"]):
                        st.image(match["AwayCrest"], width=30)
                    st.metric(f"Away ({away[:3].upper()})", f"{p_away:.1f}%")

                st.progress(int(p_home), text=f"Home Advantage Margin: {p_home:.1f}%")
    else:
        st.info("No remaining unplayed fixtures for this season!")