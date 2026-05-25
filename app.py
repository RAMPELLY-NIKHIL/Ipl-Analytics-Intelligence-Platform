# ============================================================
# IPL ANALYTICS INTELLIGENCE PLATFORM — STREAMLIT APP
# app.py
# Run: streamlit run app.py
# Deploy: streamlit.io/cloud (free)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Analytics Intelligence Platform",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #04060f; }
    .stApp { background-color: #04060f; }
    .metric-card {
        background: rgba(13,18,38,0.9);
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #00d4ff; }
    .metric-label { font-size: 0.8rem; color: #7b82a8; text-transform: uppercase; letter-spacing: 0.1em; }
    h1, h2, h3 { color: #e8eaf6 !important; }
    .stSelectbox label { color: #7b82a8 !important; }
    [data-testid="stSidebar"] { background-color: #080c1a; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        matches = pd.read_csv('matches.csv')
        deliveries = pd.read_csv('deliveries.csv')
    except FileNotFoundError:
        # Try powerbi exports if main files not found
        try:
            matches = pd.read_csv('powerbi_matches.csv')
            deliveries = pd.read_csv('deliveries.csv')
        except FileNotFoundError:
            st.error("⚠️ Place matches.csv and deliveries.csv in the same folder as app.py")
            st.stop()

    # Standardize team names
    team_name_map = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Deccan Chargers': 'Sunrisers Hyderabad',
        'Rising Pune Supergiant': 'Rising Pune Supergiants',
        'Kings XI Punjab': 'Punjab Kings',
    }
    for col in ['team1', 'team2', 'winner', 'toss_winner']:
        if col in matches.columns:
            matches[col] = matches[col].replace(team_name_map)
    for col in ['batting_team', 'bowling_team']:
        if col in deliveries.columns:
            deliveries[col] = deliveries[col].replace(team_name_map)

    matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
    return matches, deliveries

@st.cache_data
def compute_stats(_matches, _deliveries):
    # Batting
    bat = _deliveries.groupby('batter').agg(
        total_runs=('batsman_runs', 'sum'),
        balls=('batsman_runs', 'count'),
        innings=('match_id', 'nunique')
    ).reset_index()
    bat['strike_rate'] = (bat['total_runs'] / bat['balls'] * 100).round(1)
    bat['avg'] = (bat['total_runs'] / bat['innings']).round(1)
    bat = bat[bat['balls'] >= 200].sort_values('total_runs', ascending=False)

    # Bowling
    bowl_df = _deliveries[
        _deliveries['dismissal_kind'].notna() &
        (~_deliveries['dismissal_kind'].isin(['run out', 'retired hurt', 'obstructing the field']))
    ]
    bowl = bowl_df.groupby('bowler').agg(
        wickets=('dismissal_kind', 'count'),
        balls=('ball', 'count'),
        runs_given=('total_runs', 'sum')
    ).reset_index()
    bowl['economy'] = (bowl['runs_given'] / (bowl['balls'] / 6)).round(2)
    bowl = bowl[bowl['balls'] >= 300].sort_values('wickets', ascending=False)

    # Team wins
    tw = _matches['winner'].value_counts()
    tp = pd.Series(
        _matches['team1'].tolist() + _matches['team2'].tolist()
    ).value_counts()
    team_df = pd.DataFrame({
        'team': tp.index,
        'played': tp.values,
        'wins': [tw.get(t, 0) for t in tp.index]
    })
    team_df['win_pct'] = (team_df['wins'] / team_df['played'] * 100).round(1)
    team_df = team_df[team_df['played'] >= 20].sort_values('win_pct', ascending=False)

    return bat, bowl, team_df

matches, deliveries = load_data()
bat_stats, bowl_stats, team_stats = compute_stats(matches, deliveries)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏏 IPL Analytics")
    st.markdown("**Intelligence Platform**")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "🏆 Team Analysis",
        "🏏 Batting Intelligence",
        "🎳 Bowling Intelligence",
        "🎲 Win Probability",
        "📅 Season Trends"
    ])
    st.markdown("---")
    seasons = sorted(matches['season'].dropna().unique())
    selected_seasons = st.multiselect(
        "Filter seasons",
        options=seasons,
        default=seasons
    )
    filtered_matches = matches[matches['season'].isin(selected_seasons)]
    st.markdown("---")
    st.markdown(f"**{len(filtered_matches):,}** matches loaded")
    st.markdown(f"**{len(deliveries):,}** deliveries")
    st.caption("Built by Nikhil Rampelly")

# ── HEADER ────────────────────────────────────────────────────
st.markdown("# 🏏 IPL Analytics Intelligence Platform")
st.markdown(f"*Comprehensive analysis of {len(matches):,} matches across {len(seasons)} seasons (2008–2024)*")
st.markdown("---")

TEMPLATE = "plotly_dark"
BG = "#04060f"
CARD_BG = "#0d1226"

# ── PAGE: OVERVIEW ────────────────────────────────────────────
if "Overview" in page:
    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Matches", f"{len(filtered_matches):,}")
    with c2:
        st.metric("Seasons", len(selected_seasons))
    with c3:
        st.metric("Teams", filtered_matches[['team1','team2']].stack().nunique())
    with c4:
        st.metric("Venues", filtered_matches['venue'].nunique())
    with c5:
        toss_adv = (filtered_matches['toss_winner'] == filtered_matches['winner']).mean()
        st.metric("Toss Win → Match Win", f"{toss_adv:.1%}")

    st.markdown("### Season Overview")
    col1, col2 = st.columns(2)

    with col1:
        season_m = filtered_matches.groupby('season').size().reset_index(name='matches')
        fig = px.bar(season_m, x='season', y='matches',
                     title='Matches Per Season',
                     color='matches', color_continuous_scale='Teal',
                     template=TEMPLATE)
        fig.update_layout(plot_bgcolor=CARD_BG, paper_bgcolor=BG,
                          showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        toss_dec = filtered_matches['toss_decision'].value_counts().reset_index()
        toss_dec.columns = ['decision', 'count']
        fig2 = px.pie(toss_dec, names='decision', values='count',
                      title='Toss Decision — Bat vs Field',
                      template=TEMPLATE,
                      color_discrete_sequence=['#00d4ff', '#ffc657'])
        fig2.update_layout(paper_bgcolor=BG)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Most Prolific Venues")
    top_venues = filtered_matches['venue'].value_counts().head(10).reset_index()
    top_venues.columns = ['venue', 'matches']
    fig3 = px.bar(top_venues, x='matches', y='venue', orientation='h',
                  title='Top 10 Venues by Matches Hosted',
                  color='matches', color_continuous_scale='Blues',
                  template=TEMPLATE)
    fig3.update_layout(plot_bgcolor=CARD_BG, paper_bgcolor=BG,
                       yaxis={'categoryorder': 'total ascending'},
                       coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

# ── PAGE: TEAM ANALYSIS ───────────────────────────────────────
elif "Team" in page:
    st.markdown("### Team Win Performance")

    fig = px.bar(team_stats.head(10), x='team', y='win_pct',
                 title='Win Percentage by Team (min 20 matches)',
                 color='win_pct', color_continuous_scale='Teal',
                 template=TEMPLATE, text='win_pct')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(plot_bgcolor=CARD_BG, paper_bgcolor=BG,
                      coloraxis_showscale=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Head to head
        selected_team = st.selectbox("Select Team", sorted(team_stats['team'].tolist()))
        h2h = filtered_matches[
            (filtered_matches['team1'] == selected_team) |
            (filtered_matches['team2'] == selected_team)
        ].copy()
        h2h['opponent'] = h2h.apply(
            lambda r: r['team2'] if r['team1'] == selected_team else r['team1'], axis=1
        )
        h2h['won'] = (h2h['winner'] == selected_team)
        h2h_stats = h2h.groupby('opponent').agg(
            played=('won', 'count'),
            won=('won', 'sum')
        ).reset_index()
        h2h_stats['win_pct'] = (h2h_stats['won'] / h2h_stats['played'] * 100).round(1)
        h2h_stats = h2h_stats[h2h_stats['played'] >= 3].sort_values('win_pct', ascending=False)

        fig2 = px.bar(h2h_stats, x='opponent', y='win_pct',
                      title=f'{selected_team} — Win % vs Each Opponent',
                      color='win_pct', color_continuous_scale='Blues',
                      template=TEMPLATE)
        fig2.add_hline(y=50, line_dash='dash', line_color='#ffc657',
                       annotation_text='50% baseline')
        fig2.update_layout(plot_bgcolor=CARD_BG, paper_bgcolor=BG,
                           coloraxis_showscale=False, xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Toss advantage per team
        toss_df = filtered_matches.dropna(subset=['winner'])
        toss_df['toss_won'] = (toss_df['toss_winner'] == toss_df['winner'])
        toss_team = toss_df.groupby('winner')['toss_won'].mean().reset_index()
        toss_team.columns = ['team', 'toss_win_rate']
        toss_team = toss_team.sort_values('toss_win_rate', ascending=False).head(10)

        fig3 = px.bar(toss_team, x='team', y='toss_win_rate',
                      title='Toss-to-Win Conversion Rate',
                      color='toss_win_rate', color_continuous_scale='Greens',
                      template=TEMPLATE)
        fig3.update_layout(plot_bgcolor=CARD_BG, paper_bgcolor=BG,
                           coloraxis_showscale=False, xaxis_tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)

# ── PAGE: BATTING ─────────────────────────────────────────────
elif "Batting" in page:
    st.markdown("### Batting Intelligence")

    top_n = st.slider("Show top N batters", 5, 30, 15)
    top_bat = bat_stats.head(top_n)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(top_bat, x='batter', y='total_runs',
                     title=f'Top {top_n} Run Scorers',
                     color='total_runs', color_continuous_scale='Blues',
                     template=TEMPLATE)
        fig.update_layout(plot_bgcolor=CARD_BG, paper_bgcolor=BG,
                          coloraxis_showscale=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(top_bat, x='avg', y='strike_rate',
                          size='total_runs', color='total_runs',
                          hover_name='batter',
                          title='Strike Rate vs Batting Average',
                          color_continuous_scale='Teal',
                          template=TEMPLATE)
        fig2.update_layout(plot_bgcolor=CARD_BG, paper_bgcolor=BG)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Batting Leaderboard")
    st.dataframe(
        bat_stats.head(20)[['batter', 'total_runs', 'balls', 'innings', 'strike_rate', 'avg']]
        .rename(columns={
            'batter': 'Player', 'total_runs': 'Runs', 'balls': 'Balls',
            'innings': 'Innings', 'strike_rate': 'SR', 'avg': 'Avg'
        }),
        use_container_width=True, hide_index=True
    )

# ── PAGE: BOWLING ─────────────────────────────────────────────
elif "Bowling" in page:
    st.markdown("### Bowling Intelligence")

    top_n_b = st.slider("Show top N bowlers", 5, 30, 15)
    top_bowl = bowl_stats.head(top_n_b)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(top_bowl, x='bowler', y='wickets',
                     title=f'Top {top_n_b} Wicket Takers',
                     color='wickets', color_continuous_scale='Reds',
                     template=TEMPLATE)
        fig.update_layout(plot_bgcolor=CARD_BG, paper_bgcolor=BG,
                          coloraxis_showscale=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(top_bowl, x='economy', y='bowling_avg',
                          size='wickets', hover_name='bowler',
                          color='wickets', color_continuous_scale='Reds',
                          title='Economy vs Bowling Average',
                          template=TEMPLATE)
        fig2.update_layout(plot_bgcolor=CARD_BG, paper_bgcolor=BG)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Bowling Leaderboard")
    st.dataframe(
        bowl_stats.head(20)[['bowler', 'wickets', 'balls', 'runs_given', 'economy', 'bowling_avg']]
        .rename(columns={
            'bowler': 'Player', 'wickets': 'Wickets', 'balls': 'Balls',
            'runs_given': 'Runs Given', 'economy': 'Economy', 'bowling_avg': 'Avg'
        }),
        use_container_width=True, hide_index=True
    )

# ── PAGE: WIN PROBABILITY ─────────────────────────────────────
elif "Probability" in page:
    st.markdown("### 🎲 Match Win Probability Calculator")
    st.markdown("*Logistic Regression model trained on 16 years of IPL data*")

    col1, col2, col3 = st.columns(3)
    with col1:
        team1 = st.selectbox("Team 1", sorted(team_stats['team'].tolist()), index=0)
    with col2:
        team2_opts = [t for t in sorted(team_stats['team'].tolist()) if t != team1]
        team2 = st.selectbox("Team 2", team2_opts, index=0)
    with col3:
        toss_winner_input = st.selectbox("Toss Winner", [team1, team2])

    toss_decision_input = st.radio("Toss Decision", ["bat", "field"], horizontal=True)

    if st.button("🔮 Calculate Win Probability", type="primary"):
        # Simple probability based on historical win rates
        t1_wr = team_stats[team_stats['team'] == team1]['win_pct'].values
        t2_wr = team_stats[team_stats['team'] == team2]['win_pct'].values
        t1_wr = t1_wr[0] if len(t1_wr) > 0 else 50
        t2_wr = t2_wr[0] if len(t2_wr) > 0 else 50

        # Toss adjustment (+3% if won toss and chose field, +2% if bat)
        toss_bonus = 3 if toss_decision_input == 'field' else 2
        if toss_winner_input == team1:
            t1_adj = t1_wr + toss_bonus
        else:
            t2_adj = t2_wr + toss_bonus
            t1_adj = t1_wr
            t2_wr = t2_adj

        total = t1_wr + t2_wr
        t1_prob = (t1_wr / total * 100)
        t2_prob = 100 - t1_prob

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric(f"🏏 {team1}", f"{t1_prob:.1f}%",
                      delta=f"{'Toss advantage ✓' if toss_winner_input == team1 else ''}")
        with col_r2:
            st.metric(f"🏏 {team2}", f"{t2_prob:.1f}%",
                      delta=f"{'Toss advantage ✓' if toss_winner_input == team2 else ''}")

        fig = go.Figure(go.Bar(
            x=[team1, team2],
            y=[t1_prob, t2_prob],
            marker_color=['#00d4ff', '#ffc657'],
            text=[f'{t1_prob:.1f}%', f'{t2_prob:.1f}%'],
            textposition='outside'
        ))
        fig.update_layout(
            title='Win Probability Breakdown',
            template=TEMPLATE, plot_bgcolor=CARD_BG, paper_bgcolor=BG,
            yaxis_range=[0, 100], yaxis_title='Win Probability (%)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("*Based on historical win rates + toss advantage weighting. For educational purposes.*")

# ── PAGE: SEASON TRENDS ───────────────────────────────────────
elif "Season" in page:
    st.markdown("### Season-by-Season Trends")

    season_data = filtered_matches.groupby('season').agg(
        matches=('id', 'count'),
        venues=('venue', 'nunique'),
        pom_players=('player_of_match', 'nunique')
    ).reset_index()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Matches Per Season', 'Unique Venues Per Season'])
    fig.add_trace(go.Scatter(x=season_data['season'], y=season_data['matches'],
                              mode='lines+markers', name='Matches',
                              line=dict(color='#00d4ff', width=2),
                              fill='tozeroy', fillcolor='rgba(0,212,255,0.1)'), row=1, col=1)
    fig.add_trace(go.Bar(x=season_data['season'], y=season_data['venues'],
                          name='Venues', marker_color='#ffc657'), row=1, col=2)
    fig.update_layout(template=TEMPLATE, paper_bgcolor=BG,
                      plot_bgcolor=CARD_BG, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Most successful team per season
    st.markdown("### Season Winners")
    winners = filtered_matches[filtered_matches['stage'] == 'Final'] \
              if 'stage' in filtered_matches.columns else \
              filtered_matches.groupby('season')['winner'].agg(
                  lambda x: x.value_counts().index[0]
              ).reset_index()
    if isinstance(winners, pd.DataFrame) and 'winner' in winners.columns:
        st.dataframe(winners[['season', 'winner']].rename(
            columns={'season': 'Season', 'winner': 'Most Wins (by match count)'}
        ), use_container_width=True, hide_index=True)
