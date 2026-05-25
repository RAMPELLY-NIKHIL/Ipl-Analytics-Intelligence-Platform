# ============================================================
# IPL ANALYTICS INTELLIGENCE PLATFORM
# Complete Python EDA + Feature Engineering
# Author: Nikhil Rampelly
# Dataset: IPL 2008-2024 (Kaggle)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── STYLE ──────────────────────────────────────────────────────
plt.style.use('dark_background')
COLORS = ['#00d4ff', '#ffc657', '#00ff9d', '#ff6b6b', '#c77dff',
          '#ff9f43', '#48dbfb', '#ff6b9d', '#1dd1a1', '#ffeaa7']
sns.set_palette(COLORS)

# ============================================================
# STEP 1 — LOAD DATA
# ============================================================
# Download from: https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020
# Place matches.csv and deliveries.csv in same folder as this script

matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')

print("=== DATASET OVERVIEW ===")
print(f"Matches shape   : {matches.shape}")
print(f"Deliveries shape: {deliveries.shape}")
print(f"\nMatches columns :\n{list(matches.columns)}")
print(f"\nDeliveries columns:\n{list(deliveries.columns)}")
print(f"\nSeasons covered : {sorted(matches['season'].unique())}")
print(f"Total matches   : {len(matches)}")
print(f"Total deliveries: {len(deliveries)}")

# ============================================================
# STEP 2 — DATA CLEANING
# ============================================================

# Standardize team names (they changed over years)
team_name_map = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Kings XI Punjab': 'Punjab Kings',
}

for col in ['team1', 'team2', 'winner', 'toss_winner']:
    if col in matches.columns:
        matches[col] = matches[col].replace(team_name_map)

if 'batting_team' in deliveries.columns:
    deliveries['batting_team'] = deliveries['batting_team'].replace(team_name_map)
if 'bowling_team' in deliveries.columns:
    deliveries['bowling_team'] = deliveries['bowling_team'].replace(team_name_map)

# Fix date column
matches['date'] = pd.to_datetime(matches['date'], errors='coerce')

print(f"\nNull values in matches:\n{matches.isnull().sum()[matches.isnull().sum() > 0]}")
print(f"\nNull values in deliveries:\n{deliveries.isnull().sum()[deliveries.isnull().sum() > 0]}")

# ============================================================
# STEP 3 — FEATURE ENGINEERING
# ============================================================

# 3a. Win % per team
team_wins = matches['winner'].value_counts()
team_played = pd.Series(
    matches['team1'].tolist() + matches['team2'].tolist()
).value_counts()
team_winpct = (team_wins / team_played * 100).round(2).sort_values(ascending=False)
print("\n=== TEAM WIN PERCENTAGE ===")
print(team_winpct)

# 3b. Toss advantage
matches['toss_match_winner'] = (matches['toss_winner'] == matches['winner'])
toss_win_rate = matches['toss_match_winner'].mean() * 100
print(f"\nToss winner wins match: {toss_win_rate:.1f}% of the time")

# 3c. Batting stats per player
bat_stats = deliveries.groupby('batter').agg(
    total_runs=('batsman_runs', 'sum'),
    balls_faced=('batsman_runs', 'count'),
    innings=('match_id', 'nunique')
).reset_index()
bat_stats['strike_rate'] = (bat_stats['total_runs'] / bat_stats['balls_faced'] * 100).round(2)
bat_stats['avg_per_innings'] = (bat_stats['total_runs'] / bat_stats['innings']).round(2)
bat_stats = bat_stats[bat_stats['balls_faced'] >= 200].sort_values('total_runs', ascending=False)
print("\n=== TOP 10 RUN SCORERS ===")
print(bat_stats.head(10)[['batter', 'total_runs', 'strike_rate', 'avg_per_innings']].to_string(index=False))

# 3d. Bowling stats per player — fully robust across all IPL dataset versions
# ── STEP 1: Print columns for debugging ──────────────────────
print("\n=== DELIVERIES COLUMN NAMES ===")
print(deliveries.columns.tolist())

# ── STEP 2: Detect which wicket column exists ─────────────────
EXCLUDE = ['run out', 'retired hurt', 'obstructing the field']

# Check all possible column name variants
dismissal_col  = next((c for c in deliveries.columns if c in
                       ['dismissal_kind','wicket_type','kind','dismissal']), None)
is_wicket_col  = next((c for c in deliveries.columns if c in
                       ['isWicket','is_wicket','wicket','IsWicket','is_wicket_delivery']), None)
player_dis_col = next((c for c in deliveries.columns if c in
                       ['player_dismissed','dismissed_batter','out_batter']), None)

print(f"  dismissal_kind column : {dismissal_col}")
print(f"  is_wicket column      : {is_wicket_col}")
print(f"  player_dismissed col  : {player_dis_col}")

if dismissal_col:
    non_null = deliveries[dismissal_col].notna().sum()
    unique_vals = deliveries[dismissal_col].dropna().unique()[:8]
    print(f"  Non-null rows         : {non_null}")
    print(f"  Sample values         : {unique_vals}")

if is_wicket_col:
    wicket_count = (deliveries[is_wicket_col] == 1).sum()
    print(f"  isWicket==1 rows      : {wicket_count}")

# ── STEP 3: Compute all balls for economy (ALL deliveries) ────
all_balls_df = deliveries.groupby('bowler').agg(
    balls_bowled=('ball', 'count'),
    runs_conceded=('total_runs', 'sum')
).reset_index()

# ── STEP 4: Compute wickets using best available column ────────
wicket_counts = None

# Method A: dismissal_kind string column (most common older datasets)
if dismissal_col and deliveries[dismissal_col].notna().sum() > 100:
    wk_df = deliveries[
        deliveries[dismissal_col].notna() &
        (~deliveries[dismissal_col].isin(EXCLUDE))
    ]
    if len(wk_df) > 0:
        wicket_counts = wk_df.groupby('bowler').size().reset_index(name='wickets')
        print(f"  Method A (dismissal_kind): {wicket_counts['wickets'].sum()} total wickets")

# Method B: binary isWicket column
if wicket_counts is None and is_wicket_col:
    wk_df = deliveries[deliveries[is_wicket_col] == 1]
    # Exclude run outs if dismissal column also exists
    if dismissal_col:
        wk_df = wk_df[~wk_df[dismissal_col].isin(EXCLUDE)]
    if len(wk_df) > 0:
        wicket_counts = wk_df.groupby('bowler').size().reset_index(name='wickets')
        print(f"  Method B (isWicket=1): {wicket_counts['wickets'].sum()} total wickets")

# Method C: player_dismissed column — non-null means a wicket fell
if wicket_counts is None and player_dis_col:
    wk_df = deliveries[deliveries[player_dis_col].notna()]
    # Exclude run outs using dismissal col if available
    if dismissal_col:
        wk_df = wk_df[~wk_df[dismissal_col].isin(EXCLUDE)]
    if len(wk_df) > 0:
        wicket_counts = wk_df.groupby('bowler').size().reset_index(name='wickets')
        print(f"  Method C (player_dismissed notna): {wicket_counts['wickets'].sum()} total wickets")

# ── STEP 5: Merge balls + wickets ─────────────────────────────
if wicket_counts is not None and len(wicket_counts) > 0:
    bowl_stats = all_balls_df.merge(wicket_counts, on='bowler', how='left')
    bowl_stats['wickets'] = bowl_stats['wickets'].fillna(0).astype(int)
    bowl_stats['economy'] = (bowl_stats['runs_conceded'] /
                             (bowl_stats['balls_bowled'] / 6)).round(2)
    bowl_stats['bowling_avg'] = (bowl_stats['runs_conceded'] /
                                 bowl_stats['wickets'].replace(0, np.nan)).round(2)
    # Lower threshold to 120 balls (~20 overs) to capture more bowlers
    bowl_stats = bowl_stats[
        (bowl_stats['balls_bowled'] >= 120) &
        (bowl_stats['wickets'] > 0)
    ].sort_values('wickets', ascending=False).reset_index(drop=True)
    print(f"\n=== TOP 10 WICKET TAKERS ===")
    print(bowl_stats.head(10)[['bowler','wickets','economy','bowling_avg']].to_string(index=False))
    print(f"Total bowlers exported: {len(bowl_stats)}")
else:
    print("\n  COULD NOT DETECT WICKET COLUMN.")
    print("  Please paste your deliveries column names (printed above) in the chat.")
    print("  Saving empty file — Power BI bowling page will be blank until fixed.")
    bowl_stats = pd.DataFrame(columns=['bowler','wickets','balls_bowled',
                                        'runs_conceded','economy','bowling_avg'])

# 3e. Venue analysis
venue_stats = matches.groupby('venue').agg(
    matches_played=('id', 'count'),
    avg_first_innings=('target_runs', 'mean')
).reset_index().sort_values('matches_played', ascending=False)

# 3f. Season trends
season_stats = matches.groupby('season').agg(
    total_matches=('id', 'count'),
    total_boundaries=('id', 'count')  # placeholder
).reset_index()

# ============================================================
# STEP 4 — WIN PROBABILITY MODEL
# ============================================================

print("\n=== BUILDING WIN PROBABILITY MODEL ===")

# Feature set: toss win, batting first, home advantage proxy
model_df = matches.copy()
model_df = model_df.dropna(subset=['winner', 'toss_winner', 'team1', 'team2'])

# Binary: did team1 win?
model_df['team1_win'] = (model_df['winner'] == model_df['team1']).astype(int)
model_df['toss_team1']= (model_df['toss_winner'] == model_df['team1']).astype(int)

# Encode toss decision
le = LabelEncoder()
model_df['toss_dec_enc'] = le.fit_transform(model_df['toss_decision'].fillna('field'))

# Encode venue
model_df['venue_enc'] = LabelEncoder().fit_transform(model_df['venue'].fillna('Unknown'))

features = ['toss_team1', 'toss_dec_enc', 'venue_enc']
X = model_df[features]
y = model_df['team1_win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

print(f"Model accuracy : {acc:.3f}")
print(f"ROC-AUC score  : {roc:.3f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# ============================================================
# STEP 5 — VISUALIZATIONS (save as PNG for README)
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor('#04060f')
fig.suptitle('IPL Analytics Intelligence Platform — 2008 to 2024',
             fontsize=16, color='white', fontweight='bold', y=1.01)

# Plot 1: Team win %
ax1 = axes[0, 0]
top_teams = team_winpct.head(8)
bars = ax1.barh(top_teams.index, top_teams.values, color=COLORS[:8])
ax1.set_facecolor('#0d1226')
ax1.set_title('Team Win Percentage (%)', color='white', fontsize=11)
ax1.tick_params(colors='white')
ax1.set_xlabel('Win %', color='#7b82a8')
for bar, val in zip(bars, top_teams.values):
    ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', color='white', fontsize=9)

# Plot 2: Top run scorers
ax2 = axes[0, 1]
top_bat = bat_stats.head(8)
bars2 = ax2.bar(top_bat['batter'].str.split().str[-1], top_bat['total_runs'], color='#00d4ff')
ax2.set_facecolor('#0d1226')
ax2.set_title('Top Run Scorers (All Time)', color='white', fontsize=11)
ax2.tick_params(colors='white', axis='both')
ax2.tick_params(axis='x', rotation=45)
ax2.set_ylabel('Total Runs', color='#7b82a8')
for bar, val in zip(bars2, top_bat['total_runs']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'{val:,}', ha='center', color='white', fontsize=8)

# Plot 3: Top wicket takers
ax3 = axes[0, 2]
top_bowl = bowl_stats.head(8)
bars3 = ax3.bar(top_bowl['bowler'].str.split().str[-1], top_bowl['wickets'], color='#ffc657')
ax3.set_facecolor('#0d1226')
ax3.set_title('Top Wicket Takers (All Time)', color='white', fontsize=11)
ax3.tick_params(colors='white', axis='both')
ax3.tick_params(axis='x', rotation=45)
ax3.set_ylabel('Wickets', color='#7b82a8')
for bar, val in zip(bars3, top_bowl['wickets']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(val), ha='center', color='white', fontsize=8)

# Plot 4: Toss decision trend
ax4 = axes[1, 0]
toss_dec = matches['toss_decision'].value_counts()
wedges, texts, autotexts = ax4.pie(
    toss_dec.values,
    labels=toss_dec.index,
    autopct='%1.1f%%',
    colors=['#00d4ff', '#ffc657'],
    startangle=90,
    textprops={'color': 'white'}
)
ax4.set_facecolor('#0d1226')
ax4.set_title('Toss Decision Distribution', color='white', fontsize=11)

# Plot 5: Matches per season
ax5 = axes[1, 1]
season_matches = matches.groupby('season').size()
ax5.plot(season_matches.index, season_matches.values,
         color='#00ff9d', marker='o', linewidth=2, markersize=6)
ax5.fill_between(season_matches.index, season_matches.values,
                 alpha=0.2, color='#00ff9d')
ax5.set_facecolor('#0d1226')
ax5.set_title('Matches Per Season', color='white', fontsize=11)
ax5.tick_params(colors='white')
ax5.set_xlabel('Season', color='#7b82a8')
ax5.set_ylabel('Matches', color='#7b82a8')

# Plot 6: Strike rate vs avg (scatter) — top 30 batters
ax6 = axes[1, 2]
top30 = bat_stats.head(30)
scatter = ax6.scatter(
    top30['avg_per_innings'], top30['strike_rate'],
    c=top30['total_runs'], cmap='cool', s=80, alpha=0.8
)
ax6.set_facecolor('#0d1226')
ax6.set_title('Strike Rate vs Batting Avg (Top 30)', color='white', fontsize=11)
ax6.tick_params(colors='white')
ax6.set_xlabel('Avg Runs per Innings', color='#7b82a8')
ax6.set_ylabel('Strike Rate', color='#7b82a8')
plt.colorbar(scatter, ax=ax6, label='Total Runs').ax.yaxis.label.set_color('white')

plt.tight_layout()
plt.savefig('ipl_eda_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#04060f')
print("\n✓ Saved ipl_eda_dashboard.png — use this as your GitHub README banner")

# ============================================================
# STEP 6 — EXPORT CLEAN DATA FOR POWER BI
# ============================================================

# Export 1: Match summary
# ── AUTO-DETECT columns (different Kaggle datasets use different names) ──
print("\n=== YOUR DATASET COLUMNS ===")
print(list(matches.columns))

# Build column list dynamically — only include columns that actually exist
base_cols = ['id', 'season', 'date', 'city', 'venue',
             'team1', 'team2', 'toss_winner', 'toss_decision',
             'winner', 'player_of_match']

# Common name variations for margin columns
run_col = next((c for c in matches.columns
                if c in ['win_by_runs', 'result_margin', 'margin']), None)
wkt_col = next((c for c in matches.columns
                if c in ['win_by_wickets', 'wickets']), None)
result_col = next((c for c in matches.columns
                   if c in ['result', 'method', 'dl_applied']), None)

# Add optional columns only if they exist
for optional in [run_col, wkt_col, result_col]:
    if optional and optional not in base_cols:
        base_cols.append(optional)

# Filter to only columns that exist in this dataset
available_cols = [c for c in base_cols if c in matches.columns]
print(f"\nExporting columns: {available_cols}")

matches_clean = matches[available_cols].copy()
matches_clean.to_csv('powerbi_matches.csv', index=False)

# Export 2: Batting stats
bat_stats.to_csv('powerbi_batting.csv', index=False)

# Export 3: Bowling stats
bowl_stats.to_csv('powerbi_bowling.csv', index=False)

# Export 4: Team summary
team_summary = pd.DataFrame({
    'team': team_played.index,
    'matches_played': team_played.values,
    'wins': [team_wins.get(t, 0) for t in team_played.index],
    'win_pct': [team_winpct.get(t, 0) for t in team_played.index]
})
team_summary.to_csv('powerbi_teams.csv', index=False)

# Export 5: Season scorecard
season_scorecard = matches.groupby('season').agg(
    total_matches=('id', 'count'),
    unique_venues=('venue', 'nunique'),
    unique_players=('player_of_match', 'nunique')
).reset_index()
season_scorecard.to_csv('powerbi_seasons.csv', index=False)

print("\n✓ Exported 5 CSV files for Power BI:")
print("  powerbi_matches.csv")
print("  powerbi_batting.csv")
print("  powerbi_bowling.csv")
print("  powerbi_teams.csv")
print("  powerbi_seasons.csv")
print("\n✓ All steps complete. Ready for Power BI + Streamlit.")
