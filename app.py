import os
import json
import requests

import pandas as pd
from flask import Flask, render_template, request
from bs4 import BeautifulSoup

# ─── APP SETUP ───────────────────────────────────────────────────────────────
app = Flask(__name__)
HERE     = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "data")

# ─── LOAD & NORMALIZE DATA ───────────────────────────────────────────────────
matches    = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
deliveries = pd.read_csv(os.path.join(DATA_DIR, "deliveries.csv"))
deliveries.columns = deliveries.columns.str.lower()

bat_col  = "batsman" if "batsman" in deliveries.columns else "batter"
runs_col = "batsman_runs" if "batsman_runs" in deliveries.columns else next(
    c for c in deliveries.columns
    if "runs" in c and c not in ["extra_runs","total_runs"]
)
bowl_col = "bowler"
out_col  = "player_dismissed" if "player_dismissed" in deliveries.columns else None

# Precompute career totals
batsman_runs   = deliveries.groupby(bat_col)[runs_col].sum().to_dict()
bowler_wickets = {}
if out_col:
    bowler_wickets = (
        deliveries.dropna(subset=[out_col])
                  .groupby(bowl_col)[out_col]
                  .count()
                  .to_dict()
    )

all_players = set(batsman_runs) | set(bowler_wickets)

# Historical overall wins (fallback)
wins_count = matches["winner"].value_counts().to_dict()

# ─── TEAM SLUGS & INVERSE MAP ────────────────────────────────────────────────
TEAM_SLUGS = {
    "Chennai Super Kings":         "chennai-super-kings",
    "Delhi Capitals":              "delhi-capitals",
    "Gujarat Titans":              "gujarat-titans",
    "Kolkata Knight Riders":       "kolkata-knight-riders",
    "Lucknow Super Giants":        "lucknow-super-giants",
    "Mumbai Indians":              "mumbai-indians",
    "Punjab Kings":                "punjab-kings",
    "Rajasthan Royals":            "rajasthan-royals",
    "Royal Challengers Bengaluru": "royal-challengers-bengaluru",
    "Sunrisers Hyderabad":         "sunrisers-hyderabad",
}
SLUG_TO_NAME = {v: k for k, v in TEAM_SLUGS.items()}

# ─── SCRAPE LIVE SQUAD ────────────────────────────────────────────────────────
def fetch_squad(slug):
    url  = f"https://www.iplt20.com/teams/{slug}"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    js   = soup.find("script", id="__NEXT_DATA__")
    if not js:
        return []
    data = json.loads(js.string)
    squad = (data.get("props", {})
                 .get("pageProps", {})
                 .get("team", {})
                 .get("squad", []))
    return [p["playerFullName"].strip()
            for p in squad if p.get("playerFullName")]

# ─── COMPUTE SQUAD STATS ─────────────────────────────────────────────────────
def compute_player_stats(players):
    stats, total = [], 0
    for name in players:
        runs    = batsman_runs.get(name, 0)
        wkts    = bowler_wickets.get(name, 0)
        s       = runs + wkts
        if s > 0:
            stats.append({"player": name, "runs": runs, "wickets": wkts, "total": s})
            total += s
    return total, stats

# ─── DASHBOARD DATA ─────────────────────────────────────────────────────────
mp = (matches.groupby("season")["id"].count()
           .reset_index().rename(columns={"id":"matches"})
           .sort_values("season"))
tw = (matches.groupby("winner")["id"].count()
           .reset_index().rename(columns={"winner":"team","id":"wins"})
           .sort_values("wins", ascending=False).head(10))
tb = (deliveries.groupby(bat_col)[runs_col].sum()
           .reset_index().rename(columns={bat_col:"batsman", runs_col:"runs"})
           .sort_values("runs", ascending=False).head(10))
bl = pd.DataFrame()
if out_col:
    bl = (deliveries.dropna(subset=[out_col])
              .groupby(bowl_col)[out_col].count()
              .reset_index().rename(columns={bowl_col:"bowler", out_col:"wickets"})
              .sort_values("wickets", ascending=False).head(10))

# JSON blobs for Chart.js
mp_json = json.dumps(mp.to_dict("records"))
tw_json = json.dumps(tw.to_dict("records"))
tb_json = json.dumps(tb.to_dict("records"))
bl_json = json.dumps(bl.to_dict("records"))

def dashboard_ctx():
    return {
        "TEAM_SLUGS":         TEAM_SLUGS,
        "mp_json":            mp_json,
        "tw_json":            tw_json,
        "tb_json":            tb_json,
        "bl_json":            bl_json,
        "matches_per_season": mp.to_dict("records"),
        "team_wins":          tw.to_dict("records"),
        "top_batsmen":        tb.to_dict("records"),
        "top_bowlers":        bl.to_dict("records"),
    }

# ─── ROUTES ─────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", **dashboard_ctx())

@app.route("/predict", methods=["POST"])
def predict():
    t1 = request.form.get("team1_slug")
    t2 = request.form.get("team2_slug")
    if not t1 or not t2:
        return render_template("index.html",
                               error="Please select both teams.",
                               **dashboard_ctx())

    squad1 = fetch_squad(t1)
    squad2 = fetch_squad(t2)
    s1, stats1 = compute_player_stats(squad1)
    s2, stats2 = compute_player_stats(squad2)

    team1 = SLUG_TO_NAME[t1]
    team2 = SLUG_TO_NAME[t2]

    # 1) prefer squad-stat totals
    if s1 > s2:
        winner = team1
    elif s2 > s1:
        winner = team2
    else:
        # 2) fallback to historical overall wins
        w1 = wins_count.get(team1, 0)
        w2 = wins_count.get(team2, 0)
        winner = team1 if w1 >= w2 else team2

    return render_template("index.html",
                           pred=True,
                           team1_name=team1,
                           team2_name=team2,
                           stats1=stats1,
                           stats2=stats2,
                           strength1=s1,
                           strength2=s2,
                           predicted_group=winner,
                           **dashboard_ctx())

if __name__ == "__main__":
    app.run(debug=True)