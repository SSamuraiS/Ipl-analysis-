import os, json, re
import pandas as pd
import easyocr
from flask import Flask, render_template, request

app = Flask(__name__)
HERE        = os.path.dirname(__file__)
DATA_DIR    = os.path.join(HERE, "data")
UPLOAD_DIR  = os.path.join(HERE, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── OCR SETUP ────────────────────────────────────────────────────────────────
reader = easyocr.Reader(["en"], gpu=False)

# ─── LOAD & NORMALIZE DATA ───────────────────────────────────────────────────
matches    = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
deliveries = pd.read_csv(os.path.join(DATA_DIR, "deliveries.csv"))
deliveries.columns = deliveries.columns.str.lower()

# figure out which column is which
bat_col = "batsman" if "batsman" in deliveries.columns else "batter"
runs_col = "batsman_runs" if "batsman_runs" in deliveries.columns else next(c for c in deliveries.columns if "runs" in c and c not in ["extra_runs","total_runs"])
bowl_col = "bowler"
out_col  = "player_dismissed" if "player_dismissed" in deliveries.columns else None

# ─── DASHBOARD AGGREGATIONS ───────────────────────────────────────────────────
mp = (matches
      .groupby("season")["id"].count()
      .reset_index().rename(columns={"id":"matches"})
      .sort_values("season"))
tw = (matches
      .groupby("winner")["id"].count()
      .reset_index().rename(columns={"winner":"team","id":"wins"})
      .sort_values("wins", ascending=False).head(10))
tb = (deliveries
      .groupby(bat_col)[runs_col].sum()
      .reset_index().rename(columns={bat_col:"batsman", runs_col:"runs"})
      .sort_values("runs", ascending=False).head(10))
bl = pd.DataFrame()
if out_col:
    bl = (deliveries.dropna(subset=[out_col])
          .groupby(bowl_col)[out_col].count()
          .reset_index().rename(columns={bowl_col:"bowler", out_col:"wickets"})
          .sort_values("wickets", ascending=False).head(10))

mp_json = json.dumps(mp.to_dict("records"))
tw_json = json.dumps(tw.to_dict("records"))
tb_json = json.dumps(tb.to_dict("records"))
bl_json = json.dumps(bl.to_dict("records"))

# ─── PREDICTOR SETUP ─────────────────────────────────────────────────────────
teams = sorted(set(matches["team1"]).union(matches["team2"]))
wins_count = matches["winner"].value_counts().to_dict()

def extract_team_from_ocr(raw):
    # look for the first line that contains *two* known suffixes
    for ln in raw.splitlines():
        ln_low = ln.strip().lower()
        found = [t for t in teams if t.lower().split()[-1] in ln_low]
        if len(found) >= 2:
            return found[0], found[1]
    return None, None

def dashboard_ctx():
    return dict(
        mp_json=mp_json, tw_json=tw_json,
        tb_json=tb_json, bl_json=bl_json,
        matches_per_season=mp.to_dict("records"),
        team_wins=tw.to_dict("records"),
        top_batsmen=tb.to_dict("records"),
        top_bowlers=bl.to_dict("records")
    )

# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", **dashboard_ctx())

@app.route("/predict", methods=["POST"])
def predict():
    f1 = request.files.get("team1")
    f2 = request.files.get("team2")
    if not f1 or not f2 or not f1.filename or not f2.filename:
        return render_template("index.html", error="Please paste both screenshots.", **dashboard_ctx())

    p1 = os.path.join(UPLOAD_DIR, f1.filename); f1.save(p1)
    p2 = os.path.join(UPLOAD_DIR, f2.filename); f2.save(p2)

    raw1 = "\n".join(reader.readtext(p1, detail=0, paragraph=True))
    raw2 = "\n".join(reader.readtext(p2, detail=0, paragraph=True))
    t1a, t2a = extract_team_from_ocr(raw1)
    t1b, t2b = extract_team_from_ocr(raw2)
    t1 = t1a or t1b
    t2 = t2a or t2b

    if not t1 or not t2:
        return render_template("index.html", error="Could not detect both teams from your screenshots.", **dashboard_ctx())

    s1 = wins_count.get(t1, 0)
    s2 = wins_count.get(t2, 0)
    winner = t1 if s1 >= s2 else t2

    return render_template("index.html",
        pred=True,
        team1=t1, strength1=s1,
        team2=t2, strength2=s2,
        predicted=winner,
        **dashboard_ctx()
    )

if __name__ == "__main__":
    app.run(debug=True)
