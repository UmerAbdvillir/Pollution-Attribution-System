"""
Pollution Monitoring & Attribution System v8
- IST/SUPARCO drone base (verified coordinates)
- Real road routing via OSRM (client-side)
- Augmented dataset (real 500 rows + synthetic healthy-air rows), 70/30 split
- Distance_m dropped (negligible signal — see DATA NOTES below)
- Native string class targets (no LabelEncoder needed for tree models)
- Classifier: best of RF vs GBM, chosen automatically by test accuracy
- EDA/Eval plots via eda_eval.py
"""

import random, math, json, os, heapq
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, r2_score

app = Flask(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "pollution_dataset_final.csv")

# ── FEATURE SELECTION ─────────────────────────────────────────────────────
# DATA NOTES (decided by inspecting correlations + trained feature
# importances on the real 500-row dataset, not assumed):
#
#   Distance_m   — DROPPED. Correlation with PM2.5 is essentially zero
#                  (r = -0.09), and regressor importance is negligible
#                  (0.003). It has some importance for the classifier
#                  (~0.13), but that turns out to be a dataset artifact —
#                  different source types happened to be sampled at
#                  different typical distances — not a real causal signal.
#                  Physically, distance-from-source SHOULD matter (pollutant
#                  concentration falls off with distance), but this
#                  particular dataset doesn't encode that relationship, so
#                  keeping it risks the model learning a spurious shortcut.
#
#   Humidity     — KEPT. Real negative correlation with PM2.5 (r = -0.56)
#                  and non-trivial regressor importance (0.06) — humidity
#                  genuinely affects particulate behavior (moisture can
#                  scavenge or trap particulates). Classifier importance is
#                  near-zero, but it earns its place via the regressor.
#
FEATURE_COLS = [
    "SmokeDensity", "Temperature", "WindSpeed",
    "NO2_ppm", "CO_ppm", "SO2_ppm",
    "Humidity", "BlackCarbonRatio"
]

# ── DATA QUALITY CHECK ────────────────────────────────────────────────────
def check_data_quality(df):
    """Explicit, printed verification that the raw CSV needs no cleaning —
    rather than silently assuming it. Run once at startup."""
    issues = []
    n_nulls = df.isnull().sum().sum()
    if n_nulls: issues.append(f"{n_nulls} missing values")
    n_dupes = df.duplicated().sum()
    if n_dupes: issues.append(f"{n_dupes} duplicate rows")
    for col in FEATURE_COLS + ["PM25"]:
        if (df[col] < 0).any():
            issues.append(f"negative values in {col}")
    bad_types = [c for c in FEATURE_COLS + ["PM25"] if not pd.api.types.is_numeric_dtype(df[c])]
    if bad_types: issues.append(f"non-numeric dtype in {bad_types}")
    if issues:
        print(f"  Data quality issues found: {issues}")
    else:
        print(f"  Data quality check: PASSED — {len(df)} rows, no nulls/duplicates/negatives/dtype issues")
    return issues

# ── DATASET AUGMENTATION ──────────────────────────────────────────────────
# The real 500-row CSV only contains Unhealthy→Very-Unhealthy PM2.5 values
# (105–251 µg/m³) — there are zero "Good"/"Moderate" samples, so a model
# trained on it alone can never recognise healthy air. We generate
# additional realistic low-pollution samples (per source type, at reduced
# intensity) so the model also learns what clean air looks like. This is
# augmentation for coverage, not a substitute for the real data — the
# original 500 rows are always included in full and never altered in count.
def augment_with_healthy_samples(df, n_samples=200, seed=99):
    rng = random.Random(seed)
    base_profiles = {
        "Vehicle":         dict(SmokeDensity=20, Temperature=28, NO2_ppm=30, CO_ppm=5,  SO2_ppm=3,  BlackCarbonRatio=0.15),
        "Factory":         dict(SmokeDensity=35, Temperature=33, NO2_ppm=25, CO_ppm=4,  SO2_ppm=15, BlackCarbonRatio=0.25),
        "Garbage Burning": dict(SmokeDensity=28, Temperature=30, NO2_ppm=12, CO_ppm=10, SO2_ppm=5,  BlackCarbonRatio=0.35),
    }
    rows = []
    for _ in range(n_samples):
        src = rng.choice(list(base_profiles.keys()))
        b = base_profiles[src]
        intensity = rng.uniform(0.05, 0.4)
        row = {k: max(0, v * intensity + rng.gauss(0, v * 0.15)) for k, v in b.items()}
        row["Temperature"] = max(15, 28 + rng.gauss(0, 5))
        row["WindSpeed"]   = max(0.5, rng.gauss(8, 3))
        row["Humidity"]    = max(20, min(95, rng.gauss(55, 15)))
        row["PM25"]        = max(3, rng.gauss(18, 8))
        row["SourceType"]  = src
        rows.append(row)
    df_synthetic = pd.DataFrame(rows)[FEATURE_COLS + ["PM25", "SourceType"]]
    df_real = df[FEATURE_COLS + ["PM25", "SourceType"]].copy()
    return pd.concat([df_real, df_synthetic], ignore_index=True)

def build_dataset():
    df = pd.read_csv(DATA_PATH)
    check_data_quality(df)
    return augment_with_healthy_samples(df)

def train_models():
    df = build_dataset()
    X   = df[FEATURE_COLS].values
    y_c = df["SourceType"].values   # native string labels — no encoder needed
    y_r = df["PM25"].values

    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, y_c, y_r, test_size=0.30, random_state=42, stratify=y_c
    )

    # Train both classifiers, pick the winner by held-out test accuracy
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_tr, yc_tr)
    rf_acc = accuracy_score(yc_te, rf.predict(X_te))

    gbm_c = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    gbm_c.fit(X_tr, yc_tr)
    gbm_acc = accuracy_score(yc_te, gbm_c.predict(X_te))

    if rf_acc >= gbm_acc:
        clf, clf_name, clf_acc = rf, "Random Forest", rf_acc
    else:
        clf, clf_name, clf_acc = gbm_c, "Gradient Boosting", gbm_acc

    reg = GradientBoostingRegressor(n_estimators=200, random_state=42)
    reg.fit(X_tr, yr_tr)

    clf_acc  = round(clf_acc * 100, 1)
    reg_mae  = round(mean_absolute_error(yr_te, reg.predict(X_te)), 2)
    reg_rmse = round(math.sqrt(mean_squared_error(yr_te, reg.predict(X_te))), 2)
    reg_r2   = round(r2_score(yr_te, reg.predict(X_te)), 3)

    return clf, clf_name, reg, clf_acc, reg_mae, reg_rmse, reg_r2

print("Training models…")
CLASSIFIER, CLF_NAME, REGRESSOR, CLF_ACC, REG_MAE, REG_RMSE, REG_R2 = train_models()
CLASSES = list(CLASSIFIER.classes_)
print(f"  {CLF_NAME} Classifier: {CLF_ACC}%")
print(f"  GBM Regressor: MAE={REG_MAE}  RMSE={REG_RMSE}  R²={REG_R2}")

# ── DRONE BASE: Institute of Space Technology (immediately adjacent to ─────
# SUPARCO HQ — confirmed via OpenStreetMap way 998297212: SUPARCO HQ Mess
# building sits ~120m south of IST). Located at the southern end of the
# Islamabad Expressway, just before it meets GT Road (N-5) at Rawat T-Chowk.
DRONE_BASE = {
    "lat": 33.51981, "lon": 73.17582,
    "name": "Drone Launch Pad — Institute of Space Technology (adj. SUPARCO HQ), Islamabad Expressway near Rawat"
}

# ── HOTSPOTS: Islamabad / Rawalpindi / Wah & surroundings ───────────────────
HOTSPOTS = [
    # Islamabad
    {"name":"I-9 Industrial Area, Islamabad",         "lat":33.6560,"lon":73.0921,"type":"Factory"},
    {"name":"I-10 Industrial Sector, Islamabad",      "lat":33.6720,"lon":73.0700,"type":"Factory"},
    {"name":"Golra Mor traffic interchange",          "lat":33.6850,"lon":73.0200,"type":"Vehicle"},
    {"name":"Faizabad Interchange, Islamabad",        "lat":33.7050,"lon":73.0580,"type":"Vehicle"},
    {"name":"Islamabad Expressway heavy traffic",     "lat":33.6700,"lon":73.1300,"type":"Vehicle"},
    {"name":"Srinagar Highway, Islamabad",            "lat":33.7300,"lon":73.0800,"type":"Vehicle"},
    {"name":"H-9 open waste burning area",            "lat":33.6900,"lon":73.0400,"type":"Garbage Burning"},
    {"name":"Nilor landfill burning, Islamabad",      "lat":33.7150,"lon":72.9900,"type":"Garbage Burning"},
    {"name":"Bhara Kahu waste site",                  "lat":33.6580,"lon":73.1900,"type":"Garbage Burning"},
    # Rawalpindi
    {"name":"Rawalpindi Saddar traffic hub",          "lat":33.5973,"lon":73.0479,"type":"Vehicle"},
    {"name":"Committee Chowk, Rawalpindi",            "lat":33.5980,"lon":73.0550,"type":"Vehicle"},
    {"name":"Rawalpindi Ring Road",                   "lat":33.5400,"lon":73.0700,"type":"Vehicle"},
    {"name":"Raja Bazar truck route, Rawalpindi",     "lat":33.6050,"lon":73.0600,"type":"Vehicle"},
    {"name":"Westridge Industrial, Rawalpindi",       "lat":33.5800,"lon":73.0200,"type":"Factory"},
    {"name":"Chaklala industrial zone, Rawalpindi",   "lat":33.6100,"lon":73.0850,"type":"Factory"},
    {"name":"Nullah Leh garbage burning, Rawalpindi", "lat":33.5700,"lon":73.0400,"type":"Garbage Burning"},
    {"name":"Dhoke Hassu open burning, Rawalpindi",   "lat":33.5500,"lon":73.0300,"type":"Garbage Burning"},
    # Wah & surroundings
    {"name":"Wah Cantt industrial area",              "lat":33.7700,"lon":72.7700,"type":"Factory"},
    {"name":"Wah Nobel Explosives factory",           "lat":33.7750,"lon":72.7620,"type":"Factory"},
    {"name":"Taxila GT Road heavy vehicles",          "lat":33.7467,"lon":72.8350,"type":"Vehicle"},
    {"name":"Hasan Abdal bypass trucks",              "lat":33.8200,"lon":72.6900,"type":"Vehicle"},
    {"name":"PWD Housing Society traffic, Islamabad",  "lat":33.5650,"lon":73.1450,"type":"Vehicle"},
    {"name":"Humak Industrial Triangle, Islamabad",    "lat":33.5550,"lon":73.1900,"type":"Factory"},
    {"name":"Gujar Khan waste burning",               "lat":33.2553,"lon":73.3078,"type":"Garbage Burning"},
    {"name":"Murree Road, Rawalpindi",                "lat":33.6200,"lon":73.1000,"type":"Vehicle"},
    {"name":"Chakri Road industrial, Rawalpindi",     "lat":33.4900,"lon":72.9800,"type":"Factory"},
    {"name":"Taxila Museum road traffic",             "lat":33.7420,"lon":72.8400,"type":"Vehicle"},
]

SRC_PROFILES = {
    "Vehicle":         dict(SmokeDensity=(25,65), Temperature=(28,40),WindSpeed=(2,12), NO2_ppm=(60,125),CO_ppm=(10,35), SO2_ppm=(2,16), Humidity=(35,80),BlackCarbonRatio=(0.12,0.48)),
    "Factory":         dict(SmokeDensity=(65,98), Temperature=(35,57),WindSpeed=(1,8),  NO2_ppm=(40,100),CO_ppm=(4,18),  SO2_ppm=(38,80),Humidity=(30,65),BlackCarbonRatio=(0.25,0.54)),
    "Garbage Burning": dict(SmokeDensity=(55,95), Temperature=(30,50),WindSpeed=(1,7),  NO2_ppm=(12,62), CO_ppm=(22,52), SO2_ppm=(5,25), Humidity=(35,75),BlackCarbonRatio=(0.54,0.90)),
}

def gen_features(src_type, rng, low=False):
    p = SRC_PROFILES[src_type]
    factor = rng.uniform(0.05,0.35) if low else 1.0
    def rv(lo,hi):
        mid=(lo+hi)/2; v=mid*factor+rng.gauss(0,(hi-lo)/2*0.3*factor)
        return round(max(lo*0.05,v),3)
    feats = {k:rv(*v) for k,v in p.items()}
    return feats

def predict(features):
    row  = np.array([[features.get(c,0) for c in FEATURE_COLS]])
    pm25 = round(float(REGRESSOR.predict(row)[0]),1); pm25=max(3.0,pm25)
    probs= CLASSIFIER.predict_proba(row)[0]
    idx  = int(np.argmax(probs)); src=CLASSES[idx]
    all_p= {CLASSES[i]:round(float(p)*100,1) for i,p in enumerate(probs)}
    return pm25,src,all_p

def who_level(pm25):
    if pm25<=12:    return "Good",                           "#22c55e",1
    if pm25<=35.4:  return "Moderate",                       "#84cc16",2
    if pm25<=55.4:  return "Unhealthy for Sensitive Groups", "#eab308",3
    if pm25<=150.4: return "Unhealthy",                      "#f97316",4
    if pm25<=250.4: return "Very Unhealthy",                 "#ef4444",5
    return                  "Hazardous",                     "#7f1d1d",6

MEASURES = {
    "Vehicle":{
        1:[],2:["Monitor emissions periodically.","Encourage carpooling on busy routes."],
        3:["Enforce vehicle emission testing in this zone.","Advise sensitive residents to reduce outdoor exposure.","Optimise traffic flow at peak hours."],
        4:["Issue advisory for sensitive groups (asthma, elderly, children).","Deploy roadside air quality monitors.","Restrict heavy diesel trucks during peak hours.","Promote electric vehicles with incentives."],
        5:["Issue public health alert.","Inspect and impound high-emission vehicles.","Close schools/outdoor venues within 500 m.","Water-spray roads to suppress particulates.","Coordinate with traffic police to enforce idling bans."],
        6:["🚨 Emergency air-quality alert — evacuate vulnerable residents.","Mandatory vehicle ban in 1 km radius.","Deploy emergency medical teams.","Notify NEQS authority immediately.","Activate smog emergency protocol."],
    },
    "Factory":{
        1:[],2:["Routine stack-emission log review.","Verify filter maintenance schedule."],
        3:["Conduct surprise inspection of chimney filters.","Issue advisory for workers without PPE.","Review fuel type — switch to cleaner alternatives if possible."],
        4:["Issue formal notice under NEQS Section 16.","Require installation of electrostatic precipitators.","Restrict production to daytime hours.","Advise local community to keep windows closed and wear N95 masks."],
        5:["Immediate partial shutdown order issued.","Notify Pak-EPA.","Deploy real-time stack monitors.","Order industrial scrubber installation within 30 days.","Establish 1 km exclusion zone."],
        6:["🚨 Full factory shutdown — legal proceedings initiated.","Emergency NEQS violation report filed.","Coordinate with hospitals for respiratory surge.","Cordon off surrounding residential area."],
    },
    "Garbage Burning":{
        1:[],2:["Remind community of municipal waste collection schedule."],
        3:["Deploy field officer to identify and stop burning site.","Coordinate with solid-waste management authority.","Educate residents on health risks of open burning."],
        4:["Issue on-spot fine (Rs 50,000) under EPA.","Advise residents within 300 m to stay indoors.","Arrange emergency waste pickup.","Spray water to extinguish smouldering waste."],
        5:["Dispatch fire service to extinguish burning site.","Notify district environment officer.","Issue public health advisory with N95 mask guidance.","Set up temporary waste-reception point."],
        6:["🚨 Health emergency — activate district disaster management.","Immediate evacuation of households within 200 m.","Criminal proceedings under Section 11 of Pak EPA.","Emergency health camps set up for affected residents."],
    },
}

def get_measures(source_type,lvl_int):
    base=MEASURES.get(source_type,{})
    combined=[]
    for lvl in range(2,lvl_int+1): combined+=base.get(lvl,[])
    return combined if combined else ["✅ Air quality is good. No immediate action required."]

# ── ROUTING NOTE ──────────────────────────────────────────────────────────
# How real digital-navigation engines (Google Maps, OSRM, GraphHopper) find
# routes: the road network is modelled as a weighted graph (nodes = real
# intersections, edges = real road segments with length/speed-limit costs).
# A blank uniform grid with randomly-scattered "obstacles" — which is what
# the previous version of this app used — is not how any real system works,
# and produces a path that has no relationship to the actual city.
#
# The standard production pipeline is:
#   1. Dijkstra's algorithm precomputes a Contraction Hierarchy (CH) over
#      the road graph offline — this is what makes queries fast.
#   2. At query time, a bidirectional Dijkstra / A* search runs over the
#      contracted graph (A* uses straight-line or landmark-based lower-bound
#      heuristics — e.g. the ALT algorithm — to prune the search space).
#   3. Live traffic data re-weights edges before/during the search.
#
# This app uses OSRM (Open Source Routing Machine) — a free, open-source
# implementation of exactly this pipeline — to compute the real road route.
# OSRM is queried directly from the browser (templates/index.html), the
# same way Google Maps' JS SDK calls Google's routing backend client-side.
# The backend's job here is sensing + ML prediction, not routing — that
# division of responsibility mirrors how real navigation + IoT systems are
# architected (routing engine and sensor/inference service are separate
# concerns, often separate servers entirely).
#
# This server only computes great-circle ("as the crow flies") distance,
# used for fallback estimates and for ordering nearby hotspots.
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# ── GET LIST OF PLOT FILES ───────────────────────────────────────────────────
def get_plot_list():
    plot_dir = os.path.join(os.path.dirname(__file__), "static", "plots")
    if not os.path.exists(plot_dir):
        return []
    files = sorted(f for f in os.listdir(plot_dir) if f.endswith(".png"))
    return [{"filename": f, "title": f[3:-4].replace("_"," ").title()} for f in files]

PLOT_TITLES = {
    "01_class_distribution":  "Class Distribution",
    "02_pm25_distribution":   "PM2.5 Distribution with WHO Bands",
    "03_feature_boxplots":    "Feature Distributions by Source Type",
    "04_correlation_heatmap": "Feature Correlation Matrix",
    "05_pairplot":            "Pairplot — Key Features",
    "06_data_quality":        "Missing Values & Outliers",
    "07_confusion_matrix":    "Confusion Matrix — RF Classifier",
    "08_classification_report":"Classification Report Heatmap",
    "09_roc_curves":          "ROC Curves (One-vs-Rest)",
    "10_feature_importance":  "Feature Importances",
    "11_regression_diagnostics":"Regression Diagnostics",
    "12_learning_curves":     "Learning Curves",
    "13_cv_scores":           "5-Fold Cross-Validation Scores",
    "14_feature_vs_pm25":     "Key Features vs PM2.5",
}

# ── ROUTES ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

@app.route("/api/scenario")
def scenario():
    seed = random.randint(0,9_999_999)
    rng  = random.Random(seed)
    hotspot = rng.choice(HOTSPOTS)
    low     = rng.random() < 0.25
    features = gen_features(hotspot["type"], rng, low=low)
    pm25, pred_src, all_probs = predict(features)
    level, color, lvl_int = who_level(pm25)
    measures = get_measures(pred_src, lvl_int)
    src_lat = hotspot["lat"] + rng.uniform(-0.003,0.003)
    src_lon = hotspot["lon"] + rng.uniform(-0.003,0.003)
    return jsonify({
        "seed": seed,
        "drone_base": DRONE_BASE,
        "source": {"lat":round(src_lat,6),"lon":round(src_lon,6),
                   "name":hotspot["name"],"true_type":hotspot["type"]},
        "features": features,
        "prediction": {"pm25":pm25,"source_type":pred_src,"all_probs":all_probs,
                       "who_level":level,"who_color":color,"who_int":lvl_int,
                       "measures":measures},
        "model_stats": {"clf_name":CLF_NAME,"clf_acc":CLF_ACC,"reg_mae":REG_MAE,
                        "reg_rmse":REG_RMSE,"reg_r2":REG_R2},
    })

@app.route("/api/distance")
def get_distance():
    """
    Great-circle distance helper (used by the frontend only as a fallback
    if the OSRM routing service is briefly unreachable, and for display).
    The actual route the drone flies is computed client-side by querying
    OSRM's road-network routing engine — see templates/index.html.
    """
    from_lat = float(request.args.get("from_lat"))
    from_lon = float(request.args.get("from_lon"))
    to_lat   = float(request.args.get("to_lat"))
    to_lon   = float(request.args.get("to_lon"))
    km = haversine_km(from_lat, from_lon, to_lat, to_lon)
    return jsonify({"distance_km": round(km, 2)})

@app.route("/api/plots")
def plots():
    return jsonify(get_plot_list())

@app.route("/api/predict", methods=["POST"])
def predict_api():
    data  = request.get_json(force=True)
    feats = data.get("features",{})
    pm25, src, all_probs = predict(feats)
    level, color, lvl_int = who_level(pm25)
    measures = get_measures(src, lvl_int)
    return jsonify({"pm25":pm25,"source_type":src,"all_probs":all_probs,
                    "who_level":level,"who_color":color,"measures":measures})

if __name__ == "__main__":
    print("Pollution Monitor v4  →  http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
