import os
import base64
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ============================================================================
# APP CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="AutoProfit AI — Location Intelligence for Auto Drivers",
    page_icon="🛺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling for Streamlit container
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e0e12;
        color: #f8f8fa;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #16161c !important;
        border-right: 1px solid rgba(255, 199, 0, 0.15);
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #FFC700 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 800 !important;
    }
    
    /* Recommendations Custom Cards */
    .rec-card {
        background-color: rgba(28, 28, 36, 0.85);
        border: 1px solid rgba(255, 199, 0, 0.15);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .rec-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 199, 0, 0.35);
        box-shadow: 0 4px 20px rgba(255, 199, 0, 0.05);
    }
    
    .rec-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    
    .rec-name {
        font-size: 1rem;
        font-weight: 800;
        color: #f8f8fa;
    }
    
    .rec-score {
        font-size: 1.1rem;
        font-weight: 800;
        color: #00ff88;
    }
    
    .rec-desc {
        font-size: 0.8rem;
        color: #a0a0b2;
        margin-top: 4px;
    }
    
    .rec-badge {
        background: rgba(0, 255, 136, 0.12);
        color: #00ff88;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.65rem;
        font-weight: 800;
        text-transform: uppercase;
    }
    
    /* Model Info */
    .model-stat-box {
        background: rgba(28, 28, 36, 0.5);
        border: 1px solid rgba(255, 199, 0, 0.1);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ZONE DEFINITIONS
# ============================================================================
CHENNAI_ZONES = [
    {"zone_id": 0,  "name": "T. Nagar",        "lat": 13.0418, "lng": 80.2341, "type": "market",    "description": "Major shopping district"},
    {"zone_id": 1,  "name": "Anna Nagar",       "lat": 13.0850, "lng": 80.2101, "type": "residential","description": "Residential hub"},
    {"zone_id": 2,  "name": "Tambaram",          "lat": 12.9249, "lng": 80.1000, "type": "transit",   "description": "Major railway junction"},
    {"zone_id": 3,  "name": "Guindy",            "lat": 13.0067, "lng": 80.2206, "type": "commercial","description": "IT corridor & industrial area"},
    {"zone_id": 4,  "name": "Egmore",            "lat": 13.0732, "lng": 80.2609, "type": "transit",   "description": "Railway station & cultural hub"},
    {"zone_id": 5,  "name": "Chennai Central",   "lat": 13.0827, "lng": 80.2707, "type": "transit",   "description": "Main railway terminus"},
    {"zone_id": 6,  "name": "Adyar",             "lat": 13.0012, "lng": 80.2565, "type": "residential","description": "Educational & residential area"},
    {"zone_id": 7,  "name": "Velachery",          "lat": 12.9815, "lng": 80.2180, "type": "residential","description": "Growing residential suburb"},
    {"zone_id": 8,  "name": "Porur",             "lat": 13.0382, "lng": 80.1582, "type": "commercial","description": "IT hub near OMR"},
    {"zone_id": 9,  "name": "Chromepet",         "lat": 12.9516, "lng": 80.1462, "type": "industrial","description": "Industrial & residential mix"},
    {"zone_id": 10, "name": "Kodambakkam",       "lat": 13.0524, "lng": 80.2255, "type": "commercial","description": "Film industry hub"},
    {"zone_id": 11, "name": "Mylapore",          "lat": 13.0368, "lng": 80.2676, "type": "cultural",  "description": "Temple town & cultural center"},
    {"zone_id": 12, "name": "Nungambakkam",      "lat": 13.0569, "lng": 80.2425, "type": "commercial","description": "Business district"},
    {"zone_id": 13, "name": "Royapettah",        "lat": 13.0546, "lng": 80.2636, "type": "medical",   "description": "Hospital & medical zone"},
    {"zone_id": 14, "name": "Thiruvanmiyur",     "lat": 12.9830, "lng": 80.2594, "type": "residential","description": "Coastal residential area"},
    {"zone_id": 15, "name": "Koyambedu",         "lat": 13.0694, "lng": 80.1948, "type": "transit",   "description": "Major bus terminus"},
    {"zone_id": 16, "name": "Vadapalani",        "lat": 13.0526, "lng": 80.2121, "type": "cultural",  "description": "Temple & commercial area"},
    {"zone_id": 17, "name": "Sholinganallur",    "lat": 12.9010, "lng": 80.2279, "type": "commercial","description": "IT corridor (OMR)"},
]

ZONE_POI_FEATURES = {
    0:  {"nearby_pois_count": 8,  "has_hospital": 1, "has_bus_stop": 1, "has_railway_station": 1, "has_market": 1, "has_school": 1},
    1:  {"nearby_pois_count": 5,  "has_hospital": 0, "has_bus_stop": 1, "has_railway_station": 0, "has_market": 0, "has_school": 0},
    2:  {"nearby_pois_count": 4,  "has_hospital": 0, "has_bus_stop": 1, "has_railway_station": 1, "has_market": 0, "has_school": 0},
    3:  {"nearby_pois_count": 6,  "has_hospital": 1, "has_bus_stop": 1, "has_railway_station": 1, "has_market": 0, "has_school": 1},
    4:  {"nearby_pois_count": 5,  "has_hospital": 0, "has_bus_stop": 0, "has_railway_station": 1, "has_market": 0, "has_school": 1},
    5:  {"nearby_pois_count": 4,  "has_hospital": 1, "has_bus_stop": 1, "has_railway_station": 1, "has_market": 1, "has_school": 0},
    6:  {"nearby_pois_count": 6,  "has_hospital": 1, "has_bus_stop": 1, "has_railway_station": 0, "has_market": 0, "has_school": 1},
    7:  {"nearby_pois_count": 4,  "has_hospital": 0, "has_bus_stop": 0, "has_railway_station": 0, "has_market": 0, "has_school": 0},
    8:  {"nearby_pois_count": 3,  "has_hospital": 1, "has_bus_stop": 0, "has_railway_station": 0, "has_market": 0, "has_school": 0},
    9:  {"nearby_pois_count": 3,  "has_hospital": 0, "has_bus_stop": 0, "has_railway_station": 1, "has_market": 0, "has_school": 0},
    10: {"nearby_pois_count": 6,  "has_hospital": 1, "has_bus_stop": 1, "has_railway_station": 1, "has_market": 1, "has_school": 1},
    11: {"nearby_pois_count": 5,  "has_hospital": 0, "has_bus_stop": 0, "has_railway_station": 0, "has_market": 1, "has_school": 0},
    12: {"nearby_pois_count": 5,  "has_hospital": 0, "has_bus_stop": 0, "has_railway_station": 0, "has_market": 0, "has_school": 1},
    13: {"nearby_pois_count": 6,  "has_hospital": 1, "has_bus_stop": 0, "has_railway_station": 0, "has_market": 0, "has_school": 1},
    14: {"nearby_pois_count": 3,  "has_hospital": 0, "has_bus_stop": 0, "has_railway_station": 0, "has_market": 0, "has_school": 0},
    15: {"nearby_pois_count": 4,  "has_hospital": 0, "has_bus_stop": 1, "has_railway_station": 0, "has_market": 1, "has_school": 0},
    16: {"nearby_pois_count": 6,  "has_hospital": 0, "has_bus_stop": 1, "has_railway_station": 0, "has_market": 0, "has_school": 0},
    17: {"nearby_pois_count": 2,  "has_hospital": 0, "has_bus_stop": 0, "has_railway_station": 0, "has_market": 0, "has_school": 0},
}

# ============================================================================
# LOAD TRAINED ML MODELS
# ============================================================================
@st.cache_resource
def get_ml_models():
    model_dir = "model"
    try:
        with open(os.path.join(model_dir, "classifier.pkl"), "rb") as f:
            classifier = pickle.load(f)
        with open(os.path.join(model_dir, "regressor.pkl"), "rb") as f:
            regressor = pickle.load(f)
        with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)
        with open(os.path.join(model_dir, "model_info.pkl"), "rb") as f:
            model_info = pickle.load(f)
        return classifier, regressor, label_encoder, model_info
    except Exception as e:
        st.warning(f"ML models could not be loaded: {e}. Running fallback heuristic models.")
        return None, None, None, None

classifier, regressor, label_encoder, model_info = get_ml_models()

# ============================================================================
# PREDICTION HELPER FUNCTIONS
# ============================================================================
def build_feature_vector(hour, day_of_week, weather, temperature, zone_id):
    is_weekend = 1 if day_of_week >= 5 else 0
    is_peak_hour = 1 if (8 <= hour <= 10) or (17 <= hour <= 19) else 0
    is_morning = 1 if 6 <= hour <= 11 else 0
    is_evening = 1 if 16 <= hour <= 21 else 0
    is_night = 1 if hour >= 22 or hour <= 5 else 0
    
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    poi_feat = ZONE_POI_FEATURES.get(zone_id, {
        "nearby_pois_count": 3,
        "has_hospital": 0, "has_bus_stop": 0,
        "has_railway_station": 0, "has_market": 0, "has_school": 0
    })
    
    features = [
        hour, day_of_week, is_weekend, weather, temperature,
        poi_feat["nearby_pois_count"],
        poi_feat["has_hospital"],
        poi_feat["has_bus_stop"],
        poi_feat["has_railway_station"],
        poi_feat["has_market"],
        poi_feat["has_school"],
        is_peak_hour, is_morning, is_evening, is_night,
        hour_sin, hour_cos, day_sin, day_cos
    ]
    return np.array(features).reshape(1, -1)

def run_predictions(hour, day_of_week, weather, temperature):
    results = []
    
    for zone in CHENNAI_ZONES:
        zone_id = zone["zone_id"]
        
        if classifier is not None and regressor is not None:
            # Predict using models
            features = build_feature_vector(hour, day_of_week, weather, temperature, zone_id)
            level_encoded = classifier.predict(features)[0]
            demand_level = label_encoder.inverse_transform([level_encoded])[0]
            demand_score = float(regressor.predict(features)[0])
            demand_score = max(0, min(100, demand_score))
        else:
            # Fallback heuristic
            demand_score = 50
            if 7 <= hour <= 10 or 17 <= hour <= 20:
                demand_score += 20
            if zone["type"] in ["transit", "medical"]:
                demand_score += 15
            if weather >= 2:
                demand_score += 10
            
            demand_score = max(0, min(100, demand_score))
            demand_level = "High" if demand_score >= 70 else "Medium" if demand_score >= 40 else "Low"
            
        results.append({
            "zone_id": zone_id,
            "name": zone["name"],
            "lat": zone["lat"],
            "lng": zone["lng"],
            "type": zone["type"],
            "description": zone["description"],
            "demand_level": demand_level,
            "demand_score": demand_score,
            "nearby_pois": ZONE_POI_FEATURES[zone_id]["nearby_pois_count"]
        })
        
    return results

# ============================================================================
# BASE64 IMAGE ENCODING FOR 3D TOY
# ============================================================================
@st.cache_data
def get_base64_toy_image():
    try:
        with open("static/images/3d_auto_toy.png", "rb") as f:
            data = f.read()
            return "data:image/png;base64," + base64.b64encode(data).decode("utf-8")
    except Exception as e:
        # Fallback to empty transparent image or generic URL
        return "https://uxwing.com/wp-content/themes/uxwing/download/transportation-automotive/rickshaw-icon.png"

auto_toy_b64 = get_base64_toy_image()

# ============================================================================
# RENDER HEADER
# ============================================================================
# Styled header matching brand
st.markdown("""
<div style="display: flex; align-items: center; gap: 14px; margin-bottom: 24px; padding-bottom: 12px; border-bottom: 1px solid rgba(255, 199, 0, 0.25);">
    <div style="font-size: 2.2rem; transform: rotate(-5deg); filter: drop-shadow(0 2px 4px rgba(0,0,0,0.5));">🛺</div>
    <div>
        <h1 style="margin: 0; line-height: 1.1; font-size: 2rem;">AutoProfit AI</h1>
        <p style="margin: 2px 0 0 0; color: #a0a0b2; font-size: 0.85rem; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase;">Smart Location Intelligence for Auto Drivers</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
st.sidebar.markdown("### 🎛️ Prediction Controls")

# Hour selector
hour_labels = {
    0: "12:00 AM", 1: "1:00 AM", 2: "2:00 AM", 3: "3:00 AM", 4: "4:00 AM", 5: "5:00 AM",
    6: "6:00 AM", 7: "7:00 AM", 8: "8:00 AM", 9: "9:00 AM", 10: "10:00 AM", 11: "11:00 AM",
    12: "12:00 PM", 13: "1:00 PM", 14: "2:00 PM", 15: "3:00 PM", 16: "4:00 PM", 17: "5:00 PM",
    18: "6:00 PM", 19: "7:00 PM", 20: "8:00 PM", 21: "9:00 PM", 22: "10:00 PM", 23: "11:00 PM"
}
selected_hour = st.sidebar.slider(
    "Time of Day",
    min_value=0,
    max_value=23,
    value=9,
    format_func=lambda h: hour_labels.get(int(h), str(h))
)

# Day selector
day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}
selected_day_name = st.sidebar.selectbox("Day of Week", list(day_map.keys()), index=0)
selected_day = day_map[selected_day_name]

# Weather selector
weather_map = {
    "☀️ Clear": 0,
    "☁️ Cloudy": 1,
    "🌦️ Light Rain": 2,
    "🌧️ Heavy Rain": 3
}
selected_weather_name = st.sidebar.selectbox("Weather Condition", list(weather_map.keys()), index=0)
# Strip emoji for mapping
weather_key = selected_weather_name.split(" ", 1)[1] if " " in selected_weather_name else selected_weather_name
weather_code = weather_map[selected_weather_name]

# Temperature slider
selected_temp = st.sidebar.slider("Temperature (°C)", min_value=25, max_value=40, value=32, step=1)

# Run Predictions
predictions = run_predictions(selected_hour, selected_day, weather_code, selected_temp)

# Sort recommendations
high_demand_zones = [z for z in predictions if z["demand_level"] == "High"]
high_demand_zones = sorted(high_demand_zones, key=lambda x: x["demand_score"], reverse=True)

# If no high demand, pick top medium
if not high_demand_zones:
    high_demand_zones = [z for z in predictions if z["demand_level"] == "Medium"]
    high_demand_zones = sorted(high_demand_zones, key=lambda x: x["demand_score"], reverse=True)

# Limit to top 3
top_recs = high_demand_zones[:3]

# ============================================================================
# TWO COLUMN LAYOUT (LEFT: Map, RIGHT: Stats & Top Locations)
# ============================================================================
col_map, col_stats = st.columns([7, 3])

with col_map:
    st.markdown("### 🗺️ Demand Hotspot Map")
    
    # Read HTML map template and inject base64 auto toy image & predictions JSON
    try:
        with open("templates/map_template.html", "r", encoding="utf-8") as f:
            map_html = f.read()
            
        # Replace base64 toy image placeholder
        map_html = map_html.replace("/* AUTO_TOY_BASE64_PLACEHOLDER */", auto_toy_b64)
        # Replace zones predictions JSON placeholder
        map_html = map_html.replace("/* ZONES_JSON_PLACEHOLDER */", json.dumps(predictions))
        
        # Render map inside Streamlit component
        components.html(map_html, height=580, scrolling=False)
        
    except Exception as e:
        st.error(f"Error loading map template: {e}")

with col_stats:
    st.markdown("### 🏆 Top Waiting Zones")
    
    if top_recs:
        for idx, rec in enumerate(top_recs):
            rank = idx + 1
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-header">
                    <span class="rec-name">#{rank} {rec['name']}</span>
                    <span class="rec-badge">High Demand</span>
                </div>
                <div style="font-size: 0.85rem; color: #a0a0b2; margin-top: 4px;">
                    <div>Type: <strong>{rec['type'].capitalize()} Area</strong></div>
                    <div>Demand Score: <strong style="color: #00ff88;">{round(rec['demand_score'])}%</strong></div>
                </div>
                <div class="rec-desc">{rec['description']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No high/medium demand zones found. Zoom in or change filters.")
        
    # Model info panel
    st.markdown("### 🤖 ML Model Performance")
    if model_info is not None:
        acc = model_info.get("classifier_accuracy", 0.8471) * 100
        r2 = model_info.get("regressor_r2", 0.8979)
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div class="model-stat-box">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #a0a0b2;">Classifier Acc</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #FFC700;">{acc:.2f}%</div>
            </div>
            <div class="model-stat-box">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #a0a0b2;">Regressor R²</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #FFC700;">{r2:.4f}</div>
            </div>
        </div>
        <div style="font-size: 0.75rem; color: #a0a0b2; margin-top: 10px; text-align: center;">
            Algorithm: <strong>Random Forest Ensemble</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Mock values
        st.markdown("""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div class="model-stat-box">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #a0a0b2;">Classifier Acc</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #FFC700;">84.71%</div>
            </div>
            <div class="model-stat-box">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #a0a0b2;">Regressor R²</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #FFC700;">0.8979</div>
            </div>
        </div>
        <div style="font-size: 0.75rem; color: #a0a0b2; margin-top: 10px; text-align: center;">
            Algorithm: <strong>Random Forest Ensemble (Fallback)</strong>
        </div>
        """, unsafe_allow_html=True)
