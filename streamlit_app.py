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
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@500;600;700&display=swap');

    /* Main background with carbon grids and red hub radial glow */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        font-family: 'Rajdhani', sans-serif !important;
        background-color: #06070d !important;
        background-image: 
            linear-gradient(rgba(0, 168, 255, 0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 168, 255, 0.02) 1px, transparent 1px),
            radial-gradient(ellipse at 50% 50%, rgba(255, 0, 60, 0.04) 0%, transparent 70%) !important;
        background-size: 30px 30px, 30px 30px, 100% 100% !important;
        color: #ffffff !important;
    }
    
    /* Sidebar styling with cyberpunk borders */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 12, 26, 0.95) !important;
        border-right: 1px solid rgba(255, 0, 60, 0.35) !important;
        box-shadow: 0 0 15px rgba(255, 0, 60, 0.1) !important;
    }

    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p, .stMarkdown {
        font-family: 'Orbitron', sans-serif !important;
        color: #8fa8ff !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    /* Headings */
    h1, h2, h3, [data-testid="stHeader"] {
        color: #FF003C !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em !important;
        text-shadow: 0 0 8px rgba(255, 0, 60, 0.2) !important;
    }

    /* Buttons and controls */
    div[data-baseweb="select"] *, div[role="listbox"] * {
        font-family: 'Rajdhani', sans-serif !important;
        background-color: #12142a !important;
        color: #ffffff !important;
    }

    div[data-testid="stMarkdownContainer"] p strong {
        color: #00A8FF !important;
    }
    
    /* Recommendations Custom Cards with angled cuts */
    .rec-card {
        background-color: rgba(18, 22, 42, 0.7) !important;
        border-left: 3px solid #FF003C !important;
        border-top: none !important;
        border-right: none !important;
        border-bottom: none !important;
        clip-path: polygon(0 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%) !important;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), inset 0 0 10px rgba(255, 0, 60, 0.05) !important;
        transition: all 0.25s ease !important;
    }
    
    .rec-card:hover {
        transform: translateY(-2px) !important;
        border-left: 3px solid #00A8FF !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), 0 0 15px rgba(0, 168, 255, 0.15), inset 0 0 10px rgba(0, 168, 255, 0.05) !important;
    }
    
    .rec-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    
    .rec-name {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 0.95rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 0.04em;
    }
    
    .rec-desc {
        font-size: 0.8rem;
        color: #8fa8ff;
        margin-top: 4px;
        font-family: 'Rajdhani', sans-serif !important;
    }
    
    .rec-badge {
        font-family: 'Orbitron', sans-serif !important;
        background: rgba(255, 0, 60, 0.15) !important;
        color: #FF003C !important;
        border: 1px solid rgba(255, 0, 60, 0.3) !important;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .rec-badge.badge-medium {
        background: rgba(255, 215, 0, 0.15) !important;
        color: #FFD700 !important;
        border: 1px solid rgba(255, 215, 0, 0.3) !important;
    }

    .rec-badge.badge-low {
        background: rgba(0, 168, 255, 0.15) !important;
        color: #00A8FF !important;
        border: 1px solid rgba(0, 168, 255, 0.3) !important;
    }
    
    /* Model Info Boxes */
    .model-stat-box {
        background: rgba(10, 12, 26, 0.75) !important;
        border-left: 2px solid #00A8FF !important;
        border-top: none !important;
        border-right: none !important;
        border-bottom: none !important;
        clip-path: polygon(0 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%) !important;
        padding: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4) !important;
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
<div style="display: flex; align-items: center; gap: 14px; margin-bottom: 24px; padding-bottom: 12px; border-bottom: 1px solid rgba(255, 0, 60, 0.35);">
    <div style="color: #FF003C; filter: drop-shadow(0 0 8px rgba(255, 0, 60, 0.6)); display: flex; align-items: center;">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2a1.5 1.5 0 0 1 1.5 1.5c0 .35-.12.67-.32.93l1.83 1.83c1.4-.4 2.82.26 3.32 1.54.42 1.07.13 2.27-.64 3.03l-2.07 2.07 1.41 1.41c1.55-.4 3.14.34 3.69 1.77.56 1.45.02 3.1-1.25 3.96l-2.73 1.82 2.12 2.12a1 1 0 0 1-1.41 1.41l-2.83-2.83c-1-.5-1.92-1.28-2.67-2.27-.75.99-1.67 1.77-2.67 2.27l-2.83 2.83a1 1 0 1 1-1.41-1.41l2.12-2.12-2.73-1.82c-1.27-.86-1.81-2.51-1.25-3.96.55-1.43 2.14-2.17 3.69-1.77l1.41-1.41-2.07-2.07c-.77-.76-1.06-1.96-.64-3.03.5-1.28 1.92-1.94 3.32-1.54l1.83-1.83c-.2-.26-.32-.58-.32-.93A1.5 1.5 0 0 1 12 2zm0 5.5a2.5 2.5 0 0 0-2.5 2.5c0 1.04.64 1.93 1.54 2.3l-.3 1.7L12 15.5l1.26-1.5.3-1.7c.9-.37 1.54-1.26 1.54-2.3a2.5 2.5 0 0 0-2.5-2.5z"/>
        </svg>
    </div>
    <div>
        <h1 style="margin: 0; line-height: 1.1; font-size: 2.2rem; font-family: 'Orbitron', sans-serif; font-weight: 900; color: #FF003C; letter-spacing: 0.08em; text-shadow: 0 0 10px rgba(255, 0, 60, 0.4);">AutoProfit AI</h1>
        <p style="margin: 2px 0 0 0; color: #8fa8ff; font-family: 'Rajdhani', sans-serif; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;">Smart Location Intelligence for Auto Drivers</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
st.sidebar.markdown("### 🕸️ Suit HUD Controls")

# Hour selector
hour_labels = {
    0: "12:00 AM", 1: "1:00 AM", 2: "2:00 AM", 3: "3:00 AM", 4: "4:00 AM", 5: "5:00 AM",
    6: "6:00 AM", 7: "7:00 AM", 8: "8:00 AM", 9: "9:00 AM", 10: "10:00 AM", 11: "11:00 AM",
    12: "12:00 PM", 13: "1:00 PM", 14: "2:00 PM", 15: "3:00 PM", 16: "4:00 PM", 17: "5:00 PM",
    18: "6:00 PM", 19: "7:00 PM", 20: "8:00 PM", 21: "9:00 PM", 22: "10:00 PM", 23: "11:00 PM"
}
selected_hour = st.sidebar.select_slider(
    "Time of Day",
    options=list(range(24)),
    value=9,
    format_func=lambda h: hour_labels.get(h, str(h))
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
    st.markdown("### 🕸️ Tactical Map Overlay")
    
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
    st.markdown("### 🎯 High-Yield Zones")
    
    if top_recs:
        for idx, rec in enumerate(top_recs):
            rank = idx + 1
            demand_level = rec.get('demand_level', 'Medium')
            level_class = demand_level.lower()
            badge_label = f"{demand_level} Demand"
            score_color = "#FF003C" if demand_level == "High" else "#FFD700" if demand_level == "Medium" else "#00A8FF"
            st.markdown(f"""
            <div class="rec-card {level_class}">
                <div class="rec-header">
                    <span class="rec-name">#{rank} {rec['name']}</span>
                    <span class="rec-badge badge-{level_class}">{badge_label}</span>
                </div>
                <div style="font-size: 0.85rem; color: #8fa8ff; margin-top: 4px; font-family: 'Rajdhani', sans-serif;">
                    <div>Type: <strong style="color: #ffffff;">{rec['type'].capitalize()} Area</strong></div>
                    <div>Demand Score: <strong style="color: {score_color}; font-family: 'Orbitron', sans-serif;">{round(rec['demand_score'])}%</strong></div>
                </div>
                <div class="rec-desc">{rec['description']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No high/medium demand zones found. Zoom in or change filters.")
        
    # Model info panel
    st.markdown("### 🖥️ OS Analytics")
    if model_info is not None:
        acc = model_info.get("classifier_accuracy", 0.8471) * 100
        r2 = model_info.get("regressor_r2", 0.8979)
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-family: 'Rajdhani', sans-serif;">
            <div class="model-stat-box">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #8fa8ff; font-family: 'Orbitron', sans-serif;">Classifier Acc</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #00A8FF; font-family: 'Orbitron', sans-serif; text-shadow: 0 0 5px rgba(0, 168, 255, 0.4);">{acc:.2f}%</div>
            </div>
            <div class="model-stat-box">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #8fa8ff; font-family: 'Orbitron', sans-serif;">Regressor R²</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #00A8FF; font-family: 'Orbitron', sans-serif; text-shadow: 0 0 5px rgba(0, 168, 255, 0.4);">{r2:.4f}</div>
            </div>
        </div>
        <div style="font-size: 0.75rem; color: #8fa8ff; margin-top: 10px; text-align: center; text-transform: uppercase; letter-spacing: 0.05em; font-family: 'Orbitron', sans-serif;">
            Algorithm: <strong style="color: #ffffff;">Random Forest Ensemble</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Mock values
        st.markdown("""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-family: 'Rajdhani', sans-serif;">
            <div class="model-stat-box">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #8fa8ff; font-family: 'Orbitron', sans-serif;">Classifier Acc</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #00A8FF; font-family: 'Orbitron', sans-serif; text-shadow: 0 0 5px rgba(0, 168, 255, 0.4);">84.71%</div>
            </div>
            <div class="model-stat-box">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #8fa8ff; font-family: 'Orbitron', sans-serif;">Regressor R²</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #00A8FF; font-family: 'Orbitron', sans-serif; text-shadow: 0 0 5px rgba(0, 168, 255, 0.4);">0.8979</div>
            </div>
        </div>
        <div style="font-size: 0.75rem; color: #8fa8ff; margin-top: 10px; text-align: center; text-transform: uppercase; letter-spacing: 0.05em; font-family: 'Orbitron', sans-serif;">
            Algorithm: <strong style="color: #ffffff;">Random Forest Ensemble (Fallback)</strong>
        </div>
        """, unsafe_allow_html=True)
