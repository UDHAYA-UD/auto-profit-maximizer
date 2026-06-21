"""
=============================================================================
AUTO-RICKSHAW PROFIT MAXIMIZER - FLASK WEB APPLICATION
=============================================================================
This is the main web server that:
1. Serves the frontend (map-based UI)
2. Loads the trained ML models
3. Provides API endpoints for real-time demand predictions

API Endpoints:
- GET  /                -> Serves the main web page
- POST /api/predict     -> Predict demand for all zones given time/weather
- GET  /api/zones       -> Get all zone information
- GET  /api/model-info  -> Get model accuracy and feature importances

Author: Auto-Driver Profit Maximizer Project
Date: 2026
=============================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# ============================================================================
# ZONE DEFINITIONS (same as scraper.py - kept consistent)
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

# Pre-computed zone features (from data analysis)
# These represent the POI proximity features for each zone
ZONE_POI_FEATURES = {
    0:  {"nearby_pois_count": 8,  "has_hospital": 1, "has_bus_stop": 1, "has_railway_station": 1, "has_market": 1, "has_school": 1},  # T. Nagar
    1:  {"nearby_pois_count": 5,  "has_hospital": 0, "has_bus_stop": 1, "has_railway_station": 0, "has_market": 0, "has_school": 0},  # Anna Nagar
    2:  {"nearby_pois_count": 4,  "has_hospital": 0, "has_bus_stop": 1, "has_railway_station": 1, "has_market": 0, "has_school": 0},  # Tambaram
    3:  {"nearby_pois_count": 6,  "has_hospital": 1, "has_bus_stop": 1, "has_railway_station": 1, "has_market": 0, "has_school": 1},  # Guindy
    4:  {"nearby_pois_count": 5,  "has_hospital": 0, "has_bus_stop": 0, "has_railway_station": 1, "has_market": 0, "has_school": 1},  # Egmore
    5:  {"nearby_pois_count": 4,  "has_hospital": 1, "has_bus_stop": 1, "has_railway_station": 1, "has_market": 1, "has_school": 0},  # Central
    6:  {"nearby_pois_count": 6,  "has_hospital": 1, "has_bus_stop": 1, "has_railway_station": 0, "has_market": 0, "has_school": 1},  # Adyar
    7:  {"nearby_pois_count": 4,  "has_hospital": 0, "has_bus_stop": 0, "has_railway_station": 0, "has_market": 0, "has_school": 0},  # Velachery
    8:  {"nearby_pois_count": 3,  "has_hospital": 1, "has_bus_stop": 0, "has_railway_station": 0, "has_market": 0, "has_school": 0},  # Porur
    9:  {"nearby_pois_count": 3,  "has_hospital": 0, "has_bus_stop": 0, "has_railway_station": 1, "has_market": 0, "has_school": 0},  # Chromepet
    10: {"nearby_pois_count": 6,  "has_hospital": 1, "has_bus_stop": 1, "has_railway_station": 1, "has_market": 1, "has_school": 1},  # Kodambakkam
    11: {"nearby_pois_count": 5,  "has_hospital": 0, "has_bus_stop": 0, "has_railway_station": 0, "has_market": 1, "has_school": 0},  # Mylapore
    12: {"nearby_pois_count": 5,  "has_hospital": 0, "has_bus_stop": 0, "has_railway_station": 0, "has_market": 0, "has_school": 1},  # Nungambakkam
    13: {"nearby_pois_count": 6,  "has_hospital": 1, "has_bus_stop": 0, "has_railway_station": 0, "has_market": 0, "has_school": 1},  # Royapettah
    14: {"nearby_pois_count": 3,  "has_hospital": 0, "has_bus_stop": 0, "has_railway_station": 0, "has_market": 0, "has_school": 0},  # Thiruvanmiyur
    15: {"nearby_pois_count": 4,  "has_hospital": 0, "has_bus_stop": 1, "has_railway_station": 0, "has_market": 1, "has_school": 0},  # Koyambedu
    16: {"nearby_pois_count": 6,  "has_hospital": 0, "has_bus_stop": 1, "has_railway_station": 0, "has_market": 0, "has_school": 0},  # Vadapalani
    17: {"nearby_pois_count": 2,  "has_hospital": 0, "has_bus_stop": 0, "has_railway_station": 0, "has_market": 0, "has_school": 0},  # Sholinganallur
}

# Weather labels for human-readable output
WEATHER_LABELS = {0: "Clear", 1: "Cloudy", 2: "Light Rain", 3: "Heavy Rain"}


# ============================================================================
# FLASK APP SETUP
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests (needed for API calls from frontend)

# Global variables for loaded models
classifier = None
regressor = None
label_encoder = None
model_info = None


def load_models():
    """
    Load the trained ML models from disk.
    Called once when the server starts.
    """
    global classifier, regressor, label_encoder, model_info
    
    model_dir = "model"
    
    try:
        with open(os.path.join(model_dir, "classifier.pkl"), "rb") as f:
            classifier = pickle.load(f)
        print("  [OK] Loaded classifier model")
        
        with open(os.path.join(model_dir, "regressor.pkl"), "rb") as f:
            regressor = pickle.load(f)
        print("  [OK] Loaded regressor model")
        
        with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)
        print("  [OK] Loaded label encoder")
        
        with open(os.path.join(model_dir, "model_info.pkl"), "rb") as f:
            model_info = pickle.load(f)
        print("  [OK] Loaded model info")
        
        print(f"\n  Model accuracy: {model_info['classifier_accuracy']*100:.2f}%")
        
    except FileNotFoundError as e:
        print(f"\n  [ERR] Model files not found: {e}")
        print("  -> Please run 'python scraper.py' and 'python train_model.py' first!")
        print("  -> Starting with dummy predictions for now...\n")


def build_feature_vector(hour, day_of_week, weather, temperature, zone_id):
    """
    Build a feature vector for a single prediction.
    
    This creates the same features that the model was trained on:
    hour, day_of_week, is_weekend, weather, temperature,
    nearby_pois_count, has_hospital, has_bus_stop, has_railway_station,
    has_market, has_school, is_peak_hour, is_morning, is_evening, is_night,
    hour_sin, hour_cos, day_sin, day_cos
    
    Args:
        hour: Hour of day (0-23)
        day_of_week: Day (0=Monday to 6=Sunday)
        weather: Weather code (0=Clear, 1=Cloudy, 2=Light Rain, 3=Heavy Rain)
        temperature: Temperature in Celsius
        zone_id: Zone identifier
    
    Returns:
        numpy array: Feature vector ready for model prediction
    """
    is_weekend = 1 if day_of_week >= 5 else 0
    is_peak_hour = 1 if (8 <= hour <= 10) or (17 <= hour <= 19) else 0
    is_morning = 1 if 6 <= hour <= 11 else 0
    is_evening = 1 if 16 <= hour <= 21 else 0
    is_night = 1 if hour >= 22 or hour <= 5 else 0
    
    # Cyclical encoding of time features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    # Get zone-specific POI features
    poi_feat = ZONE_POI_FEATURES.get(zone_id, {
        "nearby_pois_count": 3,
        "has_hospital": 0, "has_bus_stop": 0,
        "has_railway_station": 0, "has_market": 0, "has_school": 0
    })
    
    # Build feature vector in the same order as training
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


def generate_recommendation(zone, demand_level, demand_score, hour, weather):
    """
    Generate a human-readable recommendation for a zone.
    
    Examples:
    - "High demand expected near Chennai Central during evening rush hour"
    - "Low activity in Sholinganallur - consider moving to a busier zone"
    """
    time_label = "morning rush" if 7 <= hour <= 10 else \
                 "lunch hour" if 11 <= hour <= 13 else \
                 "afternoon" if 14 <= hour <= 16 else \
                 "evening rush" if 17 <= hour <= 20 else \
                 "night" if 21 <= hour <= 23 else "late night/early morning"
    
    weather_label = WEATHER_LABELS.get(weather, "Clear")
    
    if demand_level == "High":
        reasons = []
        if zone.get("type") == "transit":
            reasons.append("transit hub")
        if zone.get("type") == "medical":
            reasons.append("hospital area")
        if 7 <= hour <= 10 or 17 <= hour <= 20:
            reasons.append(f"{time_label}")
        if weather >= 2:
            reasons.append(f"{weather_label.lower()} driving demand up")
        
        reason_text = " + ".join(reasons) if reasons else time_label
        return f"🟢 High demand! {zone['description']} — {reason_text}. Head here now!"
    
    elif demand_level == "Medium":
        return f"🟡 Moderate demand near {zone['name']} during {time_label}. Decent pickup chances."
    
    else:
        return f"🔴 Low demand in {zone['name']}. Consider moving to a higher-demand zone."


# ============================================================================
# ROUTES / API ENDPOINTS
# ============================================================================

@app.route("/")
def index():
    """Serve the main web page."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict demand for all zones given current conditions.
    
    Request JSON:
    {
        "hour": 9,           # 0-23
        "day_of_week": 0,    # 0=Monday to 6=Sunday
        "weather": 0,        # 0=Clear, 1=Cloudy, 2=Light Rain, 3=Heavy Rain
        "temperature": 32    # Celsius
    }
    
    Response JSON:
    [
        {
            "zone_id": 5,
            "zone_name": "Chennai Central",
            "lat": 13.0827,
            "lng": 80.2707,
            "demand_level": "High",
            "demand_score": 85.5,
            "recommendation": "...",
            "description": "Main railway terminus"
        },
        ...
    ]
    """
    # Get input parameters from request
    data = request.get_json()
    hour = int(data.get("hour", 9))
    day_of_week = int(data.get("day_of_week", 0))
    temperature = float(data.get("temperature", 32))
    
    # Weather can come as string ("Clear") or int (0) from the frontend
    # Map string values to integer codes used by the ML model
    weather_raw = data.get("weather", 0)
    weather_map = {"Clear": 0, "Cloudy": 1, "Light Rain": 2, "Heavy Rain": 3}
    if isinstance(weather_raw, str):
        weather = weather_map.get(weather_raw, 0)
    else:
        weather = int(weather_raw)

    
    results = []
    
    for zone in CHENNAI_ZONES:
        zone_id = zone["zone_id"]
        
        if classifier is not None and regressor is not None:
            # Use ML model for predictions
            features = build_feature_vector(hour, day_of_week, weather, temperature, zone_id)
            
            # Predict demand level (classification)
            level_encoded = classifier.predict(features)[0]
            demand_level = label_encoder.inverse_transform([level_encoded])[0]
            
            # Predict demand score (regression)
            demand_score = float(regressor.predict(features)[0])
            demand_score = max(0, min(100, demand_score))  # Clamp to 0-100
        else:
            # Fallback: simple heuristic if models not loaded
            demand_score = 50  # Default
            if 7 <= hour <= 10 or 17 <= hour <= 20:
                demand_score += 20
            if zone["type"] in ["transit", "medical"]:
                demand_score += 15
            if weather >= 2:
                demand_score += 10
            
            demand_score = max(0, min(100, demand_score))
            demand_level = "High" if demand_score >= 70 else "Medium" if demand_score >= 40 else "Low"
        
        # Generate recommendation text
        recommendation = generate_recommendation(zone, demand_level, demand_score, hour, weather)
        
        results.append({
            "zone_id": zone_id,
            "zone_name": zone["name"],
            "lat": zone["lat"],
            "lng": zone["lng"],
            "demand_level": demand_level,
            "demand_score": round(demand_score, 1),
            "recommendation": recommendation,
            "description": zone["description"],
            "zone_type": zone["type"]
        })
    
    # Sort by demand score (highest first)
    results.sort(key=lambda x: x["demand_score"], reverse=True)
    
    return jsonify(results)


@app.route("/api/zones", methods=["GET"])
def get_zones():
    """Return all zone information."""
    return jsonify(CHENNAI_ZONES)


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    """
    Return model performance metrics and feature importances.
    Used by the frontend to display model statistics.
    """
    if model_info is not None:
        # Get top 8 features for display
        top_features = sorted(
            model_info["feature_importances"],
            key=lambda x: x["importance"],
            reverse=True
        )[:8]
        
        return jsonify({
            "accuracy": round(model_info["classifier_accuracy"] * 100, 2),
            "mae": round(model_info["regressor_mae"], 2),
            "r2": round(model_info["regressor_r2"], 4),
            "algorithm": model_info["algorithm"],
            "n_estimators": model_info["n_estimators"],
            "top_features": [
                {
                    "name": f["feature"],
                    "importance": round(f["importance"] * 100, 1)
                }
                for f in top_features
            ]
        })
    else:
        return jsonify({
            "accuracy": 0,
            "mae": 0,
            "r2": 0,
            "algorithm": "Not trained yet",
            "n_estimators": 0,
            "top_features": []
        })


# ============================================================================
# START THE SERVER
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AUTO-RICKSHAW PROFIT MAXIMIZER")
    print("  Starting Web Server...")
    print("=" * 60)
    
    # Load trained models
    print("\n  Loading ML models...")
    load_models()
    
    print("\n" + "=" * 60)
    print(f"  Server running at: http://localhost:5000")
    print(f"  Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    # Start Flask development server
    app.run(debug=True, host="0.0.0.0", port=5000)
