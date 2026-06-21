"""
=============================================================================
AUTO-RICKSHAW PROFIT MAXIMIZER - WEB SCRAPER & DATA GENERATOR
=============================================================================
This script does two things:
1. Scrapes real Points of Interest (POIs) from OpenStreetMap for Chennai, India
2. Generates synthetic ride demand data based on realistic patterns

The generated dataset is used to train our ML model that predicts
passenger demand across different zones in Chennai.

Author: Auto-Driver Profit Maximizer Project
Date: 2026
=============================================================================
"""

import os
import json
import random
import math
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# ============================================================================
# ZONE DEFINITIONS - Key areas in Chennai with real GPS coordinates
# ============================================================================
# Each zone represents a major neighborhood/landmark in Chennai.
# These are the zones where we predict auto-rickshaw passenger demand.

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


# ============================================================================
# PART 1: WEB SCRAPING - Scrape POIs from OpenStreetMap
# ============================================================================

def scrape_pois_from_osm():
    """
    Scrapes Points of Interest (POIs) from OpenStreetMap using the Overpass API.
    
    The Overpass API allows us to query OpenStreetMap data. We search for
    hospitals, bus stops, railway stations, markets, schools, and temples
    within Chennai city boundaries.
    
    Returns:
        pandas.DataFrame: POI data with columns [name, lat, lng, category]
    """
    print("=" * 60)
    print("STEP 1: Web Scraping POIs from OpenStreetMap")
    print("=" * 60)
    
    # Overpass API endpoint - this is a free API to query OpenStreetMap data
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Define the bounding box for Chennai city
    # Format: (south_lat, west_lng, north_lat, east_lng)
    chennai_bbox = "12.85,80.05,13.20,80.35"
    
    # Categories of POIs we want to scrape
    # Each category maps to OpenStreetMap tags
    poi_categories = {
        "hospital": ("amenity", "hospital"),
        "bus_stop": ("highway", "bus_stop"),
        "railway_station": ("railway", "station"),
        "market": ("shop", "supermarket"),
        "school": ("amenity", "school"),
        "temple": ("amenity", "place_of_worship"),
        "mall": ("shop", "mall"),
    }
    
    all_pois = []
    
    for category, (key, value) in poi_categories.items():
        print(f"\n  Scraping {category}s from OpenStreetMap...")
        
        # Overpass QL query to find nodes with specific tags in Chennai
        # Using compact single-line format to avoid whitespace issues
        query = (
            f'[out:json][timeout:30];'
            f'(node["{key}"="{value}"]({chennai_bbox});'
            f'way["{key}"="{value}"]({chennai_bbox}););'
            f'out center body;'
        )
        
        try:
            # Send the query to Overpass API using GET with params
            response = requests.get(
                overpass_url,
                params={"data": query},
                timeout=30,
                headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse the response - extract name, lat, lng for each POI
            count = 0
            for element in data.get("elements", []):
                # Get coordinates (nodes have lat/lon directly, ways have center)
                if element["type"] == "node":
                    lat = element.get("lat")
                    lng = element.get("lon")
                elif element["type"] == "way":
                    center = element.get("center", {})
                    lat = center.get("lat")
                    lng = center.get("lon")
                else:
                    continue
                
                if lat and lng:
                    name = element.get("tags", {}).get("name", f"Unknown {category}")
                    all_pois.append({
                        "name": name,
                        "lat": round(lat, 6),
                        "lng": round(lng, 6),
                        "category": category
                    })
                    count += 1
            
            print(f"    [OK] Found {count} {category}(s)")
            
        except Exception as e:
            print(f"    [ERR] Error scraping {category}: {e}")
            print(f"    -> Will use fallback data for {category}")
    
    # If scraping got some results, great! Otherwise use fallback
    if len(all_pois) > 0:
        print(f"\n  [OK] Total POIs scraped: {len(all_pois)}")
        df = pd.DataFrame(all_pois)
    else:
        print("\n  [WARN] Scraping failed. Using fallback POI data...")
        df = get_fallback_pois()
    
    return df


def get_fallback_pois():
    """
    Fallback POI data in case web scraping fails (e.g., no internet).
    These are real locations in Chennai, manually curated.
    
    Returns:
        pandas.DataFrame: Hardcoded POI data
    """
    print("  -> Loading hardcoded POI data for Chennai...")
    
    fallback_data = [
        # Hospitals
        {"name": "Apollo Hospital", "lat": 13.0067, "lng": 80.2565, "category": "hospital"},
        {"name": "MIOT Hospital", "lat": 13.0155, "lng": 80.1870, "category": "hospital"},
        {"name": "Fortis Malar Hospital", "lat": 13.0065, "lng": 80.2540, "category": "hospital"},
        {"name": "Government General Hospital", "lat": 13.0760, "lng": 80.2740, "category": "hospital"},
        {"name": "Royapettah Government Hospital", "lat": 13.0520, "lng": 80.2620, "category": "hospital"},
        {"name": "Sri Ramachandra Hospital", "lat": 13.0382, "lng": 80.1582, "category": "hospital"},
        {"name": "Kauvery Hospital", "lat": 13.0450, "lng": 80.2350, "category": "hospital"},
        
        # Bus Stops
        {"name": "Koyambedu Bus Terminus", "lat": 13.0694, "lng": 80.1948, "category": "bus_stop"},
        {"name": "T.Nagar Bus Stop", "lat": 13.0418, "lng": 80.2341, "category": "bus_stop"},
        {"name": "Adyar Bus Depot", "lat": 13.0050, "lng": 80.2550, "category": "bus_stop"},
        {"name": "Broadway Bus Stand", "lat": 13.0870, "lng": 80.2800, "category": "bus_stop"},
        {"name": "Tambaram Bus Stop", "lat": 12.9250, "lng": 80.1010, "category": "bus_stop"},
        {"name": "Guindy Bus Stop", "lat": 13.0067, "lng": 80.2200, "category": "bus_stop"},
        {"name": "Vadapalani Bus Stop", "lat": 13.0526, "lng": 80.2121, "category": "bus_stop"},
        {"name": "Anna Nagar Bus Stop", "lat": 13.0850, "lng": 80.2101, "category": "bus_stop"},
        
        # Railway Stations
        {"name": "Chennai Central", "lat": 13.0827, "lng": 80.2707, "category": "railway_station"},
        {"name": "Chennai Egmore", "lat": 13.0732, "lng": 80.2609, "category": "railway_station"},
        {"name": "Tambaram Railway Station", "lat": 12.9249, "lng": 80.1000, "category": "railway_station"},
        {"name": "Guindy Railway Station", "lat": 13.0090, "lng": 80.2130, "category": "railway_station"},
        {"name": "Chromepet Railway Station", "lat": 12.9516, "lng": 80.1462, "category": "railway_station"},
        {"name": "Mambalam Railway Station", "lat": 13.0390, "lng": 80.2310, "category": "railway_station"},
        
        # Markets / Shopping
        {"name": "T.Nagar Ranganathan Street", "lat": 13.0410, "lng": 80.2330, "category": "market"},
        {"name": "Pondy Bazaar", "lat": 13.0440, "lng": 80.2370, "category": "market"},
        {"name": "Koyambedu Wholesale Market", "lat": 13.0710, "lng": 80.1950, "category": "market"},
        {"name": "Parry's Corner Market", "lat": 13.0920, "lng": 80.2870, "category": "market"},
        {"name": "Mylapore Market", "lat": 13.0340, "lng": 80.2690, "category": "market"},
        
        # Schools / Colleges
        {"name": "IIT Madras", "lat": 12.9941, "lng": 80.2336, "category": "school"},
        {"name": "Anna University", "lat": 13.0108, "lng": 80.2354, "category": "school"},
        {"name": "Loyola College", "lat": 13.0530, "lng": 80.2500, "category": "school"},
        {"name": "Stella Maris College", "lat": 13.0589, "lng": 80.2439, "category": "school"},
        {"name": "Presidency College", "lat": 13.0660, "lng": 80.2570, "category": "school"},
        
        # Temples
        {"name": "Kapaleeshwarar Temple", "lat": 13.0339, "lng": 80.2695, "category": "temple"},
        {"name": "Vadapalani Murugan Temple", "lat": 13.0520, "lng": 80.2120, "category": "temple"},
        {"name": "Parthasarathy Temple", "lat": 13.0580, "lng": 80.2730, "category": "temple"},
        
        # Malls
        {"name": "Phoenix Marketcity", "lat": 12.9900, "lng": 80.2170, "category": "mall"},
        {"name": "Express Avenue Mall", "lat": 13.0590, "lng": 80.2640, "category": "mall"},
        {"name": "VR Chennai", "lat": 13.0630, "lng": 80.2090, "category": "mall"},
        {"name": "Forum Vijaya Mall", "lat": 13.0510, "lng": 80.2130, "category": "mall"},
    ]
    
    print(f"    [OK] Loaded {len(fallback_data)} fallback POIs")
    return pd.DataFrame(fallback_data)


# ============================================================================
# PART 2: COUNT POIs NEAR EACH ZONE
# ============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two GPS points using
    the Haversine formula. Returns distance in kilometers.
    
    This is used to determine which POIs are "near" each zone.
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def count_pois_near_zone(zone, pois_df, radius_km=2.0):
    """
    Count how many POIs of each category are within `radius_km` of a zone center.
    
    Args:
        zone: dict with 'lat' and 'lng' keys
        pois_df: DataFrame of all POIs
        radius_km: search radius in kilometers (default 2km)
    
    Returns:
        dict: counts and boolean flags for each POI category
    """
    nearby = {"total": 0}
    categories = ["hospital", "bus_stop", "railway_station", "market", "school"]
    
    for cat in categories:
        nearby[f"has_{cat}"] = 0
    
    for _, poi in pois_df.iterrows():
        dist = haversine_distance(zone["lat"], zone["lng"], poi["lat"], poi["lng"])
        if dist <= radius_km:
            nearby["total"] += 1
            cat = poi["category"]
            key = f"has_{cat}"
            if key in nearby:
                nearby[key] = 1  # Binary flag: at least one nearby
    
    return nearby


# ============================================================================
# PART 3: SYNTHETIC DEMAND DATA GENERATION
# ============================================================================

def calculate_demand_score(hour, day_of_week, is_weekend, weather, temperature,
                           zone_type, has_hospital, has_bus_stop, 
                           has_railway_station, has_market, has_school):
    """
    Generate a realistic demand score (0-100) based on multiple factors.
    
    This simulates how passenger demand varies with:
    - Time of day (peak hours have more passengers)
    - Day of week (weekdays vs weekends have different patterns)
    - Weather (rain increases demand for auto-rickshaws)
    - Nearby POIs (hospitals, stations etc. attract passengers)
    
    The logic is based on real-world patterns observed in Indian cities.
    """
    score = 30  # Base demand score
    
    # ---------- TIME OF DAY EFFECTS ----------
    # Morning peak: 7-10 AM (office/school commute)
    if 7 <= hour <= 9:
        score += 25
    elif hour == 10:
        score += 15
    
    # Evening peak: 5-8 PM (return commute)
    if 17 <= hour <= 19:
        score += 25
    elif hour == 20:
        score += 15
    
    # Lunch hour mini-peak: 12-1 PM
    if 12 <= hour <= 13:
        score += 10
    
    # Late night / early morning: very low demand
    if 0 <= hour <= 5:
        score -= 25
    elif hour == 23:
        score -= 15
    
    # ---------- DAY OF WEEK EFFECTS ----------
    if is_weekend:
        # Weekends: less commute demand, but more shopping/leisure
        score -= 5
        # Weekend mornings are slower
        if 7 <= hour <= 9:
            score -= 10
    else:
        # Weekdays: strong commute patterns
        score += 5
    
    # ---------- WEATHER EFFECTS ----------
    # 0=Clear, 1=Cloudy, 2=Light Rain, 3=Heavy Rain
    if weather == 2:  # Light rain - people avoid walking, take autos
        score += 15
    elif weather == 3:  # Heavy rain - very high demand
        score += 25
    elif weather == 1:  # Cloudy - slight increase
        score += 5
    
    # ---------- TEMPERATURE EFFECTS ----------
    # Very hot weather (>35°C) increases demand (people don't want to walk)
    if temperature > 35:
        score += 10
    elif temperature > 38:
        score += 15
    
    # ---------- POI-BASED EFFECTS ----------
    # Railway stations: huge demand during commute hours
    if has_railway_station:
        score += 10
        if 7 <= hour <= 10 or 17 <= hour <= 20:
            score += 15  # Extra boost during peak commute
    
    # Hospitals: consistent demand throughout the day
    if has_hospital:
        score += 12
    
    # Bus stops: general transit demand
    if has_bus_stop:
        score += 8
    
    # Markets: demand peaks during shopping hours
    if has_market:
        score += 5
        if 10 <= hour <= 13:  # Morning shopping
            score += 10
        if 16 <= hour <= 20:  # Evening shopping
            score += 12
        if is_weekend:  # Weekends = shopping time
            score += 10
    
    # Schools/colleges: demand during school hours
    if has_school:
        if 7 <= hour <= 9:  # Going to school
            score += 12
        if 15 <= hour <= 17:  # Coming from school
            score += 12
        if is_weekend:  # No school on weekends
            score -= 5
    
    # ---------- ZONE TYPE EFFECTS ----------
    zone_bonuses = {
        "transit": 10,      # Transit hubs always have some demand
        "commercial": 8,    # Business areas
        "medical": 12,      # Medical zones
        "market": 7,        # Shopping areas
        "cultural": 5,      # Temples, cultural centers
        "residential": 3,   # Base residential
        "industrial": 4,    # Industrial areas
    }
    score += zone_bonuses.get(zone_type, 0)
    
    # ---------- ADD RANDOM NOISE ----------
    # Real-world demand isn't perfectly predictable
    noise = random.gauss(0, 8)  # Gaussian noise with std=8
    score += noise
    
    # Clamp to valid range [0, 100]
    score = max(0, min(100, score))
    
    return round(score, 1)


def generate_demand_data(pois_df):
    """
    Generate synthetic ride demand data for all zones across different
    times, days, and weather conditions.
    
    This creates ~10,000 training samples for our ML model.
    
    Args:
        pois_df: DataFrame of POIs (used to calculate zone features)
    
    Returns:
        pandas.DataFrame: Complete training dataset
    """
    print("\n" + "=" * 60)
    print("STEP 2: Generating Synthetic Ride Demand Data")
    print("=" * 60)
    
    # First, calculate POI features for each zone
    print("\n  Calculating POI proximity for each zone...")
    zone_features = {}
    for zone in CHENNAI_ZONES:
        poi_counts = count_pois_near_zone(zone, pois_df)
        zone_features[zone["zone_id"]] = poi_counts
        print(f"    - {zone['name']}: {poi_counts['total']} nearby POIs")
    
    # Generate data rows
    print("\n  Generating demand samples...")
    rows = []
    
    # For each zone, generate samples across different conditions
    for zone in CHENNAI_ZONES:
        zid = zone["zone_id"]
        poi_feat = zone_features[zid]
        
        # Sample across hours (all 24 hours)
        for hour in range(24):
            # Sample across days of week (0=Mon to 6=Sun)
            for day in range(7):
                is_weekend = 1 if day >= 5 else 0
                
                # Sample across weather conditions (we generate multiple per combo)
                for weather in range(4):
                    # Generate 2-3 samples per combination with different temperatures
                    num_samples = random.choice([2, 3])
                    for _ in range(num_samples):
                        temperature = round(random.uniform(25, 40), 1)
                        
                        # Calculate the demand score
                        demand_score = calculate_demand_score(
                            hour=hour,
                            day_of_week=day,
                            is_weekend=is_weekend,
                            weather=weather,
                            temperature=temperature,
                            zone_type=zone["type"],
                            has_hospital=poi_feat.get("has_hospital", 0),
                            has_bus_stop=poi_feat.get("has_bus_stop", 0),
                            has_railway_station=poi_feat.get("has_railway_station", 0),
                            has_market=poi_feat.get("has_market", 0),
                            has_school=poi_feat.get("has_school", 0),
                        )
                        
                        # Classify demand level
                        if demand_score >= 70:
                            demand_level = "High"
                        elif demand_score >= 40:
                            demand_level = "Medium"
                        else:
                            demand_level = "Low"
                        
                        rows.append({
                            "zone_id": zid,
                            "zone_name": zone["name"],
                            "lat": zone["lat"],
                            "lng": zone["lng"],
                            "zone_type": zone["type"],
                            "hour": hour,
                            "day_of_week": day,
                            "is_weekend": is_weekend,
                            "weather": weather,
                            "temperature": temperature,
                            "nearby_pois_count": poi_feat["total"],
                            "has_hospital": poi_feat.get("has_hospital", 0),
                            "has_bus_stop": poi_feat.get("has_bus_stop", 0),
                            "has_railway_station": poi_feat.get("has_railway_station", 0),
                            "has_market": poi_feat.get("has_market", 0),
                            "has_school": poi_feat.get("has_school", 0),
                            "demand_score": demand_score,
                            "demand_level": demand_level,
                        })
    
    df = pd.DataFrame(rows)
    
    # Print summary statistics
    print(f"\n  [OK] Generated {len(df)} demand samples")
    print(f"    - Zones: {df['zone_name'].nunique()}")
    print(f"    - Demand distribution:")
    print(f"      High:   {(df['demand_level'] == 'High').sum()} ({(df['demand_level'] == 'High').mean()*100:.1f}%)")
    print(f"      Medium: {(df['demand_level'] == 'Medium').sum()} ({(df['demand_level'] == 'Medium').mean()*100:.1f}%)")
    print(f"      Low:    {(df['demand_level'] == 'Low').sum()} ({(df['demand_level'] == 'Low').mean()*100:.1f}%)")
    
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function - run the complete data pipeline."""
    
    print("\n" + "=" * 60)
    print("  AUTO-RICKSHAW PROFIT MAXIMIZER")
    print("  Data Scraping & Generation Pipeline")
    print("=" * 60)
    
    # Create output directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    
    # STEP 1: Scrape POIs from OpenStreetMap
    pois_df = scrape_pois_from_osm()
    
    # Save scraped POIs
    pois_path = os.path.join("data", "chennai_pois.csv")
    pois_df.to_csv(pois_path, index=False)
    print(f"\n  [OK] POIs saved to {pois_path}")
    
    # STEP 2: Generate synthetic demand data
    demand_df = generate_demand_data(pois_df)
    
    # Save demand data
    demand_path = os.path.join("data", "ride_demand_data.csv")
    demand_df.to_csv(demand_path, index=False)
    print(f"\n  [OK] Demand data saved to {demand_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("  DATA PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"  Files created:")
    print(f"    1. {pois_path} ({len(pois_df)} POIs)")
    print(f"    2. {demand_path} ({len(demand_df)} samples)")
    print(f"\n  Next step: Run 'python train_model.py' to train the ML model")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    main()
