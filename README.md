# 🛺 AutoProfit AI — Chennai Auto-Rickshaw Demand Predictor

AutoProfit AI is an intelligent location-based decision-support web application designed to help auto-rickshaw drivers in Chennai maximize their daily earnings by predicting passenger demand hotspots in real-time. By minimizing empty cruising (driving around searching for passengers), the application helps drivers save expensive LPG/petrol, minimize fuel waste, and optimize their working hours.

---

## 🎯 Project Objectives

1. **Minimize Idle Fuel Consumption:** Reduce the fuel wasted by Chennai auto drivers driving empty between rides during low-demand periods.
2. **Maximize Daily Earnings:** Recommend optimal waiting locations (hotspots) in real-time to connect drivers with passengers faster.
3. **Data-Driven Decision Support:** Leverage historical patterns, points of interest (POIs), weather conditions, and temporal features to make intelligent zone recommendations.
4. **Dual Interface Deployment:** Provide a map-first interface (Flask) for driver dispatch and a clean dashboard interface (Streamlit) for deployment on Streamlit Cloud.

---

## 📊 Data Sources & Features

The machine learning models are trained on a comprehensive dataset of **30,316 demand samples** representing trips across Chennai, engineered from the following data sources:

1. **Geographic Points of Interest (POIs):** Real geographic locations of 18 major transit, commercial, residential, and medical hubs in Chennai (e.g., Chennai Central, T. Nagar, Egmore, Koyambedu Bus Terminus, Guindy IT corridor, Adyar, Velachery, and Mylapore).
2. **POI Density & Proximity Features:** Features indicating the presence of key passenger generators:
   * Total nearby POIs count
   * Presence of hospitals, schools, markets, railway stations, and bus stops
3. **Weather Condition Inputs:** Live weather condition codes (Clear, Cloudy, Light Rain, Heavy Rain) and temperature ranges (25°C to 40°C) which heavily impact public transit usage.
4. **Cyclical Temporal Encoding:** Time of day (0-23 hours) and Day of week (0-6) mapped using trigonometric functions (`sine` and `cosine` transforms) to help the machine learning models recognize periodic daily and weekly patterns.

---

## 🤖 Machine Learning Pipeline & Algorithms

To make accurate real-time recommendations, the system uses two **Supervised Machine Learning Ensemble Models** (Random Forest) trained on the engineered dataset:

### 1. Random Forest Classifier (Demand Level Predictor)
* **Objective:** Categorize the demand level of each Chennai zone into `High` (🟢), `Medium` (🟡), or `Low` (🔴).
* **Accuracy:** **84.71%**
* **Usage:** Determines color-coding of markers on the map and identifies recommendation filters.

### 2. Random Forest Regressor (Demand Score Predictor)
* **Objective:** Predict a continuous demand score (0-100) representing passenger density.
* **Performance:** **R² Score of 0.8979** (with an average Mean Absolute Error of only 5.79 points).
* **Usage:** Controls the dynamic scale radius of demand circles on the map.

### 3. Feature Importance
The ML models rank features in the following order of importance:
1. Hour of the Day (Peak hours vs Off-peak)
2. Weather Conditions (Rain increases auto demand)
3. Temperature
4. Proximity to Transit Hubs (Railway Stations, Bus Termini)

---

## 🛠️ Technology Stack & Frameworks

* **Frontend Visualization:** 
  * **Leaflet.js:** Renders the map using dark theme CartoDB tiles.
  * **Custom CSS3 Animations:** Implements a floating **3D Auto-Rickshaw Toy** marker on high-demand zones. The toy auto bobs in 3D space (`bobFloat` animation) and casts a realistic synced floor shadow (`shadowShrink` animation).
* **Backend Frameworks:** 
  * **Flask** (served locally on port 5000 with interactive REST APIs).
  * **Streamlit** (served on port 8501 and deployed to Streamlit Cloud).
* **Core Libraries:** `scikit-learn` (model training), `pandas` & `numpy` (data pipeline), `joblib` (model serialization), `Pillow` (image background processing).

---

## 🚀 How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data & Scrape POIs (Optional)
```bash
python scraper.py
```

### 3. Train the ML Models
```bash
python train_model.py
```

### 4. Run the Web App

#### Flask Server (Map-First Interface)
```bash
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)**.

#### Streamlit Dashboard (Community Cloud deployment)
```bash
streamlit run streamlit_app.py
```
Open **[http://localhost:8501](http://localhost:8501)**.
