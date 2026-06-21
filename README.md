# 🛺 AutoProfit AI — Location Intelligence for Chennai Auto Drivers

AutoProfit AI is a predictive machine learning web application designed to help auto-rickshaw drivers in Chennai maximize their daily earnings by predicting passenger demand hotspots in real-time. By minimizing empty driving (cruising for passengers), the application saves expensive LPG fuel and increases driver profits.

---

## 🌟 Key Features

*   **Interactive Hotspot Map:** Displays passenger demand levels (High 🟢, Medium 🟡, Low 🔴) across 18 major zones in Chennai.
*   **3D Auto Toy Markers:** High-demand zones feature cute **3D auto-rickshaw toy models** that float and bob in 3D perspective above the map, casting realistic synced shadows underneath.
*   **Location Recommendation Engine:** Shows the Top 3 recommended waiting zones with specific area descriptions (e.g., Transit Hub, Market District, Hospital Zone).
*   **Custom Predictor Controls:** Filter demand dynamically by adjusting:
    *   Time of day (Hour)
    *   Day of week
    *   Weather conditions (Clear, Cloudy, Light Rain, Heavy Rain)
    *   Temperature
*   **Machine Learning Under the Hood:** Uses custom-trained Random Forest Classifier (84.71% accuracy) and Regressor models trained on local event schedules, weather patterns, and POIs.
*   **Dual Deployment Options:** Can be run locally as a Flask application or deployed directly to **Streamlit Cloud** for web sharing.

---

## 🛠️ Technology Stack

*   **Frontend:** HTML5, CSS3 (Custom 3D Animations & Transitions), JavaScript (Leaflet.js map integration).
*   **Backend:** Python 3, Flask (API Server) & Streamlit (Dashboard).
*   **Machine Learning:** Scikit-Learn (Random Forest Ensemble), Pandas, NumPy, Joblib.

---

## 🚀 How to Run Locally

### 1. Install Dependencies
Ensure you have Python installed, then clone the repository and run:
```bash
pip install -r requirements.txt
```

### 2. Generate Data & Scrape POIs (Optional)
If you want to re-generate the dataset or scrape Chennai POIs from OpenStreetMap, run:
```bash
python scraper.py
```

### 3. Train the ML Models
To train the Random Forest models on the dataset and save them, run:
```bash
python train_model.py
```

### 4. Start the Application
You can start the app in two different modes:

#### Option A: Flask Server (Map-First Interface)
```bash
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your web browser.

#### Option B: Streamlit Dashboard (Tactile Sidebar Panel)
```bash
streamlit run streamlit_app.py
```
Open the local URL displayed in your terminal (usually **[http://localhost:8501](http://localhost:8501)**).

---

## 🌐 How to Deploy to Streamlit Cloud

1.  **Push the repository to GitHub** (see instructions below).
2.  Go to **[Streamlit Cloud](https://streamlit.io/cloud)** and log in with your GitHub account.
3.  Click **New App**.
4.  Select your repository, branch (e.g., `main`), and set the main file path to `streamlit_app.py`.
5.  Click **Deploy!**

---

## 🐙 Push to your GitHub Repository

Run these commands in your project terminal to push the codebase to your GitHub:

```bash
# 1. Initialize git
git init

# 2. Commit the changes
git add .
git commit -m "Initial commit: AutoProfit AI Chennai Driver Optimizer"

# 3. Add your remote repository link (Replace with your actual GitHub URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 4. Rename default branch to main and push
git branch -M main
git push -u origin main
```
*(Note: Make sure you create a new repository on GitHub first before running step 3).*
