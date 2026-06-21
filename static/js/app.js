/* ============================================================
   AUTOPROFIT AI — Frontend Application
   ============================================================ */

(function () {
    'use strict';

    // ── Constants ───────────────────────────────────────────
    const CHENNAI_CENTER = [13.0827, 80.2707];
    const DEFAULT_ZOOM   = 12;
    const API_PREDICT    = '/api/predict';
    const API_MODEL_INFO = '/api/model-info';

    const DEMAND_COLORS = {
        High:   '#00ff88',
        Medium: '#ffaa00',
        Low:    '#ff4444'
    };

    const DEMAND_FILL_OPACITY = {
        High:   0.35,
        Medium: 0.3,
        Low:    0.25
    };

    // ── DOM References ──────────────────────────────────────
    const $map             = document.getElementById('map');
    const $form            = document.getElementById('predictionForm');
    const $hourSelect      = document.getElementById('hourSelect');
    const $daySelect       = document.getElementById('daySelect');
    const $weatherSelect   = document.getElementById('weatherSelect');
    const $tempSlider      = document.getElementById('tempSlider');
    const $tempValue       = document.getElementById('tempValue');
    const $predictBtn      = document.getElementById('predictBtn');
    const $mapLoading      = document.getElementById('mapLoading');
    const $recList         = document.getElementById('recommendationsList');
    const $accuracyBar     = document.getElementById('accuracyBar');
    const $accuracyValue   = document.getElementById('accuracyValue');
    const $modelFeatures   = document.getElementById('modelFeatures');
    const $toastContainer  = document.getElementById('toastContainer');
    const $sidebarToggle   = document.getElementById('sidebarToggle');
    const $sidebar         = document.getElementById('sidebar');

    // ── Initialize Leaflet Map ──────────────────────────────
    const map = L.map('map', {
        center: CHENNAI_CENTER,
        zoom: DEFAULT_ZOOM,
        zoomControl: true,
        attributionControl: true
    });

    // Dark CartoDB tiles
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(map);

    // Layer group for zone markers
    const markersLayer = L.layerGroup().addTo(map);

    // ── Helpers ─────────────────────────────────────────────

    /** Get color for demand level */
    function getDemandColor(level) {
        return DEMAND_COLORS[level] || '#8888aa';
    }

    /** Show / hide loading overlay */
    function setLoading(active) {
        if (active) {
            $mapLoading.classList.add('active');
            $predictBtn.disabled = true;
        } else {
            $mapLoading.classList.remove('active');
            $predictBtn.disabled = false;
        }
    }

    /** Toast notification */
    function showToast(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        const icon = type === 'error' ? '⚠️' : '✅';
        toast.innerHTML = `<span class="toast-icon">${icon}</span><span>${message}</span>`;
        $toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('hiding');
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    /** Set default form values based on current date/time */
    function setDefaults() {
        const now = new Date();
        $hourSelect.value = now.getHours();
        // JS: 0=Sun, 1=Mon ... 6=Sat → map to our 0=Mon..6=Sun
        const jsDay = now.getDay();
        const mappedDay = jsDay === 0 ? 6 : jsDay - 1;
        $daySelect.value = mappedDay;
        $tempSlider.value = 32;
        $tempValue.textContent = '32°C';
    }

    /** Collect form values */
    function getFormData() {
        return {
            hour:        parseInt($hourSelect.value, 10),
            day_of_week: parseInt($daySelect.value, 10),
            weather:     $weatherSelect.value,
            temperature: parseInt($tempSlider.value, 10)
        };
    }

    // ── Build Popup HTML ────────────────────────────────────
    function buildPopup(zone) {
        const levelClass = (zone.demand_level || '').toLowerCase();
        const score = typeof zone.demand_score === 'number'
            ? zone.demand_score.toFixed(1)
            : zone.demand_score;

        return `
            <div class="popup-content">
                <div class="popup-zone">${zone.zone || zone.zone_name || 'Zone'}</div>
                <div class="popup-stat">
                    <span class="popup-stat-label">Demand</span>
                    <span class="popup-stat-value ${levelClass}">${zone.demand_level || '—'}</span>
                </div>
                <div class="popup-stat">
                    <span class="popup-stat-label">Score</span>
                    <span class="popup-stat-value ${levelClass}">${score}</span>
                </div>
                ${zone.recommendation ? `<div class="popup-recommendation">"${zone.recommendation}"</div>` : ''}
            </div>
        `;
    }

    // ── Render Zone Markers ─────────────────────────────────
    function renderMarkers(zones) {
        markersLayer.clearLayers();

        if (!zones || zones.length === 0) return;

        const bounds = [];

        zones.forEach(zone => {
            const lat = zone.lat || zone.latitude;
            const lng = zone.lng || zone.longitude;
            if (lat == null || lng == null) return;

            const latlng = [lat, lng];
            bounds.push(latlng);

            const color  = getDemandColor(zone.demand_level);
            const score  = typeof zone.demand_score === 'number' ? zone.demand_score : 50;
            const radius = Math.max(300, score * 8); // scale radius

            const circle = L.circleMarker(latlng, {
                radius:      Math.max(8, score / 5),
                color:       color,
                weight:      2,
                opacity:     0.9,
                fillColor:   color,
                fillOpacity: DEMAND_FILL_OPACITY[zone.demand_level] || 0.3,
                className:   zone.demand_level === 'High' ? 'pulse-marker' : ''
            });

            circle.bindPopup(buildPopup(zone), { maxWidth: 240, closeButton: true });
            markersLayer.addLayer(circle);

            // For high-demand zones, add a larger translucent circle as a "glow" and a custom auto-rickshaw marker badge!
            if (zone.demand_level === 'High') {
                const glow = L.circle(latlng, {
                    radius:      radius,
                    color:       color,
                    weight:      0,
                    fillColor:   color,
                    fillOpacity: 0.08
                });
                markersLayer.addLayer(glow);

                const customIcon = L.divIcon({
                    html: `<div class="auto-marker-badge animate-marker-pulse">
                        <img src="/static/images/3d_auto_toy.png" alt="3D Auto Toy" />
                    </div>`,
                    className: 'auto-div-marker',
                    iconSize: [48, 48],
                    iconAnchor: [24, 24]
                });

                const autoMarker = L.marker(latlng, { icon: customIcon });
                autoMarker.bindPopup(buildPopup(zone), { maxWidth: 240, closeButton: true });
                markersLayer.addLayer(autoMarker);
            }
        });

        // Fit map bounds
        if (bounds.length > 0) {
            map.fitBounds(bounds, { padding: [50, 50], maxZoom: 14 });
        }
    }

    // ── Render Top Recommendations Panel ────────────────────
    function renderRecommendations(zones) {
        if (!zones || zones.length === 0) {
            $recList.innerHTML = `
                <div class="empty-state">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3">
                        <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
                        <circle cx="12" cy="10" r="3"/>
                    </svg>
                    <p>No predictions available</p>
                </div>`;
            return;
        }

        // Sort by demand_score descending and take top 3
        const sorted = [...zones].sort((a, b) => (b.demand_score || 0) - (a.demand_score || 0));
        const top3 = sorted.slice(0, 3);
        const maxScore = top3[0]?.demand_score || 100;

        $recList.innerHTML = top3.map((zone, i) => {
            const rank   = i + 1;
            const score  = typeof zone.demand_score === 'number' ? zone.demand_score.toFixed(1) : zone.demand_score;
            const pct    = Math.min(100, ((zone.demand_score || 0) / maxScore) * 100);
            const color  = getDemandColor(zone.demand_level);
            const level  = zone.demand_level || 'Medium';
            const name   = zone.zone || zone.zone_name || `Zone ${rank}`;
            const rec    = zone.recommendation || 'Consider heading to this area.';

            return `
                <div class="rec-card" data-level="${level}">
                    <div class="rec-rank rec-rank--${rank}">${rank}</div>
                    <div class="rec-body">
                        <div class="rec-zone" title="${name}">${name}</div>
                        <div class="rec-score-container">
                            <div class="rec-score-bar-bg">
                                <div class="rec-score-bar" style="width:${pct}%;background:${color}"></div>
                            </div>
                            <span class="rec-score-value" style="color:${color}">${score}</span>
                        </div>
                        <div class="rec-recommendation">${rec}</div>
                    </div>
                </div>
            `;
        }).join('');
    }

    // ── Fetch Predictions ───────────────────────────────────
    async function fetchPredictions() {
        const data = getFormData();
        setLoading(true);

        try {
            const res = await fetch(API_PREDICT, {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify(data)
            });

            if (!res.ok) {
                const errBody = await res.json().catch(() => ({}));
                throw new Error(errBody.error || `Server error (${res.status})`);
            }

            const result = await res.json();
            const zones  = result.predictions || result.zones || result;

            if (!Array.isArray(zones)) {
                throw new Error('Unexpected response format');
            }

            renderMarkers(zones);
            renderRecommendations(zones);
            showToast(`Predictions updated — ${zones.length} zones analyzed`, 'success');

        } catch (err) {
            console.error('Prediction error:', err);
            showToast(err.message || 'Failed to fetch predictions', 'error');
        } finally {
            setLoading(false);
        }
    }

    // ── Fetch Model Info ────────────────────────────────────
    async function fetchModelInfo() {
        try {
            const res = await fetch(API_MODEL_INFO);
            if (!res.ok) throw new Error(`Status ${res.status}`);

            const info = await res.json();

            // Accuracy
            const accuracy = info.accuracy ?? info.score ?? null;
            if (accuracy !== null) {
                const pct = (accuracy * (accuracy <= 1 ? 100 : 1)).toFixed(1);
                $accuracyValue.textContent = pct + '%';
                setTimeout(() => { $accuracyBar.style.width = pct + '%'; }, 200);
            }

            // Top features
            const features = info.top_features || info.features || info.feature_importances || [];
            if (Array.isArray(features) && features.length > 0) {
                const maxImp = Math.max(...features.map(f => f.importance || f.value || 0));

                const featureHTML = features.slice(0, 5).map(f => {
                    const name = f.name || f.feature || 'Feature';
                    const imp  = f.importance || f.value || 0;
                    const pct  = maxImp > 0 ? ((imp / maxImp) * 100).toFixed(0) : 0;
                    const display = typeof imp === 'number' ? imp.toFixed(3) : imp;
                    return `
                        <div class="feature-item">
                            <span class="feature-name">${name}</span>
                            <div class="feature-bar-bg">
                                <div class="feature-bar" style="width:${pct}%"></div>
                            </div>
                            <span class="feature-value">${display}</span>
                        </div>
                    `;
                }).join('');

                $modelFeatures.innerHTML = `<span class="features-label">Top Features</span>${featureHTML}`;

                // Animate bars in
                setTimeout(() => {
                    document.querySelectorAll('.feature-bar').forEach(bar => {
                        bar.style.transition = 'width 1.2s ease-out';
                    });
                }, 100);
            }

        } catch (err) {
            console.warn('Could not load model info:', err);
            $accuracyValue.textContent = 'N/A';
        }
    }

    // ── Event Listeners ─────────────────────────────────────

    // Form submit
    $form.addEventListener('submit', function (e) {
        e.preventDefault();
        // Remove pulse animation class after first click
        $predictBtn.classList.remove('pulse-animation');
        fetchPredictions();
    });

    // Temperature slider live update
    $tempSlider.addEventListener('input', function () {
        $tempValue.textContent = this.value + '°C';
    });

    // Mobile sidebar toggle
    $sidebarToggle.addEventListener('click', function () {
        $sidebar.classList.toggle('collapsed');
    });

    // Fix map size when sidebar toggles on mobile
    $sidebarToggle.addEventListener('click', function () {
        setTimeout(() => map.invalidateSize(), 350);
    });

    // Fix map size on window resize
    window.addEventListener('resize', () => map.invalidateSize());

    // ── Initialization ──────────────────────────────────────
    function init() {
        setDefaults();
        fetchModelInfo();

        // Small delay before initial prediction so the map renders first
        setTimeout(() => {
            fetchPredictions();
        }, 600);
    }

    // Start when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
