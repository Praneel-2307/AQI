import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------
# 1. PAGE CONFIG & FUTURISTIC GLASSMORPHISM STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="AERO-PULSE // AI AQI Telemetry & Prediction",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Dark Cyber Theme */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0f172a 0%, #050811 60%, #02040a 100%);
        color: #e2e8f0;
        font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
    }
    
    /* Neon Gradient Title */
    .cyber-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        background: linear-gradient(90deg, #00f5d4 0%, #00bbf9 50%, #9b5de5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .cyber-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }

    /* Glassmorphism Metric Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 245, 212, 0.25);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(0, 245, 212, 0.7);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-delta {
        font-size: 0.82rem;
        margin-top: 4px;
    }

    /* Advisory Box */
    .advisory-box {
        border-left: 4px solid #00f5d4;
        background: rgba(0, 245, 212, 0.08);
        padding: 16px 20px;
        border-radius: 0 12px 12px 0;
        margin-top: 12px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(7, 11, 22, 0.95);
        border-right: 1px solid rgba(0, 187, 249, 0.2);
    }
    
    /* Custom Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(15, 23, 42, 0.6);
        border-radius: 8px;
        padding: 10px 20px;
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(0, 245, 212, 0.18), rgba(0, 187, 249, 0.18));
        border: 1px solid #00f5d4 !important;
        color: #00f5d4 !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. CPCB SUB-INDEX, DATA & MODEL FUNCTIONS
# ---------------------------------------------------------
def get_PM10_subindex(x):
    if x <= 100: return x
    elif x <= 250: return 100 + (x - 100) * 100 / 150
    elif x <= 350: return 200 + (x - 250)
    elif x <= 430: return 300 + (x - 350) * 100 / 80
    elif x > 430: return 400 + (x - 430) * 100 / 80
    return 0

def get_PM25_subindex(x):
    if x <= 30: return x * 50 / 30
    elif x <= 60: return 50 + (x - 30) * 50 / 30
    elif x <= 90: return 100 + (x - 60) * 100 / 30
    elif x <= 120: return 200 + (x - 90) * 100 / 30
    elif x <= 250: return 300 + (x - 120) * 100 / 130
    elif x > 250: return 400 + (x - 250) * 100 / 130
    return 0

def get_SO2_subindex(x):
    if x <= 40: return x * 50 / 40
    elif x <= 80: return 50 + (x - 40) * 50 / 40
    elif x <= 380: return 100 + (x - 80) * 100 / 300
    elif x <= 800: return 200 + (x - 380) * 100 / 420
    elif x <= 1600: return 300 + (x - 800) * 100 / 800
    elif x > 1600: return 400 + (x - 1600) * 100 / 800
    return 0

def get_NOx_subindex(x):
    if x <= 40: return x * 50 / 40
    elif x <= 80: return 50 + (x - 40) * 50 / 40
    elif x <= 180: return 100 + (x - 80) * 100 / 100
    elif x <= 280: return 200 + (x - 180) * 100 / 100
    elif x <= 400: return 300 + (x - 280) * 100 / 120
    elif x > 400: return 400 + (x - 400) * 100 / 120
    return 0

def get_NH3_subindex(x):
    if x <= 200: return x * 50 / 200
    elif x <= 400: return 50 + (x - 200) * 50 / 200
    elif x <= 800: return 100 + (x - 400) * 100 / 400
    elif x <= 1200: return 200 + (x - 800) * 100 / 400
    elif x <= 1800: return 300 + (x - 1200) * 100 / 600
    elif x > 1800: return 400 + (x - 1800) * 100 / 600
    return 0

def get_CO_subindex(x):
    if x <= 1: return x * 50 / 1
    elif x <= 2: return 50 + (x - 1) * 50 / 1
    elif x <= 10: return 100 + (x - 2) * 100 / 8
    elif x <= 17: return 200 + (x - 10) * 100 / 7
    elif x <= 34: return 300 + (x - 17) * 100 / 17
    elif x > 34: return 400 + (x - 34) * 100 / 17
    return 0

def get_O3_subindex(x):
    if x <= 50: return x * 50 / 50
    elif x <= 100: return 50 + (x - 50) * 50 / 50
    elif x <= 168: return 100 + (x - 100) * 100 / 68
    elif x <= 208: return 200 + (x - 168) * 100 / 40
    elif x <= 748: return 300 + (x - 208) * 100 / 539
    elif x > 748: return 400 + (x - 400) * 100 / 539
    return 0

def classify_aqi(aqi):
    if aqi <= 50:
        return "Good", "#00f5d4", "Minimal impact. Ideal conditions for outdoor activities and natural ventilation."
    elif aqi <= 100:
        return "Satisfactory", "#38b000", "Minor breathing discomfort to sensitive individuals. Safe for general outdoor activity."
    elif aqi <= 200:
        return "Moderate", "#fee440", "Breathing discomfort to people with lung/heart disease, children, and older adults."
    elif aqi <= 300:
        return "Poor", "#f77f00", "Breathing discomfort to most people on prolonged exposure. Wear an N95 mask outdoors."
    elif aqi <= 400:
        return "Very Poor", "#d62828", "Respiratory illness on prolonged exposure. Avoid outdoor exertion; run indoor HEPA filtration."
    else:
        return "Severe", "#9b5de5", "Health emergency: affects healthy people and seriously impacts those with existing diseases. Stay indoors."

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("Air_Quality.csv", parse_dates=["Date"])
    df.rename(columns={"AQI_Bucket": "Air_quality"}, inplace=True)
    cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    df[cols] = df[cols].fillna(df[cols].mean())

    df["PM10_SubIndex"] = df["PM10"].astype(int).apply(get_PM10_subindex)
    df["PM2.5_SubIndex"] = df["PM2.5"].astype(int).apply(get_PM25_subindex)
    df["SO2_SubIndex"] = df["SO2"].astype(int).apply(get_SO2_subindex)
    df["NOx_SubIndex"] = df["NOx"].astype(int).apply(get_NOx_subindex)
    df["NH3_SubIndex"] = df["NH3"].astype(int).apply(get_NH3_subindex)
    df["CO_SubIndex"] = df["CO"].astype(int).apply(get_CO_subindex)
    df["O3_SubIndex"] = df["O3"].astype(int).apply(get_O3_subindex)

    sub_cols = ["PM2.5_SubIndex", "PM10_SubIndex", "SO2_SubIndex", "NOx_SubIndex", "NH3_SubIndex", "CO_SubIndex", "O3_SubIndex"]
    df["AQI"] = df["AQI"].fillna(round(df[sub_cols].max(axis=1)))
    df["Air_quality"] = df["Air_quality"].fillna(df["AQI"].apply(lambda x: classify_aqi(x)[0]))
    df.drop(columns=sub_cols, inplace=True)
    return df

@st.cache_resource
def get_trained_model(data):
    features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    X = data[features]
    y = data['AQI']
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf

df = load_and_clean_data()
model = get_trained_model(df)

# ---------------------------------------------------------
# 3. SIDEBAR TELEMETRY CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌍 REGIONAL TELEMETRY")
    selected_city = st.selectbox(
        "Target Monitoring Station",
        ["All Cities"] + sorted(df["City"].unique().tolist())
    )
    
    filtered_df = df if selected_city == "All Cities" else df[df["City"] == selected_city]
    
    min_date, max_date = filtered_df["Date"].min().date(), filtered_df["Date"].max().date()
    date_range = st.date_input("Observation Window", [min_date, max_date], min_value=min_date, max_value=max_date)
    if len(date_range) == 2:
        filtered_df = filtered_df[(filtered_df["Date"].dt.date >= date_range[0]) & (filtered_df["Date"].dt.date <= date_range[1])]

    st.divider()
    st.caption("ENGINE: RANDOM FOREST REGRESSOR")
    st.caption("STATUS: ONLINE // CPCB STANDARDS ACTIVE")

# ---------------------------------------------------------
# 4. HEADER & TOP KPI TELEMETRY STRIP
# ---------------------------------------------------------
st.markdown('<div class="cyber-title">AERO-PULSE // AI AIR QUALITY COMMAND</div>', unsafe_allow_html=True)
st.markdown('<div class="cyber-subtitle">Predictive Atmospheric Inference • CPCB Sub-Index Engine • Multi-City Telemetry</div>', unsafe_allow_html=True)

avg_aqi = filtered_df["AQI"].mean()
max_aqi = filtered_df["AQI"].max()
dominant_bucket = filtered_df["Air_quality"].mode()[0] if not filtered_df.empty else "N/A"
total_records = len(filtered_df)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Mean Station AQI</div>
        <div class="metric-value" style="color:#00f5d4;">{avg_aqi:.1f}</div>
        <div class="metric-delta" style="color:#94a3b8;">Region: {selected_city}</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Peak Recorded AQI</div>
        <div class="metric-value" style="color:#f77f00;">{max_aqi:.0f}</div>
        <div class="metric-delta" style="color:#94a3b8;">Critical Threshold Alert</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Dominant Air Profile</div>
        <div class="metric-value" style="color:#00bbf9;">{dominant_bucket}</div>
        <div class="metric-delta" style="color:#94a3b8;">Modal Classification</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Telemetry Samples</div>
        <div class="metric-value" style="color:#9b5de5;">{total_records:,}</div>
        <div class="metric-delta" style="color:#94a3b8;">Validated Sensor Logs</div>
    </div>""", unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
# 5. MAIN COMMAND TABS
# ---------------------------------------------------------
tab_predict, tab_analytics, tab_cities, tab_data = st.tabs([
    "🔮 AI Prediction & Sub-Index Simulator",
    "📈 Atmospheric Trends & Distributions",
    "🏙️ Cross-City Intelligence",
    "🗄️ Sensor Data Explorer"
])

# =========================================================
# TAB 1: PREDICTIVE INFERENCE & FULL DIAGNOSTICS
# =========================================================
with tab_predict:
    st.markdown("#### 1. Configure Atmospheric Sensor Inputs")
    
    preset_col1, preset_col2 = st.columns([1, 3])
    with preset_col1:
        preset = st.selectbox(
            "Load Sensor Preset",
            ["Custom Input", "Clean Morning Breeze", "Urban Rush Hour", "Industrial Smog Spike", "Station Average"]
        )
    
    defaults = {
        "Custom Input":          [72.8, 127.6, 17.6, 18.2, 17.1, 25.6, 0.92, 27.6, 133.3, 0.0, 0.02, 0.0],
        "Clean Morning Breeze":  [18.0, 42.0,  6.5,  11.0, 14.0, 12.0, 0.35, 8.0,  32.0,  0.1, 0.5,  0.1],
        "Urban Rush Hour":       [95.0, 185.0, 42.0, 58.0, 88.0, 34.0, 2.40, 24.0, 68.0,  3.2, 11.5, 2.8],
        "Industrial Smog Spike": [265.0, 440.0, 85.0, 110.0, 195.0, 75.0, 14.5, 95.0, 145.0, 9.8, 28.0, 7.4],
        "Station Average": [
            round(float(filtered_df["PM2.5"].mean()), 2),
            round(float(filtered_df["PM10"].mean()), 2),
            round(float(filtered_df["NO"].mean()), 2),
            round(float(filtered_df["NO2"].mean()), 2),
            round(float(filtered_df["NOx"].mean()), 2),
            round(float(filtered_df["NH3"].mean()), 2),
            round(float(filtered_df["CO"].mean()), 2),
            round(float(filtered_df["SO2"].mean()), 2),
            round(float(filtered_df["O3"].mean()), 2),
            round(float(filtered_df["Benzene"].mean()), 2),
            round(float(filtered_df["Toluene"].mean()), 2),
            round(float(filtered_df["Xylene"].mean()), 2),
        ]
    }[preset]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 1000.0, defaults[0], step=1.0)
        pm10 = st.number_input("PM10 (µg/m³)", 0.0, 1000.0, defaults[1], step=1.0)
        no   = st.number_input("NO (µg/m³)", 0.0, 500.0, defaults[2], step=0.5)
    with c2:
        no2  = st.number_input("NO2 (µg/m³)", 0.0, 500.0, defaults[3], step=0.5)
        nox  = st.number_input("NOx (ppb)", 0.0, 500.0, defaults[4], step=0.5)
        nh3  = st.number_input("NH3 (µg/m³)", 0.0, 2000.0, defaults[5], step=0.5)
    with c3:
        co   = st.number_input("CO (mg/m³)", 0.0, 100.0, defaults[6], step=0.1)
        so2  = st.number_input("SO2 (µg/m³)", 0.0, 2000.0, defaults[7], step=0.5)
        o3   = st.number_input("O3 (µg/m³)", 0.0, 1000.0, defaults[8], step=1.0)
    with c4:
        benzene = st.number_input("Benzene (µg/m³)", 0.0, 100.0, defaults[9], step=0.1)
        toluene = st.number_input("Toluene (µg/m³)", 0.0, 200.0, defaults[10], step=0.1)
        xylene  = st.number_input("Xylene (µg/m³)", 0.0, 100.0, defaults[11], step=0.1)

    st.divider()

    # Run Live Reactive Inference
    input_df = pd.DataFrame([{
        'PM2.5': pm25, 'PM10': pm10, 'NO': no, 'NO2': no2,
        'NOx': nox, 'NH3': nh3, 'CO': co, 'SO2': so2,
        'O3': o3, 'Benzene': benzene, 'Toluene': toluene, 'Xylene': xylene
    }])
    ml_pred = float(model.predict(input_df)[0])
    model_status = "Random Forest Regressor (100 Trees)"

    # Compute Exact CPCB Sub-Indices for Complete Explainability
    sub_indices = {
        "PM2.5": round(get_PM25_subindex(pm25), 1),
        "PM10":  round(get_PM10_subindex(pm10), 1),
        "NOx":   round(get_NOx_subindex(nox), 1),
        "NH3":   round(get_NH3_subindex(nh3), 1),
        "CO":    round(get_CO_subindex(co), 1),
        "SO2":   round(get_SO2_subindex(so2), 1),
        "O3":    round(get_O3_subindex(o3), 1),
    }
    cpcb_calculated_aqi = max(sub_indices.values())
    dominant_pollutant = max(sub_indices, key=sub_indices.get)
    
    final_aqi = ml_pred
    category, badge_color, health_advice = classify_aqi(final_aqi)
    delta_vs_city = final_aqi - avg_aqi

    # Results Output Grid
    res_col1, res_col2, res_col3 = st.columns([1.2, 1.3, 1.1])

    with res_col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=final_aqi,
            delta={'reference': avg_aqi, 'increasing': {'color': "#f77f00"}, 'decreasing': {'color': "#00f5d4"}},
            title={'text': f"PREDICTED AQI<br><span style='font-size:0.8em;color:{badge_color}'>{category.upper()}</span>", 'font': {'color': '#e2e8f0', 'size': 18}},
            number={'font': {'color': badge_color, 'size': 48}},
            gauge={
                'axis': {'range': [0, 500], 'tickcolor': "#94a3b8"},
                'bar': {'color': badge_color, 'thickness': 0.28},
                'bgcolor': "rgba(15,23,42,0.8)",
                'borderwidth': 1,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(0, 245, 212, 0.15)'},
                    {'range': [50, 100], 'color': 'rgba(56, 176, 0, 0.15)'},
                    {'range': [100, 200], 'color': 'rgba(254, 228, 64, 0.15)'},
                    {'range': [200, 300], 'color': 'rgba(247, 127, 0, 0.15)'},
                    {'range': [300, 400], 'color': 'rgba(214, 40, 40, 0.15)'},
                    {'range': [400, 500], 'color': 'rgba(155, 93, 229, 0.15)'}
                ],
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "#e2e8f0"},
            height=310,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        sub_df = pd.DataFrame({
            "Pollutant": list(sub_indices.keys()),
            "Sub-Index": list(sub_indices.values())
        }).sort_values("Sub-Index", ascending=True)

        fig_sub = px.bar(
            sub_df, x="Sub-Index", y="Pollutant", orientation="h",
            title="CPCB Pollutant Sub-Index Breakdown",
            color="Sub-Index",
            color_continuous_scale=["#00f5d4", "#00bbf9", "#fee440", "#f77f00", "#9b5de5"],
            text="Sub-Index"
        )
        fig_sub.update_traces(textposition="outside")
        fig_sub.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.4)",
            font={'color': "#e2e8f0"},
            coloraxis_showscale=False,
            height=310,
            margin=dict(l=10, r=30, t=45, b=20),
            xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.1)")
        )
        st.plotly_chart(fig_sub, use_container_width=True)

    with res_col3:
        st.markdown(f"""
        <div class="glass-card" style="border-color:{badge_color};">
            <div class="metric-label">Diagnostic Summary</div>
            <div style="font-size:1.1rem; margin-bottom:8px;">
                <b>Primary Driver:</b> <span style="color:#00f5d4;">{dominant_pollutant}</span> ({sub_indices[dominant_pollutant]})
            </div>
            <div style="font-size:0.95rem; margin-bottom:6px;">
                <b>ML Predicted AQI:</b> {ml_pred:.2f}
            </div>
            <div style="font-size:0.95rem; margin-bottom:6px;">
                <b>CPCB Formula AQI:</b> {cpcb_calculated_aqi:.1f}
            </div>
            <div style="font-size:0.95rem; margin-bottom:6px;">
                <b>Vs. {selected_city} Avg:</b> {delta_vs_city:+.1f} pts
            </div>
            <div style="font-size:0.8rem; color:#94a3b8; margin-top:8px;">
                Engine: {model_status}
            </div>
        </div>
        <div class="advisory-box" style="border-left-color:{badge_color};">
            <div style="font-weight:700; color:{badge_color}; margin-bottom:4px;">HEALTH & PROTOCOL ADVISORY</div>
            <div style="font-size:0.9rem; color:#e2e8f0;">{health_advice}</div>
        </div>
        """, unsafe_allow_html=True)

    # Feature Importance Diagnostics
    st.markdown("#### 2. Tree Ensemble Feature Weight Diagnostics")
    imp_df = pd.DataFrame({
        "Feature": model.feature_names_in_,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    fig_imp = px.bar(
        imp_df, x="Feature", y="Importance",
        color="Importance",
        color_continuous_scale=["#00bbf9", "#00f5d4", "#9b5de5"]
    )
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        font={'color': "#e2e8f0"},
        coloraxis_showscale=False,
        height=260,
        margin=dict(l=10, r=10, t=20, b=20)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# =========================================================
# TAB 2: ATMOSPHERIC TRENDS & DISTRIBUTIONS
# =========================================================
with tab_analytics:
    col_a, col_b = st.columns(2)
    with col_a:
        trend_df = filtered_df.groupby("Date")["AQI"].mean().reset_index()
        fig_line = px.area(
            trend_df, x="Date", y="AQI",
            title=f"Temporal AQI Trajectory ({selected_city})",
            color_discrete_sequence=["#00f5d4"]
        )
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.4)",
            font={'color': "#e2e8f0"},
            xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.1)")
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col_b:
        bucket_counts = filtered_df["Air_quality"].value_counts().reset_index()
        bucket_counts.columns = ["Air_quality", "Count"]
        color_map = {
            "Good": "#00f5d4", "Satisfactory": "#38b000", "Moderate": "#fee440",
            "Poor": "#f77f00", "Very Poor": "#d62828", "Severe": "#9b5de5"
        }
        fig_bar = px.bar(
            bucket_counts, x="Air_quality", y="Count", color="Air_quality",
            title="Air Quality Category Distribution",
            color_discrete_map=color_map
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.4)",
            font={'color': "#e2e8f0"},
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### Multi-Pollutant Correlation Matrix")
    pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'AQI']
    corr_matrix = filtered_df[pollutant_cols].corr().round(2)
    fig_corr = px.imshow(
        corr_matrix, text_auto=True, aspect="auto",
        color_continuous_scale=["#0f172a", "#00bbf9", "#00f5d4", "#9b5de5"]
    )
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e2e8f0"},
        height=420
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# =========================================================
# TAB 3: CROSS-CITY INTELLIGENCE
# =========================================================
with tab_cities:
    city_summary = df.groupby("City").agg(
        Mean_AQI=("AQI", "mean"),
        Max_AQI=("AQI", "max"),
        Mean_PM25=("PM2.5", "mean"),
        Mean_PM10=("PM10", "mean")
    ).reset_index().sort_values("Mean_AQI", ascending=False)

    fig_cities = px.bar(
        city_summary, x="City", y="Mean_AQI", color="Mean_AQI",
        title="National Station Ranking by Mean AQI",
        color_continuous_scale=["#00f5d4", "#fee440", "#f77f00", "#9b5de5"]
    )
    fig_cities.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        font={'color': "#e2e8f0"}
    )
    st.plotly_chart(fig_cities, use_container_width=True)
    st.dataframe(city_summary.style.format({"Mean_AQI": "{:.1f}", "Max_AQI": "{:.0f}", "Mean_PM25": "{:.1f}", "Mean_PM10": "{:.1f}"}), use_container_width=True)

# =========================================================
# TAB 4: SENSOR DATA EXPLORER
# =========================================================
with tab_data:
    st.markdown(f"#### Raw & Imputed Telemetry Records ({selected_city})")
    st.dataframe(filtered_df.head(200), use_container_width=True)
    csv_export = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export Filtered Telemetry CSV", csv_export, f"aqi_telemetry_{selected_city}.csv", "text/csv")