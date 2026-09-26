# ❄ AI-Driven Smart Energy Management System for Polar Research Stations

**Organization**: Ministry of Earth Sciences (MoES)  
**Department**: National Centre for Polar and Ocean Research (NCPOR)  
**Category**: Clean & Green Technology (Software Prototype)  

---

## 1. Problem Statement & Context

Isolated Antarctic research stations (such as **Maitri** and **Bharati** operated by NCPOR) operate under extreme sub-zero weather conditions (-45°C cold, blizzards, polar night). Electricity and heating supply relies on a hybrid microgrid combining:
* Solar Photovoltaic (PV) Panels
* Wind Turbines
* Battery Energy Storage System (BESS)
* Diesel Generators (using difficult-to-replenish polar diesel fuel)

The goal of this software prototype is to provide an **intelligent decision-support and energy-management system** that:
1. Forecasts future station electricity demand.
2. Forecasts renewable energy availability (Solar & Wind).
3. Optimizes the use of solar, wind, battery, and diesel power.
4. Minimizes diesel fuel consumption and carbon emissions.
5. Preserves emergency battery reserves.
6. Protects critical life-support and heating loads.
7. Detects upcoming energy deficits through a **24/48/72-Hour Energy Survival Forecast**.
8. Provides automated operator recommendations through a professional interactive dashboard.

---

## 2. System Architecture & Pipeline

```
Historical Data + Weather Data
          ↓
   Data Processing & Feature Engineering (Cyclical Encodings + Lags)
          ↓
   ML Forecasting Engine (XGBoost Regressors)
   ┌───────────────────────────────────────────┐
   │ 1. Station Load Demand Forecast (kW)      │
   │ 2. Solar PV Generation Forecast (kW)      │
   │ 3. Wind Turbine Generation Forecast (kW)  │
   └─────────────────────┬─────────────────────┘
                         ↓
         PuLP MILP Energy Optimizer (Cost Minimization)
                         ↓
   ┌─────────────────────┼─────────────────────┐
   ↓                     ↓                     ↓
Solar PV              Battery               Diesel
   ↓                     ↓                     ↓
┌──────────────────────────────────────────────┐
│           Station Load Management            │
│ Tier 1 Critical | Tier 2 Important | Tier 3  │
└──────────────────────┬───────────────────────┘
                       ↓
         Dashboard + Alerts + Recommendations
```

> **Design Principle**: ML models predict future demand and generation potential. A separate **Mixed-Integer Linear Programming (MILP)** optimization layer decides optimal dispatch actions across generation and storage assets.

---

## 3. Dataset & Data Sources

### Simulated Polar Research Station Dataset
Due to security and logistical constraints surrounding live station operational feeds, this prototype includes a physics-informed **"Simulated Polar Research Station Dataset"** (`data/sample/polar_station_data.csv`) modeled on coastal Antarctic station specifications:
* **Solar PV Peak**: 60 kW
* **Wind Turbine Peak**: 80 kW
* **Battery Storage**: 200 kWh capacity (min SOC 20%, max SOC 95%)
* **Diesel Generator**: 100 kW max capacity (min stable load 15 kW)
* **Time Resolution**: 1-Hour intervals (8,760 hours = 1 full year)

> **Data Transparency Notice**: In accordance with system guidelines, simulated dataset values are strictly labeled as **"Simulated Polar Research Station Dataset"**.

---

## 4. Machine Learning & Forecasting Methodology

### Models & Algorithms
* **Load Forecasting Model**: `XGBoostRegressor` predicting station power demand in kW.
* **Solar Forecasting Model**: `XGBoostRegressor` trained on solar irradiance, ambient temperature, and diurnal solar elevation angles.
* **Wind Forecasting Model**: `XGBoostRegressor` trained on Weibull wind speed distributions and turbine power curves.

### Preprocessing & Feature Engineering
* **Cyclical Time Transformations**: `sin_hour`, `cos_hour`, `sin_month`, `cos_month`, `sin_day_of_year`, `cos_day_of_year`.
* **Lag Features**: `previous_hour_load`, `previous_24_hour_load`, `rolling_mean_load_6h`, `rolling_mean_load_24h`.
* **Chronological Splitting**: 70% Train, 15% Validation, 15% Test without random shuffling.
* **Metrics**: MAE, RMSE, R², and MAPE computed strictly from test set.

---

## 5. Energy Optimization Engine (PuLP MILP)

The optimizer solves a Mixed-Integer Linear Program minimizing:

$$\text{Total Cost} = \text{Diesel Fuel Cost} + \text{Battery Degradation Cost} + \text{Renewable Curtailment Cost} + \text{Unmet Load Penalty}$$

### Priority Load Tiers
1. **Tier 1 - Critical Loads (50% of total)**: Heating, medical systems, life support, emergency comms. Penalty: Extremely High ($10,000/kWh).
2. **Tier 2 - Important Loads (30% of total)**: Lab tools, computers, water processing. Penalty: High ($500/kWh).
3. **Tier 3 - Non-Critical Loads (20% of total)**: EV charging, optional heating, extra tools. Penalty: Moderate ($50/kWh).

---

## 6. Physical Asset Models

* **Battery Model**: Configurable 200 kWh BESS with 92% round-trip efficiency, 20% emergency reserve protection, and max 40 kW charge / 50 kW discharge rate limits.
* **Diesel Generator Model**: 100 kW max capacity, 15 kW min stable loading, specific fuel consumption ~0.28 L/kWh, fuel tank tracking, and endurance estimation.

---

## 7. Installation & Quick Start

### Prerequisites
* Python 3.10+ (Tested on Python 3.13)

### Installation
```bash
# 1. Clone repository & change directory
cd polar_energy_system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate sample dataset & train initial ML models
python src/data_generator.py
python train_initial_models.py

# 4. Launch Streamlit Application
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 8. Application Navigation & Features

1. **📊 Main Dashboard**: Real-time KPI cards, live 24h trajectory chart, energy survival forecast, and alerts.
2. **📡 Real-Time Monitoring**: Live IoT sensor telemetry, 32-cell BESS voltage distribution, diesel engine gauges, and 3-phase AC power quality waveforms.
3. **📈 AI Forecast**: Actual vs Predicted Load, Solar, and Wind charts with MAE/RMSE/R² metrics.
4. **⚡ Energy Optimization**: PuLP MILP hourly dispatch schedule, stacked area breakdown, and fuel savings.
5. **🔀 Energy Flow**: Visual microgrid topology and Plotly Sankey energy routing diagram.
6. **🚨 Alerts & Recommendations**: Dynamic AI advisories and polar risk warnings.
7. **🧪 Simulation Mode**: Stress-test station resilience under 8 extreme polar scenarios (e.g. *Severe Polar Storm*, *Polar Night*).
8. **⚖️ Before vs After**: Side-by-side comparison of Baseline Strategy vs AI Optimization Strategy.
9. **📁 Data Upload**: Upload custom CSV station files with automated schema validation.
10. **⚙️ Model Training**: Retrain load and renewable models and download `.pkl` model binaries.

---

## 9. Future Extensions
* Real-time IoT MQTT sensor streaming
* Satellite weather data integration (Copernicus / ECMWF)
* Deep Reinforcement Learning (DRL) microgrid control
* Multi-station microgrid coordination (Maitri + Bharati)
