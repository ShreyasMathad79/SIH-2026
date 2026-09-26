import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
"""
Data Upload & Management Page
CSV uploader with schema validation, missing column checks, interactive data preview, dataset activation, and disk persistence across system reboots.
"""

import streamlit as st
import pandas as pd

try:
    from src.state_helper import initialize_system_state, save_active_dataset, reset_to_sample_dataset, load_sample_dataset_cached
    from src.preprocessing import load_and_preprocess_data
except ModuleNotFoundError:
    from state_helper import initialize_system_state, save_active_dataset, reset_to_sample_dataset, load_sample_dataset_cached
    from preprocessing import load_and_preprocess_data
initialize_system_state()

st.title("📁 Data Upload & Station Dataset Management")
st.caption("Upload Custom Station Historical CSV Files or Reset to Default Polar Station Dataset | MoES / NCPOR")

st.markdown("""
### Upload Historical Station CSV
Upload a CSV file containing station microgrid historical time-series data. Data uploaded here will be **saved persistently** to disk and restored automatically whenever the device or server reboots.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.success(f"✓ File successfully read: **{uploaded_file.name}** ({len(raw_df)} rows, {len(raw_df.columns)} columns)")
        
        st.subheader("Data Preview (First 10 Rows)")
        st.dataframe(raw_df.head(10), use_container_width=True)
        
        st.subheader("Schema Validation")
        required_cols = ['timestamp', 'historical_load', 'temperature', 'wind_speed', 'solar_irradiance']
        missing_cols = [c for c in required_cols if c not in raw_df.columns]
        
        if missing_cols:
            st.error(f"❌ Dataset is missing required columns: {missing_cols}")
            st.warning("Required columns: timestamp, historical_load, temperature, wind_speed, solar_irradiance")
        else:
            st.success("✓ Schema Validation Passed! All required time-series columns are present.")
            
            # Preprocess test
            processed_df = load_and_preprocess_data(raw_df)
            
            st.subheader("Summary Statistics")
            st.dataframe(processed_df.describe(), use_container_width=True)
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("🚀 Set as Active Station Dataset (Persists on Reboot)", use_container_width=True):
                    label = f"Uploaded Dataset: {uploaded_file.name}"
                    st.session_state['data_df'] = processed_df
                    st.session_state['data_source_label'] = label
                    save_active_dataset(processed_df, label)
                    st.success("✓ Active dataset updated & saved to disk! It will now automatically load after any system restart.")
                    st.rerun()
                    
    except Exception as e:
        st.error(f"Error processing uploaded CSV file: {str(e)}")

st.markdown("---")

st.subheader("ℹ️ Current Active Dataset & Data Sources")
st.info(f"**Active Dataset Label**: {st.session_state.get('data_source_label')}")

if st.session_state.get('data_df') is not None:
    st.markdown(f"**Total Records**: {len(st.session_state['data_df'])} hours")
    st.dataframe(st.session_state['data_df'].head(5), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔄 Reset to Default Station Dataset", use_container_width=True):
    reset_to_sample_dataset()
    df = load_sample_dataset_cached()
    st.session_state['data_df'] = df
    st.session_state['data_source_label'] = "Simulated Polar Station Telemetry Dataset (Default)"
    st.success("✓ Custom dataset cleared. Reverted to default polar station dataset.")
    st.rerun()

st.markdown("""
> [!NOTE]
> **Data Transparency & Persistence Notice**:
> All active station datasets are saved directly to disk (`data/active_dataset.csv`). When you restart your device tomorrow, the system will automatically restore your data without needing to upload again.
""")
