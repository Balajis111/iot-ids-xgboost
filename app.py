import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(
    page_title="IoT Intrusion Detection System",
    layout="wide"
)

@st.cache_resource
def load_models():
    model  = joblib.load('models/xgboost_ids.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

@st.cache_resource
def load_multiclass_models():
    mc_model  = joblib.load('models/xgboost_multiclass.pkl')
    mc_scaler = joblib.load('models/scaler_multiclass.pkl')
    le_target = joblib.load('models/label_encoder_target.pkl')
    return mc_model, mc_scaler, le_target

model, scaler       = load_models()
mc_model, mc_scaler, le_target = load_multiclass_models()

st.title("Lightweight Explainable IDS for IoT Traffic")
st.markdown("Detects network intrusions using XGBoost + SHAP explainability")
st.divider()

st.sidebar.header("Upload Network Traffic Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV or Parquet file",
    type=["csv", "parquet"]
)

if uploaded_file is not None:

    # Load file
    if uploaded_file.name.endswith('.parquet'):
        df = pd.read_parquet(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    # Drop label columns if present
    for col in ['label', 'attack_cat']:
        if col in df.columns:
            df = df.drop(columns=[col])

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Preprocess
    df_clean = df.copy()
    cat_cols = df_clean.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    df_clean = df_clean.astype(float)

    # Scale for binary model
    df_scaled = pd.DataFrame(
        scaler.transform(df_clean),
        columns=df_clean.columns
    )

    # Scale for multiclass model
    df_mc_scaled = pd.DataFrame(
        mc_scaler.transform(df_clean),
        columns=df_clean.columns
    )

    # Binary predictions
    predictions   = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)

    # Metrics
    st.subheader("Prediction Results")
    col1, col2, col3 = st.columns(3)
    total   = len(predictions)
    attacks = int(predictions.sum())
    normals = total - attacks
    col1.metric("Total connections", total)
    col2.metric("Attacks detected",  attacks,
                delta=f"{attacks/total*100:.1f}%",
                delta_color="inverse")
    col3.metric("Normal traffic",    normals,
                delta=f"{normals/total*100:.1f}%")

    # Prediction table
    result_df = df_clean.copy()
    result_df['prediction'] = [
        'Attack' if p == 1 else 'Normal' for p in predictions
    ]
    result_df['confidence'] = [
        f"{max(p)*100:.1f}%" for p in probabilities
    ]
    st.dataframe(result_df[['prediction', 'confidence']].head(20))

    # Download button
    csv = result_df.to_csv(index=False)
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name="ids_predictions.csv",
        mime="text/csv"
    )

    # Attack type breakdown
    st.subheader("Attack Type Breakdown")
    attack_type_preds  = mc_model.predict(df_mc_scaled)
    attack_labels      = le_target.inverse_transform(attack_type_preds)
    type_counts        = pd.Series(attack_labels).value_counts().reset_index()
    type_counts.columns = ['Attack Type', 'Count']

    import plotly.express as px
    fig = px.bar(
        type_counts,
        x='Attack Type',
        y='Count',
        title='Detected Attack Types',
        color='Attack Type'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
    st.dataframe(type_counts)

    # SHAP
    st.subheader("SHAP Explainability — Why did the model decide this?")
    st.markdown("Top features driving the predictions:")
    with st.spinner("Generating SHAP explanation..."):
        explainer   = shap.TreeExplainer(model)
        sample      = df_scaled.iloc[:200]
        shap_values = explainer.shap_values(sample)
        fig2, ax    = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, sample,
                          plot_type="bar", show=False)
        st.pyplot(fig2)

else:
    st.info("Upload a network traffic file from the sidebar to get started")

    st.subheader("How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1. Upload")
        st.markdown("Upload network traffic data as CSV or Parquet")
    with col2:
        st.markdown("### 2. Detect")
        st.markdown("XGBoost classifies each connection as Normal or Attack")
    with col3:
        st.markdown("### 3. Explain")
        st.markdown("SHAP shows exactly why each prediction was made")