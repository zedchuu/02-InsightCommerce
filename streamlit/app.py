import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- 1. Page Configuration ---
st.set_page_config(page_title="InsightCommerce AI", layout="wide")
st.title("🛍️ InsightCommerce: AI Customer Intelligence")
st.markdown("Predictive Churn & LTV Segmentation Engine built by [Your Name]")

# --- 2. Load the AI Brain (Cached for speed) ---
@st.cache_resource
def load_models():
    scaler = joblib.load('../sql/scaler.joblib')
    rf_model = joblib.load('../sql/rf_model.joblib')
    df = pd.read_parquet('../sql/dashboard_data.parquet')
    return scaler, rf_model, df

scaler, rf_model, df = load_models()

# Mapping the cluster numbers to Business Labels (Adjust these based on your K=4 results)
cluster_names = {
    0: "The Lost (High Churn)",
    1: "At-Risk Loyalists",
    2: "The Champions",
    3: "Recent Bargain Hunters"
}

# --- 3. Sidebar: User Input (The "What-If" Engine) ---
st.sidebar.header("Target Customer Profile")
st.sidebar.markdown("Input a customer's behavior to predict their segment.")

recency_input = st.sidebar.slider("Days Since Last Purchase (Recency)", 0, 365, 30)
frequency_input = st.sidebar.slider("Number of Shopping Trips (Frequency)", 1, 50, 5)
monetary_input = st.sidebar.number_input("Total Lifetime Spend (£)", min_value=1.0, value=500.0, step=50.0)

# --- 4. The Live Prediction Engine ---
# Apply the exact same "Systems Thinking" transformations we used in training
raw_data = np.array([[recency_input, frequency_input, monetary_input]])
log_data = np.log1p(raw_data) # 1. Log Transform
scaled_data = scaler.transform(log_data) # 2. Standard Scale

# Run the prediction
prediction = rf_model.predict(scaled_data)[0]
probabilities = rf_model.predict_proba(scaled_data)[0]
confidence = probabilities[prediction] * 100

# --- 5. Dashboard Layout: The Results ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("AI Prediction")
    st.info(f"**Assigned Segment:** {cluster_names[prediction]}")
    st.metric(label="Model Confidence", value=f"{confidence:.1f}%")
    
    st.markdown("---")
    st.subheader("Probability Breakdown")
    for i, prob in enumerate(probabilities):
        st.write(f"**{cluster_names[i]}:** {prob*100:.1f}%")

with col2:
    st.subheader("Customer Base 3D Topology")
    # Prepare data for 3D plot
    viz_df = df.copy()
    viz_df['Log_Recency'] = np.log1p(viz_df['recency'])
    viz_df['Log_Frequency'] = np.log1p(viz_df['frequency'])
    viz_df['Log_Monetary'] = np.log1p(viz_df['monetary'])
    viz_df['Segment'] = viz_df['Cluster'].map(cluster_names)
    
    # Generate the 3D Plot (Increased height and lowered opacity of the crowd)
    fig = px.scatter_3d(
        viz_df, x='Log_Recency', y='Log_Frequency', z='Log_Monetary',
        color='Segment', opacity=0.2,
        height=700,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # THE UPGRADE: Drop a giant red pin exactly where the sidebar slider is!
    fig.add_scatter3d(
        x=[log_data[0][0]], 
        y=[log_data[0][1]], 
        z=[log_data[0][2]],
        mode='markers',
        marker=dict(size=15, color='red', symbol='diamond'),
        name='🎯 Target Customer'
    )
    
    st.plotly_chart(fig, use_container_width=True)