import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import datetime as dt

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Customer Segmentation Tool")

# --- 2. DATA SIMULATION (REPLACE with actual data loading) ---
@st.cache_data
def load_and_prepare_data():
    # Simulate transactional data
    data = {
        'CustomerID': np.repeat(range(101, 201), np.random.randint(5, 20, 100)),
        'InvoiceDate': pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(1, 365, 1250), unit='D'),
        'Sales': np.random.uniform(10, 500, 1250)
    }
    df = pd.DataFrame(data)
    
    # Calculate latest date for Recency calculation
    NOW = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    # --- RFM CALCULATION ---
    rfm_df = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (NOW - x.max()).days),
        Frequency=('InvoiceDate', 'count'),
        Monetary=('Sales', 'sum')
    ).reset_index()

    return rfm_df, NOW

# --- 3. RFM SCORING AND K-MEANS MODEL ---
def run_kmeans_model(rfm_df, k):
    
    # 3.1 Data Preparation (Log Transformation & Scaling)
    rfm_log = np.log1p(rfm_df[['Recency', 'Frequency', 'Monetary']])
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    # 3.2 K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # 3.3 Cluster Profiling (Analysis)
    cluster_profiles = rfm_df.groupby('Cluster').agg(
        Count=('CustomerID', 'count'),
        Mean_Recency=('Recency', 'mean'),
        Mean_Frequency=('Frequency', 'mean'),
        Mean_Monetary=('Monetary', 'mean')
    ).sort_values(by='Mean_Monetary', ascending=False)
    
    return rfm_df, cluster_profiles

# --- 4. STREAMLIT UI LAYOUT ---

st.title("🛍️ Customer Segmentation (RFM & K-Means)")
st.caption("A working model to analyze customer behavior and identify high-value segments.")

# Load Data
rfm_data, latest_date = load_and_prepare_data()

st.sidebar.header("Model Configuration")
k_clusters = st.sidebar.slider("Select Number of Clusters (k)", min_value=3, max_value=6, value=4)
st.sidebar.markdown(f"**Data Analyzed Up To:** {latest_date.strftime('%Y-%m-%d')}")

# Run Model
final_df, profiles = run_kmeans_model(rfm_data.copy(), k_clusters)

# --- RESULTS SECTION ---
st.header("1. Cluster Profiles")
st.markdown("Average RFM values for each segment, ordered by Mean Monetary Value.")

# Rename columns for clarity in display
profiles.index.name = 'Segment ID'
profiles.columns = ['Count', 'Avg. Recency (Days)', 'Avg. Frequency (Txns)', 'Avg. Monetary ($)']

st.dataframe(profiles.style.highlight_max(axis=0), use_container_width=True)

st.markdown("""
<style>
.low-r { color: green; font-weight: bold; }
.high-r { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

with st.expander("Interpretation Guide"):
    st.markdown("""
    - **Champions:** Usually **low** Avg. Recency (bought recently), **high** Avg. Frequency and Monetary. **(Segment ID 0 in this run)**
    - **At Risk:** High Avg. Recency (long time since last purchase), but decent Avg. Frequency and Monetary.
    - **New Customers:** Low Avg. Recency, but low F & M.
    """)

st.header("2. Cluster Visualization (3D Scatter)")
st.markdown("Visualizing segments across Recency, Frequency, and Monetary dimensions.")

# Create the 3D plot using Plotly
fig = px.scatter_3d(
    final_df, 
    x='Recency', 
    y='Frequency', 
    z='Monetary',
    color='Cluster',
    title="Customer Segments in RFM Space",
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.info("This application demonstrates the use of Pandas for Feature Engineering (RFM), Scikit-learn for Unsupervised Learning (K-Means), and Streamlit for web deployment.")
