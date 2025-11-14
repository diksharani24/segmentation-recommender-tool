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
    
    # --- FIX: Define fixed size and ensure all arrays match ---
    TOTAL_ROWS = 1500  # Define the total number of simulated transactions
    NUM_CUSTOMERS = 100
    
    # Generate Customer IDs, ensuring the final length matches TOTAL_ROWS
    customer_ids_temp = np.repeat(range(101, 101 + NUM_CUSTOMERS), 
                                  np.random.randint(5, 30, NUM_CUSTOMERS))
    
    # Handle length mismatch: trim if too long, pad if too short
    if len(customer_ids_temp) >= TOTAL_ROWS:
        customer_ids = customer_ids_temp[:TOTAL_ROWS]
    else:
        # If too short, randomly pick from the existing IDs to pad
        fill_count = TOTAL_ROWS - len(customer_ids_temp)
        random_fill = np.random.choice(customer_ids_temp, fill_count)
        customer_ids = np.concatenate((customer_ids_temp, random_fill))

    # Simulate transactional data
    data = {
        'CustomerID': customer_ids, # Length is now TOTAL_ROWS
        'InvoiceDate': pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(1, 365, TOTAL_ROWS), unit='D'),
        'Sales': np.random.uniform(10, 500, TOTAL_ROWS)
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
    # Using 'auto' for n_init to suppress the deprecation warning
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto') 
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


# -------------------------------------------------------------
# FIX: Dynamically determine the Champion Segment ID
# Champions are the segment with high M (top of the sorted profiles) and low R (most recent).
top_monetary_segments = profiles.head(3)
# Find the ID among the top monetary segments that has the LOWEST Recency
champion_id = top_monetary_segments['Avg. Recency (Days)'].idxmin()

# -------------------------------------------------------------

st.markdown("""
<style>
.low-r { color: green; font-weight: bold; }
.high-r { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

with st.expander("Interpretation Guide"):
    st.markdown(f"""
    - **Champions:** Usually **low** Avg. Recency (bought recently), **high** Avg. Frequency and Monetary. **(Segment ID {champion_id} in this run)**
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
