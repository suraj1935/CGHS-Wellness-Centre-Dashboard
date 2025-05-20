#app
import streamlit as st
st.set_page_config(layout="wide")

# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_excel("cleaned dataset.xlsx")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'wellnessCentreCode': 'wellnessCenterCode',
        'WellnessCentreAddress': 'wellnessCenterAddress'
    })
    df['DoctorCount'] = pd.to_numeric(df['DoctorCount'], errors='coerce').fillna(0)
    df.dropna(subset=['latitude', 'longitude'], inplace=True)
    return df

df = load_data()

# --- App Title ---
st.title("CGHS Wellness Centers: Healthcare Optimization Dashboard")
st.markdown("Explore doctor availability, center distribution, underserved zones, and actionable insights.")

# --- Sidebar Filter ---
st.sidebar.header("City Filter")
cities = sorted(df['City'].dropna().unique())
selected_city = st.sidebar.selectbox("Select City", options=["All Cities"] + cities)
filtered_df = df if selected_city == "All Cities" else df[df['City'] == selected_city]

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. EDA", "2. Geospatial", "3. Clustering", "4. AI Modeling", "5. Recommendations"
])

# -------- TAB 1: EDA --------
with tab1:
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Centers by City")
        city_counts = df['City'].value_counts().reset_index()
        city_counts.columns = ['City', 'Number of Centers']
        st.bar_chart(city_counts.set_index('City'))

    with col2:
        st.subheader("Category Breakdown")
        category_counts = df['Category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        st.dataframe(category_counts)

    st.subheader("Doctor Count per City")
    city_doctors = df.groupby('City')['DoctorCount'].sum().reset_index().sort_values(by='DoctorCount', ascending=False)
    st.bar_chart(city_doctors.set_index('City'))

# -------- TAB 2: Geospatial --------
with tab2:
    st.subheader("Geographic Distribution of Centers")

    st.map(df[['latitude', 'longitude']].rename(columns={'latitude': 'lat', 'longitude': 'lon'}))

    st.markdown("### 🔍 Explore a Specific City")
    selected_geo_city = st.selectbox("Choose city to zoom into:", options=cities)
    city_subset = df[df["City"] == selected_geo_city]
    st.info(f"Total centers in **{selected_geo_city}**: {len(city_subset)}")
    st.map(city_subset[['latitude', 'longitude']].rename(columns={'latitude': 'lat', 'longitude': 'lon'}))

# -------- TAB 3: Clustering --------
with tab3:
    st.subheader("Cluster Analysis (Underserved Zones)")

    kmeans_data = df[['latitude', 'longitude']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(kmeans_data)

    n_clusters = min(5, len(df) // 2) if len(df) > 5 else 1
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(scaled)
    df['Cluster'] = kmeans.labels_

    st.dataframe(df[['wellnessCenterName', 'City', 'Cluster']].sort_values('Cluster'))

# -------- TAB 4: AI Modeling --------
with tab4:
    st.subheader("AI-Based Doctor Allocation Prediction")

    model_data = pd.get_dummies(df[['latitude', 'longitude', 'Category']], drop_first=True)
    y = df['DoctorCount']

    if len(df) > 10:
        model = LinearRegression()
        model.fit(model_data, y)
        df['PredictedDoctors'] = model.predict(model_data)
        df['Gap'] = df['PredictedDoctors'] - df['DoctorCount']

        st.subheader("Actual vs Predicted (First 20 Rows)")
        st.dataframe(df[['City', 'wellnessCenterName', 'DoctorCount', 'PredictedDoctors', 'Gap']].head(20))

        st.subheader("Under-Resourced Centers (Gap > 2)")
        under = df[df['Gap'] > 2].sort_values('Gap', ascending=False)
        st.dataframe(under[['City', 'wellnessCenterName', 'DoctorCount', 'PredictedDoctors', 'Gap']].head(10))
    else:
        st.warning("Not enough data for predictions in this city.")

# -------- TAB 5: Recommendations --------
with tab5:
    st.subheader("Strategic Optimization Recommendations")

    rec1 = f"Geographic Coverage: {n_clusters} clusters identified. Improve access between clusters."

    rec2 = "Doctor Allocation: "
    if 'Gap' in df.columns and df['Gap'].max() > 0:
        rec2 += f"Some centers may need up to {df['Gap'].max():.1f} more doctors."
    else:
        rec2 += "Current allocation seems balanced."

    rec3 = "Center Type Diversity: "
    category_dist = df['Category'].value_counts(normalize=True)
    if len(category_dist) > 0:
        dominant = category_dist.idxmax()
        rec3 += f"{dominant} dominates ({category_dist[dominant]*100:.1f}%). Diversification may help."

    st.markdown(f"""
    **{rec1}**

    **{rec2}**

    **{rec3}**

    **Additional Suggestions:**
    - Expand into areas with no current clusters.
    - Balance alternative and allopathic medicine centers.
    - Prioritize high-need locations based on prediction gap.
    """)

# --- Footer ---
st.markdown("---")
st.caption("CGHS Wellness Center Dashboard | Source: MoHFW, India")
