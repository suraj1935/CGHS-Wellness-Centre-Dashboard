#streamlit_app
#streamlit_app
# MUST BE FIRST: Streamlit configuration and imports
import streamlit as st
st.set_page_config(layout="wide")

# Import other required packages
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LinearRegression

# Load Data
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

# App Title
st.title("CGHS Wellness Centers: Healthcare Optimization Dashboard")
st.markdown("Explore doctor availability, center distribution, underserved zones, and actionable insights.")

# Sidebar filter
st.sidebar.header("City Filter")
cities = sorted(df['City'].dropna().unique())
selected_city = st.sidebar.selectbox("Select City", options=["All Cities"] + cities)

if selected_city != "All Cities":
    df = df[df['City'] == selected_city]

# Tabs for Sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. EDA",
    "2. Geospatial",
    "3. Clustering",
    "4. AI Modeling",
    "5. Recommendations"
])

# -------- TAB 1: EDA --------
with tab1:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Centers by City")
        city_counts = df['City'].value_counts().reset_index()
        city_counts.columns = ['City', 'count']
        st.plotly_chart(px.bar(
            city_counts,
            x='City',
            y='count',
            labels={'count': 'Number of Centers'},
            height=400
        ), use_container_width=True)

    with col2:
        st.subheader("Category Breakdown")
        category_counts = df['Category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'count']
        st.plotly_chart(px.pie(
            category_counts,
            names='Category',
            values='count',
            height=400
        ), use_container_width=True)

    st.subheader("Doctor Count per City")
    city_doctors = df.groupby('City')['DoctorCount'].sum().reset_index().sort_values(by='DoctorCount', ascending=False)
    st.plotly_chart(px.bar(
        city_doctors,
        x='City',
        y='DoctorCount',
        labels={'DoctorCount': 'Total Doctors'},
        height=500
    ), use_container_width=True)

# -------- TAB 2: Geospatial --------
with tab2:
    st.subheader("Geographic Distribution of Centers")

    map1 = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=5)
    for _, row in df.iterrows():
        popup = f"""
        <b>{row['wellnessCenterName']}</b><br>
        City: {row['City']}<br>
        Doctors: {row['DoctorCount']}<br>
        Category: {row['Category']}
        """
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5 + row['DoctorCount'] * 0.5,
            popup=popup,
            color='blue',
            fill=True,
            fill_opacity=0.6
        ).add_to(map1)
    folium_static(map1, width=1000, height=600)

# -------- TAB 3: Clustering --------
with tab3:
    st.subheader("Cluster Analysis (Underserved Zones)")

    kmeans_data = df[['latitude', 'longitude']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(kmeans_data)

    n_clusters = min(5, len(df) // 2) if len(df) > 5 else 1

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(scaled)
    df['Cluster'] = kmeans.labels_

    cluster_map = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=5)
    colors = ['red', 'green', 'blue', 'purple', 'orange']

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            color=colors[row['Cluster'] % len(colors)],
            fill=True,
            fill_opacity=0.7,
            popup=f"Cluster {row['Cluster']}"
        ).add_to(cluster_map)

    for i, center in enumerate(scaler.inverse_transform(kmeans.cluster_centers_)):
        folium.Marker(
            location=[center[0], center[1]],
            icon=folium.Icon(color=colors[i % len(colors)]),
            popup=f"Center of Cluster {i}"
        ).add_to(cluster_map)

    folium_static(cluster_map, width=1000, height=600)

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

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Actual vs Predicted")
            st.plotly_chart(px.scatter(
                df,
                x='DoctorCount',
                y='PredictedDoctors',
                color='City',
                height=400,
                labels={
                    'DoctorCount': 'Actual Doctors',
                    'PredictedDoctors': 'Predicted Doctors'
                }
            ), use_container_width=True)

        with col2:
            st.subheader("Under-Resourced Centers")
            under = df[df['Gap'] > 2].sort_values('Gap', ascending=False)
            st.dataframe(under[['City', 'wellnessCenterName', 'DoctorCount', 'PredictedDoctors', 'Gap']].head(10))
    else:
        st.warning("Not enough data for predictions in this city.")

# -------- TAB 5: Recommendations --------
with tab5:
    st.subheader("Strategic Optimization Recommendations")

    rec1 = "Geographic Coverage: "
    rec1 += f"{n_clusters} clusters identified. Improve access between clusters."

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

# Footer
st.markdown("---")
st.caption("CGHS Wellness Center Dashboard | Source: MoHFW, India")
