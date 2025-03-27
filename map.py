import streamlit as st
import geopandas as gpd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import random

# Predefined list of arrondissements
arrondissements = ["Douala III", 'Douala', 'Douala V', 'Douala I', 'Yaoundé', 'Douala II', 
                   'Douala IV', 'Bafoussam', 'Dschang', 'Buea', 'Non precise', 'Kribi', 
                   'Njombe', 'Tiko', 'Edéa', 'Manjo', 'West', 'Oyack', 'Deido', 'Douala VI', 
                   'Batie', 'Bomono ba mbegue', 'Meiganga', 'Sud ouest tombel', 
                   'Ngodi bakoko', 'LimbÃ©', 'Dcankongmondo', 'Boko']

# Function to generate a random color
def random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f'rgb({r},{g},{b})'

# Streamlit app
st.title('Cameroon Arrondissements Map with Blood Markers')

# File uploader
#uploaded_file = st.file_uploader("Choose a GeoJSON file", type="json")
file_path = "data/geoBoundaries-CMR-ADM3.geojson"
# Open and read the JSON file
with open(file_path, 'r') as json_file:
    data = gpd.read_file(json_file)

if data is not None:
    # Read the GeoJSON file with GeoPandas
    gdf = data
    
    # Filter to only include predefined arrondissements
    gdf_filtered = gdf[gdf['shapeName'].isin(arrondissements)]
    
    # Create color map
    color_map = {arr: random_color() for arr in arrondissements}
    gdf_filtered['color'] = gdf_filtered['shapeName'].map(color_map)
    
    # Multiselect for places
    selected_places = st.multiselect(
        'Select places to focus on', 
        arrondissements,
        default=arrondissements
    )
    
    # Further filter if places are selected
    if selected_places:
        gdf_filtered = gdf_filtered[gdf_filtered['shapeName'].isin(selected_places)]
    
    # Calculate centroid for each feature
    gdf_filtered['centroid'] = gdf_filtered.geometry.centroid
    
    # Create plotly figure
    fig = px.choropleth_mapbox(
        gdf_filtered,
        geojson=gdf_filtered.geometry,
        locations=gdf_filtered.index,
        color='shapeName',
        color_discrete_map={name: color_map[name] for name in color_map if name in gdf_filtered['shapeName'].unique()},
        hover_name='shapeName',
        mapbox_style="open-street-map",
        center={
            "lat": gdf_filtered.geometry.centroid.y.mean(), 
            "lon": gdf_filtered.geometry.centroid.x.mean()
        },
        zoom=7,
        opacity=0.5
    )
    
    # Add blood drop markers
    blood_markers = []
    for idx, row in gdf_filtered.iterrows():
        marker = go.Scattermapbox(
            lat=[row.centroid.y],
            lon=[row.centroid.x],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                symbol='water',
            ),
            text=row['shapeName'],
            hoverinfo='text'
        )
        blood_markers.append(marker)
    
    # Add blood markers to the figure
    for marker in blood_markers:
        fig.add_trace(marker)
    
    # Customize layout
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        height=600
    )
    
    # Display the map
    st.plotly_chart(fig, use_container_width=True)
    
    # Display color legend
    st.subheader('Arrondissement Colors')
    color_display = []
    for arr in selected_places:
        color = color_map.get(arr, '#888888')
        color_display.append(f"<div style='background-color:{color};padding:5px;margin:2px;'>{arr}</div>")
    
    st.markdown(''.join(color_display), unsafe_allow_html=True)