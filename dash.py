import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import pickle
from wordcloud import WordCloud
import io
from datetime import datetime, date


config = {
    "toImageButtonOptions": {
        "format": "png",
        "filename": "custom_image",
        "height": 720,
        "width": 480,
        "scale": 6,
    }
}

# Initialization Functions
def initialize_session_state():
    if 'new_candidates' not in st.session_state:
        st.session_state.new_candidates = pd.DataFrame()
    if 'new_donors' not in st.session_state:
        st.session_state.new_donors = pd.DataFrame()

# Configuration of the Web page
def configure_page():
    st.set_page_config(
        page_title="Blood Donation Dashboard",
        page_icon="🩸",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# UI Component Functions
def render_header():
    col1,col2 = st.columns(2)
    with col1:
        st.image("Images/blood2.png", width= 200)
    with col2:
        st.markdown("""
            <style>
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: linear-gradient(to right, #fff1f2, #fee2e2);
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .logo-section {
                display: flex;
                align-items: center;
            }
            .title {
                color: #991b1b;
                font-size: 2.5em;
                font-weight: bold;
            }
            .subtitle {
                color: #dc2626;
                font-size: 1.2em;
            }
            .tagline {
                text-align: right;
                color: #7f1d1d;
            }
            </style>
            <div class="header">
                <div class="logo-section">
                    <div>
                        <div class="subtitle">Blood Donation Platform</div>
                    </div>
                </div>
                <div class="tagline">
                    <div style="font-size: 1.2em; color: #dc2626;">
                        ❤️ Every Drop Saves Lives
                    </div>
                    <div style="font-size: 1em; color: #7f1d1d;">
                        Connecting Donors | Saving Communities
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

def render_styles():
    st.markdown("""
        <style>
            .main-header {font-size: 2.5rem; color: #B22222; text-align: center; margin-bottom: 1rem;}
            .sub-header {font-size: 1.8rem; color: #8B0000; margin-top: 1rem;}
            .metric-container {background-color: #F8F8F8; border-radius: 5px; padding: 1rem; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
            .metric-container2 {background-color: red; border-radius: 5px; padding: 1rem; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
            .highlight {color: black; font-weight: bold;}
            .chart-container {background-color: white; border-radius: 5px; padding: 1rem; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); margin: 1rem 0;}
            .footer {text-align: center; font-size: 0.8rem; color: gray; margin-top: 2rem;}
            body {
                background: linear-gradient(-45deg, darkred, red, orangered, tomato);
                background-size: 400% 400%;
                animation: gradient 15s ease infinite;
            }
            @keyframes gradient {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
        </style>
        <div class="main-header">🩸 Blood Donation Campaign Dashboard</div>
    """, unsafe_allow_html=True)

def process_candidate_data(df):
    candidates_columns = {
        "Date de remplissage de la fiche": "form_fill_date",
        "Date de naissance": "birth_date",
        "Age": "age",
        "Niveau d'etude": "education_level",
        "Genre": "gender",
        "Taille": "height",
        "Poids": "weight",
        "Situation Matrimoniale (SM)": "marital_status",
        "Profession": "profession",
        "Arrondissement de residence": "residence_district",
        "Quartier de Residence": "residence_neighborhood",
        "Nationalite": "nationality",
        "Religion": "religion",
        "A-t-il (elle) deja donne le sang": "has_donated_before",
        "Si oui preciser la date du dernier don.": "last_donation_date",
        "Taux d'hemoglobine": "hemoglobin_level",
        "ELIGIBILITE AU DON.": "eligibility",
        "Raison indisponibilite  [Est sous anti-biotherapie  ]": "ineligible_antibiotics",
        "Raison indisponibilite  [Taux d'hemoglobine bas ]": "ineligible_low_hemoglobin",
        "Raison indisponibilite  [date de dernier Don < 3 mois ]": "ineligible_recent_donation",
        "Raison indisponibilite  [IST recente (Exclu VIH, Hbs, Hcv)]": "ineligible_recent_sti",
        "Date de dernieres regles (DDR)": "last_menstrual_date",
        "Raison de l'indisponibilite de la femme [La DDR est mauvais si <14 jour avant le don]": "female_ineligible_menstrual",
        "Raison de l'indisponibilite de la femme [Allaitement ]": "female_ineligible_breastfeeding",
        "Raison de l'indisponibilite de la femme [A accoucher ces 6 derniers mois  ]": "female_ineligible_postpartum",
        "Raison de l'indisponibilite de la femme [Interruption de grossesse  ces 06 derniers mois]": "female_ineligible_miscarriage",
        "Raison de l'indisponibilite de la femme [est enceinte ]": "female_ineligible_pregnant",
        "Autre raisons,  preciser": "other_reasons",
        "Selectionner \"ok\" pour envoyer": "submission_status",
        "Raison de non-eligibilite totale  [Antecedent de transfusion]": "total_ineligible_transfusion_history",
        "Raison de non-eligibilite totale  [Porteur(HIV,hbs,hcv)]": "total_ineligible_hiv_hbs_hcv",
        "Raison de non-eligibilite totale  [Opere]": "total_ineligible_surgery",
        "Raison de non-eligibilite totale  [Drepanocytaire]": "total_ineligible_sickle_cell",
        "Raison de non-eligibilite totale  [Diabetique]": "total_ineligible_diabetes",
        "Raison de non-eligibilite totale  [Hypertendus]": "total_ineligible_hypertension",
        "Raison de non-eligibilite totale  [Asthmatiques]": "total_ineligible_asthma",
        "Raison de non-eligibilite totale  [Cardiaque]": "total_ineligible_heart_disease",
        "Raison de non-eligibilite totale  [Tatoue]": "total_ineligible_tattoo",
        "Raison de non-eligibilite totale  [Scarifie]": "total_ineligible_scarification",
        "Si autres raison preciser": "other_total_ineligible_reasons"
    }
    df.rename(columns=candidates_columns, inplace=True)
    
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    if 'eligibility' in df.columns:
        df['is_eligible'] = df['eligibility'].apply(
            lambda x: 1 if str(x).lower() in ['yes', 'oui', '1', 'true', 'eligible'] else 0
        )
    return df

def process_donor_data(df):
    donors_columns = {
        "Horodateur": "timestamp",
        "Sexe": "gender",
        "Age": "age",
        "Type de donation": "donation_type",
        "Groupe Sanguin ABO/Rhesus": "blood_group",
        "Phenotype": "phenotype"
    }
    df.rename(columns=donors_columns, inplace=True)
    return df

def render_sidebar():
    with st.sidebar:
        st.image("Images/codeflow.png", width=200)
            
        st.markdown("## **Navigation**")
        expander = st.expander("⚒ **Menu**")
        with expander:
            page = st.radio("Select Dashboard Page", [
                "Home Page","Overview", "Geographic Distribution", "Health Conditions", 
                "Donor Profiles", "Campaign Effectiveness", "Donor Retention", 
                "Sentiment Analysis", "Eligibility Prediction", "Data Collection"
            ])

        expander = st.expander("🗀 File Input")
        with expander:
            file_uploader_key = "file_uploader_{}".format(
                st.session_state.get("file_uploader_key", False)
            )

            uploaded_files = st.file_uploader(
                "Upload local files:",
                type=["csv","xls"],
                key=file_uploader_key,
                accept_multiple_files=True,
            )

            #uploaded_files = st.file_uploader("Choose CSV files", type=['csv','xls'], accept_multiple_files=True)
            donors, donor_candidates_birth, campaigns = None, None, None
            
            if uploaded_files:
                for file in uploaded_files:
                    if file.name.lower().startswith('donor'):
                        donors = pd.read_csv(file)
                        st.success(f"Successfully loaded donors dataset from {file.name}")
                        #st.dataframe(donors.head())
                    elif file.name.lower().startswith('candidates'):
                        donor_candidates_birth = pd.read_csv(file)
                        st.success(f"Successfully loaded candidates dataset from {file.name}")
                        #st.dataframe(donor_candidates_birth.head())
                    elif file.name.lower().startswith('campaign'):
                        campaigns = pd.read_excel(file)
                        st.success(f"Successfully loaded campaigns dataset from {file.name}")
                        #st.dataframe(campaigns.head())
                    else:
                        st.warning(f"Unrecognized file: {file.name}")

            if uploaded_files is not None:
                st.session_state["uploaded_files"] = uploaded_files
            if donor_candidates_birth is not None and donors is not None:
                donor_candidates_birth = process_candidate_data(donor_candidates_birth)
                donors = process_donor_data(donors)

        ## Definition of filters
        expander = st.expander("**Filters**")
        with expander:
            if donor_candidates_birth is not None:
                age_min, age_max = int(donor_candidates_birth['age'].min()), int(donor_candidates_birth['age'].max())
                age_range = st.slider("Age Range", age_min, age_max, (age_min, age_max))

                weight_min, weight_max = int(donor_candidates_birth['weight'].min()), int(donor_candidates_birth['weight'].max())
                weight_range = st.slider("Weight Range", weight_min, weight_max, (weight_min, weight_max))
                
                gender_options = ['All'] + donor_candidates_birth['gender'].unique().tolist()
                gender = st.selectbox("Gender", gender_options)
                
                district_options = ['All'] + donor_candidates_birth['residence_district'].unique().tolist()
                district = st.selectbox("District", district_options)
            else:
                age_range, weight_range, gender, district = 0, 0, 'All', 'All'
            
        data = [donor_candidates_birth, donors]
        return page, age_range, weight_range, gender, district, data

def apply_filters(age_range, weight_range, gender, district, df):
    if df[0] is not None and df[1] is not None:
        filtered_df1 = df[0].copy()
        filtered_df2 = df[1].copy()
        if 'age' in filtered_df1.columns:
            filtered_df1 = filtered_df1[
                (filtered_df1['age'] >= age_range[0]) & 
                (filtered_df1['age'] <= age_range[1])
            ]
        if 'weight' in filtered_df1.columns:
            filtered_df1 = filtered_df1[
                (filtered_df1['weight'] >= weight_range[0]) & 
                (filtered_df1['weight'] <= weight_range[1])
            ]
        if 'gender' in filtered_df1.columns and gender != 'All':
            filtered_df1 = filtered_df1[filtered_df1['gender'] == gender]
        if 'residence_district' in filtered_df1.columns and district != 'All':
            filtered_df1 = filtered_df1[filtered_df1['residence_district'] == district]
        print("Definition of Filters OK !")
        return filtered_df1, filtered_df2
    else:
        print("Data frame is empty !")
        return None, None

# Page Rendering Functions
def home_page(donor_candidates_birth, donors):
    st.title("Home Page")

    with st.expander('**Data**'):
        st.write('**Raw Data**')
        st.write('**Candidates Donors data**')
        donor_candidates_birth
        st.write('**Donors data**')
        donors
    with st.expander('**Description:Summary statistics**'):
        if donor_candidates_birth is not None and donors is not None:
            st.write('**Candidates Donors data**')
            st.write(donor_candidates_birth.describe())
            st.write('**Donors data**')
            st.write(donors.describe())
        else:
            st.write("😠 Upload the data first !")

def render_overview(donor_candidates_birth, donors):
    st.markdown(f'<div class="metric-container"></div>', unsafe_allow_html=True)
    metrics_row = st.columns(4)
    if donor_candidates_birth is not None and donors is not None:
        with metrics_row[0]:
            total_candidates = len(donor_candidates_birth)
            st.markdown(f'<div class="metric-container2"><h3>Total Candidates</h3><p class="highlight" style="font-size: 2rem;">Total Candidates: {total_candidates}</p></div>', unsafe_allow_html=True)
        with metrics_row[1]:
            total_donors = len(donors)
            conversion_rate = round((total_donors / total_candidates) * 100, 1)
            st.markdown(f'<div class="metric-container2"><h3>Total Donors</h3><p class="highlight" style="font-size: 2rem;">Total Donors: {total_donors}</p></div>', unsafe_allow_html=True)
        with metrics_row[2]:
            if 'gender' in donors.columns:
                gender_counts = donors['gender'].value_counts()
                male_pct = round((gender_counts.get('M', 0) / total_donors) * 100, 1)
                female_pct = round((gender_counts.get('F', 0) / total_donors) * 100, 1)
                st.markdown(f'<div class="metric-container2"><h3>Gender Distribution</h3><p>Male: <span class="highlight">{male_pct}%</span></p><p> Female: <span class="highlight">{female_pct}%</span></p></div>', unsafe_allow_html=True)
        with metrics_row[3]:
            if 'donation_type' in donors.columns:
                donation_type_counts = donors['donation_type'].value_counts()
                voluntary_pct = round((donation_type_counts.get('B', 0) / total_donors) * 100, 1)
                family_pct = round((donation_type_counts.get('F', 0) / total_donors) * 100, 1)
                st.markdown(f'<div class="metric-container2"><h3>Donation Types</h3><p>Voluntary: <span class="highlight">{voluntary_pct}%</span></p><p>Family: <span class="highlight">{family_pct}%</span></p></div>', unsafe_allow_html=True)
    
    st.markdown("<div class='sub-header'>Key Insights</div>", unsafe_allow_html=True)
    charts_row = st.columns(2)
    if donors is not None:
        with charts_row[0]:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Blood Group Distribution")
            if 'blood_group' in donors.columns:
                blood_group_counts = donors['blood_group'].value_counts()
                fig = px.pie(values=blood_group_counts.values, names=blood_group_counts.index, color_discrete_sequence=px.colors.sequential.Reds, hole=0.5, title="Distribution of Blood Groups")
                fig.update_traces(textinfo='percent+label', pull=[0.1]*len(donors['blood_group'].unique()))
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with charts_row[1]:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Age Distribution of Donors")
            if 'age' in donors.columns:
                fig = px.histogram(donors, x='age', nbins=20, color_discrete_sequence=['#B22222'], title="Age Distribution of Blood Donors")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sub-header'>Donation Trends</div>", unsafe_allow_html=True)
    if donor_candidates_birth is not None:
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        donations = [45, 42, 50, 55, 60, 62, 58, 65, 70, 68, 72, 75]
        eligible_rates = [0.75, 0.73, 0.76, 0.78, 0.80, 0.79, 0.77, 0.81, 0.83, 0.82, 0.84, 0.85]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=donations, mode='lines+markers', name='Donations', line=dict(color='#B22222', width=3), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=months, y=[rate * 100 for rate in eligible_rates], mode='lines+markers', name='Eligibility Rate (%)', line=dict(color='#FFA07A', width=3, dash='dot'), marker=dict(size=8), yaxis='y2'))
        fig.update_layout(title='Monthly Donations and Eligibility Rates in 2019', xaxis_title='Month', yaxis_title='Number of Donations', yaxis2=dict(title='Eligibility Rate (%)', overlaying='y', side='right'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=500)
        st.plotly_chart(fig, use_container_width=True)

def main():
    configure_page()
    render_header()
    render_styles()
    L = render_sidebar()
    page = L[0]
    age_range = L[1]
    weight_range = L[2]
    gender = L[3]
    district = L[4]
    data = L[5]

    if L is not None:
        donor_candidates_birth, donors = apply_filters(age_range, weight_range, gender, district, data)
    else:
        donor_candidates_birth, donors = None, None
 

    page_functions = {
        "Home Page": home_page,
        "Overview": render_overview,
        #"Geographic Distribution": render_geographic_distribution,
        #"Health Conditions": render_health_conditions,
        #"Donor Profiles": render_donor_profiles,
        #"Campaign Effectiveness": render_campaign_effectiveness,
        #"Donor Retention": render_donor_retention,
        #"Sentiment Analysis": render_sentiment_analysis,
        #"Eligibility Prediction": render_eligibility_prediction,
        #"Data Collection": render_data_collection
    }
    if page in ["Home Page"]:
        page_functions[page](donor_candidates_birth, donors)
    elif page in ["Overview"]:
        page_functions[page](donor_candidates_birth, donors)
    elif page in ["Geographic Distribution"]:
        page_functions[page](donor_candidates_birth, geo_data)
    elif page in ["Health Conditions", "Donor Profiles"]:
        page_functions[page](donor_candidates_birth, age_range, gender, district)
    elif page in ["Eligibility Prediction"]:
        page_functions[page](models)
    elif page in ["Data Collection"]:
        page_functions[page](geo_data)
    elif page in ["Sentiment Analysis"]:
        page_functions[page](donor_candidates_birth)
    else:
        page_functions[page]()
    
    st.markdown('<div class="footer">Developed by Team [CodeFlow] | March 2025</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()