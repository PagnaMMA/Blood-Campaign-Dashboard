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
    st.title("IndabaX Cameroon Hackaton: Blood Donation Management System")
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
                    <div class="title">CodeFlow</div>
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
            .highlight {color: #B22222; font-weight: bold;}
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
    #st.image("/home/student24/Documents/AIMS_Folder/IndabaX_Cam/Project_test/Images/blood2.png", width=200)

    image_url = ("Images/codeflow.png")

    with st.sidebar:
        st.markdown(
        f"""
        <div style='display: flex; align-items: center;'>
            <img src='{image_url}' style='width: 50px; height: 50px; margin-right: 30px;'>
            <h1 style='margin: 0;'>CodeFLow</h1>
        </div>
        """,
        unsafe_allow_html=True,
        )
            
        st.markdown("## Navigation")
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

        expander = st.expander("⚒ Menu")
        with expander:
            page = st.radio("Select Dashboard Page", [
                "Overview", "Geographic Distribution", "Health Conditions", 
                "Donor Profiles", "Campaign Effectiveness", "Donor Retention", 
                "Sentiment Analysis", "Eligibility Prediction", "Data Collection"
            ])
        
    return page, donor_candidates_birth, donors
    

def main():
    configure_page()
    render_header()
    render_styles()
    render_sidebar()


if __name__ == "__main__":
    main()