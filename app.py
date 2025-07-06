import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform, jaccard
from scipy import stats
from scipy.stats import normaltest, jarque_bera, shapiro
import networkx as nx
import ast
import re
from io import StringIO, BytesIO
import base64
from datetime import datetime
import warnings
import hashlib
import time
import urllib.parse
import uuid
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="HRMS Analyse",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base de données des catégories (nombre total de molécules par catégorie)
DATABASE_CATEGORIES = {
    'BIOCID': 136,
    'DOA': 475,
    'DW': 4414,
    'FOODC': 6034,
    'HUME': 49949,
    'HUTOX': 1285,
    'IND': 258,
    'INDOOR': 1569,
    'NATOX': 1549,
    'PCP': 2855,
    'PFAS': 3662,
    'PHARMA': 7861,
    'PLAST': 2795,
    'PMT': 367,
    'PPP': 964,
    'SMOKE': 92,
    'SURF': 477,
    'UNCLASSIFIED': 12040
}

# Palette de couleurs distinctes et contrastées pour les groupes
DISTINCT_COLORS = [
    '#FF6B6B',  # Rouge corail
    '#4ECDC4',  # Turquoise
    '#45B7D1',  # Bleu ciel
    '#96CEB4',  # Vert menthe
    '#FFEAA7',  # Jaune pastel
    '#DDA0DD',  # Prune
    '#98D8C8',  # Vert d'eau
    '#F7DC6F',  # Jaune doré
    '#BB8FCE',  # Violet pastel
    '#85C1E9',  # Bleu pastel
    '#F8C471',  # Orange pastel
    '#82E0AA',  # Vert clair
    '#F1948A',  # Rose saumon
    '#85D4E3',  # Cyan clair
    '#F4D03F',  # Jaune moutarde
    '#D7BDE2',  # Lilas
    '#A3E4D7',  # Menthe glacée
    '#FAD7A0',  # Pêche
    '#D5A6BD',  # Rose poudré
    '#AED6F1'   # Bleu poudré
]

# CSS personnalisé avec navigation améliorée
st.markdown("""
<style>
    /* Styles généraux */
    .main > div {
        padding-top: 1rem;
    }
    
    /* Navigation principale */
    .nav-container {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-background-color) 100%);
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .nav-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        padding: 0.5rem;
        background: var(--background-color);
        border-radius: 10px;
    }
    
    .nav-button {
        flex: 1;
        min-width: 120px;
        padding: 0.75rem 1.25rem;
        border: 2px solid transparent;
        border-radius: 8px;
        background: var(--secondary-background-color);
        color: var(--text-color);
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
        position: relative;
        overflow: hidden;
    }
    
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        background: var(--primary-color);
        color: white;
    }
    
    .nav-button.active {
        background: var(--primary-color);
        color: white;
        border-color: var(--primary-color);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .nav-button.active::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: white;
        animation: slideIn 0.3s ease;
    }
    
    @keyframes slideIn {
        from { width: 0; }
        to { width: 100%; }
    }
    
    .nav-icon {
        font-size: 1.2em;
        margin-bottom: 0.25rem;
        display: block;
    }
    
    .nav-text {
        font-size: 0.85em;
        display: block;
    }
    
    /* Breadcrumb */
    .breadcrumb {
        background: var(--secondary-background-color);
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        margin-bottom: 1.5rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .breadcrumb-item {
        color: var(--text-color);
        opacity: 0.7;
        transition: opacity 0.2s ease;
    }
    
    .breadcrumb-item:hover {
        opacity: 1;
    }
    
    .breadcrumb-separator {
        color: var(--text-color);
        opacity: 0.5;
    }
    
    .breadcrumb-active {
        color: var(--primary-color);
        font-weight: 600;
        opacity: 1;
    }
    
    /* Progress indicator */
    .progress-indicator {
        height: 4px;
        background: var(--secondary-background-color);
        border-radius: 2px;
        margin-bottom: 1rem;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: var(--primary-color);
        border-radius: 2px;
        transition: width 0.5s ease;
    }
    
    /* Section headers avec style amélioré */
    .section-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, transparent 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        position: relative;
    }
    
    .section-header h2 {
        color: var(--text-color);
        margin: 0;
        font-size: 1.8em;
    }
    
    .section-header p {
        color: var(--text-color);
        opacity: 0.8;
        margin: 0.5rem 0 0 0;
    }
    
    /* Quick stats */
    .quick-stats {
        background: var(--secondary-background-color);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-around;
        align-items: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .stat-item {
        text-align: center;
        flex: 1;
    }
    
    .stat-value {
        font-size: 1.5em;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    .stat-label {
        font-size: 0.85em;
        color: var(--text-color);
        opacity: 0.7;
    }
    
    /* Autres styles existants */
    .stMetric {
        background-color: var(--background-color);
        border: 1px solid var(--border-color);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .highlight-box {
        background-color: var(--secondary-background-color);
        border-left: 5px solid var(--primary-color);
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .info-box {
        background-color: var(--secondary-background-color);
        border: 2px solid var(--primary-color);
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .success-box {
        background-color: var(--secondary-background-color);
        border: 2px solid #34a853;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .warning-box {
        background-color: var(--secondary-background-color);
        border: 2px solid #fbbc04;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .error-box {
        background-color: var(--secondary-background-color);
        border: 2px solid #ea4335;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .molecule-card {
        background-color: var(--background-color);
        border: 1px solid var(--border-color);
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .assignment-explanation {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .detection-factor {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .confidence-table {
        background-color: var(--background-color);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    .confidence-table table {
        width: 100%;
        border-collapse: collapse;
        margin: 0;
    }
    .confidence-table th {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 15px;
        text-align: center;
        font-weight: bold;
        border-bottom: 1px solid var(--border-color);
    }
    .confidence-table td {
        padding: 12px;
        border-bottom: 1px solid var(--border-color);
        vertical-align: top;
        color: var(--text-color);
    }
    .confidence-table tr:hover {
        background-color: var(--secondary-background-color);
    }
    .explanation-text {
        background-color: var(--secondary-background-color);
        border-left: 4px solid var(--primary-color);
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: var(--text-color);
    }
    .criteria-success {
        color: #28a745;
        font-weight: bold;
    }
    .criteria-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .criteria-error {
        color: #dc3545;
        font-weight: bold;
    }
    .ecotox-card {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .toxic { 
        background-color: var(--secondary-background-color) !important; 
        border-left: 5px solid #f44336; 
        color: var(--text-color);
    }
    .moderate { 
        background-color: var(--secondary-background-color) !important; 
        border-left: 5px solid #ff9800; 
        color: var(--text-color);
    }
    .safe { 
        background-color: var(--secondary-background-color) !important; 
        border-left: 5px solid #4caf50; 
        color: var(--text-color);
    }
    .pubchem-button {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    .pubchem-button:hover {
        background-color: #45a049;
    }
    .welcome-card {
        background-color: var(--secondary-background-color);
        border: 2px solid var(--primary-color);
        padding: 2rem;
        margin: 2rem 0;
        border-radius: 1rem;
        text-align: center;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .nav-pills {
            flex-direction: column;
        }
        .nav-button {
            width: 100%;
        }
        .quick-stats {
            flex-direction: column;
            gap: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Fonctions utilitaires améliorées et optimisées
def generate_unique_key(base_key):
    """Génère une clé unique en utilisant UUID pour éviter les collisions"""
    unique_id = str(uuid.uuid4())[:8]
    timestamp = str(int(time.time() * 1000))[-6:]
    return f"{base_key}_{unique_id}_{timestamp}"

@st.cache_data(ttl=3600, max_entries=3)  # Cache optimisé
def parse_list_column(series):
    """Convertit les colonnes de listes string en vraies listes"""
    def safe_eval(x):
        if pd.isna(x) or x == '[]' or x == '' or x == 'nan':
            return []
        try:
            if isinstance(x, str):
                x = x.strip()
                if x.startswith('[') and x.endswith(']'):
                    return ast.literal_eval(x)
                else:
                    return [float(i.strip()) for i in x.split(',') if i.strip()]
            elif isinstance(x, list):
                return x
            return []
        except:
            return []
    return series.apply(safe_eval)

@st.cache_data(ttl=3600, max_entries=3)  # Cache optimisé
def parse_dict_column(series):
    """Convertit les colonnes de dictionnaires string en vrais dictionnaires"""
    def safe_eval_dict(x):
        if pd.isna(x) or x == '{}' or x == '' or x == 'nan':
            return {}
        try:
            if isinstance(x, str):
                return ast.literal_eval(x)
            elif isinstance(x, dict):
                return x
            return {}
        except:
            return {}
    return series.apply(safe_eval_dict)

@st.cache_data(ttl=1800)  # Cache optimisé
def get_intensity_for_sample(row, sample_name):
    """Extrait l'intensité spécifique pour un échantillon donné"""
    try:
        # Vérifier si les colonnes nécessaires existent
        if 'intensities_by_sample' not in row or 'sample_names_order' not in row:
            return row.get('intensity', 0)
        
        intensities = row['intensities_by_sample']
        sample_names = row['sample_names_order']
        
        # Parser les listes si elles sont en format string
        if isinstance(intensities, str):
            intensities = ast.literal_eval(intensities)
        if isinstance(sample_names, str):
            sample_names = ast.literal_eval(sample_names)
        
        # Trouver l'index de l'échantillon
        if sample_name in sample_names:
            idx = sample_names.index(sample_name)
            if idx < len(intensities):
                return intensities[idx]
        
        # Si pas trouvé, retourner l'intensité générale
        return row.get('intensity', 0)
    
    except:
        # En cas d'erreur, retourner l'intensité générale
        return row.get('intensity', 0)

@st.cache_data(ttl=1800)  # Cache optimisé
def count_unique_molecules(df, sample_filter=None, confidence_levels=None):
    """Compte les molécules uniques (pas les occurrences)"""
    filtered_df = df[df['match_name'].notna()].copy()
    
    if sample_filter:
        filtered_df = filtered_df[filtered_df['samples'].str.contains(sample_filter, na=False)]
    
    if confidence_levels and 'confidence_level' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['confidence_level'].isin(confidence_levels)]
    
    return len(filtered_df['match_name'].unique())

@st.cache_data(ttl=1800)  # Cache optimisé
def aggregate_molecules_by_name_enhanced(df, sample_filter=None):
    """Agrège les données par nom de molécule en conservant toutes les informations importantes et utilise les intensités par échantillon"""
    filtered_df = df[df['match_name'].notna()].copy()
    
    if sample_filter:
        filtered_df = filtered_df[filtered_df['samples'].str.contains(sample_filter, na=False)]
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # Grouper par nom de molécule
    aggregated_data = []
    
    for molecule_name, group in filtered_df.groupby('match_name'):
        # Prendre la ligne avec la plus haute intensité comme base
        best_row = group.loc[group['intensity'].idxmax()].copy()
        
        # Agréger les adduits (enlever les doublons)
        all_adducts = sorted(list(set(group['match_adduct'].dropna().tolist())))
        best_row['match_adduct'] = all_adducts
        
        # Prendre le meilleur niveau de confiance (le plus bas)
        if 'confidence_level' in group.columns:
            best_row['confidence_level'] = group['confidence_level'].min()
        
        # Prendre le meilleur score MS2
        if 'ms2_similarity_score' in group.columns:
            best_row['ms2_similarity_score'] = group['ms2_similarity_score'].max()
        
        # Pour les données MS2, prendre celles de la ligne avec le meilleur score MS2
        ms2_columns = ['ms2_mz_experimental', 'ms2_intensities_experimental', 
                       'ms2_mz_reference', 'ms2_intensities_reference']
        
        if 'ms2_similarity_score' in group.columns:
            # Trouver la ligne avec le meilleur score MS2
            best_ms2_idx = group['ms2_similarity_score'].idxmax()
            if pd.notna(group.loc[best_ms2_idx, 'ms2_similarity_score']) and group.loc[best_ms2_idx, 'ms2_similarity_score'] > 0:
                for col in ms2_columns:
                    if col in group.columns:
                        best_row[col] = group.loc[best_ms2_idx, col]
                
                # Aussi prendre les autres colonnes MS2 importantes de cette ligne
                other_ms2_cols = ['has_ms2_db', 'collision_energy_reference']
                for col in other_ms2_cols:
                    if col in group.columns:
                        best_row[col] = group.loc[best_ms2_idx, col]
        
        # Agréger les échantillons
        all_samples = set()
        for samples_str in group['samples'].dropna():
            all_samples.update([s.strip() for s in samples_str.split(',')])
        best_row['samples'] = ', '.join(sorted(all_samples))
        
        # Calculer l'intensité spécifique pour l'échantillon si sample_filter est fourni
        if sample_filter:
            sample_intensity = 0
            for idx, row in group.iterrows():
                intensity_for_sample = get_intensity_for_sample(row, sample_filter)
                sample_intensity += intensity_for_sample
            best_row['sample_specific_intensity'] = sample_intensity
        
        # Sommer les intensités de tous les adduits (total)
        best_row['total_intensity'] = group['intensity'].sum()
        
        # Ajouter un identifiant unique pour éviter les problèmes de sélection
        best_row['unique_id'] = f"{molecule_name}_{hash(str(all_adducts))}"
        
        aggregated_data.append(best_row)
    
    return pd.DataFrame(aggregated_data)

@st.cache_data(ttl=1800)  # Cache optimisé
def aggregate_molecules_by_name(df, sample_filter=None):
    """Version standard pour la compatibilité"""
    return aggregate_molecules_by_name_enhanced(df, sample_filter)

# Fonction pour créer le lien PubChem
def create_pubchem_link(molecule_name, smiles=None):
    """Crée un lien vers PubChem basé sur le nom de la molécule ou SMILES"""
    if smiles and pd.notna(smiles) and smiles != 'N/A':
        # Utiliser SMILES si disponible
        encoded_smiles = urllib.parse.quote(smiles)
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/search/#collection=compounds&query_type=smiles&query={encoded_smiles}"
    else:
        # Utiliser le nom de la molécule
        encoded_name = urllib.parse.quote(molecule_name)
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{encoded_name}"
    
    return search_url

def display_pubchem_link(molecule_name, smiles=None):
    """Affiche le bouton PubChem"""
    pubchem_url = create_pubchem_link(molecule_name, smiles)
    
    st.markdown(f"""
    <a href="{pubchem_url}" target="_blank" class="pubchem-button">
        🔗 Voir sur PubChem
    </a>
    """, unsafe_allow_html=True)

# Fonctions de chargement des données avec cache optimisé
@st.cache_data(ttl=7200, show_spinner="Chargement des données features...")  # Cache optimisé
def load_features_data(uploaded_file):
    """Charge et traite le fichier features_complete.csv."""
    df = pd.read_csv(uploaded_file)
    
    # Traitement des colonnes de listes
    list_columns = ['ms2_mz_experimental', 'ms2_intensities_experimental', 
                   'ms2_mz_reference', 'ms2_intensities_reference', 'categories',
                   'intensities_by_sample', 'sample_names_order']
    
    for col in list_columns:
        if col in df.columns:
            df[col] = parse_list_column(df[col])
    
    # Traitement des colonnes de dictionnaires
    if 'individual_scores' in df.columns:
        df['individual_scores'] = parse_dict_column(df['individual_scores'])
    
    return df

@st.cache_data(ttl=7200, show_spinner="Chargement de la matrice...")  # Cache optimisé
def load_matrix_data(uploaded_file):
    """Charge le fichier feature_matrix.csv."""
    df = pd.read_csv(uploaded_file, index_col=0)
    return df

@st.cache_data(ttl=3600)  # Cache optimisé
def calculate_detection_factor(df, samples_list, confidence_levels=None):
    """Calcule le facteur de détection par catégorie pour chaque échantillon avec filtrage par niveau de confiance - MOLÉCULES UNIQUES"""
    detection_factors = {}
    
    for sample in samples_list:
        sample_data = df[df['samples'].str.contains(sample, na=False)]
        identified_data = sample_data[sample_data['match_name'].notna()]
        
        # Filtrer par niveau de confiance si spécifié
        if confidence_levels and 'confidence_level' in identified_data.columns:
            identified_data = identified_data[identified_data['confidence_level'].isin(confidence_levels)]
        
        sample_factors = {}
        
        for category, total_db in DATABASE_CATEGORIES.items():
            # Compter les molécules UNIQUES détectées dans cette catégorie
            unique_molecules = set()
            
            for idx, row in identified_data.iterrows():
                categories = row.get('categories', [])
                if isinstance(categories, list) and category in categories:
                    unique_molecules.add(row['match_name'])
            
            detected_in_category = len(unique_molecules)
            
            # Calculer le facteur de détection
            detection_factor = (detected_in_category / total_db) * 100 if total_db > 0 else 0
            sample_factors[category] = detection_factor
        
        detection_factors[sample] = sample_factors
    
    return detection_factors

def plot_detection_factor_radar(detection_factors):
    """Graphique radar du facteur de détection par échantillon avec couleurs visibles dans les deux thèmes"""
    if not detection_factors:
        st.warning("Aucun facteur de détection calculé")
        return
    
    fig = go.Figure()
    
    categories = list(DATABASE_CATEGORIES.keys())
    
    for idx, (sample, factors) in enumerate(detection_factors.items()):
        values = [factors.get(cat, 0) for cat in categories]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=sample,
            line=dict(width=2, color=DISTINCT_COLORS[idx % len(DISTINCT_COLORS)]),
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max([max(factors.values()) for factors in detection_factors.values()]) if detection_factors else 1],
                ticksuffix='%',
                tickfont=dict(size=12, color='#555555')  # Gris foncé visible sur les deux thèmes
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='#555555')  # Gris foncé visible sur les deux thèmes
            )),
        showlegend=True,
        title="Facteur de détection par catégorie et échantillon (%)<br><sub>Nombre détecté / Nombre total dans la base × 100</sub>",
        height=600,
        font=dict(color='#555555')  # Gris foncé pour tout le texte
    )
    
    st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("detection_factor_radar"))

def plot_category_distribution_radar(df, samples_list, confidence_levels=None):
    """Radar de distribution des catégories par échantillon avec filtrage par niveau de confiance et couleurs visibles dans les deux thèmes"""
    fig = go.Figure()
    
    categories = list(DATABASE_CATEGORIES.keys())
    
    for idx, sample in enumerate(samples_list):
        sample_data = df[df['samples'].str.contains(sample, na=False)]
        identified_data = sample_data[sample_data['match_name'].notna()]
        
        # Filtrer par niveau de confiance si spécifié
        if confidence_levels and 'confidence_level' in identified_data.columns:
            identified_data = identified_data[identified_data['confidence_level'].isin(confidence_levels)]
        
        category_counts = {}
        for category in categories:
            unique_molecules = set()
            for idx_row, row in identified_data.iterrows():
                cats = row.get('categories', [])
                if isinstance(cats, list) and category in cats:
                    unique_molecules.add(row['match_name'])
            category_counts[category] = len(unique_molecules)
        
        values = [category_counts[cat] for cat in categories]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=sample,
            line=dict(width=2, color=DISTINCT_COLORS[idx % len(DISTINCT_COLORS)]),
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max([max([category_counts[cat] for cat in categories]) for sample in samples_list]) if samples_list else 1],
                tickfont=dict(size=12, color='#555555')  # Gris foncé visible sur les deux thèmes
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='#555555')  # Gris foncé visible sur les deux thèmes
            )),
        showlegend=True,
        title="Distribution des molécules uniques par catégorie et échantillon",
        height=600,
        font=dict(color='#555555')  # Gris foncé pour tout le texte
    )
    
    st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("category_distribution_radar"))

def plot_level1_bubble_plot_sample(df, sample_name):
    """Bubble plot pour les molécules d'un échantillon spécifique avec choix de niveaux et intensités spécifiques"""
    if 'confidence_level' not in df.columns:
        st.warning("Colonne confidence_level non trouvée")
        return
    
    # Sélecteur de niveaux de confiance
    col1, col2 = st.columns([1, 3])
    with col1:
        available_levels = sorted([1, 2, 3])  # Seulement 1, 2, 3
        selected_levels = st.multiselect(
            "Niveaux de confiance",
            options=available_levels,
            default=[1],
            help="Sélectionnez les niveaux à inclure",
            key=f"bubble_levels_{sample_name}_fixed"  # CLÉ FIXE avec nom d'échantillon
        )
    
    with col2:
        if not selected_levels:
            st.warning("Veuillez sélectionner au moins un niveau de confiance")
            return
        st.info(f"Niveaux sélectionnés : {', '.join(map(str, selected_levels))}")
    
    # Filtrer les données de l'échantillon et niveaux sélectionnés
    sample_data = df[df['samples'].str.contains(sample_name, na=False)]
    level_data = sample_data[sample_data['confidence_level'].isin(selected_levels)].copy()
    
    if level_data.empty:
        st.warning(f"Aucune molécule des niveaux {selected_levels} trouvée dans l'échantillon {sample_name}")
        return
    
    # Agréger par molécule unique avec intensités spécifiques
    aggregated_level = aggregate_molecules_by_name(level_data, sample_name)
    
    if aggregated_level.empty:
        st.warning(f"Aucune molécule des niveaux {selected_levels} trouvée dans l'échantillon {sample_name}")
        return
    
    # Utiliser l'intensité spécifique à l'échantillon si disponible
    size_column = 'sample_specific_intensity' if 'sample_specific_intensity' in aggregated_level.columns else 'total_intensity'
    
    # Préparer les données pour le bubble plot
    fig = px.scatter(
        aggregated_level,
        x=[sample_name] * len(aggregated_level),
        y='match_name',
        size=size_column,
        color='confidence_level',
        hover_data=['mz', 'retention_time', size_column],
        title=f"Molécules uniques niveaux {selected_levels} - Intensités spécifiques - {sample_name}",
        size_max=30,
        labels={'x': 'Échantillon', 'y': 'Molécule'},
        color_discrete_sequence=DISTINCT_COLORS[:3]
    )
    
    fig.update_layout(
        height=max(400, len(aggregated_level) * 25),
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Échantillon",
        yaxis_title=f"Molécules (Niveaux {selected_levels})"
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"level_bubble_{sample_name}_fixed")  # CLÉ FIXE avec nom d'échantillon
    
    # Afficher quelques statistiques
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"Molécules uniques niveaux {selected_levels}", len(aggregated_level))
    with col2:
        avg_intensity = aggregated_level[size_column].mean()
        st.metric("Intensité moyenne", f"{avg_intensity:.2e}")
    with col3:
        max_intensity_mol = aggregated_level.loc[aggregated_level[size_column].idxmax(), 'match_name']
        st.metric("Plus intense", max_intensity_mol)

def explain_compound_confidence(compound_data):
    """Explique pourquoi un composé a reçu son niveau de confiance avec un meilleur rendu"""
    if pd.isna(compound_data['match_name']):
        return "❌ Aucune identification trouvée"
    
    level = compound_data.get('confidence_level', 5)
    reason = compound_data.get('confidence_reason', 'Non spécifié')
    
    # Extraction des informations
    mz_error = abs(compound_data.get('mz_error_ppm', 999))
    rt_obs = not pd.isna(compound_data.get('match_rt_obs'))
    rt_pred = not pd.isna(compound_data.get('match_rt_pred'))
    ccs_exp = not pd.isna(compound_data.get('match_ccs_exp'))
    ccs_pred = not pd.isna(compound_data.get('match_ccs_pred'))
    ms2_score = compound_data.get('ms2_similarity_score', 0)
    has_ms2_db = compound_data.get('has_ms2_db', 0) == 1
    
    rt_error = compound_data.get('rt_error_min', 999)
    ccs_error = compound_data.get('ccs_error_percent', 999)
    
    # Créer l'explication avec du HTML stylé
    explanation_html = f"""
    <div class="explanation-text">
        <h3>🔍 Explication du niveau {level} pour {compound_data['match_name']}</h3>
        <p><strong>Raison officielle :</strong> <em>{reason}</em></p>
        <p><strong>Critères vérifiées :</p>
        
    """
    
    # Vérification m/z
    if mz_error <= 5:
        explanation_html += f'<p><span class="criteria-success">✅ Masse exacte</span> : {mz_error:.2f} ppm (≤ 5 ppm)</p>'
    else:
        explanation_html += f'<p><span class="criteria-error">❌ Masse exacte</span> : {mz_error:.2f} ppm (> 5 ppm)</p>'
    
    # Vérification RT
    if rt_obs:
        if abs(rt_error) <= 0.1:
            explanation_html += f'<p><span class="criteria-success">✅ RT observé</span> : {abs(rt_error):.3f} min (≤ 0.1 min) - Standard de référence</p>'
        else:
            explanation_html += f'<p><span class="criteria-warning">⚠️ RT observé</span> : {abs(rt_error):.3f} min (> 0.1 min) - Standard disponible mais déviation</p>'
    elif rt_pred:
        explanation_html += f'<p><span class="criteria-warning">⚠️ RT prédit</span> : {abs(rt_error):.3f} min - Prédiction utilisée</p>'
    else:
        explanation_html += f'<p><span class="criteria-error">❌ RT</span> : Non disponible</p>'
    
    # Vérification CCS
    if ccs_exp:
        if abs(ccs_error) <= 8:
            explanation_html += f'<p><span class="criteria-success">✅ CCS expérimental</span> : {abs(ccs_error):.2f}% (≤ 8%) - Standard de référence</p>'
        else:
            explanation_html += f'<p><span class="criteria-warning">⚠️ CCS expérimental</span> : {abs(ccs_error):.2f}% (> 8%) - Standard disponible mais déviation</p>'
    elif ccs_pred:
        explanation_html += f'<p><span class="criteria-warning">⚠️ CCS prédit</span> : {abs(ccs_error):.2f}% - Prédiction utilisée</p>'
    else:
        explanation_html += f'<p><span class="criteria-error">❌ CCS</span> : Non disponible</p>'
    
    # Vérification MS2
    if has_ms2_db:
        if ms2_score >= 0.7:
            explanation_html += f'<p><span class="criteria-success">✅ MS/MS parfait</span> : Score {ms2_score:.3f} (≥ 0.7)</p>'
        elif ms2_score >= 0.4:
            explanation_html += f'<p><span class="criteria-warning">⚠️ MS/MS bon</span> : Score {ms2_score:.3f} (0.4-0.7)</p>'
        elif ms2_score >= 0.2:
            explanation_html += f'<p><span class="criteria-warning">⚠️ MS/MS partiel</span> : Score {ms2_score:.3f} (0.2-0.4)</p>'
        else:
            explanation_html += f'<p><span class="criteria-error">❌ MS/MS faible</span> : Score {ms2_score:.3f} (< 0.2)</p>'
    else:
        explanation_html += f'<p><span class="criteria-error">❌ MS/MS</span> : Non disponible dans la base de données</p>'
    
    # Conclusion selon le niveau
    if level == 1:
        explanation_html += '<p><strong>🥇 Conclusion :</strong> Identification de niveau 1 - Tous les critères remplis avec standards de référence</p>'
    elif level == 2:
        explanation_html += '<p><strong>🥈 Conclusion :</strong> Identification de niveau 2 - Standards disponibles mais MS/MS non parfait</p>'
    elif level == 3:
        explanation_html += '<p><strong>🥉 Conclusion :</strong> Identification de niveau 3 - Utilisation de prédictions ou MS/MS partiel</p>'
    elif level == 4:
        explanation_html += '<p><strong>⚠️ Conclusion :</strong> Identification de niveau 4 - CCS aide à la distinction</p>'
    else:
        explanation_html += '<p><strong>❓ Conclusion :</strong> Identification de niveau 5 - Masse exacte uniquement</p>'
    
    explanation_html += '</div>'
    
    return explanation_html

def analyze_ecotoxicity(compound_data):
    """Analyse et visualise les données écotoxicologiques avec axes adaptés"""
    # Colonnes écotoxicologiques
    ecotox_columns = {
        'daphnia_LC50_48_hr_ug/L': 'Daphnia (LC50 48h)',
        'algae_EC50_72_hr_ug/L': 'Algues (EC50 72h)',
        'pimephales_LC50_96_hr_ug/L': 'Poisson (LC50 96h)'
    }
    
    ecotox_data = {}
    
    for col, label in ecotox_columns.items():
        if col in compound_data and pd.notna(compound_data[col]):
            try:
                value = float(compound_data[col])
                ecotox_data[label] = value
            except:
                continue
    
    if not ecotox_data:
        st.info("Aucune donnée écotoxicologique disponible pour cette molécule")
        return
    
    st.markdown("""
    <div class="ecotox-card">
    <h3>🌿 Analyse Écotoxicologique</h3>
    <p>Évaluation de la toxicité sur différents organismes aquatiques</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Seuils de toxicité (µg/L)
    thresholds = {
        'Très toxique': 100,
        'Toxique': 1000,
        'Modérément toxique': 10000,
        'Peu toxique': float('inf')
    }
    
    # Analyse par organisme
    for organism, value in ecotox_data.items():
        # Déterminer le niveau de toxicité
        toxicity_level = None
        for level, threshold in thresholds.items():
            if value <= threshold:
                toxicity_level = level
                break
        
        # Choisir la classe CSS
        css_class = 'toxic'
        if toxicity_level in ['Modérément toxique', 'Peu toxique']:
            css_class = 'moderate' if toxicity_level == 'Modérément toxique' else 'safe'
        
        # Affichage stylé
        st.markdown(f"""
        <div class="{css_class}" style="padding: 10px; margin: 5px 0; border-radius: 5px;">
            <strong>{organism}</strong><br>
            Valeur: {value:.2f} µg/L<br>
            Classification: <strong>{toxicity_level}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphique comparatif avec axes adaptés
    if len(ecotox_data) > 1:
        values = list(ecotox_data.values())
        min_val = min(values)
        max_val = max(values)
        
        # Adapter les axes selon les valeurs
        y_min = max(1, min_val * 0.1)  # Ne pas descendre sous 1
        y_max = max_val * 10  # Un peu d'espace au-dessus
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(ecotox_data.keys()),
                y=values,
                marker_color=DISTINCT_COLORS[:len(values)],
                text=[f'{v:.2f} µg/L' for v in values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Comparaison de la toxicité par organisme",
            yaxis_title="Concentration (µg/L)",
            yaxis_type="log",
            yaxis=dict(range=[np.log10(y_min), np.log10(y_max)]),
            height=400
        )
        
        # Ajouter des lignes de seuil seulement si elles sont dans la plage visible
        for i, (level, threshold) in enumerate(list(thresholds.items())[:-1]):
            if y_min <= threshold <= y_max:
                fig.add_hline(y=threshold, line_dash="dash", 
                             annotation_text=f"Seuil: {level}",
                             line_color=DISTINCT_COLORS[(i+3) % len(DISTINCT_COLORS)])
        
        st.plotly_chart(fig, use_container_width=True, key=generate_unique_key(f"ecotox_comparison_{compound_data.get('match_name', 'unknown')}"))

def plot_ms2_comparison_enhanced(compound_data):
    """Version améliorée de la comparaison des spectres MS2 avec seulement 2 couleurs"""
    exp_mz = compound_data.get('ms2_mz_experimental', [])
    exp_int = compound_data.get('ms2_intensities_experimental', [])
    ref_mz = compound_data.get('ms2_mz_reference', [])
    ref_int = compound_data.get('ms2_intensities_reference', [])
    
    if not exp_mz and not ref_mz:
        st.warning("Aucun spectre MS2 disponible")
        return
    
    # Normalisation des intensités
    if exp_int:
        max_exp = max(exp_int) if exp_int else 1
        exp_int_norm = [i/max_exp*100 for i in exp_int]
    else:
        exp_int_norm = []
    
    if ref_int:
        max_ref = max(ref_int) if ref_int else 1
        ref_int_norm = [i/max_ref*100 for i in ref_int]
    else:
        ref_int_norm = []
    
    # Créer le graphique miroir avec seulement 2 couleurs fixes
    fig = go.Figure()
    
    # Couleurs fixes pour expérimental et référence
    exp_color = '#FF6B6B'  # Rouge pour expérimental
    ref_color = '#4ECDC4'  # Turquoise pour référence
    
    # Spectre expérimental (positif)
    if exp_mz and exp_int_norm:
        for mz, intensity in zip(exp_mz, exp_int_norm):
            fig.add_trace(go.Scatter(
                x=[mz, mz],
                y=[0, intensity],
                mode='lines',
                line=dict(color=exp_color, width=2),
                showlegend=False,
                hovertemplate=f"Exp: m/z={mz:.2f}<br>Int={intensity:.1f}%<extra></extra>"
            ))
    
    # Spectre de référence (négatif)
    if ref_mz and ref_int_norm:
        for mz, intensity in zip(ref_mz, ref_int_norm):
            fig.add_trace(go.Scatter(
                x=[mz, mz],
                y=[0, -intensity],
                mode='lines',
                line=dict(color=ref_color, width=2),
                showlegend=False,
                hovertemplate=f"Ref: m/z={mz:.2f}<br>Int={intensity:.1f}%<extra></extra>"
            ))
    
    # Ajouter les légendes avec les mêmes couleurs
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color=exp_color, width=2),
        name='Expérimental'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color=ref_color, width=2),
        name='Référence'
    ))
    
    fig.update_layout(
        title=f"Comparaison MS2 en miroir - {compound_data.get('match_name', 'Composé inconnu')}",
        xaxis_title="m/z",
        yaxis_title="Intensité relative (%)",
        height=500,
        yaxis=dict(range=[-110, 110]),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True, key=generate_unique_key(f"ms2_comparison_{compound_data.get('match_name', 'unknown')}"))
    
    # Calcul et affichage des métriques de similarité
    if exp_mz and ref_mz:
        similarity = compound_data.get('ms2_similarity_score', 0)
        energy = compound_data.get('collision_energy_reference', 'N/A')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Score de similarité", f"{similarity:.3f}")
        with col2:
            st.metric("Énergie de collision", f"{energy}")
        with col3:
            n_peaks_exp = len(exp_mz)
            st.metric("Pics expérimentaux", n_peaks_exp)
        with col4:
            n_peaks_ref = len(ref_mz)
            st.metric("Pics référence", n_peaks_ref)

def plot_sample_overview(df, sample_name):
    """Vue d'ensemble d'un échantillon avec intensités spécifiques"""
    sample_data = df[df['samples'].str.contains(sample_name, na=False)]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Features détectées", len(sample_data))
    
    with col2:
        # CHANGEMENT ICI : compter les molécules uniques
        identified = len(sample_data[sample_data['match_name'].notna()]['match_name'].unique())
        st.metric("Molécules identifiées", identified)
    
    with col3:
        if 'confidence_level' in sample_data.columns:
            # CHANGEMENT ICI : compter les molécules uniques niveau 1
            level1_molecules = sample_data[sample_data['confidence_level'] == 1]['match_name'].dropna().unique()
            st.metric("Identifications niveau 1", len(level1_molecules))
    
    with col4:
        # Calculer l'intensité moyenne spécifique à l'échantillon si possible
        total_sample_intensity = 0
        count = 0
        for idx, row in sample_data.iterrows():
            sample_intensity = get_intensity_for_sample(row, sample_name)
            if sample_intensity > 0:
                total_sample_intensity += sample_intensity
                count += 1
        
        if count > 0:
            avg_intensity = total_sample_intensity / count
            st.metric("Intensité moyenne (échantillon)", f"{avg_intensity:.0f}")
        else:
            avg_intensity = sample_data['intensity'].mean()
            st.metric("Intensité moyenne", f"{avg_intensity:.0f}")
    
    return sample_data

def plot_confidence_levels_distribution(df):
    """Distribution des niveaux de confiance"""
    if 'confidence_level' in df.columns:
        conf_data = df['confidence_level'].value_counts().sort_index()
        
        fig = px.bar(
            x=conf_data.index,
            y=conf_data.values,
            labels={'x': 'Niveau de confiance', 'y': 'Nombre de molécules'},
            title="Distribution des niveaux de confiance",
            color=conf_data.index,
            color_discrete_sequence=DISTINCT_COLORS[:len(conf_data)]
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        return fig
    return None

def plot_confidence_comparison_across_samples(df, samples_list, selected_levels=None):
    """Visualisation des niveaux de confiance à travers les échantillons avec filtres"""
    if 'confidence_level' not in df.columns:
        st.warning("Colonne confidence_level non trouvée")
        return
    
    # Filtres pour sélectionner les niveaux
    st.subheader("🎯 Filtres de niveaux de confiance")
    
    col1, col2 = st.columns(2)
    with col1:
        # Options prédéfinies courantes
        preset_options = {
            "Niveau 1 uniquement": [1],
            "Niveaux 1+2": [1, 2],
            "Niveaux 1+2+3": [1, 2, 3],
            "Niveaux 1+2+3+4": [1, 2, 3, 4],
            "Tous les niveaux": [1, 2, 3, 4, 5]
        }
        
        preset_choice = st.selectbox(
            "Sélection rapide",
            options=list(preset_options.keys()),
            index=2,  # Par défaut "Niveaux 1+2+3"
            key="preset_confidence_levels_fixed"  # Clé FIXE au lieu de dynamique
        )
        
        selected_levels_from_preset = preset_options[preset_choice]
    
    with col2:
        # Sélection manuelle avec les niveaux du preset comme défaut
        manual_levels = st.multiselect(
            "Sélection manuelle (remplace la sélection rapide)",
            options=[1, 2, 3, 4, 5],
            default=selected_levels_from_preset,
            help="Personnalisez votre sélection de niveaux",
            key="manual_confidence_levels_fixed"  # Clé FIXE au lieu de dynamique
        )
        
        # Utiliser la sélection manuelle si elle diffère du preset
        if manual_levels:  # S'assurer que manual_levels n'est pas vide
            selected_levels_final = manual_levels
        else:
            selected_levels_final = selected_levels_from_preset
    
    if not selected_levels_final:
        st.warning("Veuillez sélectionner au moins un niveau de confiance")
        return
    
    st.info(f"Niveaux sélectionnés : {', '.join(map(str, selected_levels_final))}")
    
    # Préparer les données pour le graphique
    conf_data = []
    for sample in samples_list:
        sample_data = df[df['samples'].str.contains(sample, na=False)]
        for level in selected_levels_final:
            level_molecules = sample_data[sample_data['confidence_level'] == level]['match_name'].dropna().unique()
            count = len(level_molecules)
            conf_data.append({
                'Échantillon': sample,
                'Niveau': f'Niveau {level}',
                'Niveau_num': level,
                'Nombre': count
            })
    
    conf_df = pd.DataFrame(conf_data)
    
    if conf_df.empty:
        st.warning("Aucune donnée pour les niveaux sélectionnés")
        return
    
    # Graphique en barres empilées
    fig = px.bar(
        conf_df,
        x='Échantillon',
        y='Nombre',
        color='Niveau',
        title=f"Distribution des niveaux de confiance par échantillon (molécules uniques) - Niveaux {selected_levels_final}",
        color_discrete_sequence=DISTINCT_COLORS[:len(selected_levels_final)]
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True, key="confidence_comparison_across_samples_fixed")  # Clé FIXE

def plot_level1_bubble_plot(df, samples_list):
    """Bubble plot pour les molécules avec choix de niveaux de confiance"""
    if 'confidence_level' not in df.columns:
        st.warning("Colonne confidence_level non trouvée")
        return
    
    # Sélecteur de niveaux de confiance
    col1, col2 = st.columns([1, 3])
    with col1:
        available_levels = sorted([1, 2, 3])  # Seulement 1, 2, 3
        selected_levels = st.multiselect(
            "Niveaux de confiance",
            options=available_levels,
            default=[1],
            help="Sélectionnez les niveaux à inclure",
            key="bubble_levels_comparison_fixed"  # CLÉ FIXE au lieu de dynamique
        )
    
    with col2:
        if not selected_levels:
            st.warning("Veuillez sélectionner au moins un niveau de confiance")
            return
        st.info(f"Niveaux sélectionnés : {', '.join(map(str, selected_levels))}")
    
    # Filtrer les molécules des niveaux sélectionnés
    level_data = df[df['confidence_level'].isin(selected_levels)].copy()
    
    if level_data.empty:
        st.warning(f"Aucune molécule des niveaux {selected_levels} trouvée")
        return
    
    # Préparer les données pour le bubble plot avec intensités spécifiques
    bubble_data = []
    for idx, row in level_data.iterrows():
        samples = row['samples'].split(',') if pd.notna(row['samples']) else []
        for sample in samples:
            sample = sample.strip()
            if sample in samples_list:
                # Utiliser l'intensité spécifique à l'échantillon
                sample_intensity = get_intensity_for_sample(row, sample)
                
                bubble_data.append({
                    'Molécule': row['match_name'],
                    'Échantillon': sample,
                    'Intensité': sample_intensity,
                    'log_Intensité': np.log10(sample_intensity) if sample_intensity > 0 else 0,
                    'Niveau': row['confidence_level']
                })
    
    if not bubble_data:
        st.warning("Aucune donnée pour le bubble plot")
        return
    
    bubble_df = pd.DataFrame(bubble_data)
    
    # Agréger par molécule et échantillon (sommer les intensités des adduits)
    aggregated_bubble = bubble_df.groupby(['Molécule', 'Échantillon']).agg({
        'Intensité': 'sum',
        'Niveau': 'min'  # Prendre le meilleur niveau
    }).reset_index()
    
    aggregated_bubble['log_Intensité'] = np.log10(aggregated_bubble['Intensité'])
    
    # Créer le bubble plot
    fig = px.scatter(
        aggregated_bubble,
        x='Échantillon',
        y='Molécule',
        size='log_Intensité',
        color='Niveau',
        hover_data=['Intensité'],
        title=f"Intensités des molécules uniques niveaux {selected_levels} par échantillon",
        size_max=30,
        color_discrete_sequence=DISTINCT_COLORS[:3]
    )
    
    fig.update_layout(
        height=max(400, len(aggregated_bubble['Molécule'].unique()) * 20),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True, key="level_bubble_plot_comparison_fixed")  # CLÉ FIXE

def calculate_jaccard_similarity(df, samples_list):
    """Calcule la similarité de Jaccard entre échantillons"""
    # Créer une matrice binaire (présence/absence des molécules)
    molecule_matrix = {}
    
    all_molecules = set()
    for sample in samples_list:
        sample_data = df[df['samples'].str.contains(sample, na=False)]
        identified_molecules = set(sample_data[sample_data['match_name'].notna()]['match_name'])
        molecule_matrix[sample] = identified_molecules
        all_molecules.update(identified_molecules)
    
    # Calculer les similarités de Jaccard
    jaccard_matrix = pd.DataFrame(index=samples_list, columns=samples_list)
    
    for i, sample1 in enumerate(samples_list):
        for j, sample2 in enumerate(samples_list):
            set1 = molecule_matrix[sample1]
            set2 = molecule_matrix[sample2]
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            jaccard_similarity = intersection / union if union > 0 else 0
            jaccard_matrix.loc[sample1, sample2] = jaccard_similarity
    
    return jaccard_matrix.astype(float)

def plot_hierarchical_clustering(df, samples_list, confidence_levels=None, selected_samples=None):
    """Clustering hiérarchique des échantillons avec filtrage par niveau de confiance"""
    # Filtrer par échantillons sélectionnés
    filtered_samples = samples_list
    if selected_samples and len(selected_samples) > 0:
        filtered_samples = selected_samples
    
    # Ne continuer que si nous avons au moins 2 échantillons
    if len(filtered_samples) < 2:
        st.warning("Au moins 2 échantillons sont nécessaires pour l'analyse de similarité")
        return
    
    # Créer une matrice binaire (présence/absence des molécules) avec filtrage par niveau de confiance
    molecule_matrix = {}
    
    for sample in filtered_samples:
        sample_data = df[df['samples'].str.contains(sample, na=False)]
        
        # Filtrer par niveau de confiance si spécifié
        if confidence_levels and 'confidence_level' in sample_data.columns:
            sample_data = sample_data[sample_data['confidence_level'].isin(confidence_levels)]
            
        identified_molecules = set(sample_data[sample_data['match_name'].notna()]['match_name'])
        molecule_matrix[sample] = identified_molecules
    
    # Calculer les similarités de Jaccard
    jaccard_matrix = pd.DataFrame(index=filtered_samples, columns=filtered_samples)
    
    for i, sample1 in enumerate(filtered_samples):
        for j, sample2 in enumerate(filtered_samples):
            set1 = molecule_matrix[sample1]
            set2 = molecule_matrix[sample2]
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            jaccard_similarity = intersection / union if union > 0 else 0
            jaccard_matrix.loc[sample1, sample2] = jaccard_similarity
    
    # Convertir en matrice de distances (1 - similarité)
    distance_matrix = 1 - jaccard_matrix
    
    # Clustering hiérarchique
    condensed_distances = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_distances, method='ward')
    
    # Créer le dendrogramme
    fig = go.Figure()
    
    # Utiliser scipy pour créer le dendrogramme
    dendro = dendrogram(linkage_matrix, labels=filtered_samples, no_plot=True)
    
    # Convertir en plotly
    for i, d in zip(dendro['icoord'], dendro['dcoord']):
        fig.add_trace(go.Scatter(
            x=i, y=d,
            mode='lines',
            line=dict(width=2, color=DISTINCT_COLORS[0]),
            showlegend=False
        ))
    
    # Ajouter les labels
    fig.add_trace(go.Scatter(
        x=dendro['icoord'][0],
        y=[0] * len(dendro['ivl']),
        mode='text',
        text=dendro['ivl'],
        textposition='bottom center',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"Clustering hiérarchique des échantillons (basé sur la similarité de Jaccard){' - Filtré par niveau(x) ' + str(confidence_levels) if confidence_levels else ''}",
        xaxis_title="Échantillons",
        yaxis_title="Distance",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("hierarchical_clustering"))
    
    # Afficher la matrice de Jaccard
    st.subheader("Matrice de similarité de Jaccard")
    fig_heatmap = px.imshow(
        jaccard_matrix,
        text_auto=".3f",
        aspect="auto",
        title=f"Similarité de Jaccard entre échantillons{' - Filtré par niveau(x) ' + str(confidence_levels) if confidence_levels else ''}",
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True, key=generate_unique_key("jaccard_heatmap"))

def safe_pca_analysis(matrix_df, n_components=3):
    """PCA sécurisée qui gère les erreurs de dimensions"""
    if matrix_df is None or matrix_df.empty:
        return None, None, None
    
    n_samples, n_features = matrix_df.shape
    max_components = min(n_samples, n_features)
    
    if max_components < 2:
        st.error(f"Impossible de faire une PCA : seulement {max_components} composante(s) possible(s)")
        return None, None, None
    
    # Ajuster le nombre de composantes
    n_components = min(n_components, max_components)
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(matrix_df.values)
    
    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return pca, X_pca, X_scaled

def plot_3d_pca(matrix_df):
    """PCA en 3D avec gestion d'erreur"""
    if matrix_df is None or matrix_df.empty:
        st.warning("Aucune matrice chargée")
        return
    
    pca, X_pca, X_scaled = safe_pca_analysis(matrix_df, n_components=3)
    
    if pca is None:
        return
    
    if X_pca.shape[1] < 3:
        st.warning(f"PCA 3D impossible : seulement {X_pca.shape[1]} composante(s) disponible(s)")
        # Afficher PCA 2D à la place
        if X_pca.shape[1] >= 2:
            fig = px.scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                text=matrix_df.index,
                title=f"PCA 2D (PC1: {pca.explained_variance_ratio_[0]*100:.1f}%, "
                      f"PC2: {pca.explained_variance_ratio_[1]*100:.1f}%)",
                labels={
                    'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                    'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'
                },
                color_discrete_sequence=DISTINCT_COLORS
            )
            fig.update_traces(textposition="top center", marker=dict(size=12))
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("pca_2d_plot"))
        return
    
    # PCA 3D normale
    fig = px.scatter_3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=X_pca[:, 2],
        text=matrix_df.index,
        title=f"PCA 3D (PC1: {pca.explained_variance_ratio_[0]*100:.1f}%, "
              f"PC2: {pca.explained_variance_ratio_[1]*100:.1f}%, "
              f"PC3: {pca.explained_variance_ratio_[2]*100:.1f}%)",
        labels={
            'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
            'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
            'z': f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)'
        },
        color_discrete_sequence=DISTINCT_COLORS
    )
    
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("pca_3d_plot"))

def plot_tsne_analysis(matrix_df):
    """Analyse t-SNE avec gestion d'erreur"""
    if matrix_df is None or matrix_df.empty:
        st.warning("Aucune matrice chargée")
        return
    
    n_samples = len(matrix_df)
    
    if n_samples < 3:
        st.error(f"t-SNE impossible : seulement {n_samples} échantillon(s). Minimum 3 requis.")
        return
    
    # Ajuster la perplexité selon le nombre d'échantillons
    max_perplexity = min(30, (n_samples - 1) // 3)
    
    if max_perplexity < 5:
        max_perplexity = min(5, n_samples - 1)
    
    # Paramètres t-SNE
    perplexity = st.slider(
        "Perplexité t-SNE", 
        1, 
        max_perplexity, 
        min(max_perplexity, 5),
        help=f"Maximum possible: {max_perplexity} (basé sur {n_samples} échantillons)",
        key=f"tsne_perplexity_{generate_unique_key('tsne')}"
    )
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(matrix_df.values)
    
    try:
        # t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        X_tsne = tsne.fit_transform(X_scaled)
        
        fig = px.scatter(
            x=X_tsne[:, 0],
            y=X_tsne[:, 1],
            text=matrix_df.index,
            title=f"Analyse t-SNE (perplexité={perplexity})",
            labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
            color_discrete_sequence=DISTINCT_COLORS
        )
        
        fig.update_traces(textposition="top center", marker=dict(size=12))
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("tsne_plot"))
        
    except Exception as e:
        st.error(f"Erreur t-SNE : {str(e)}")

def perform_pca_analysis(matrix_df):
    """Effectue une analyse PCA sécurisée sur la matrice des features"""
    if matrix_df is None or matrix_df.empty:
        st.error("Aucune matrice de features chargée")
        return
    
    pca, X_pca, X_scaled = safe_pca_analysis(matrix_df, n_components=min(10, matrix_df.shape[1], matrix_df.shape[0]))
    
    if pca is None:
        return
    
    # Variance expliquée
    variance_ratio = pca.explained_variance_ratio_
    cumsum_variance = np.cumsum(variance_ratio)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique de variance expliquée
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=list(range(1, len(variance_ratio) + 1)),
            y=variance_ratio * 100,
            name='Variance individuelle',
            marker_color=DISTINCT_COLORS[0]
        ))
        fig1.add_trace(go.Scatter(
            x=list(range(1, len(cumsum_variance) + 1)),
            y=cumsum_variance * 100,
            mode='lines+markers',
            name='Variance cumulée',
            line=dict(color=DISTINCT_COLORS[1], width=3)
        ))
        fig1.update_layout(
            title="Variance expliquée par les composantes principales",
            xaxis_title="Composante principale",
            yaxis_title="Variance expliquée (%)"
        )
        st.plotly_chart(fig1, use_container_width=True, key=generate_unique_key("pca_variance_plot"))
    
    with col2:
        # Score plot PCA
        if X_pca.shape[1] >= 2:
            fig2 = px.scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                text=matrix_df.index,
                title=f"Score plot PCA (PC1: {variance_ratio[0]*100:.1f}%, PC2: {variance_ratio[1]*100:.1f}%)",
                labels={'x': f'PC1 ({variance_ratio[0]*100:.1f}%)', 'y': f'PC2 ({variance_ratio[1]*100:.1f}%)'},
                color_discrete_sequence=DISTINCT_COLORS
            )
            fig2.update_traces(textposition="top center")
            st.plotly_chart(fig2, use_container_width=True, key=generate_unique_key("pca_score_plot"))
    
    # Affichage des métriques
    st.subheader("Métriques PCA")
    cols = st.columns(min(3, len(variance_ratio)))
    
    for i, col in enumerate(cols):
        if i < len(variance_ratio):
            with col:
                st.metric(f"PC{i+1} variance expliquée", f"{variance_ratio[i]*100:.1f}%")

# NOUVELLES FONCTIONS POUR LES ANALYSES STATISTIQUES AVANCÉES

def plot_correlation_heatmap(matrix_df):
    """Heatmap de corrélation entre échantillons"""
    if matrix_df is None or matrix_df.empty:
        st.warning("Aucune matrice chargée")
        return
    
    # Calculer la matrice de corrélation
    corr_matrix = matrix_df.T.corr()  # Transpose pour avoir échantillons vs échantillons
    
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        title="Matrice de corrélation entre échantillons",
        color_continuous_scale='RdBu_r',
        range_color=[-1, 1]
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("correlation_heatmap"))
    
    return corr_matrix

def perform_kmeans_clustering(matrix_df):
    """Clustering K-means sur la matrice des features avec validation du nombre de clusters"""
    if matrix_df is None or matrix_df.empty:
        st.warning("Aucune matrice chargée")
        return
    
    # CORRECTION : Vérifier le nombre d'échantillons
    n_samples = len(matrix_df)
    max_clusters = min(10, n_samples)
    
    if n_samples < 2:
        st.error(f"Impossible de faire du clustering : seulement {n_samples} échantillon(s). Minimum 2 requis.")
        return
    
    # CORRECTION ICI: Gérer le cas où n_samples = 2
    if n_samples == 2:
        # Si seulement 2 échantillons, on ne peut avoir que 2 clusters
        n_clusters = 2
        st.info(f"Avec seulement {n_samples} échantillons, le nombre de clusters est fixé à {n_clusters}.")
    else:
        # Sélection du nombre de clusters avec validation
        n_clusters = st.slider(
            "Nombre de clusters", 
            2, 
            max_clusters, 
            min(3, max_clusters), 
            help=f"Maximum possible: {max_clusters} (basé sur {n_samples} échantillons)",
            key=f"kmeans_clusters_{generate_unique_key('kmeans')}"
        )
    
    # Vérification de sécurité
    if n_clusters > n_samples:
        st.error(f"Le nombre de clusters ({n_clusters}) ne peut pas être supérieur au nombre d'échantillons ({n_samples})")
        return
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(matrix_df.values)
    
    try:
        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # PCA pour visualisation
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Graphique
        fig = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=cluster_labels,
            text=matrix_df.index,
            title=f"Clustering K-means (k={n_clusters})",
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
                    'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'},
            color_discrete_sequence=DISTINCT_COLORS
        )
        
        fig.update_traces(textposition="top center", marker=dict(size=12))
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("kmeans_plot"))
        
        # Afficher les clusters
        st.subheader("Composition des clusters")
        cluster_df = pd.DataFrame({
            'Échantillon': matrix_df.index,
            'Cluster': cluster_labels
        })
        
        for i in range(n_clusters):
            cluster_samples = cluster_df[cluster_df['Cluster'] == i]['Échantillon'].tolist()
            st.write(f"**Cluster {i}:** {', '.join(cluster_samples)}")
    
    except Exception as e:
        st.error(f"Erreur lors du clustering K-means : {str(e)}")

def plot_boxplot_by_category(df, samples_list):
    """Boxplots des intensités par catégorie"""
    st.subheader("📦 Distribution des intensités par catégorie")
    
    # Préparer les données
    intensity_data = []
    
    for idx, row in df[df['match_name'].notna()].iterrows():
        categories = row.get('categories', [])
        if isinstance(categories, list):
            for category in categories:
                intensity_data.append({
                    'Catégorie': category,
                    'Intensité': row['intensity'],
                    'log_Intensité': np.log10(row['intensity']) if row['intensity'] > 0 else 0
                })
    
    if not intensity_data:
        st.warning("Aucune donnée d'intensité par catégorie")
        return
    
    intensity_df = pd.DataFrame(intensity_data)
    
    # Filtre pour les principales catégories
    top_categories = intensity_df['Catégorie'].value_counts().head(10).index.tolist()
    filtered_df = intensity_df[intensity_df['Catégorie'].isin(top_categories)]
    
    # Boxplot
    fig = px.box(
        filtered_df,
        x='Catégorie',
        y='log_Intensité',
        title="Distribution des intensités par catégorie (Top 10, échelle log)",
        color='Catégorie',
        color_discrete_sequence=DISTINCT_COLORS
    )
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("boxplot_categories"))

# NOUVELLE FONCTION : Boxplot par catégorie pour un échantillon spécifique
def plot_sample_boxplot_by_category(df, sample_name):
    """Boxplots des intensités par catégorie pour un échantillon spécifique"""
    st.subheader(f"📦 Distribution des intensités par catégorie - {sample_name}")
    
    # Filtrer les données de l'échantillon
    sample_data = df[df['samples'].str.contains(sample_name, na=False)]
    identified_data = sample_data[sample_data['match_name'].notna()]
    
    if identified_data.empty:
        st.warning(f"Aucune molécule identifiée dans l'échantillon {sample_name}")
        return
    
    # Préparer les données avec intensités spécifiques
    intensity_data = []
    
    for idx, row in identified_data.iterrows():
        categories = row.get('categories', [])
        if isinstance(categories, list):
            # Utiliser l'intensité spécifique à l'échantillon
            sample_intensity = get_intensity_for_sample(row, sample_name)
            
            for category in categories:
                intensity_data.append({
                    'Catégorie': category,
                    'Intensité': sample_intensity,
                    'log_Intensité': np.log10(sample_intensity) if sample_intensity > 0 else 0,
                    'Molécule': row['match_name']
                })
    
    if not intensity_data:
        st.warning(f"Aucune donnée d'intensité par catégorie pour {sample_name}")
        return
    
    intensity_df = pd.DataFrame(intensity_data)
    
    # Compter les molécules par catégorie
    category_counts = intensity_df['Catégorie'].value_counts()
    
    # Filtrer pour ne garder que les catégories avec au moins 2 molécules
    valid_categories = category_counts[category_counts >= 2].index.tolist()
    
    if not valid_categories:
        st.warning("Pas assez de données pour créer des boxplots (minimum 2 molécules par catégorie)")
        # Afficher un graphique en barres à la place
        fig = px.bar(
            category_counts.head(10),
            x=category_counts.head(10).index,
            y=category_counts.head(10).values,
            title=f"Nombre de molécules par catégorie - {sample_name}",
            color=category_counts.head(10).index,
            color_discrete_sequence=DISTINCT_COLORS
        )
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45,
            showlegend=False,
            xaxis_title="Catégorie",
            yaxis_title="Nombre de molécules"
        )
        st.plotly_chart(fig, use_container_width=True, key=generate_unique_key(f"category_counts_{sample_name}"))
        return
    
    # Filtrer les données pour les catégories valides
    filtered_df = intensity_df[intensity_df['Catégorie'].isin(valid_categories)]
    
    # Prendre seulement les top 8 catégories pour la lisibilité
    top_categories = category_counts[category_counts >= 2].head(8).index.tolist()
    filtered_df = filtered_df[filtered_df['Catégorie'].isin(top_categories)]
    
    # Créer le boxplot
    fig = px.box(
        filtered_df,
        x='Catégorie',
        y='log_Intensité',
        title=f"Distribution des intensités par catégorie - {sample_name} (intensités spécifiques, échelle log)",
        color='Catégorie',
        color_discrete_sequence=DISTINCT_COLORS,
        hover_data=['Molécule', 'Intensité']
    )
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45,
        showlegend=False,
        xaxis_title="Catégorie",
        yaxis_title="log10(Intensité spécifique)"
    )
    
    st.plotly_chart(fig, use_container_width=True, key=generate_unique_key(f"sample_boxplot_categories_{sample_name}"))
    
    # Afficher quelques statistiques
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Catégories représentées", len(top_categories))
    
    with col2:
        most_abundant_cat = category_counts.head(1).index[0]
        st.metric("Catégorie dominante", most_abundant_cat)
    
    with col3:
        avg_intensity = filtered_df['Intensité'].mean()
        st.metric("Intensité moyenne", f"{avg_intensity:.2e}")

def show_confidence_levels_table():
    """Affiche le tableau des niveaux de confiance avec un style amélioré"""
    st.markdown("### 📊 Tableau des niveaux de confiance pour l'identification des composés")
    
    # HTML pour un tableau stylé
    table_html = """
    <div class="confidence-table">
        <table>
            <thead>
                <tr>
                    <th>Niveau</th>
                    <th>Statut</th>
                    <th>Critères OBLIGATOIRES</th>
                    <th>Critères OPTIONNELS</th>
                    <th>Raison d'assignement</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>🥇 1</strong></td>
                    <td><strong>Confirmé<br>(Gold Standard)</strong></td>
                    <td>• m/z ≤ 5 ppm<br>• RT observé ≤ 0.1 min<br>• CCS expérimental ≤ 8%<br>• MS/MS parfait ≥ 0.7<br>• Spectre MS2 en base</td>
                    <td>-</td>
                    <td>Match parfait avec standards de référence<br>(m/z + RT obs + CCS exp + MS/MS parfait)</td>
                </tr>
                <tr>
                    <td><strong>🥈 2</strong></td>
                    <td><strong>Probable</strong></td>
                    <td>• m/z ≤ 5 ppm<br>• CCS expérimental ≤ 8%<br>• MS/MS bon 0.4-0.7<br>• Spectre MS2 en base</td>
                    <td>• RT observé<br>(peut être absent)</td>
                    <td>Match probable avec références<br>(m/z + CCS exp + MS/MS base de données)</td>
                </tr>
                <tr>
                    <td><strong>🥉 3</strong></td>
                    <td><strong>Tentative</strong></td>
                    <td>• m/z ≤ 5 ppm<br>• CCS (exp OU prédit) ≤ 8%<br>• MS/MS partiel 0.2-0.4<br>• Spectre MS2 en base</td>
                    <td>• RT prédit autorisé<br>• Prédictions acceptées</td>
                    <td>Match possible avec prédictions<br>(m/z + CCS + MS/MS partiel)</td>
                </tr>
                <tr>
                    <td><strong>⚠️ 4</strong></td>
                    <td><strong>Équivoque</strong></td>
                    <td>• m/z ≤ 5 ppm<br>• CCS (exp OU prédit)</td>
                    <td>• MS/MS < 0.2 ou absent<br>• RT peut aider</td>
                    <td>Match tentatif avec aide CCS<br>(m/z + CCS pour distinction)</td>
                </tr>
                <tr>
                    <td><strong>❓ 5</strong></td>
                    <td><strong>Incertain</strong></td>
                    <td>• m/z ≤ 5 ppm</td>
                    <td>• Aucune autre donnée fiable</td>
                    <td>Match incertain<br>(m/z uniquement)</td>
                </tr>
            </tbody>
        </table>
    </div>
    """
    
    st.markdown(table_html, unsafe_allow_html=True)

def display_molecule_details_sidebar(molecule_data):
    """Affiche les détails d'une molécule dans la sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔍 Détails de la molécule sélectionnée")
        
        st.markdown(f"**Nom :** {molecule_data.get('match_name', 'N/A')}")
        st.markdown(f"**m/z :** {molecule_data.get('mz', 'N/A'):.4f}")
        st.markdown(f"**RT :** {molecule_data.get('retention_time', 'N/A'):.2f} min")
        st.markdown(f"**Intensité :** {molecule_data.get('intensity', 'N/A'):.2e}")
        st.markdown(f"**Niveau confiance :** {molecule_data.get('confidence_level', 'N/A')}")
        
        if st.button("📊 Voir l'analyse complète"):
            st.session_state.selected_molecule_for_analysis = molecule_data.get('match_name')

def show_home_page():
    """Affiche la page d'accueil avec toutes les informations principales"""
    # En-tête de bienvenue
    st.markdown("""
    <div class="welcome-card">
        <h1>🧪 Bienvenue dans l'outil d'analyse HRMS</h1>
        <p style="font-size: 1.2em; margin: 1rem 0;">
            Analysez et visualisez vos données de spectrométrie de masse haute résolution
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Afficher les tableaux des niveaux de confiance
    show_confidence_levels_table()
    
    st.markdown("---")
    
    # Instructions détaillées
    st.markdown("""
    ## 🎯 Guide d'utilisation
    
    ### 📁 Fichiers requis
    
    1. **features_complete.csv** (obligatoire)
       - Contient les résultats d'identification MS
       - Colonnes requises : mz, retention_time, intensity, match_name, confidence_level, etc.
       - **Nouvelles colonnes écotoxicologiques** : daphnia_LC50_48_hr_ug/L, algae_EC50_72_hr_ug/L, pimephales_LC50_96_hr_ug/L
       - **Colonnes d'intensités spécifiques** : intensities_by_sample, sample_names_order
    
    2. **feature_matrix.csv** (optionnel)
       - Matrice d'intensités pour analyses statistiques
       - Format : échantillons en lignes, features en colonnes
    
    """)
    
    # Affichage de la base de données de référence
    st.subheader("📚 Base de données de référence")
    
    # Créer un DataFrame pour la base de données
    db_df = pd.DataFrame([
        {'Catégorie': cat, 'Nombre_molécules': count, 'Pourcentage': f"{count/sum(DATABASE_CATEGORIES.values())*100:.1f}%"}
        for cat, count in sorted(DATABASE_CATEGORIES.items(), key=lambda x: x[1], reverse=True)
    ])
    
    # Affichage du tableau
    st.dataframe(db_df, use_container_width=True)
    
    # Graphique de la distribution
    fig_db = px.pie(
        db_df.head(10),  # Top 10 pour la lisibilité
        values='Nombre_molécules',
        names='Catégorie',
        title="Distribution des 10 principales catégories dans la base de données",
        color_discrete_sequence=DISTINCT_COLORS[:10]
    )
    fig_db.update_layout(height=500)
    st.plotly_chart(fig_db, use_container_width=True, key=generate_unique_key("database_distribution_pie"))
    
    # Métriques de la base
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total molécules", "72,577")
    
    with col2:
        st.metric("Catégories", len(DATABASE_CATEGORIES))
    
    with col3:
        largest_cat = max(DATABASE_CATEGORIES.items(), key=lambda x: x[1])
        st.metric("Plus grande catégorie", f"{largest_cat[0]} ({largest_cat[1]:,})")
    
    with col4:
        smallest_cat = min(DATABASE_CATEGORIES.items(), key=lambda x: x[1])
        st.metric("Plus petite catégorie", f"{smallest_cat[0]} ({smallest_cat[1]})")

# Fonction pour créer la navigation améliorée
def create_navigation(current_tab):
    """Crée une navigation améliorée avec design moderne"""
    tab_items = [
        {"id": "home", "icon": "🏠", "label": "Accueil", "description": "Vue d'ensemble et guide"},
        {"id": "overview", "icon": "📊", "label": "Vue d'ensemble", "description": "Statistiques globales"},
        {"id": "sample", "icon": "🔍", "label": "Analyse par échantillon", "description": "Détails par échantillon"},
        {"id": "molecules", "icon": "🧬", "label": "Molécules individuelles", "description": "Analyse des composés"},
        {"id": "detection", "icon": "📡", "label": "Facteurs de détection", "description": "Efficacité de détection"},
        {"id": "comparison", "icon": "⚖️", "label": "Comparaison", "description": "Comparer les échantillons"},
        {"id": "statistics", "icon": "📈", "label": "Analyses statistiques", "description": "Analyses avancées"},
        {"id": "reports", "icon": "📋", "label": "Rapports & Export", "description": "Générer des rapports"},
        {"id": "confidence", "icon": "ℹ️", "label": "Système de confiance", "description": "Niveaux de confiance"}
    ]
    
    # Créer le HTML de navigation
    nav_html = """<div class="nav-container">"""
    nav_html += """<div class="nav-pills">"""
    
    for item in tab_items:
        active_class = "active" if item["id"] == current_tab else ""
        nav_html += f"""
        <div class="nav-button {active_class}" onclick="window.location.hash='{item["id"]}'">
            <span class="nav-icon">{item["icon"]}</span>
            <span class="nav-text">{item["label"]}</span>
        </div>
        """
    
    nav_html += """</div></div>"""
    
    return nav_html, tab_items

# Interface principale optimisée avec navigation moderne
def main():
    st.title("🧪 Analyse et visualisation des données de HRMS")
    
    # Initialiser les états de session
    if 'features_df' not in st.session_state:
        st.session_state.features_df = None
    if 'matrix_df' not in st.session_state:
        st.session_state.matrix_df = None
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "home"
    
    # Sidebar pour les uploads
    st.sidebar.header("📁 Chargement des fichiers")
    
    # Upload features_complete.csv
    features_file = st.sidebar.file_uploader(
        "features_complete.csv",
        type=['csv'],
        help="Fichier contenant les identifications complètes"
    )
    
    # Upload feature_matrix.csv
    matrix_file = st.sidebar.file_uploader(
        "feature_matrix.csv (optionnel)",
        type=['csv'],
        help="Matrice d'intensités pour analyses statistiques"
    )
    
    # Chargement des données avec persistance
    features_df = st.session_state.features_df
    matrix_df = st.session_state.matrix_df
    
    # Charger features_complete.csv
    if features_file is not None:
        try:
            with st.spinner("Chargement des features..."):
                features_df = load_features_data(features_file)
                st.session_state.features_df = features_df
            st.sidebar.success(f"✅ Features chargées : {len(features_df)} lignes")
        except Exception as e:
            st.sidebar.error(f"❌ Erreur de chargement : {str(e)}")
            features_df = None
            st.session_state.features_df = None
    
    # Charger feature_matrix.csv
    if matrix_file is not None:
        try:
            with st.spinner("Chargement de la matrice..."):
                matrix_df = load_matrix_data(matrix_file)
                st.session_state.matrix_df = matrix_df
            st.sidebar.success(f"✅ Matrice chargée : {matrix_df.shape}")
        except Exception as e:
            st.sidebar.error(f"❌ Erreur de chargement : {str(e)}")
            matrix_df = None
            st.session_state.matrix_df = None
    
    # Quick stats si des données sont chargées
    if features_df is not None:
        # Calcul des statistiques
        total_features = len(features_df)
        identified = len(features_df[features_df['match_name'].notna()]['match_name'].unique())
        samples_list = list(set([s for samples in features_df['samples'].dropna() 
                            for s in samples.split(',')]))
        n_samples = len(samples_list)
        
        # Créer tout le HTML en une seule chaîne sans sauts de ligne problématiques
        quick_stats_html = "<div class=\"quick-stats\">"
        
        quick_stats_html += f"""
        <div class="stat-item">
            <div class="stat-value">{total_features:,}</div>
            <div class="stat-label">Features totales</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{identified:,}</div>
            <div class="stat-label">Molécules identifiées</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{n_samples}</div>
            <div class="stat-label">Échantillons</div>
        </div>"""
        
        if 'confidence_level' in features_df.columns:
            level1_molecules = features_df[features_df['confidence_level'] == 1]['match_name'].dropna().unique()
            # Utiliser une chaîne simple sans indentation ni retours à la ligne
            quick_stats_html += f"<div class=\"stat-item\"><div class=\"stat-value\">{len(level1_molecules):,}</div><div class=\"stat-label\">Niveau 1</div></div>"
        
        # Fermer la div sans retour à la ligne
        quick_stats_html += "</div>"
        st.markdown(quick_stats_html, unsafe_allow_html=True)
    
    # Navigation avec design moderne
    tab_mapping = {
        "🏠 Accueil": "home",
        "📊 Vue d'ensemble": "overview",
        "🔍 Analyse par échantillon": "sample",
        "🧬 Molécules individuelles": "molecules",
        "📡 Facteurs de détection": "detection",
        "⚖️ Comparaison échantillons": "comparison",
        "📈 Analyses statistiques": "statistics",
        "📋 Rapports & Export": "reports",
        "ℹ️ Système de confiance": "confidence"
    }
    
    # Créer la navigation en colonnes
    st.markdown("""<div class="nav-container">""", unsafe_allow_html=True)
    cols = st.columns(len(tab_mapping))
    
    selected_tab = None
    for idx, (tab_name, tab_id) in enumerate(tab_mapping.items()):
        with cols[idx]:
            # Extraire l'icône et le label
            icon = tab_name.split()[0]
            label = ' '.join(tab_name.split()[1:])
            
            # Créer un bouton stylé
            if st.button(f"{icon}\n{label}", key=f"nav_{tab_id}", 
                        use_container_width=True,
                        help=f"Aller à {label}"):
                st.session_state.active_tab = tab_id
                selected_tab = tab_name
    
    # Si aucun bouton n'est cliqué, utiliser l'onglet actif de la session
    if selected_tab is None:
        for tab_name, tab_id in tab_mapping.items():
            if tab_id == st.session_state.active_tab:
                selected_tab = tab_name
                break
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Breadcrumb
    if selected_tab:
        st.markdown(f"""
        <div class="breadcrumb">
            <span class="breadcrumb-item">HRMS Analyse</span>
            <span class="breadcrumb-separator">›</span>
            <span class="breadcrumb-active">{selected_tab}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress indicator
    if features_df is not None:
        progress = (list(tab_mapping.values()).index(st.session_state.active_tab) + 1) / len(tab_mapping) * 100
        st.markdown(f"""
        <div class="progress-indicator">
            <div class="progress-bar" style="width: {progress}%"></div>
        </div>
        """, unsafe_allow_html=True)
    
    # Contenu principal basé sur l'onglet sélectionné
    if st.session_state.active_tab == "home":
        if features_df is None:
            show_home_page()
        else:
            st.markdown("""
            <div class="section-header">
                <h2>🎉 Données chargées avec succès !</h2>
                <p>Vos données sont maintenant disponibles dans toutes les sections d'analyse.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Résumé rapide
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total features", len(features_df))
            
            with col2:
                identified = len(features_df[features_df['match_name'].notna()]['match_name'].unique())
                st.metric("Molécules identifiées", identified)
            
            with col3:
                samples_list = list(set([s for samples in features_df['samples'].dropna() 
                                       for s in samples.split(',')]))
                st.metric("Échantillons", len(samples_list))
            
            with col4:
                matrix_status = "✅ Chargée" if matrix_df is not None else "❌ Non chargée"
                st.metric("Matrice", matrix_status)
            
            st.markdown("---")
            show_home_page()
    
    elif st.session_state.active_tab == "overview":
        if features_df is None:
            st.warning("⚠️ Veuillez charger le fichier **features_complete.csv** pour accéder à l'analyse")
        else:
            st.markdown("""
            <div class="section-header">
                <h2>📊 Vue d'ensemble du dataset</h2>
                <p>Statistiques globales et distributions de vos données MS</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Info
            st.info("ℹ️ Comptage basé sur les molécules uniques. Les adduits multiples pour une même molécule sont groupés.")
            
            # Métriques principales
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.metric("Total features", len(features_df))
            
            with col2:
                identified = len(features_df[features_df['match_name'].notna()]['match_name'].unique())
                id_rate = identified/len(features_df)*100
                st.metric("Identifiées", identified, delta=f"{id_rate:.1f}%")
            
            with col3:
                if 'confidence_level' in features_df.columns:
                    level1_molecules = features_df[features_df['confidence_level'] == 1]['match_name'].dropna().unique()
                    st.metric("Niveau 1", len(level1_molecules))
            
            with col4:
                unique_samples = len(set([s for samples in features_df['samples'].dropna() 
                                        for s in samples.split(',')]))
                st.metric("Échantillons", unique_samples)
            
            with col5:
                avg_intensity = features_df['intensity'].mean()
                st.metric("Int. moyenne", f"{avg_intensity:.0f}")
            
            with col6:
                if 'categories' in features_df.columns:
                    all_cats = []
                    for cats in features_df['categories'].dropna():
                        if isinstance(cats, list):
                            all_cats.extend(cats)
                    unique_cats = len(set(all_cats))
                    st.metric("Catégories", unique_cats)
            
            # Graphiques
            col1, col2 = st.columns(2)
            
            with col1:
                conf_fig = plot_confidence_levels_distribution(features_df)
                if conf_fig:
                    st.plotly_chart(conf_fig, use_container_width=True, key=generate_unique_key("overview_confidence_dist"))
                
                if 'mz_error_ppm' in features_df.columns:
                    error_data = features_df[features_df['mz_error_ppm'].notna()]
                    if not error_data.empty:
                        fig_error = px.histogram(
                            error_data,
                            x='mz_error_ppm',
                            nbins=50,
                            title="Distribution des erreurs m/z",
                            labels={'mz_error_ppm': 'Erreur m/z (ppm)', 'count': 'Nombre'},
                            color_discrete_sequence=DISTINCT_COLORS
                        )
                        fig_error.add_vline(x=5, line_dash="dash", line_color=DISTINCT_COLORS[1],
                                          annotation_text="Seuil 5 ppm")
                        st.plotly_chart(fig_error, use_container_width=True, key=generate_unique_key("overview_mz_error"))
            
            with col2:
                if 'ms2_similarity_score' in features_df.columns:
                    ms2_data = features_df[features_df['ms2_similarity_score'] > 0]
                    if not ms2_data.empty:
                        fig_ms2 = px.histogram(
                            ms2_data,
                            x='ms2_similarity_score',
                            nbins=30,
                            title="Distribution des scores MS2",
                            labels={'ms2_similarity_score': 'Score MS2', 'count': 'Nombre'},
                            color_discrete_sequence=DISTINCT_COLORS
                        )
                        fig_ms2.add_vline(x=0.7, line_dash="dash", line_color=DISTINCT_COLORS[2],
                                        annotation_text="Excellent (≥0.7)")
                        fig_ms2.add_vline(x=0.4, line_dash="dash", line_color=DISTINCT_COLORS[3],
                                        annotation_text="Bon (≥0.4)")
                        st.plotly_chart(fig_ms2, use_container_width=True, key=generate_unique_key("overview_ms2_dist"))
            
            st.markdown("---")
            samples_list = list(set([s for samples in features_df['samples'].dropna() 
                                   for s in samples.split(',')]))
            plot_boxplot_by_category(features_df, samples_list)
    
    elif st.session_state.active_tab == "sample":
        if features_df is None:
            st.warning("⚠️ Veuillez charger le fichier **features_complete.csv** pour accéder à l'analyse")
        else:
            st.markdown("""
            <div class="section-header">
                <h2>🔍 Analyse détaillée par échantillon</h2>
                <p>Explorez les données spécifiques à chaque échantillon</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("ℹ️ Affichage automatique des molécules uniques (une ligne par molécule avec adduits groupés)")
            
            # Sélection de l'échantillon
            samples_list = list(set([s for samples in features_df['samples'].dropna() 
                                   for s in samples.split(',')]))
            
            selected_sample = st.selectbox("Choisir un échantillon", samples_list, key="sample_selector_main")
            
            if selected_sample:
                sample_data = plot_sample_overview(features_df, selected_sample)
                
                # Graphiques
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_mz = px.scatter(
                        sample_data,
                        x='mz',
                        y='intensity',
                        color='confidence_level' if 'confidence_level' in sample_data.columns else None,
                        size='intensity',
                        hover_data=['match_name', 'retention_time'],
                        title=f"Distribution m/z - {selected_sample}",
                        log_y=True,
                        color_discrete_sequence=DISTINCT_COLORS
                    )
                    st.plotly_chart(fig_mz, use_container_width=True, key=generate_unique_key(f"sample_mz_dist_{selected_sample}"))
                
                with col2:
                    fig_rt = px.scatter(
                        sample_data,
                        x='retention_time',
                        y='intensity',
                        color='confidence_level' if 'confidence_level' in sample_data.columns else None,
                        size='intensity',
                        hover_data=['match_name', 'mz'],
                        title=f"Profil chromatographique - {selected_sample}",
                        log_y=True,
                        color_discrete_sequence=DISTINCT_COLORS
                    )
                    st.plotly_chart(fig_rt, use_container_width=True, key=generate_unique_key(f"sample_rt_profile_{selected_sample}"))
                
                st.markdown("---")
                plot_sample_boxplot_by_category(features_df, selected_sample)
                
                st.subheader(f"🫧 Molécules uniques par niveau - {selected_sample}")
                plot_level1_bubble_plot_sample(features_df, selected_sample)
                
                # Tableau détaillé
                st.subheader(f"Molécules uniques - {selected_sample}")
                
                sample_identified = sample_data[sample_data['match_name'].notna()]
                
                if not sample_identified.empty:
                    aggregated_display = aggregate_molecules_by_name_enhanced(sample_identified, selected_sample)
                    
                    if not aggregated_display.empty:
                        st.info(f"Affichage de {len(aggregated_display)} molécules uniques (adduits automatiquement groupés)")
                        
                        # Filtres
                        with st.expander("Filtres avancés"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                intensity_column = 'sample_specific_intensity' if 'sample_specific_intensity' in aggregated_display.columns else 'total_intensity'
                                intensity_min = st.number_input(
                                    f"Intensité minimale ({intensity_column})",
                                    min_value=0.0,
                                    value=0.0,
                                    key="intensity_min_sample"
                                )
                            
                            with col2:
                                if 'confidence_level' in aggregated_display.columns:
                                    conf_levels = st.multiselect(
                                        "Niveaux de confiance",
                                        options=[1, 2, 3, 4, 5],
                                        default=[1],
                                        key="conf_levels_sample"
                                    )
                            
                            with col3:
                                if 'categories' in aggregated_display.columns:
                                    all_cats = []
                                    for cats in aggregated_display['categories'].dropna():
                                        if isinstance(cats, list):
                                            all_cats.extend(cats)
                                    cat_filter = st.multiselect(
                                        "Catégories",
                                        options=list(set(all_cats)),
                                        key="cat_filter_sample"
                                    )
                        
                        # Appliquer les filtres
                        filtered_data = aggregated_display[aggregated_display[intensity_column] >= intensity_min]
                        
                        if 'confidence_level' in aggregated_display.columns and conf_levels:
                            filtered_data = filtered_data[filtered_data['confidence_level'].isin(conf_levels)]
                        
                        if cat_filter:
                            filtered_data = filtered_data[
                                filtered_data['categories'].apply(
                                    lambda x: any(cat in x for cat in cat_filter) if isinstance(x, list) else False
                                )
                            ]
                        
                        # Colonnes à afficher
                        display_columns = [
                            'match_name', 'mz', 'retention_time', intensity_column, 'intensity',
                            'confidence_level', 'match_adduct',
                            'mz_error_ppm', 'rt_error_min', 'ccs_error_percent',
                            'ms2_similarity_score', 'categories'
                        ]
                        available_columns = [col for col in display_columns if col in filtered_data.columns]
                        
                        display_df = filtered_data[available_columns].copy()
                        if 'match_adduct' in display_df.columns:
                            display_df['match_adduct'] = display_df['match_adduct'].apply(
                                lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                            )
                        
                        # Affichage avec sélection
                        selected_indices = st.dataframe(
                            display_df.round(4),
                            use_container_width=True,
                            on_select="rerun",
                            selection_mode="single-row"
                        )
                        
                        # Si une ligne est sélectionnée
                        if selected_indices.selection.rows:
                            selected_idx = selected_indices.selection.rows[0]
                            selected_molecule_data = filtered_data.iloc[selected_idx]
                            
                            st.markdown("---")
                            st.subheader("🔬 Analyse détaillée de la molécule sélectionnée")
                            
                            if pd.notna(selected_molecule_data.get('match_name')):
                                st.subheader("🔗 Liens externes")
                                display_pubchem_link(
                                    selected_molecule_data.get('match_name'),
                                    selected_molecule_data.get('match_smiles')
                                )
                                
                                # Informations de base
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown(f"**Molécule :** {selected_molecule_data.get('match_name')}")
                                    st.markdown(f"**m/z :** {selected_molecule_data.get('mz', 'N/A'):.4f}")
                                    st.markdown(f"**RT :** {selected_molecule_data.get('retention_time', 'N/A'):.2f} min")
                                
                                with col2:
                                    intensity_val = selected_molecule_data.get(intensity_column, 'N/A')
                                    st.markdown(f"**Intensité ({intensity_column}) :** {intensity_val:.2e}")
                                    st.markdown(f"**Intensité max :** {selected_molecule_data.get('intensity', 'N/A'):.2e}")
                                    st.markdown(f"**Niveau confiance :** {selected_molecule_data.get('confidence_level', 'N/A')}")
                                
                                with col3:
                                    st.markdown(f"**Erreur m/z :** {selected_molecule_data.get('mz_error_ppm', 'N/A'):.2f} ppm")
                                    st.markdown(f"**Score MS2 :** {selected_molecule_data.get('ms2_similarity_score', 0):.3f}")
                                    
                                    adduits = selected_molecule_data.get('match_adduct', 'N/A')
                                    if isinstance(adduits, list):
                                        st.markdown(f"**Adduits :** {', '.join(adduits)}")
                                    else:
                                        st.markdown(f"**Adduits :** {adduits}")
                                
                                # Explication du niveau de confiance
                                with st.expander("Pourquoi ce niveau de confiance ?", expanded=True):
                                    explanation_html = explain_compound_confidence(selected_molecule_data)
                                    st.markdown(explanation_html, unsafe_allow_html=True)
                                
                                # Spectres MS2
                                st.subheader("🔬 Spectres MS2")
                                plot_ms2_comparison_enhanced(selected_molecule_data)
                                
                                # Écotoxicologie
                                st.subheader("🌿 Données écotoxicologiques")
                                analyze_ecotoxicity(selected_molecule_data)
                else:
                    st.warning("Aucune molécule identifiée dans cet échantillon")
    
    elif st.session_state.active_tab == "molecules":
        if features_df is None:
            st.warning("⚠️ Veuillez charger le fichier **features_complete.csv** pour accéder à l'analyse")
        else:
            st.markdown("""
            <div class="section-header">
                <h2>🧬 Analyse détaillée des molécules</h2>
                <p>Explorez les propriétés de chaque composé identifié</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("ℹ️ Les adduits multiples pour une même molécule sont groupés. Meilleures valeurs affichées.")
            
            identified_molecules = features_df[features_df['match_name'].notna()]
            
            if not identified_molecules.empty:
                # Filtres
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    samples_list = list(set([s for samples in features_df['samples'].dropna() 
                                           for s in samples.split(',')]))
                    selected_samples_filter = st.multiselect(
                        "Filtrer par échantillons",
                        options=samples_list,
                        default=samples_list,
                        key="samples_filter_molecules"
                    )
                
                with col2:
                    if 'confidence_level' in identified_molecules.columns:
                        conf_filter = st.multiselect(
                            "Niveaux de confiance",
                            options=sorted(identified_molecules['confidence_level'].dropna().unique()),
                            default=sorted(identified_molecules['confidence_level'].dropna().unique()),
                            key="conf_filter_molecules"
                        )
                
                with col3:
                    sort_by = st.selectbox(
                        "Trier par",
                        ["Nom", "Confiance", "Intensité"],
                        key="sort_by_molecules"
                    )
                
                # Appliquer les filtres
                filtered_molecules = identified_molecules.copy()
                
                if selected_samples_filter:
                    filtered_molecules = filtered_molecules[
                        filtered_molecules['samples'].apply(
                            lambda x: any(sample in str(x) for sample in selected_samples_filter) if pd.notna(x) else False
                        )
                    ]
                
                if conf_filter and 'confidence_level' in filtered_molecules.columns:
                    filtered_molecules = filtered_molecules[
                        filtered_molecules['confidence_level'].isin(conf_filter)
                    ]
                
                if not filtered_molecules.empty:
                    molecule_names = sorted(filtered_molecules['match_name'].unique())
                    
                    selected_molecule = st.selectbox(
                        "Choisir une molécule",
                        options=[""] + molecule_names,
                        index=0,
                        key="molecule_selector"
                    )
                    
                    if selected_molecule and selected_molecule in molecule_names:
                        molecule_occurrences = filtered_molecules[
                            filtered_molecules['match_name'] == selected_molecule
                        ]
                        
                        aggregated_molecule = aggregate_molecules_by_name_enhanced(molecule_occurrences)
                        
                        if not aggregated_molecule.empty:
                            molecule_data = aggregated_molecule.iloc[0]
                            
                            st.markdown(f"""
                            <div class="molecule-card">
                            <h2>{selected_molecule}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.subheader("🔗 Liens externes")
                            display_pubchem_link(
                                selected_molecule,
                                molecule_data.get('match_smiles')
                            )
                            
                            level = molecule_data.get('confidence_level', 'N/A')
                            st.markdown(f"""
                            <div class="info-box">
                            <h3>Niveau de confiance : {level}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Métriques
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("m/z mesuré", f"{molecule_data.get('mz', 'N/A'):.4f}")
                                st.metric("m/z théorique", f"{molecule_data.get('match_mz', 'N/A'):.4f}")
                            
                            with col2:
                                st.metric("Erreur m/z", f"{molecule_data.get('mz_error_ppm', 'N/A'):.2f} ppm")
                                st.metric("Score MS2", f"{molecule_data.get('ms2_similarity_score', 0):.3f}")
                            
                            with col3:
                                st.metric("RT observé", f"{molecule_data.get('match_rt_obs', 'N/A'):.2f} min")
                                st.metric("Erreur RT", f"{molecule_data.get('rt_error_min', 'N/A'):.3f} min")
                            
                            with col4:
                                st.metric("CCS exp", f"{molecule_data.get('match_ccs_exp', 'N/A'):.1f} Å²")
                                st.metric("Erreur CCS", f"{molecule_data.get('ccs_error_percent', 'N/A'):.1f}%")
                            
                            # Adduits
                            st.subheader("🔗 Adduits détectés")
                            adduits = molecule_data.get('match_adduct', [])
                            if isinstance(adduits, list) and adduits:
                                for adduit in adduits:
                                    st.markdown(f"• {adduit}")
                                st.metric("Intensité totale (tous adduits)", f"{molecule_data.get('total_intensity', 'N/A'):.2e}")
                            
                            # Intensités par échantillon
                            if 'intensities_by_sample' in molecule_data and 'sample_names_order' in molecule_data:
                                st.subheader("📊 Intensités par échantillon")
                                try:
                                    intensities = molecule_data['intensities_by_sample']
                                    sample_names = molecule_data['sample_names_order']
                                    
                                    if isinstance(intensities, list) and isinstance(sample_names, list):
                                        intensity_df = pd.DataFrame({
                                            'Échantillon': sample_names,
                                            'Intensité': intensities
                                        })
                                        
                                        fig_intensities = px.bar(
                                            intensity_df,
                                            x='Échantillon',
                                            y='Intensité',
                                            title=f"Intensités de {selected_molecule} par échantillon",
                                            color='Échantillon',
                                            color_discrete_sequence=DISTINCT_COLORS[:len(sample_names)]
                                        )
                                        fig_intensities.update_layout(showlegend=False)
                                        st.plotly_chart(fig_intensities, use_container_width=True, key=generate_unique_key(f"molecule_intensities_{selected_molecule}"))
                                        
                                        st.dataframe(intensity_df, use_container_width=True)
                                except:
                                    st.info("Données d'intensités par échantillon non disponibles")
                            
                            # Explication niveau de confiance
                            st.subheader("📊 Pourquoi ce niveau de confiance ?")
                            with st.expander("Voir l'explication détaillée", expanded=True):
                                explanation_html = explain_compound_confidence(molecule_data)
                                st.markdown(explanation_html, unsafe_allow_html=True)
                            
                            # Spectres MS2
                            st.subheader("🔬 Analyse des spectres MS2")
                            plot_ms2_comparison_enhanced(molecule_data)
                            
                            # Écotoxicologie
                            st.subheader("🌿 Données écotoxicologiques")
                            analyze_ecotoxicity(molecule_data)
                            
                            # Informations additionnelles
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("📋 Informations structurales")
                                st.markdown(f"""
                                **SMILES :** `{molecule_data.get('match_smiles', 'N/A')}`
                                
                                **Échantillons :** {molecule_data.get('samples', 'N/A')}
                                
                                **Intensité max :** {molecule_data.get('intensity', 'N/A'):.2e}
                                """)
                            
                            with col2:
                                st.subheader("🏷️ Catégories")
                                categories = molecule_data.get('categories', [])
                                if isinstance(categories, list) and categories:
                                    for cat in categories:
                                        st.markdown(f"- {cat}")
                                else:
                                    st.markdown("Aucune catégorie assignée")
                    else:
                        st.info("Sélectionnez une molécule dans la liste déroulante pour voir ses détails")
    
    elif st.session_state.active_tab == "detection":
        if features_df is None:
            st.warning("⚠️ Veuillez charger le fichier **features_complete.csv** pour accéder à l'analyse")
        else:
            st.markdown("""
            <div class="section-header">
                <h2>📡 Facteurs de détection par catégorie</h2>
                <p>Évaluez l'efficacité de votre méthode pour chaque famille de composés</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("ℹ️ Facteurs calculés sur les molécules uniques. Les adduits multiples ne sont comptés qu'une fois.")
            
            st.markdown("""
            <div class="detection-factor">
            <h3>🎯 Facteur de détection</h3>
            <p><strong>Formule :</strong> (Nombre de composés UNIQUES détectés dans la famille / Nombre total de composés de la famille dans la base de données) × 100</p>
            <p>Ce facteur indique l'efficacité de détection de votre méthode pour chaque famille de composés.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("⚙️ Options de filtrage")
            
            col1, col2 = st.columns(2)
            with col1:
                confidence_levels_choice = st.multiselect(
                    "Inclure les niveaux de confiance",
                    options=[1, 2, 3, 4, 5],
                    default=[1, 2, 3, 4, 5],
                    key="conf_levels_detection"
                )
            
            with col2:
                st.info(f"Niveaux sélectionnés : {', '.join(map(str, confidence_levels_choice))}")
            
            samples_list = list(set([s for samples in features_df['samples'].dropna() 
                                   for s in samples.split(',')]))
            
            if samples_list and confidence_levels_choice:
                detection_factors = calculate_detection_factor(features_df, samples_list, confidence_levels_choice)
                
                st.subheader("📡 Graphique radar des facteurs de détection")
                plot_detection_factor_radar(detection_factors)
                
                st.subheader("📊 Tableau détaillé des facteurs de détection")
                
                detection_df_data = []
                for sample, factors in detection_factors.items():
                    for category, factor in factors.items():
                        sample_data = features_df[
                            (features_df['samples'].str.contains(sample, na=False)) & 
                            (features_df['match_name'].notna()) &
                            (features_df['confidence_level'].isin(confidence_levels_choice) if 'confidence_level' in features_df.columns else True)
                        ]
                        
                        unique_molecules = set()
                        for idx, row in sample_data.iterrows():
                            if isinstance(row.get('categories', []), list) and category in row.get('categories', []):
                                unique_molecules.add(row['match_name'])
                        
                        detected = len(unique_molecules)
                        
                        detection_df_data.append({
                            'Échantillon': sample,
                            'Catégorie': category,
                            'Détectés': detected,
                            'Total_DB': DATABASE_CATEGORIES[category],
                            'Facteur_détection_%': factor
                        })
                
                detection_df = pd.DataFrame(detection_df_data)
                
                # Filtres
                col1, col2 = st.columns(2)
                with col1:
                    selected_samples = st.multiselect(
                        "Filtrer par échantillons",
                        samples_list,
                        default=samples_list,
                        key="samples_filter_detection"
                    )
                
                with col2:
                    selected_categories = st.multiselect(
                        "Filtrer par catégories",
                        list(DATABASE_CATEGORIES.keys()),
                        default=list(DATABASE_CATEGORIES.keys()),
                        key="categories_filter_detection"
                    )
                
                filtered_detection_df = detection_df[
                    (detection_df['Échantillon'].isin(selected_samples)) &
                    (detection_df['Catégorie'].isin(selected_categories))
                ]
                
                st.dataframe(
                    filtered_detection_df.style.format({
                        'Facteur_détection_%': '{:.2f}%'
                    }).background_gradient(subset=['Facteur_détection_%'], cmap='RdYlGn'),
                    use_container_width=True
                )
                
                st.subheader("🏆 Top 5 catégories les mieux détectées par échantillon")
                
                for i, sample in enumerate(samples_list):
                    sample_data = detection_df[detection_df['Échantillon'] == sample]
                    top5 = sample_data.nlargest(5, 'Facteur_détection_%')
                    
                    with st.expander(f"📈 {sample}", expanded=False):
                        fig = px.bar(
                            top5,
                            x='Catégorie',
                            y='Facteur_détection_%',
                            title=f"Top 5 catégories - {sample}",
                            color='Facteur_détection_%',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(showlegend=False, height=400)
                        st.plotly_chart(fig, use_container_width=True, key=generate_unique_key(f"top5_categories_{i}"))
            
            st.subheader("📊 Distribution des catégories par échantillon")
            
            confidence_levels_radar = st.multiselect(
                "Niveaux de confiance pour le radar des catégories",
                options=[1, 2, 3, 4, 5],
                default=[1, 2, 3, 4, 5],
                key="conf_levels_radar"
            )
            
            plot_category_distribution_radar(features_df, samples_list, confidence_levels_radar)
    
    elif st.session_state.active_tab == "comparison":
        if features_df is None:
            st.warning("⚠️ Veuillez charger le fichier **features_complete.csv** pour accéder à l'analyse")
        else:
            st.markdown("""
            <div class="section-header">
                <h2>⚖️ Comparaison multi-échantillons</h2>
                <p>Comparez les performances et similarités entre vos échantillons</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("ℹ️ Analyses basées sur les molécules uniques. Les adduits multiples ne sont comptés qu'une fois.")
            
            samples_list = list(set([s for samples in features_df['samples'].dropna() 
                                   for s in samples.split(',')]))
            
            if len(samples_list) >= 2:
                # Statistiques comparatives
                stats_data = []
                for sample in samples_list:
                    sample_data = features_df[features_df['samples'].str.contains(sample, na=False)]
                    identified_data = sample_data[sample_data['match_name'].notna()]
                    
                    unique_identified = len(identified_data['match_name'].unique())
                    non_identified = len(sample_data) - len(identified_data)
                    
                    conf_counts = {}
                    if 'confidence_level' in sample_data.columns:
                        for level in range(1, 5):
                            level_molecules = sample_data[sample_data['confidence_level'] == level]['match_name'].dropna().unique()
                            conf_counts[f'Niveau_{level}'] = len(level_molecules)
                        conf_counts['Niveau_5'] = non_identified
                    
                    total_sample_intensity = 0
                    count = 0
                    for idx, row in sample_data.iterrows():
                        sample_intensity = get_intensity_for_sample(row, sample)
                        if sample_intensity > 0:
                            total_sample_intensity += sample_intensity
                            count += 1
                    
                    avg_intensity = total_sample_intensity / count if count > 0 else sample_data['intensity'].mean()
                    median_intensity = sample_data['intensity'].median()
                    
                    total_intensity = 0
                    for idx, row in sample_data.iterrows():
                        sample_intensity = get_intensity_for_sample(row, sample)
                        total_intensity += sample_intensity
                    
                    stats = {
                        'Échantillon': sample,
                        'Total_features': len(sample_data),
                        'Identifiées': unique_identified,
                        'Non_identifiées': non_identified,
                        'Taux_identification_%': (unique_identified/len(sample_data)*100) if len(sample_data) > 0 else 0,
                        'Intensité_moyenne': avg_intensity,
                        'Intensité_médiane': median_intensity,
                        'Intensité_totale': total_intensity
                    }
                    stats.update(conf_counts)
                    stats_data.append(stats)
                
                stats_df = pd.DataFrame(stats_data)
                
                st.subheader("📊 Tableau comparatif des échantillons")
                st.dataframe(stats_df.round(2), use_container_width=True)
                
                # Visualisations
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = px.bar(
                        stats_df,
                        x='Échantillon',
                        y=['Total_features', 'Identifiées', 'Non_identifiées'],
                        title="Features totales vs identifiées vs non identifiées",
                        barmode='group',
                        color_discrete_sequence=DISTINCT_COLORS[:3]
                    )
                    st.plotly_chart(fig1, use_container_width=True, key=generate_unique_key("comparison_features_vs_identified"))
                
                with col2:
                    fig2 = px.bar(
                        stats_df.sort_values('Taux_identification_%', ascending=False),
                        x='Échantillon',
                        y='Taux_identification_%',
                        title="Taux d'identification par échantillon",
                        color='Taux_identification_%',
                        color_continuous_scale='Viridis'
                    )
                    fig2.update_layout(showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True, key=generate_unique_key("comparison_identification_rate"))
                
                st.subheader("📊 Distribution des niveaux de confiance par échantillon")
                plot_confidence_comparison_across_samples(features_df, samples_list)
                
                st.subheader("🫧 Bubble plot - Intensités des molécules par niveau")
                plot_level1_bubble_plot(features_df, samples_list)
                
                st.subheader("🎯 Comparaison multi-critères (radar)")
                
                available_metrics = [col for col in stats_df.columns if col not in ['Échantillon'] and stats_df[col].dtype in ['int64', 'float64']]
                selected_metrics = st.multiselect(
                    "Métriques pour le radar",
                    available_metrics,
                    default=['Taux_identification_%', 'Niveau_1', 'Niveau_2'][:min(3, len(available_metrics))],
                    key="metrics_radar_comparison"
                )
                
                if len(selected_metrics) >= 3:
                    fig_radar = go.Figure()
                    
                    for idx, row in stats_df.iterrows():
                        values = [row[metric] for metric in selected_metrics]
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=selected_metrics,
                            fill='toself',
                            name=row['Échantillon'],
                            line=dict(width=2, color=DISTINCT_COLORS[idx % len(DISTINCT_COLORS)]),
                            opacity=0.7
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, max([stats_df[metric].max() for metric in selected_metrics])],
                                tickfont=dict(size=12, color='#555555')
                            ),
                            angularaxis=dict(
                                tickfont=dict(size=12, color='#555555')
                            )),
                        showlegend=True,
                        title="Comparaison multi-critères des échantillons",
                        height=600,
                        font=dict(color='#555555')
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True, key=generate_unique_key("comparison_radar_multicriteria"))
                
                st.subheader("🔗 Similarité de Jaccard et Clustering hiérarchique")

                col1, col2 = st.columns(2)

                with col1:
                    if 'confidence_level' in features_df.columns:
                        confidence_levels_jaccard = st.multiselect(
                            "Filtrer par niveau de confiance",
                            options=[1, 2, 3, 4, 5],
                            default=[1, 2, 3],
                            key="conf_levels_jaccard"
                        )
                    else:
                        confidence_levels_jaccard = None

                with col2:
                    selected_samples_jaccard = st.multiselect(
                        "Sélectionner des échantillons spécifiques",
                        options=samples_list,
                        default=samples_list,
                        key="selected_samples_jaccard"
                    )

                plot_hierarchical_clustering(
                    features_df, 
                    samples_list, 
                    confidence_levels=confidence_levels_jaccard, 
                    selected_samples=selected_samples_jaccard if selected_samples_jaccard else None
                )
                
            else:
                st.warning("Au moins 2 échantillons sont nécessaires pour la comparaison")
    
    elif st.session_state.active_tab == "statistics":
        if matrix_df is not None:
            st.markdown("""
            <div class="section-header">
                <h2>📈 Analyses statistiques avancées</h2>
                <p>Explorez vos données avec des méthodes statistiques avancées</p>
            </div>
            """, unsafe_allow_html=True)
            
            stat_section = st.selectbox(
                "Choisir une analyse:",
                ["📊 PCA & t-SNE", "🔍 Clustering", "📈 Corrélations", "🎨 Heatmaps"],
                key="stat_navigation"
            )
            
            if stat_section == "📊 PCA & t-SNE":
                st.subheader("📊 Analyse en Composantes Principales (PCA)")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    pca_3d = st.checkbox("Afficher PCA 3D", value=True, key="pca_3d_checkbox")
                with col2:
                    show_loadings = st.checkbox("Afficher les loadings", value=False, key="show_loadings_checkbox")
                
                if pca_3d:
                    plot_3d_pca(matrix_df)
                else:
                    perform_pca_analysis(matrix_df)
                
                if show_loadings:
                    st.subheader("Contribution des features aux composantes principales")
                    
                    pca, X_pca, X_scaled = safe_pca_analysis(matrix_df, n_components=min(10, matrix_df.shape[1], matrix_df.shape[0]))
                    
                    if pca is not None:
                        n_components = min(3, pca.n_components_)
                        loadings_df = pd.DataFrame(
                            pca.components_[:n_components].T,
                            columns=[f'PC{i+1}' for i in range(n_components)],
                            index=matrix_df.columns
                        )
                        
                        for i in range(n_components):
                            pc = f'PC{i+1}'
                            st.write(f"**Top 10 contributeurs à {pc}:**")
                            top_features = loadings_df[pc].abs().nlargest(10)
                            st.dataframe(top_features.round(3))
                
                st.markdown("---")
                
                st.subheader("🌐 Analyse t-SNE")
                plot_tsne_analysis(matrix_df)
            
            elif stat_section == "🔍 Clustering":
                st.subheader("🔍 Analyses de clustering")
                
                st.subheader("K-means Clustering")
                perform_kmeans_clustering(matrix_df)
                
                st.markdown("---")
                
                st.subheader("Clustering hiérarchique")
                st.info("Cette analyse utilise les molécules identifiées du fichier features")
                if features_df is not None:
                    samples_list = list(set([s for samples in features_df['samples'].dropna() 
                                           for s in samples.split(',')]))
                    plot_hierarchical_clustering(features_df, samples_list)
            
            elif stat_section == "📈 Corrélations":
                st.subheader("📈 Analyses de corrélation")
                
                corr_matrix = plot_correlation_heatmap(matrix_df)
                
                if corr_matrix is not None:
                    st.subheader("Statistiques de corrélation")
                    
                    corr_values = []
                    for i in range(len(corr_matrix)):
                        for j in range(i+1, len(corr_matrix)):
                            corr_values.append(corr_matrix.iloc[i, j])
                    
                    if corr_values:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Corrélation moyenne", f"{np.mean(corr_values):.3f}")
                        with col2:
                            st.metric("Corrélation médiane", f"{np.median(corr_values):.3f}")
                        with col3:
                            st.metric("Corrélation max", f"{max(corr_values):.3f}")
                        with col4:
                            st.metric("Corrélation min", f"{min(corr_values):.3f}")
                        
                        fig_corr_dist = px.histogram(
                            x=corr_values,
                            nbins=30,
                            title="Distribution des corrélations entre échantillons",
                            labels={'x': 'Coefficient de corrélation', 'y': 'Fréquence'},
                            color_discrete_sequence=DISTINCT_COLORS
                        )
                        st.plotly_chart(fig_corr_dist, use_container_width=True, key=generate_unique_key("correlation_distribution"))
            
            elif stat_section == "🎨 Heatmaps":
                st.subheader("🎨 Heatmaps avancées")
                
                st.subheader("🔥 Heatmap des intensités")
                
                transform_option = st.selectbox(
                    "Transformation des données",
                    ["Aucune", "Log10", "Z-score", "Min-Max"],
                    key="heatmap_transform"
                )
                
                if transform_option == "Log10":
                    matrix_transformed = np.log10(matrix_df + 1)
                    title_suffix = "(échelle log)"
                elif transform_option == "Z-score":
                    scaler = StandardScaler()
                    matrix_transformed = pd.DataFrame(
                        scaler.fit_transform(matrix_df.T).T,
                        index=matrix_df.index,
                        columns=matrix_df.columns
                    )
                    title_suffix = "(Z-score)"
                elif transform_option == "Min-Max":
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    matrix_transformed = pd.DataFrame(
                        scaler.fit_transform(matrix_df.T).T,
                        index=matrix_df.index,
                        columns=matrix_df.columns
                    )
                    title_suffix = "(Min-Max normalisé)"
                else:
                    matrix_transformed = matrix_df
                    title_suffix = ""
                
                max_features = min(100, len(matrix_df.columns))
                n_features = st.slider(
                    "Nombre de features à afficher",
                    10, max_features, min(50, max_features),
                    key="heatmap_features"
                )
                
                feature_var = matrix_transformed.var(axis=0).nlargest(n_features)
                selected_features = feature_var.index
                
                matrix_subset = matrix_transformed[selected_features]
                
                fig = px.imshow(
                    matrix_subset,
                    labels=dict(x="Features", y="Échantillons", color=f"Intensité {title_suffix}"),
                    title=f"Heatmap des intensités {title_suffix}",
                    aspect="auto",
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("advanced_heatmap"))
                
                st.subheader("📏 Heatmap des distances entre échantillons")
                
                distance_metric = st.selectbox(
                    "Métrique de distance",
                    ["Euclidienne", "Cosinus"],
                    key="distance_metric"
                )
                
                from scipy.spatial.distance import pdist, squareform
                
                if distance_metric == "Euclidienne":
                    distances = pdist(matrix_df.values, metric='euclidean')
                else:
                    distances = pdist(matrix_df.values, metric='cosine')
                
                distance_matrix = squareform(distances)
                distance_df = pd.DataFrame(
                    distance_matrix,
                    index=matrix_df.index,
                    columns=matrix_df.index
                )
                
                fig_dist = px.imshow(
                    distance_df,
                    text_auto=".2f",
                    aspect="auto",
                    title=f"Matrice de distance {distance_metric.lower()} entre échantillons",
                    color_continuous_scale='Plasma'
                )
                
                fig_dist.update_layout(height=600)
                st.plotly_chart(fig_dist, use_container_width=True, key=generate_unique_key("distance_heatmap"))
            
        else:
            st.warning("""
            ⚠️ Veuillez charger le fichier **feature_matrix.csv** pour accéder aux analyses statistiques avancées.
            
            Ce fichier doit contenir une matrice avec :
            - Lignes : échantillons
            - Colonnes : features (format : F0001_mz102.9880)
            - Valeurs : intensités
            """)
    
    elif st.session_state.active_tab == "reports":
        if features_df is None:
            st.warning("⚠️ Veuillez charger le fichier **features_complete.csv** pour accéder aux fonctions d'export")
        else:
            st.markdown("""
            <div class="section-header">
                <h2>📋 Génération de rapports et export</h2>
                <p>Exportez vos résultats dans différents formats</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("ℹ️ Rapports basés sur les molécules uniques. Les adduits multiples ne sont comptés qu'une fois.")
            
            if st.button("📄 Générer le rapport de qualité", type="primary"):
                samples_list = list(set([s for samples in features_df['samples'].dropna() 
                                       for s in samples.split(',')]))
                
                unique_molecules = len(features_df[features_df['match_name'].notna()]['match_name'].unique())
                
                report = f"""
# Rapport de Qualité des Données MS

**Date de génération :** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Résumé Global

- **Total de features :** {len(features_df)}
- **Molécules uniques identifiées :** {unique_molecules} ({unique_molecules/len(features_df)*100:.1f}%)
- **Nombre d'échantillons :** {len(samples_list)}

## Distribution des Niveaux de Confiance (Molécules Uniques)

"""
                
                if 'confidence_level' in features_df.columns:
                    for level in range(1, 6):
                        level_molecules = features_df[features_df['confidence_level'] == level]['match_name'].dropna().unique()
                        count = len(level_molecules)
                        if count > 0:
                            report += f"- Niveau {level} : {count} molécules uniques ({count/unique_molecules*100:.1f}%)\n"
                
                report += f"""

## Facteurs de Détection par Catégorie (Molécules Uniques)

"""
                
                detection_factors = calculate_detection_factor(features_df, samples_list)
                
                for sample, factors in detection_factors.items():
                    report += f"\n### {sample}\n"
                    for category, factor in factors.items():
                        if factor > 0:
                            report += f"- {category}: {factor:.2f}%\n"
                
                ecotox_columns = ['daphnia_LC50_48_hr_ug/L', 'algae_EC50_72_hr_ug/L', 'pimephales_LC50_96_hr_ug/L']
                ecotox_available = any(col in features_df.columns for col in ecotox_columns)
                
                if ecotox_available:
                    report += f"""

## Données Écotoxicologiques

"""
                    for col in ecotox_columns:
                        if col in features_df.columns:
                            non_null_count = features_df[col].notna().sum()
                            report += f"- {col}: {non_null_count} valeurs disponibles\n"
                
                report += f"""

## Statistiques par Échantillon (Molécules Uniques et Intensités Spécifiques)

"""
                
                for sample in samples_list:
                    sample_data = features_df[features_df['samples'].str.contains(sample, na=False)]
                    identified = len(sample_data[sample_data['match_name'].notna()]['match_name'].unique())
                    total = len(sample_data)
                    rate = (identified/total*100) if total > 0 else 0
                    
                    total_sample_intensity = 0
                    count = 0
                    for idx, row in sample_data.iterrows():
                        sample_intensity = get_intensity_for_sample(row, sample)
                        if sample_intensity > 0:
                            total_sample_intensity += sample_intensity
                            count += 1
                    
                    avg_intensity = total_sample_intensity / count if count > 0 else sample_data['intensity'].mean()
                    
                    report += f"\n### {sample}\n"
                    report += f"- Total features: {total}\n"
                    report += f"- Molécules uniques identifiées: {identified} ({rate:.1f}%)\n"
                    report += f"- Intensité moyenne (spécifique): {avg_intensity:.2e}\n"
                
                st.text_area(
                    "Rapport de qualité",
                    report,
                    height=400
                )
                
                st.download_button(
                    label="📥 Télécharger le rapport",
                    data=report,
                    file_name=f"rapport_qualite_MS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            st.markdown("---")
            
            st.subheader("Export personnalisé des données")
            
            col1, col2 = st.columns(2)
            
            with col1:
                export_identified_only = st.checkbox("Exporter uniquement les identifiées", value=True, key="export_identified_only")
                export_aggregate = st.checkbox("Agréger par molécule (grouper adduits)", value=True, key="export_aggregate")
                
                if 'confidence_level' in features_df.columns:
                    export_conf_levels = st.multiselect(
                        "Niveaux de confiance à exporter",
                        options=[1, 2, 3, 4, 5],
                        default=[1, 2, 3],
                        key="export_conf_levels"
                    )
                
                samples_list = list(set([s for samples in features_df['samples'].dropna() 
                                       for s in samples.split(',')]))
                export_samples = st.multiselect(
                    "Échantillons à exporter",
                    options=samples_list,
                    default=samples_list,
                    key="export_samples"
                )
            
            with col2:
                all_columns = features_df.columns.tolist()
                default_export_cols = [
                    'feature_id', 'mz', 'retention_time', 'intensity',
                    'match_name', 'confidence_level', 'match_adduct',
                    'categories', 'samples', 'mz_error_ppm', 'ms2_similarity_score',
                    'intensities_by_sample', 'sample_names_order'
                ]
                
                ecotox_columns = ['daphnia_LC50_48_hr_ug/L', 'algae_EC50_72_hr_ug/L', 'pimephales_LC50_96_hr_ug/L']
                for col in ecotox_columns:
                    if col in all_columns:
                        default_export_cols.append(col)
                
                if export_aggregate:
                    default_export_cols.insert(4, 'total_intensity')
                    default_export_cols.insert(5, 'sample_specific_intensity')
                
                export_columns = st.multiselect(
                    "Colonnes à exporter", 
                    options=all_columns + (['total_intensity', 'sample_specific_intensity'] if export_aggregate else []),
                    default=[col for col in default_export_cols if col in all_columns + (['total_intensity', 'sample_specific_intensity'] if export_aggregate else [])],
                    key="export_columns"
                )
                
                include_stats = st.checkbox("Inclure les statistiques par échantillon", value=False, key="include_stats")
                include_summary = st.checkbox("Inclure un résumé en en-tête", value=True, key="include_summary")
            
            # Préparer les données
            export_df = features_df.copy()
            
            if export_identified_only:
                export_df = export_df[export_df['match_name'].notna()]
            
            if 'confidence_level' in export_df.columns and export_conf_levels:
                export_df = export_df[export_df['confidence_level'].isin(export_conf_levels)]
            
            if export_samples:
                export_df = export_df[
                    export_df['samples'].apply(
                        lambda x: any(sample in str(x) for sample in export_samples) if pd.notna(x) else False
                    )
                ]
            
            if export_aggregate and export_identified_only:
                aggregated_data = []
                for sample in export_samples:
                    sample_agg = aggregate_molecules_by_name_enhanced(export_df, sample)
                    if not sample_agg.empty:
                        aggregated_data.append(sample_agg)
                
                if aggregated_data:
                    export_df = pd.concat(aggregated_data, ignore_index=True)
                    if 'match_adduct' in export_df.columns:
                        export_df['match_adduct'] = export_df['match_adduct'].apply(
                            lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                        )
            
            if export_columns:
                available_cols = [col for col in export_columns if col in export_df.columns]
                export_df = export_df[available_cols]
            
            st.subheader("Aperçu des données à exporter")
            info_text = f"Nombre de lignes à exporter : {len(export_df)}"
            if export_aggregate and export_identified_only:
                info_text += " (molécules uniques avec adduits groupés et intensités spécifiques)"
            st.info(info_text)
            st.dataframe(export_df.head(10), use_container_width=True)
            
            # Boutons d'export
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if not export_df.empty:
                    csv_data = export_df.to_csv(index=False)
                    
                    if include_summary:
                        summary = f"""# Résumé de l'export
# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Nombre de lignes: {len(export_df)}
# Agrégé par molécule: {export_aggregate and export_identified_only}
# Intensités spécifiques incluses: {export_aggregate and export_identified_only}
# Filtres appliqués:
# - Identifiées uniquement: {export_identified_only}
# - Niveaux de confiance: {export_conf_levels if 'confidence_level' in features_df.columns else 'Tous'}
# - Échantillons: {', '.join(export_samples)}
#
"""
                        csv_data = summary + csv_data
                    
                    st.download_button(
                        label="📥 Télécharger CSV",
                        data=csv_data,
                        file_name=f"export_MS_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if not export_df.empty:
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        export_df.to_excel(writer, sheet_name='MS_Data', index=False)
                        
                        if include_stats:
                            stats_data = []
                            for sample in export_samples:
                                if export_aggregate:
                                    sample_data = export_df[export_df['samples'].str.contains(sample, na=False)]
                                    unique_molecules = len(sample_data['match_name'].unique()) if 'match_name' in sample_data.columns else 0
                                    avg_intensity = sample_data['sample_specific_intensity'].mean() if 'sample_specific_intensity' in sample_data.columns else 0
                                    total_intensity = sample_data['sample_specific_intensity'].sum() if 'sample_specific_intensity' in sample_data.columns else 0
                                else:
                                    sample_data = features_df[features_df['samples'].str.contains(sample, na=False)]
                                    sample_data = sample_data[sample_data['match_name'].notna()]
                                    unique_molecules = len(sample_data['match_name'].unique())
                                    
                                    total_sample_intensity = 0
                                    count = 0
                                    for idx, row in sample_data.iterrows():
                                        sample_intensity = get_intensity_for_sample(row, sample)
                                        if sample_intensity > 0:
                                            total_sample_intensity += sample_intensity
                                            count += 1
                                    
                                    avg_intensity = total_sample_intensity / count if count > 0 else 0
                                    total_intensity = total_sample_intensity
                                
                                stats_data.append({
                                    'Échantillon': sample,
                                    'Molécules_uniques': unique_molecules,
                                    'Intensité_moyenne_spécifique': avg_intensity,
                                    'Intensité_totale_spécifique': total_intensity
                                })
                            
                            stats_df = pd.DataFrame(stats_data)
                            stats_df.to_excel(writer, sheet_name='Statistiques', index=False)
                    
                    st.download_button(
                        label="📥 Télécharger Excel",
                        data=buffer.getvalue(),
                        file_name=f"export_MS_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col3:
                if not export_df.empty:
                    json_data = export_df.to_json(orient='records', indent=2)
                    
                    st.download_button(
                        label="📥 Télécharger JSON",
                        data=json_data,
                        file_name=f"export_MS_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    elif st.session_state.active_tab == "confidence":
        st.markdown("""
        <div class="section-header">
            <h2>ℹ️ Système de niveaux de confiance</h2>
            <p>Comprendre les critères d'identification des composés</p>
        </div>
        """, unsafe_allow_html=True)
        
        if features_df is not None:
            st.info("ℹ️ Analyses basées sur les molécules uniques. Les statistiques reflètent les meilleures identifications.")
            
            show_confidence_levels_table()
            
            st.markdown("---")
            
            st.subheader("📊 Distribution des niveaux dans vos données")
            
            if 'confidence_level' in features_df.columns:
                level_counts = {}
                for level in range(1, 6):
                    level_molecules = features_df[features_df['confidence_level'] == level]['match_name'].dropna().unique()
                    level_counts[level] = len(level_molecules)
                
                fig = go.Figure(go.Funnel(
                    y=[f"Niveau {i}" for i in level_counts.keys()],
                    x=list(level_counts.values()),
                    textposition="inside",
                    textinfo="value+percent total",
                    marker=dict(color=DISTINCT_COLORS[:len(level_counts)])
                ))
                
                fig.update_layout(
                    title="Distribution des niveaux de confiance dans le dataset (molécules uniques)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("confidence_funnel_unique"))
                
                st.subheader("🔍 Analyse détaillée par niveau")
                
                available_levels = [level for level, count in level_counts.items() if count > 0]
                selected_level = st.selectbox(
                    "Sélectionner un niveau pour analyse",
                    available_levels,
                    key="selected_level_analysis"
                )
                
                level_data = features_df[features_df['confidence_level'] == selected_level]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    unique_molecules = len(level_data['match_name'].dropna().unique())
                    st.metric("Molécules uniques", unique_molecules)
                
                with col2:
                    avg_ms2 = level_data['ms2_similarity_score'].mean()
                    st.metric("Score MS2 moyen", f"{avg_ms2:.3f}")
                
                with col3:
                    avg_mz_error = level_data['mz_error_ppm'].abs().mean()
                    st.metric("Erreur m/z moyenne", f"{avg_mz_error:.2f} ppm")
                
                with col4:
                    avg_intensity = level_data['intensity'].mean()
                    st.metric("Intensité moyenne", f"{avg_intensity:.2e}")
                
                st.subheader(f"Top 10 molécules uniques - Niveau {selected_level}")
                
                if not level_data.empty:
                    aggregated_level = aggregate_molecules_by_name_enhanced(level_data)
                    
                    if not aggregated_level.empty:
                        display_cols = ['match_name', 'total_intensity', 'intensity', 'ms2_similarity_score', 'mz_error_ppm', 'samples']
                        available_cols = [col for col in display_cols if col in aggregated_level.columns]
                        
                        top_molecules = aggregated_level.nlargest(10, 'total_intensity')[available_cols]
                        
                        if 'match_adduct' in aggregated_level.columns:
                            top_molecules_display = top_molecules.copy()
                            top_molecules_display['adduits'] = aggregated_level.nlargest(10, 'total_intensity')['match_adduct'].apply(
                                lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                            )
                            st.dataframe(top_molecules_display.round(3), use_container_width=True)
                        else:
                            st.dataframe(top_molecules.round(3), use_container_width=True)
                
                st.subheader(f"📈 Analyse des critères - Niveau {selected_level}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'mz_error_ppm' in level_data.columns and not level_data.empty:
                        fig_error = px.histogram(
                            level_data,
                            x='mz_error_ppm',
                            nbins=20,
                            title=f"Distribution erreurs m/z - Niveau {selected_level}",
                            labels={'mz_error_ppm': 'Erreur m/z (ppm)'},
                            color_discrete_sequence=DISTINCT_COLORS
                        )
                        fig_error.add_vline(x=5, line_dash="dash", line_color=DISTINCT_COLORS[1])
                        st.plotly_chart(fig_error, use_container_width=True, key=generate_unique_key(f"mz_error_dist_level_{selected_level}"))
                
                with col2:
                    if 'ms2_similarity_score' in level_data.columns and not level_data.empty:
                        ms2_level_data = level_data[level_data['ms2_similarity_score'] > 0]
                        if not ms2_level_data.empty:
                            fig_ms2 = px.histogram(
                                ms2_level_data,
                                x='ms2_similarity_score',
                                nbins=20,
                                title=f"Distribution scores MS2 - Niveau {selected_level}",
                                labels={'ms2_similarity_score': 'Score MS2'},
                                color_discrete_sequence=DISTINCT_COLORS
                            )
                            fig_ms2.add_vline(x=0.7, line_dash="dash", line_color=DISTINCT_COLORS[2])
                            fig_ms2.add_vline(x=0.4, line_dash="dash", line_color=DISTINCT_COLORS[3])
                            st.plotly_chart(fig_ms2, use_container_width=True, key=generate_unique_key(f"ms2_score_dist_level_{selected_level}"))
        else:
            show_confidence_levels_table()

if __name__ == "__main__":
    main()