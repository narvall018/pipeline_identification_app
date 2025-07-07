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
from scipy.stats import normaltest, jarque_bera, shapiro, ttest_ind, mannwhitneyu
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

# CORRECTION PRINCIPALE : Fonction pour initialiser les paramètres de filtres SANS conflits
def init_filter_params():
    """Initialise tous les paramètres de filtres dans session_state s'ils n'existent pas"""
    
    # Paramètres pour l'onglet sample
    if 'sample_intensity_min' not in st.session_state:
        st.session_state.sample_intensity_min = 0.0
    if 'sample_conf_levels' not in st.session_state:
        st.session_state.sample_conf_levels = [1]
    if 'sample_cat_filter' not in st.session_state:
        st.session_state.sample_cat_filter = []
    
    # Paramètres pour l'onglet molecules
    if 'molecules_samples_filter' not in st.session_state:
        st.session_state.molecules_samples_filter = []
    if 'molecules_conf_filter' not in st.session_state:
        st.session_state.molecules_conf_filter = []
    if 'molecules_sort_by' not in st.session_state:
        st.session_state.molecules_sort_by = "Nom"
    
    # Paramètres pour l'onglet detection
    if 'detection_conf_levels' not in st.session_state:
        st.session_state.detection_conf_levels = [1, 2, 3, 4, 5]
    if 'detection_samples_filter' not in st.session_state:
        st.session_state.detection_samples_filter = []
    if 'detection_categories_filter' not in st.session_state:
        st.session_state.detection_categories_filter = []
    if 'detection_radar_conf_levels' not in st.session_state:
        st.session_state.detection_radar_conf_levels = [1, 2, 3, 4, 5]
    
    # Paramètres pour l'onglet comparison - CORRECTION DES BUGS D'INTERACTION
    if 'comparison_preset_conf' not in st.session_state:
        st.session_state.comparison_preset_conf = "Niveaux 1+2+3"
    if 'comparison_manual_conf' not in st.session_state:
        st.session_state.comparison_manual_conf = [1, 2, 3]
    if 'comparison_bubble_levels' not in st.session_state:
        st.session_state.comparison_bubble_levels = [1]
    if 'comparison_jaccard_conf' not in st.session_state:
        st.session_state.comparison_jaccard_conf = [1, 2, 3]
    if 'comparison_jaccard_samples' not in st.session_state:
        st.session_state.comparison_jaccard_samples = []
    if 'comparison_radar_metrics' not in st.session_state:
        st.session_state.comparison_radar_metrics = []
    
    # Paramètres pour l'onglet statistics
    if 'stats_section' not in st.session_state:
        st.session_state.stats_section = "📊 PCA & t-SNE"
    if 'stats_analysis_type' not in st.session_state:
        st.session_state.stats_analysis_type = "Analyse par individus"
    if 'stats_pca_3d' not in st.session_state:
        st.session_state.stats_pca_3d = True
    if 'stats_show_loadings' not in st.session_state:
        st.session_state.stats_show_loadings = False
    if 'stats_tsne_perplexity' not in st.session_state:
        st.session_state.stats_tsne_perplexity = 5
    if 'stats_kmeans_clusters' not in st.session_state:
        st.session_state.stats_kmeans_clusters = 3
    if 'stats_heatmap_transform' not in st.session_state:
        st.session_state.stats_heatmap_transform = "Aucune"
    if 'stats_heatmap_features' not in st.session_state:
        st.session_state.stats_heatmap_features = 50
    if 'stats_distance_metric' not in st.session_state:
        st.session_state.stats_distance_metric = "Euclidienne"
    if 'volcano_group1' not in st.session_state:
        st.session_state.volcano_group1 = []
    if 'volcano_group2' not in st.session_state:
        st.session_state.volcano_group2 = []
    if 'volcano_pvalue_threshold' not in st.session_state:
        st.session_state.volcano_pvalue_threshold = 0.05
    if 'volcano_fc_threshold' not in st.session_state:
        st.session_state.volcano_fc_threshold = 2.0
    
    # Paramètres pour l'onglet reports
    if 'reports_identified_only' not in st.session_state:
        st.session_state.reports_identified_only = True
    if 'reports_aggregate' not in st.session_state:
        st.session_state.reports_aggregate = True
    if 'reports_conf_levels' not in st.session_state:
        st.session_state.reports_conf_levels = [1, 2, 3]
    if 'reports_samples' not in st.session_state:
        st.session_state.reports_samples = []
    if 'reports_columns' not in st.session_state:
        st.session_state.reports_columns = []
    if 'reports_include_stats' not in st.session_state:
        st.session_state.reports_include_stats = False
    if 'reports_include_summary' not in st.session_state:
        st.session_state.reports_include_summary = True
    
    # Paramètres pour l'onglet confidence
    if 'confidence_selected_level' not in st.session_state:
        st.session_state.confidence_selected_level = 1

# Fonctions utilitaires améliorées et optimisées
def generate_unique_key(base_key):
    """Génère une clé unique en utilisant UUID pour éviter les collisions"""
    unique_id = str(uuid.uuid4())[:8]
    timestamp = str(int(time.time() * 1000))[-6:]
    return f"{base_key}_{unique_id}_{timestamp}"

@st.cache_data(ttl=7200, max_entries=5, show_spinner=False)  # Cache optimisé
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

@st.cache_data(ttl=7200, max_entries=5, show_spinner=False)  # Cache optimisé
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

@st.cache_data(ttl=3600, show_spinner=False)  # Cache optimisé
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

@st.cache_data(ttl=3600, show_spinner=False)  # Cache optimisé
def count_unique_molecules(df, sample_filter=None, confidence_levels=None):
    """Compte les molécules uniques (pas les occurrences)"""
    filtered_df = df[df['match_name'].notna()].copy()
    
    if sample_filter:
        filtered_df = filtered_df[filtered_df['samples'].str.contains(sample_filter, na=False)]
    
    if confidence_levels and 'confidence_level' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['confidence_level'].isin(confidence_levels)]
    
    return len(filtered_df['match_name'].unique())

@st.cache_data(ttl=3600, show_spinner=False)  # Cache optimisé
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

@st.cache_data(ttl=3600, show_spinner=False)  # Cache optimisé
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
@st.cache_data(ttl=14400, show_spinner="Chargement des données features...")  # Cache optimisé
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

@st.cache_data(ttl=14400, show_spinner="Chargement de la matrice...")
def load_matrix_data(uploaded_file):
    """Charge le fichier feature_matrix.csv avec gestion des erreurs."""
    try:
        # Lire le fichier
        df = pd.read_csv(uploaded_file)
        
        # Si la première colonne est vide ou s'appelle 'Unnamed: 0', la supprimer
        if df.columns[0] in ['Unnamed: 0', ''] or df.iloc[:, 0].isnull().all():
            df = df.drop(df.columns[0], axis=1)
            # Réinitialiser l'index avec des noms d'échantillons
            df.index = [f"Echantillon_{i+1}" for i in range(len(df))]
        else:
            # Utiliser la première colonne comme index
            df = df.set_index(df.columns[0])
        
        return df
        
    except Exception as e:
        st.error(f"Erreur lors du chargement de la matrice : {str(e)}")
        try:
            # Essayer avec index_col=0 comme fallback
            df = pd.read_csv(uploaded_file, index_col=0)
            return df
        except:
            return None

@st.cache_data(ttl=7200, show_spinner=False)  # Cache optimisé
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

# CORRECTION MAJEURE : Fonction pour gérer les callbacks des filtres
def update_bubble_levels_sample(sample_name):
    """Callback pour mettre à jour les niveaux bubble pour un échantillon spécifique"""
    session_key = f"bubble_levels_{sample_name}"
    widget_key = f"bubble_levels_{sample_name}_widget"
    if widget_key in st.session_state:
        st.session_state[session_key] = st.session_state[widget_key]

def plot_level1_bubble_plot_sample(df, sample_name):
    """Bubble plot pour les molécules d'un échantillon spécifique avec choix de niveaux CORRIGÉ"""
    if 'confidence_level' not in df.columns:
        st.warning("Colonne confidence_level non trouvée")
        return
    
    # Sélecteur de niveaux de confiance avec gestion CORRIGÉE
    col1, col2 = st.columns([1, 3])
    with col1:
        available_levels = sorted([1, 2, 3])  # Seulement 1, 2, 3
        
        # Clé de session pour ce échantillon spécifique
        session_key = f"bubble_levels_{sample_name}"
        widget_key = f"bubble_levels_{sample_name}_widget"
        
        # Initialiser dans session_state si pas encore fait
        if session_key not in st.session_state:
            st.session_state[session_key] = [1]
        
        # CORRECTION : Utiliser on_change pour éviter les conflits
        selected_levels = st.multiselect(
            "Niveaux de confiance",
            options=available_levels,
            default=st.session_state[session_key],
            help="Sélectionnez les niveaux à inclure",
            key=widget_key,
            on_change=update_bubble_levels_sample,
            args=(sample_name,)
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
    
    st.plotly_chart(fig, use_container_width=True, key=f"level_bubble_{sample_name}_corrected")
    
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

# CORRECTION MAJEURE : Fonctions avec callbacks pour éviter les conflits

def update_comparison_preset_conf():
    """Callback pour preset confidence levels"""
    preset_options = {
        "Niveau 1 uniquement": [1],
        "Niveaux 1+2": [1, 2],
        "Niveaux 1+2+3": [1, 2, 3],
        "Niveaux 1+2+3+4": [1, 2, 3, 4],
        "Tous les niveaux": [1, 2, 3, 4, 5]
    }
    
    if 'preset_confidence_levels_fixed' in st.session_state:
        st.session_state.comparison_preset_conf = st.session_state.preset_confidence_levels_fixed
        st.session_state.comparison_manual_conf = preset_options[st.session_state.preset_confidence_levels_fixed]

def update_comparison_manual_conf():
    """Callback pour manual confidence levels"""
    if 'manual_confidence_levels_fixed' in st.session_state:
        st.session_state.comparison_manual_conf = st.session_state.manual_confidence_levels_fixed

def update_comparison_bubble_levels():
    """Callback pour bubble levels"""
    if 'bubble_levels_comparison_fixed' in st.session_state:
        st.session_state.comparison_bubble_levels = st.session_state.bubble_levels_comparison_fixed

def plot_confidence_comparison_across_samples(df, samples_list, selected_levels=None):
    """Visualisation des niveaux de confiance à travers les échantillons avec filtres PERSISTANTS CORRIGÉS"""
    if 'confidence_level' not in df.columns:
        st.warning("Colonne confidence_level non trouvée")
        return
    
    # Filtres pour sélectionner les niveaux avec persistance CORRIGÉE
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
        
        # CORRECTION : Utiliser des callbacks pour éviter les doubles clics
        preset_choice = st.selectbox(
            "Sélection rapide",
            options=list(preset_options.keys()),
            index=list(preset_options.keys()).index(st.session_state.comparison_preset_conf),
            key="preset_confidence_levels_fixed",
            on_change=update_comparison_preset_conf
        )
        
        selected_levels_from_preset = preset_options[preset_choice]
    
    with col2:
        # Sélection manuelle CORRIGÉE
        manual_levels = st.multiselect(
            "Sélection manuelle (remplace la sélection rapide)",
            options=[1, 2, 3, 4, 5],
            default=st.session_state.comparison_manual_conf,
            help="Personnalisez votre sélection de niveaux",
            key="manual_confidence_levels_fixed",
            on_change=update_comparison_manual_conf
        )
        
        # Utiliser la sélection manuelle si elle existe, sinon le preset
        if manual_levels:  
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
    st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("confidence_comparison_fixed"))

def plot_level1_bubble_plot(df, samples_list):
    """Bubble plot pour les molécules avec choix de niveaux de confiance CORRIGÉ et PERSISTANT"""
    if 'confidence_level' not in df.columns:
        st.warning("Colonne confidence_level non trouvée")
        return
    
    # Sélecteur de niveaux de confiance avec persistance CORRIGÉE
    col1, col2 = st.columns([1, 3])
    with col1:
        available_levels = sorted([1, 2, 3])  # Seulement 1, 2, 3
        
        # CORRECTION : Widget unique avec callback pour éviter les doubles clics
        selected_levels = st.multiselect(
            "Niveaux de confiance",
            options=available_levels,
            default=st.session_state.comparison_bubble_levels,
            help="Sélectionnez les niveaux à inclure",
            key="bubble_levels_comparison_fixed",
            on_change=update_comparison_bubble_levels
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
    
    st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("level_bubble_plot_fixed"))

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

# CORRECTION : Callbacks pour les autres filtres

def update_comparison_jaccard_conf():
    """Callback pour jaccard confidence levels"""
    if 'conf_levels_jaccard_widget' in st.session_state:
        st.session_state.comparison_jaccard_conf = st.session_state.conf_levels_jaccard_widget

def update_comparison_jaccard_samples():
    """Callback pour jaccard samples"""
    if 'selected_samples_jaccard_widget' in st.session_state:
        st.session_state.comparison_jaccard_samples = st.session_state.selected_samples_jaccard_widget

def plot_hierarchical_clustering(df, samples_list, confidence_levels=None, selected_samples=None):
    """Clustering hiérarchique des échantillons avec filtrage par niveau de confiance PERSISTANT"""
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

# CORRECTION DU PROBLÈME D'ACP - Fonction corrigée pour la direction de l'ACP
@st.cache_data(ttl=3600, show_spinner=False)
def safe_pca_analysis(matrix_df, n_components=3, analysis_type="Analyse par individus"):
    """PCA sécurisée qui gère les erreurs de dimensions et les petits datasets avec direction correcte"""
    if matrix_df is None or matrix_df.empty:
        return None, None, None
    
    # CORRECTION PRINCIPALE : Direction de l'ACP selon le type d'analyse
    if analysis_type == "Analyse par marqueurs":
        # Pour l'analyse par marqueurs, on transpose la matrice
        # Les marqueurs deviennent les "individus" à analyser
        data_for_pca = matrix_df.T  # Transposer : marqueurs en lignes, échantillons en colonnes
        n_samples, n_features = data_for_pca.shape
        entity_type = "marqueurs"
    else:
        # Pour l'analyse par individus (défaut)
        data_for_pca = matrix_df  # Normal : échantillons en lignes, features en colonnes
        n_samples, n_features = data_for_pca.shape
        entity_type = "échantillons"
    
    # Vérifications de base
    if n_samples < 2:
        st.error(f"❌ PCA impossible : seulement {n_samples} {entity_type}. Minimum 2 requis.")
        return None, None, None
    
    if n_features < 2:
        st.error(f"❌ PCA impossible : seulement {n_features} feature(s). Minimum 2 requis.")
        return None, None, None
    
    max_components = min(n_samples, n_features)
    
    # Avertissement pour 2 entités
    if n_samples == 2:
        st.warning(f"⚠️ Avec seulement 2 {entity_type}, la PCA sera limitée à 1 composante principale.")
        max_components = 1
    
    # Ajuster le nombre de composantes
    n_components = min(n_components, max_components)
    
    try:
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data_for_pca.values)
        
        # PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        return pca, X_pca, X_scaled
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la PCA : {str(e)}")
        return None, None, None

# CORRECTION : Callbacks pour les autres widgets

def update_stats_analysis_type():
    """Callback pour analysis type"""
    if 'analysis_type_widget' in st.session_state:
        st.session_state.stats_analysis_type = st.session_state.analysis_type_widget

def update_stats_pca_3d():
    """Callback pour PCA 3D"""
    if 'pca_3d_checkbox_widget' in st.session_state:
        st.session_state.stats_pca_3d = st.session_state.pca_3d_checkbox_widget

def update_stats_show_loadings():
    """Callback pour show loadings"""
    if 'show_loadings_checkbox_widget' in st.session_state:
        st.session_state.stats_show_loadings = st.session_state.show_loadings_checkbox_widget

def update_stats_tsne_perplexity():
    """Callback pour tsne perplexity"""
    if 'tsne_perplexity_widget' in st.session_state:
        st.session_state.stats_tsne_perplexity = st.session_state.tsne_perplexity_widget

def update_stats_kmeans_clusters():
    """Callback pour kmeans clusters"""
    if 'kmeans_clusters_widget' in st.session_state:
        st.session_state.stats_kmeans_clusters = st.session_state.kmeans_clusters_widget

def update_stats_section():
    """Callback pour stat section"""
    if 'stat_navigation_widget' in st.session_state:
        st.session_state.stats_section = st.session_state.stat_navigation_widget

def update_stats_heatmap_transform():
    """Callback pour heatmap transform"""
    if 'heatmap_transform_widget' in st.session_state:
        st.session_state.stats_heatmap_transform = st.session_state.heatmap_transform_widget

def update_stats_heatmap_features():
    """Callback pour heatmap features"""
    if 'heatmap_features_widget' in st.session_state:
        st.session_state.stats_heatmap_features = st.session_state.heatmap_features_widget

def update_stats_distance_metric():
    """Callback pour distance metric"""
    if 'distance_metric_widget' in st.session_state:
        st.session_state.stats_distance_metric = st.session_state.distance_metric_widget

def update_volcano_group1():
    """Callback pour volcano group1"""
    if 'volcano_group1_widget' in st.session_state:
        st.session_state.volcano_group1 = st.session_state.volcano_group1_widget

def update_volcano_group2():
    """Callback pour volcano group2"""
    if 'volcano_group2_widget' in st.session_state:
        st.session_state.volcano_group2 = st.session_state.volcano_group2_widget

def update_volcano_pvalue():
    """Callback pour volcano pvalue"""
    if 'volcano_pvalue_widget' in st.session_state:
        st.session_state.volcano_pvalue_threshold = st.session_state.volcano_pvalue_widget

def update_volcano_fc():
    """Callback pour volcano fc"""
    if 'volcano_fc_widget' in st.session_state:
        st.session_state.volcano_fc_threshold = st.session_state.volcano_fc_widget

def plot_3d_pca(matrix_df, analysis_type="Analyse par individus"):
    """PCA en 3D avec gestion d'erreur et type d'analyse CORRIGÉ"""
    if matrix_df is None or matrix_df.empty:
        st.warning("Aucune matrice chargée")
        return
    
    # CORRECTION : Déterminer les dimensions selon le type d'analyse
    if analysis_type == "Analyse par marqueurs":
        n_entities = matrix_df.shape[1]  # Nombre de marqueurs (colonnes)
        entity_type = "marqueurs"
    else:
        n_entities = matrix_df.shape[0]  # Nombre d'échantillons (lignes)
        entity_type = "échantillons"
    
    # Vérifications préalables
    if n_entities < 2:
        st.error(f"❌ PCA impossible avec moins de 2 {entity_type}")
        return
    
    if n_entities == 2:
        st.warning(f"⚠️ Avec 2 {entity_type}, seule la PCA 2D est possible")
    
    pca, X_pca, X_scaled = safe_pca_analysis(matrix_df, n_components=min(3, n_entities), analysis_type=analysis_type)
    
    if pca is None:
        return
    
    # Variance expliquée
    variance_ratio = pca.explained_variance_ratio_
    
    # CORRECTION : Adapter les labels selon le type d'analyse
    if analysis_type == "Analyse par marqueurs":
        labels_for_plot = matrix_df.columns  # Noms des marqueurs
        title_suffix = " - Focus sur les marqueurs"
    else:
        labels_for_plot = matrix_df.index  # Noms des échantillons
        title_suffix = ""
    
    # Décider du type de graphique selon le nombre de composantes
    if X_pca.shape[1] >= 3 and n_entities >= 3:
        # PCA 3D normale
        fig = px.scatter_3d(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            z=X_pca[:, 2],
            text=labels_for_plot,
            title=f"PCA 3D{title_suffix} (PC1: {variance_ratio[0]*100:.1f}%, "
                  f"PC2: {variance_ratio[1]*100:.1f}%, "
                  f"PC3: {variance_ratio[2]*100:.1f}%)",
            labels={
                'x': f'PC1 ({variance_ratio[0]*100:.1f}%)',
                'y': f'PC2 ({variance_ratio[1]*100:.1f}%)',
                'z': f'PC3 ({variance_ratio[2]*100:.1f}%)'
            },
            color_discrete_sequence=DISTINCT_COLORS
        )
        fig.update_traces(marker=dict(size=12))
        fig.update_layout(height=700)
        
    elif X_pca.shape[1] >= 2:
        # PCA 2D
        fig = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            text=labels_for_plot,
            title=f"PCA 2D{title_suffix} (PC1: {variance_ratio[0]*100:.1f}%, "
                  f"PC2: {variance_ratio[1]*100:.1f}%)",
            labels={
                'x': f'PC1 ({variance_ratio[0]*100:.1f}%)',
                'y': f'PC2 ({variance_ratio[1]*100:.1f}%)'
            },
            color_discrete_sequence=DISTINCT_COLORS
        )
        fig.update_traces(textposition="top center", marker=dict(size=12))
        fig.update_layout(height=600)
        
    else:
        # 1D seulement
        fig = px.scatter(
            x=X_pca[:, 0],
            y=[0] * len(X_pca),
            text=labels_for_plot,
            title=f"PCA 1D{title_suffix} (PC1: {variance_ratio[0]*100:.1f}%)",
            labels={
                'x': f'PC1 ({variance_ratio[0]*100:.1f}%)',
                'y': 'Position'
            },
            color_discrete_sequence=DISTINCT_COLORS
        )
        fig.update_traces(textposition="top center", marker=dict(size=12))
        fig.update_layout(height=400, yaxis=dict(showticklabels=False))
    
    st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("pca_plot"))

def plot_tsne_analysis(matrix_df, analysis_type="Analyse par individus"):
    """Analyse t-SNE avec gestion d'erreur et PERSISTANCE CORRIGÉE"""
    if matrix_df is None or matrix_df.empty:
        st.warning("Aucune matrice chargée")
        return
    
    # CORRECTION : Déterminer les dimensions selon le type d'analyse
    if analysis_type == "Analyse par marqueurs":
        n_entities = matrix_df.shape[1]  # Nombre de marqueurs
        entity_type = "marqueurs"
        data_for_tsne = matrix_df.T  # Transposer pour analyser les marqueurs
        labels_for_plot = matrix_df.columns
    else:
        n_entities = matrix_df.shape[0]  # Nombre d'échantillons
        entity_type = "échantillons"
        data_for_tsne = matrix_df
        labels_for_plot = matrix_df.index
    
    # Vérifications strictes pour t-SNE
    if n_entities < 3:
        st.error(f"❌ t-SNE impossible : seulement {n_entities} {entity_type}. **Minimum 3 requis** pour t-SNE.")
        st.info(f"💡 Suggestion : Utilisez la PCA pour analyser vos données avec moins de 3 {entity_type}.")
        return
    
    # Ajuster la perplexité selon le nombre d'entités
    max_perplexity = min(30, (n_entities - 1) // 3)
    
    if max_perplexity < 1:
        max_perplexity = 1
    
    # Paramètres t-SNE avec persistance CORRIGÉE
    perplexity = st.slider(
        "Perplexité t-SNE", 
        1, 
        max_perplexity, 
        min(max_perplexity, st.session_state.stats_tsne_perplexity),
        help=f"Maximum possible: {max_perplexity} (basé sur {n_entities} {entity_type})",
        key="tsne_perplexity_widget",
        on_change=update_stats_tsne_perplexity
    )
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_for_tsne.values)
    
    try:
        # t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        X_tsne = tsne.fit_transform(X_scaled)
        
        title_suffix = ""
        if analysis_type == "Analyse par marqueurs":
            title_suffix = " - Focus sur les marqueurs"
        
        fig = px.scatter(
            x=X_tsne[:, 0],
            y=X_tsne[:, 1],
            text=labels_for_plot,
            title=f"Analyse t-SNE{title_suffix} (perplexité={perplexity})",
            labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
            color_discrete_sequence=DISTINCT_COLORS
        )
        
        fig.update_traces(textposition="top center", marker=dict(size=12))
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("tsne_plot"))
        
    except Exception as e:
        st.error(f"❌ Erreur t-SNE : {str(e)}")
        st.info("💡 Essayez de réduire la perplexité ou d'utiliser la PCA à la place.")

def perform_pca_analysis(matrix_df, analysis_type="Analyse par individus"):
    """Effectue une analyse PCA sécurisée sur la matrice des features CORRIGÉE"""
    if matrix_df is None or matrix_df.empty:
        st.error("Aucune matrice de features chargée")
        return
    
    # CORRECTION : Adapter selon le type d'analyse
    if analysis_type == "Analyse par marqueurs":
        max_components = min(10, matrix_df.shape[1], matrix_df.shape[0])  # Marqueurs
    else:
        max_components = min(10, matrix_df.shape[1], matrix_df.shape[0])  # Échantillons
    
    pca, X_pca, X_scaled = safe_pca_analysis(matrix_df, n_components=max_components, analysis_type=analysis_type)
    
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
        
        title_suffix = ""
        if analysis_type == "Analyse par marqueurs":
            title_suffix = " - Focus sur les marqueurs"
        
        fig1.update_layout(
            title=f"Variance expliquée par les composantes principales{title_suffix}",
            xaxis_title="Composante principale",
            yaxis_title="Variance expliquée (%)"
        )
        st.plotly_chart(fig1, use_container_width=True, key=generate_unique_key("pca_variance_plot"))
    
    with col2:
        # Score plot PCA CORRIGÉ - AVEC GESTION DU CAS 1D
        if analysis_type == "Analyse par marqueurs":
            labels_for_plot = matrix_df.columns  # Noms des marqueurs
        else:
            labels_for_plot = matrix_df.index  # Noms des échantillons
        
        if X_pca.shape[1] >= 2:
            # PCA 2D
            fig2 = px.scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                text=labels_for_plot,
                title=f"Score plot PCA{title_suffix} (PC1: {variance_ratio[0]*100:.1f}%, PC2: {variance_ratio[1]*100:.1f}%)",
                labels={'x': f'PC1 ({variance_ratio[0]*100:.1f}%)', 'y': f'PC2 ({variance_ratio[1]*100:.1f}%)'},
                color_discrete_sequence=DISTINCT_COLORS
            )
            fig2.update_traces(textposition="top center", marker=dict(size=12))
            fig2.update_layout(height=500)
            st.plotly_chart(fig2, use_container_width=True, key=generate_unique_key("pca_score_plot"))
        
        elif X_pca.shape[1] == 1:
            # PCA 1D - AJOUT DE CE CAS MANQUANT
            fig2 = px.scatter(
                x=X_pca[:, 0],
                y=[0] * len(X_pca),
                text=labels_for_plot,
                title=f"Score plot PCA 1D{title_suffix} (PC1: {variance_ratio[0]*100:.1f}%)",
                labels={'x': f'PC1 ({variance_ratio[0]*100:.1f}%)', 'y': 'Position'},
                color_discrete_sequence=DISTINCT_COLORS
            )
            fig2.update_traces(textposition="top center", marker=dict(size=12))
            fig2.update_layout(height=400, yaxis=dict(showticklabels=False))
            st.plotly_chart(fig2, use_container_width=True, key=generate_unique_key("pca_score_plot_1d"))
        
        else:
            st.warning("❌ Aucune composante PCA disponible pour l'affichage")
    
    # AJOUT CORRECTION : Afficher les loadings pour l'analyse par individus
    if analysis_type == "Analyse par individus":
        # Afficher les top contributeurs (descripteurs/features) pour l'analyse par individus
        st.subheader("🏆 Top contributeurs des descripteurs/features")
        
        # Calculer les loadings (contribution des features aux composantes principales)
        n_components_loadings = min(3, pca.n_components_)
        loadings_df = pd.DataFrame(
            pca.components_[:n_components_loadings].T,
            columns=[f'PC{i+1}' for i in range(n_components_loadings)],
            index=matrix_df.columns  # Noms des features/descripteurs
        )
        
        if n_components_loadings >= 2:
            # Calculer la distance à l'origine pour PC1 et PC2
            loadings_df['Distance'] = np.sqrt(loadings_df['PC1']**2 + loadings_df['PC2']**2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Top 15 contributeurs à PC1:**")
                top_pc1 = loadings_df.reindex(loadings_df['PC1'].abs().sort_values(ascending=False).index).head(15)
                st.dataframe(
                    top_pc1[['PC1', 'Distance']].round(4),
                    use_container_width=True
                )
            
            with col2:
                st.write(f"**Top 15 contributeurs à PC2:**")
                top_pc2 = loadings_df.reindex(loadings_df['PC2'].abs().sort_values(ascending=False).index).head(15)
                st.dataframe(
                    top_pc2[['PC2', 'Distance']].round(4),
                    use_container_width=True
                )
            
            # Top contributeurs globaux (distance maximale à l'origine)
            st.subheader("🎯 Top descripteurs les plus discriminants (distance maximale)")
            
            top_discriminants = loadings_df.nlargest(20, 'Distance')
            st.dataframe(
                top_discriminants[['PC1', 'PC2', 'Distance']].round(4),
                use_container_width=True
            )
        else:
            # Pour 1 composante seulement
            st.subheader("🏆 Top contributeurs à PC1")
            top_pc1 = loadings_df.reindex(loadings_df['PC1'].abs().sort_values(ascending=False).index).head(20)
            st.dataframe(
                top_pc1[['PC1']].round(4),
                use_container_width=True
            )
    
    # Affichage des métriques
    st.subheader("Métriques PCA")
    cols = st.columns(min(3, len(variance_ratio)))
    
    for i, col in enumerate(cols):
        if i < len(variance_ratio):
            with col:
                st.metric(f"PC{i+1} variance expliquée", f"{variance_ratio[i]*100:.1f}%")

def plot_correlation_heatmap(matrix_df, analysis_type="Analyse par individus"):
    """Heatmap de corrélation entre échantillons ou marqueurs CORRIGÉE"""
    if matrix_df is None or matrix_df.empty:
        st.warning("Aucune matrice chargée")
        return
    
    # CORRECTION : Adapter selon le type d'analyse
    if analysis_type == "Analyse par marqueurs":
        # Pour les marqueurs, analyser les corrélations entre marqueurs (features)
        if matrix_df.shape[1] < 2:
            st.error("❌ Corrélation entre marqueurs impossible : moins de 2 features")
            return None
        corr_matrix = matrix_df.corr()
        title = "Matrice de corrélation entre marqueurs (features)"
        n_entities = matrix_df.shape[1]
    else:
        # Calculer la matrice de corrélation entre échantillons
        n_entities = matrix_df.shape[0]
        if n_entities < 2:
            st.error("❌ Analyse de corrélation impossible avec moins de 2 échantillons")
            return None
        if n_entities == 2:
            st.info("📊 Avec 2 échantillons, la corrélation sera limitée")
        corr_matrix = matrix_df.T.corr()
        title = "Matrice de corrélation entre échantillons"
    
    try:
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            title=title,
            color_continuous_scale='RdBu_r',
            range_color=[-1, 1]
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("correlation_heatmap"))
        
        return corr_matrix
        
    except Exception as e:
        st.error(f"❌ Erreur lors du calcul de corrélation : {str(e)}")
        return None

def perform_kmeans_clustering(matrix_df, analysis_type="Analyse par individus"):
    """Clustering K-means sur la matrice des features avec validation du nombre de clusters et PERSISTANCE CORRIGÉE"""
    if matrix_df is None or matrix_df.empty:
        st.warning("Aucune matrice chargée")
        return
    
    # CORRECTION : Adapter selon le type d'analyse
    if analysis_type == "Analyse par marqueurs":
        n_entities = matrix_df.shape[1]  # Nombre de marqueurs
        entity_type = "marqueurs"
        data_for_clustering = matrix_df.T  # Transposer pour analyser les marqueurs
        labels_for_plot = matrix_df.columns
    else:
        n_entities = matrix_df.shape[0]  # Nombre d'échantillons
        entity_type = "échantillons"
        data_for_clustering = matrix_df
        labels_for_plot = matrix_df.index
    
    # Vérifications strictes
    if n_entities < 2:
        st.error(f"❌ Impossible de faire du clustering : seulement {n_entities} {entity_type}. **Minimum 2 requis**.")
        return
    
    max_clusters = min(10, n_entities)
    
    # Cas spécial pour 2 entités
    if n_entities == 2:
        n_clusters = 2
        st.info(f"📊 Avec seulement {n_entities} {entity_type}, le nombre de clusters est fixé à {n_clusters}.")
    else:
        # Sélection du nombre de clusters avec validation et persistance
        n_clusters = st.slider(
            "Nombre de clusters", 
            2, 
            max_clusters, 
            min(st.session_state.stats_kmeans_clusters, max_clusters), 
            help=f"Maximum possible: {max_clusters} (basé sur {n_entities} {entity_type})",
            key="kmeans_clusters_widget",
            on_change=update_stats_kmeans_clusters
        )
    
    # Vérification de sécurité
    if n_clusters > n_entities:
        st.error(f"❌ Le nombre de clusters ({n_clusters}) ne peut pas être supérieur au nombre de {entity_type} ({n_entities})")
        return
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_for_clustering.values)
    
    try:
        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # PCA pour visualisation (avec gestion des petits datasets)
        pca_vis, X_pca_vis, _ = safe_pca_analysis(matrix_df, n_components=2, analysis_type=analysis_type)
        
        if pca_vis is None:
            st.error("❌ Impossible de visualiser le clustering sans PCA")
            return
        
        title_suffix = ""
        if analysis_type == "Analyse par marqueurs":
            title_suffix = " - Focus sur les marqueurs"
        
        # Graphique adapté selon le nombre de composantes PCA disponibles
        if X_pca_vis.shape[1] >= 2:
            # Visualisation 2D normale
            fig = px.scatter(
                x=X_pca_vis[:, 0],
                y=X_pca_vis[:, 1],
                color=cluster_labels,
                text=labels_for_plot,
                title=f"Clustering K-means{title_suffix} (k={n_clusters})",
                labels={'x': f'PC1 ({pca_vis.explained_variance_ratio_[0]*100:.1f}%)', 
                        'y': f'PC2 ({pca_vis.explained_variance_ratio_[1]*100:.1f}%)'},
                color_discrete_sequence=DISTINCT_COLORS
            )
            fig.update_traces(textposition="top center", marker=dict(size=12))
            fig.update_layout(height=600)
        else:
            # Visualisation 1D
            fig = px.scatter(
                x=X_pca_vis[:, 0],
                y=[0] * len(X_pca_vis),
                color=cluster_labels,
                text=labels_for_plot,
                title=f"Clustering K-means{title_suffix} (k={n_clusters}) - Vue 1D",
                labels={'x': f'PC1 ({pca_vis.explained_variance_ratio_[0]*100:.1f}%)', 'y': 'Position'},
                color_discrete_sequence=DISTINCT_COLORS
            )
            fig.update_traces(textposition="top center", marker=dict(size=12))
            fig.update_layout(height=400, yaxis=dict(showticklabels=False))
        
        st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("kmeans_plot"))
        
        # Afficher les clusters
        st.subheader(f"📋 Composition des clusters")
        cluster_df = pd.DataFrame({
            f'{entity_type.capitalize()}': labels_for_plot,
            'Cluster': cluster_labels
        })
        
        for i in range(n_clusters):
            cluster_entities = cluster_df[cluster_df['Cluster'] == i][f'{entity_type.capitalize()}'].tolist()
            st.write(f"**Cluster {i}:** {', '.join(cluster_entities)}")
    
    except Exception as e:
        st.error(f"❌ Erreur lors du clustering K-means : {str(e)}")

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

# NOUVELLES FONCTIONS POUR LE VOLCANO PLOT ET LES ANALYSES DE MARQUEURS

@st.cache_data(ttl=3600, show_spinner=False)
def prepare_volcano_data(matrix_df, group1_samples, group2_samples):
    """Prépare les données pour le volcano plot"""
    if not group1_samples or not group2_samples:
        return None
    
    # Filtrer les échantillons des deux groupes
    group1_data = matrix_df.loc[group1_samples]
    group2_data = matrix_df.loc[group2_samples]
    
    results = []
    
    for feature in matrix_df.columns:
        # Valeurs pour chaque groupe
        values1 = group1_data[feature].values
        values2 = group2_data[feature].values
        
        # Calculer les moyennes
        mean1 = np.mean(values1)
        mean2 = np.mean(values2)
        
        # Éviter les divisions par zéro
        if mean2 == 0:
            fold_change = np.inf if mean1 > 0 else 1
        else:
            fold_change = mean1 / mean2
        
        # Test statistique (t-test ou Mann-Whitney selon la distribution)
        try:
            if len(values1) >= 3 and len(values2) >= 3:
                # Test de normalité
                _, p_norm1 = shapiro(values1) if len(values1) <= 5000 else normaltest(values1)
                _, p_norm2 = shapiro(values2) if len(values2) <= 5000 else normaltest(values2)
                
                if p_norm1 > 0.05 and p_norm2 > 0.05:
                    # Données normales : t-test
                    stat, p_value = ttest_ind(values1, values2)
                else:
                    # Données non-normales : Mann-Whitney U
                    stat, p_value = mannwhitneyu(values1, values2, alternative='two-sided')
            else:
                # Pas assez d'échantillons pour un test statistique
                p_value = 1.0
                
        except:
            p_value = 1.0
        
        results.append({
            'Feature': feature,
            'Group1_mean': mean1,
            'Group2_mean': mean2,
            'Fold_change': fold_change,
            'Log2_fold_change': np.log2(fold_change) if fold_change > 0 and fold_change != np.inf else 0,
            'P_value': p_value,
            'Neg_log10_p': -np.log10(p_value) if p_value > 0 else 0
        })
    
    return pd.DataFrame(results)

def plot_volcano_plot(matrix_df):
    """Crée un volcano plot pour comparer deux groupes d'individus avec CALLBACKS CORRIGÉS"""
    st.subheader("🌋 Volcano Plot - Comparaison entre groupes")
    
    if matrix_df is None or matrix_df.empty:
        st.warning("Aucune matrice chargée")
        return
    
    n_samples = len(matrix_df)
    
    # Vérifications
    if n_samples < 2:
        st.error("❌ Volcano Plot impossible avec moins de 2 échantillons")
        return None
    
    if n_samples == 2:
        st.info("📊 Avec seulement 2 échantillons, l'un ira dans le groupe 1, l'autre dans le groupe 2")
    
    samples_list = list(matrix_df.index)
    
    # Interface pour sélectionner les groupes AVEC CALLBACKS
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Groupe 1")
        max_group1 = max(1, n_samples // 2) if n_samples > 2 else 1
        group1_samples = st.multiselect(
            "Sélectionner les échantillons du groupe 1",
            options=samples_list,
            default=st.session_state.volcano_group1 if st.session_state.volcano_group1 else samples_list[:max_group1],
            key="volcano_group1_widget",
            on_change=update_volcano_group1
        )
        
        if group1_samples:
            st.info(f"Groupe 1: {len(group1_samples)} échantillons")
    
    with col2:
        st.subheader("Groupe 2")
        # Exclure les échantillons déjà sélectionnés dans le groupe 1
        available_group2 = [s for s in samples_list if s not in group1_samples]
        group2_samples = st.multiselect(
            "Sélectionner les échantillons du groupe 2",
            options=available_group2,
            default=[s for s in st.session_state.volcano_group2 if s in available_group2] if st.session_state.volcano_group2 else available_group2,
            key="volcano_group2_widget",
            on_change=update_volcano_group2
        )
        
        if group2_samples:
            st.info(f"Groupe 2: {len(group2_samples)} échantillons")
    
    # Vérifications des groupes
    if not group1_samples or not group2_samples:
        st.warning("⚠️ Veuillez sélectionner au moins un échantillon pour chaque groupe")
        return None
    
    if len(group1_samples) == 1 and len(group2_samples) == 1:
        st.warning("⚠️ Avec 1 échantillon par groupe, les tests statistiques seront limités")
    
    # Paramètres de seuillage AVEC CALLBACKS
    col1, col2 = st.columns(2)
    
    with col1:
        p_threshold = st.number_input(
            "Seuil p-value",
            min_value=0.001,
            max_value=0.1,
            value=st.session_state.volcano_pvalue_threshold,
            step=0.005,
            format="%.3f",
            key="volcano_pvalue_widget",
            on_change=update_volcano_pvalue
        )
    
    with col2:
        fc_threshold = st.number_input(
            "Seuil fold-change",
            min_value=1.1,
            max_value=10.0,
            value=st.session_state.volcano_fc_threshold,
            step=0.1,
            format="%.1f",
            key="volcano_fc_widget",
            on_change=update_volcano_fc
        )
    
    # Calculer les données du volcano plot
    with st.spinner("Calcul des statistiques..."):
        volcano_data = prepare_volcano_data(matrix_df, group1_samples, group2_samples)
    
    if volcano_data is None or volcano_data.empty:
        st.error("❌ Erreur dans le calcul des données volcano")
        return None
    
    # Classifier les points
    volcano_data['Significance'] = 'Non significatif'
    volcano_data.loc[
        (volcano_data['P_value'] < p_threshold) & 
        (volcano_data['Fold_change'] > fc_threshold), 
        'Significance'
    ] = f'↑ Groupe 1 (FC>{fc_threshold}, p<{p_threshold})'
    
    volcano_data.loc[
        (volcano_data['P_value'] < p_threshold) & 
        (volcano_data['Fold_change'] < 1/fc_threshold), 
        'Significance'
    ] = f'↓ Groupe 2 (FC<{1/fc_threshold:.1f}, p<{p_threshold})'
    
    # Créer le volcano plot
    fig = px.scatter(
        volcano_data,
        x='Log2_fold_change',
        y='Neg_log10_p',
        color='Significance',
        hover_data=['Feature', 'P_value', 'Fold_change', 'Group1_mean', 'Group2_mean'],
        title=f"Volcano Plot: {', '.join(group1_samples)} vs {', '.join(group2_samples)}",
        labels={
            'Log2_fold_change': 'Log2(Fold Change)',
            'Neg_log10_p': '-Log10(p-value)'
        },
        color_discrete_map={
            'Non significatif': '#CCCCCC',
            f'↑ Groupe 1 (FC>{fc_threshold}, p<{p_threshold})': '#FF6B6B',
            f'↓ Groupe 2 (FC<{1/fc_threshold:.1f}, p<{p_threshold})': '#4ECDC4'
        }
    )
    
    # Ajouter les lignes de seuil
    fig.add_hline(y=-np.log10(p_threshold), line_dash="dash", line_color="black",
                  annotation_text=f"p = {p_threshold}")
    fig.add_vline(x=np.log2(fc_threshold), line_dash="dash", line_color="black",
                  annotation_text=f"FC = {fc_threshold}")
    fig.add_vline(x=-np.log2(fc_threshold), line_dash="dash", line_color="black",
                  annotation_text=f"FC = {1/fc_threshold:.1f}")
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("volcano_plot"))
    
    # Statistiques du volcano plot
    n_up = len(volcano_data[volcano_data['Significance'].str.contains('↑')])
    n_down = len(volcano_data[volcano_data['Significance'].str.contains('↓')])
    n_total = len(volcano_data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total features", n_total)
    
    with col2:
        st.metric("↑ Groupe 1", n_up)
    
    with col3:
        st.metric("↓ Groupe 2", n_down)
    
    with col4:
        st.metric("% significatifs", f"{(n_up + n_down)/n_total*100:.1f}%")
    
    # Tableau des marqueurs significatifs
    st.subheader("📋 Marqueurs significatifs")
    
    significant_markers = volcano_data[
        volcano_data['Significance'] != 'Non significatif'
    ].sort_values('P_value')
    
    if not significant_markers.empty:
        display_cols = ['Feature', 'Log2_fold_change', 'P_value', 'Fold_change', 
                       'Group1_mean', 'Group2_mean', 'Significance']
        st.dataframe(
            significant_markers[display_cols].round(4),
            use_container_width=True
        )
        
        # Export des marqueurs significatifs
        csv_markers = significant_markers.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les marqueurs significatifs",
            data=csv_markers,
            file_name=f"marqueurs_significatifs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("ℹ️ Aucun marqueur significatif trouvé avec les seuils actuels")
    
    return significant_markers if not significant_markers.empty else None

# CORRECTION DU BUG KeyError PC2 - Fonction corrigée
def plot_complete_pca_with_loadings(matrix_df, analysis_type="Analyse par individus"):
    """PCA complète avec projection de TOUS les marqueurs et analyse détaillée CORRIGÉE"""
    st.subheader("🎯 PCA complète avec projection de tous les marqueurs")
    
    if matrix_df is None or matrix_df.empty:
        st.warning("Aucune matrice chargée")
        return
    
    # CORRECTION : Déterminer les dimensions selon le type d'analyse
    if analysis_type == "Analyse par marqueurs":
        n_entities = matrix_df.shape[1]  # Nombre de marqueurs
        entity_type = "marqueurs"
    else:
        n_entities = matrix_df.shape[0]  # Nombre d'échantillons
        entity_type = "échantillons"
    
    # Vérifications préalables
    if n_entities < 2:
        st.error(f"❌ PCA impossible avec moins de 2 {entity_type}")
        return
    
    if n_entities == 2:
        st.warning(f"⚠️ Avec seulement 2 {entity_type}, l'analyse PCA sera limitée à 1 composante principale.")
    
    # Effectuer la PCA sur la matrice complète AVEC LA DIRECTION CORRECTE
    pca, X_pca, X_scaled = safe_pca_analysis(matrix_df, n_components=min(10, matrix_df.shape[1], n_entities), analysis_type=analysis_type)
    
    if pca is None:
        return
    
    # Variance expliquée
    variance_ratio = pca.explained_variance_ratio_
    
    # CORRECTION PRINCIPALE : Calculer les loadings selon le type d'analyse
    n_components = min(10, pca.n_components_)
    
    if analysis_type == "Analyse par marqueurs":
        # Pour l'analyse par marqueurs, les loadings sont les composantes principales 
        # appliquées sur les échantillons (car on a transposé)
        loadings_df = pd.DataFrame(
            pca.components_[:n_components].T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=matrix_df.index  # Noms des échantillons
        )
        labels_for_samples = matrix_df.columns  # Marqueurs pour le score plot
        labels_for_loadings = matrix_df.index   # Échantillons pour les loadings
    else:
        # Pour l'analyse par individus (échantillons)
        loadings_df = pd.DataFrame(
            pca.components_[:n_components].T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=matrix_df.columns  # Noms des features/marqueurs
        )
        labels_for_samples = matrix_df.index    # Échantillons pour le score plot
        labels_for_loadings = matrix_df.columns # Features pour les loadings
    
    # Interface pour sélectionner les composantes à visualiser (seulement si plus d'1 composante)
    if n_components > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            pc_x = st.selectbox(
                "Composante X",
                options=[f'PC{i+1}' for i in range(n_components)],
                index=0,
                key="pc_x_selector"
            )
        
        with col2:
            pc_y = st.selectbox(
                "Composante Y",
                options=[f'PC{i+1}' for i in range(n_components)],
                index=1 if n_components > 1 else 0,
                key="pc_y_selector"
            )
        
        pc_x_idx = int(pc_x.replace('PC', '')) - 1
        pc_y_idx = int(pc_y.replace('PC', '')) - 1
        
        # Affichage côte à côte : échantillons et marqueurs
        col1, col2 = st.columns(2)
        
        with col1:
            # Score plot PCA
            if analysis_type == "Analyse par marqueurs":
                title_samples = f"PCA - Projection des marqueurs ({pc_x}: {variance_ratio[pc_x_idx]*100:.1f}%, {pc_y}: {variance_ratio[pc_y_idx]*100:.1f}%)"
            else:
                title_samples = f"PCA - Projection des échantillons ({pc_x}: {variance_ratio[pc_x_idx]*100:.1f}%, {pc_y}: {variance_ratio[pc_y_idx]*100:.1f}%)"
            
            fig_samples = px.scatter(
                x=X_pca[:, pc_x_idx],
                y=X_pca[:, pc_y_idx],
                text=labels_for_samples,
                title=title_samples,
                labels={'x': f'{pc_x} ({variance_ratio[pc_x_idx]*100:.1f}%)', 'y': f'{pc_y} ({variance_ratio[pc_y_idx]*100:.1f}%)'},
                color_discrete_sequence=DISTINCT_COLORS
            )
            fig_samples.update_traces(textposition="top center", marker=dict(size=12))
            fig_samples.update_layout(height=500)
            st.plotly_chart(fig_samples, use_container_width=True, key=generate_unique_key("pca_samples_complete"))
        
        with col2:
            # CORRECTION : Vérifier que PC2 existe avant de l'utiliser
            if pc_y in loadings_df.columns:
                # Projection des loadings
                if analysis_type == "Analyse par marqueurs":
                    title_loadings = f"PCA - Contribution des échantillons aux marqueurs (Loadings)"
                else:
                    title_loadings = f"PCA - Projection de TOUS les marqueurs (Loadings)"
                
                fig_loadings = px.scatter(
                    loadings_df,
                    x=pc_x,
                    y=pc_y,
                    title=title_loadings,
                    labels={'x': f'{pc_x} ({variance_ratio[pc_x_idx]*100:.1f}%)', 'y': f'{pc_y} ({variance_ratio[pc_y_idx]*100:.1f}%)'},
                    hover_name=labels_for_loadings,
                    opacity=0.6
                )
                
                # CORRECTION : Calculer la distance seulement si PC2 existe
                loadings_df['Distance'] = np.sqrt(loadings_df[pc_x]**2 + loadings_df[pc_y]**2)
                
                # Ajouter des cercles de contribution
                theta = np.linspace(0, 2*np.pi, 100)
                for radius in [0.3, 0.6, 0.9]:
                    x_circle = radius * np.cos(theta)
                    y_circle = radius * np.sin(theta)
                    fig_loadings.add_trace(go.Scatter(
                        x=x_circle, y=y_circle,
                        mode='lines',
                        line=dict(dash='dash', color='gray', width=1),
                        showlegend=False,
                        hoverinfo='skip',
                        name=f'Contribution {radius:.1f}'
                    ))
                
                # Mettre en évidence les top contributeurs
                top_contributors = loadings_df.nlargest(20, 'Distance')
                
                fig_loadings.add_trace(go.Scatter(
                    x=top_contributors[pc_x],
                    y=top_contributors[pc_y],
                    mode='markers+text',
                    marker=dict(size=8, color='red'),
                    text=top_contributors.index,
                    textposition="top center",
                    name="Top 20 contributeurs",
                    showlegend=True
                ))
                
                fig_loadings.update_layout(
                    height=500,
                    xaxis=dict(range=[-1.1, 1.1]),
                    yaxis=dict(range=[-1.1, 1.1])
                )
                
                st.plotly_chart(fig_loadings, use_container_width=True, key=generate_unique_key("pca_loadings_complete"))
            else:
                st.warning(f"⚠️ {pc_y} n'est pas disponible. Affichage en 1D uniquement.")
    
    else:
        # Cas spécial : 1 seule composante
        st.info(f"📊 Avec 2 {entity_type}, seule 1 composante principale est disponible.")
        
        if analysis_type == "Analyse par marqueurs":
            title_1d = f"PCA 1D - Projection des marqueurs (PC1: {variance_ratio[0]*100:.1f}%)"
        else:
            title_1d = f"PCA 1D - Projection des échantillons (PC1: {variance_ratio[0]*100:.1f}%)"
        
        fig_samples = px.scatter(
            x=X_pca[:, 0],
            y=[0] * len(X_pca),
            text=labels_for_samples,
            title=title_1d,
            labels={'x': f'PC1 ({variance_ratio[0]*100:.1f}%)', 'y': 'Position'},
            color_discrete_sequence=DISTINCT_COLORS
        )
        fig_samples.update_traces(textposition="top center", marker=dict(size=12))
        fig_samples.update_layout(height=400, yaxis=dict(showticklabels=False))
        st.plotly_chart(fig_samples, use_container_width=True, key=generate_unique_key("pca_samples_1d"))
        
        # Loadings 1D
        if analysis_type == "Analyse par marqueurs":
            title_loadings_1d = "Contribution des échantillons (PC1)"
        else:
            title_loadings_1d = "Contribution des marqueurs (PC1)"
        
        fig_loadings = px.scatter(
            x=loadings_df['PC1'],
            y=[0] * len(loadings_df),
            hover_name=labels_for_loadings,
            title=title_loadings_1d,
            labels={'x': f'PC1 ({variance_ratio[0]*100:.1f}%)', 'y': 'Position'}
        )
        fig_loadings.update_layout(height=400, yaxis=dict(showticklabels=False))
        st.plotly_chart(fig_loadings, use_container_width=True, key=generate_unique_key("pca_loadings_1d"))
    
    # Variance expliquée par toutes les composantes
    st.subheader("📊 Variance expliquée par les composantes principales")
    
    fig_variance = go.Figure()
    fig_variance.add_trace(go.Bar(
        x=[f'PC{i+1}' for i in range(len(variance_ratio))],
        y=variance_ratio * 100,
        name='Variance individuelle',
        marker_color=DISTINCT_COLORS[0]
    ))
    
    cumsum_variance = np.cumsum(variance_ratio)
    fig_variance.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(len(cumsum_variance))],
        y=cumsum_variance * 100,
        mode='lines+markers',
        name='Variance cumulée',
        line=dict(color=DISTINCT_COLORS[1], width=3),
        yaxis='y2'
    ))
    
    fig_variance.update_layout(
        title="Variance expliquée par toutes les composantes principales",
        xaxis_title="Composante principale",
        yaxis_title="Variance expliquée (%)",
        yaxis2=dict(
            title="Variance cumulée (%)",
            overlaying='y',
            side='right'
        ),
        height=400
    )
    
    st.plotly_chart(fig_variance, use_container_width=True, key=generate_unique_key("pca_variance_complete"))
    
    # Tableau des top contributeurs (seulement si plus d'1 composante)
    if n_components > 1 and 'Distance' in loadings_df.columns:
        st.subheader(f"🏆 Top contributeurs pour {pc_x} et {pc_y}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Top 15 contributeurs à {pc_x}:**")
            top_pc_x = loadings_df.reindex(loadings_df[pc_x].abs().sort_values(ascending=False).index).head(15)
            st.dataframe(
                top_pc_x[[pc_x, 'Distance']].round(4),
                use_container_width=True
            )
        
        with col2:
            st.write(f"**Top 15 contributeurs à {pc_y}:**")
            top_pc_y = loadings_df.reindex(loadings_df[pc_y].abs().sort_values(ascending=False).index).head(15)
            st.dataframe(
                top_pc_y[[pc_y, 'Distance']].round(4),
                use_container_width=True
            )
        
        # Top contributeurs globaux (distance maximale à l'origine)
        st.subheader("🎯 Top marqueurs les plus discriminants (distance maximale)")
        
        top_discriminants = loadings_df.nlargest(20, 'Distance')
        st.dataframe(
            top_discriminants[[pc_x, pc_y, 'Distance']].round(4),
            use_container_width=True
        )
    else:
        st.subheader("🏆 Top contributeurs à PC1")
        top_pc1 = loadings_df.reindex(loadings_df['PC1'].abs().sort_values(ascending=False).index).head(20)
        st.dataframe(
            top_pc1[['PC1']].round(4),
            use_container_width=True
        )
    
    # Export des loadings
    if st.button("📥 Exporter les loadings complets"):
        csv_loadings = loadings_df.to_csv()
        st.download_button(
            label="📥 Télécharger les loadings PCA",
            data=csv_loadings,
            file_name=f"pca_loadings_complets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# CORRECTION DU BUG KeyError PC2 - Fonction corrigée
def plot_markers_pca_with_projections(matrix_df, significant_markers_df=None):
    """PCA focalisée sur les marqueurs significatifs avec projection de tous les marqueurs - VERSION CORRIGÉE"""
    st.subheader("🎯 PCA des marqueurs significatifs avec projection complète")
    
    if matrix_df is None or matrix_df.empty:
        st.warning("Aucune matrice chargée")
        return
    
    if significant_markers_df is None or significant_markers_df.empty:
        st.info("Aucun marqueur significatif sélectionné. Analyse de tous les features avec variance élevée.")
        # Utiliser les features avec la plus grande variance
        feature_var = matrix_df.var(axis=0).nlargest(100)
        selected_features = feature_var.index.tolist()
    else:
        selected_features = significant_markers_df['Feature'].tolist()
        selected_features = [f for f in selected_features if f in matrix_df.columns]
    
    if not selected_features:
        st.warning("Aucun marqueur valide trouvé")
        return
    
    # Effectuer la PCA sur la matrice complète (tous les features) - INDIVIDUS
    pca, X_pca, X_scaled = safe_pca_analysis(matrix_df, n_components=min(10, matrix_df.shape[1], matrix_df.shape[0]), analysis_type="Analyse par individus")
    
    if pca is None:
        return
    
    # Variance expliquée
    variance_ratio = pca.explained_variance_ratio_
    
    # CORRECTION : Vérifier le nombre de composantes disponibles
    n_components = min(5, pca.n_components_)
    
    # Calculer les loadings pour TOUS les features
    loadings_df = pd.DataFrame(
        pca.components_[:n_components].T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=matrix_df.columns
    )
    
    # Marquer les features significatifs
    loadings_df['Significatif'] = loadings_df.index.isin(selected_features)
    
    # CORRECTION : Calculer la distance seulement si PC2 existe
    if n_components >= 2:
        loadings_df['Distance'] = np.sqrt(loadings_df['PC1']**2 + loadings_df['PC2']**2)
    else:
        loadings_df['Distance'] = np.abs(loadings_df['PC1'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Score plot PCA avec échantillons
        if X_pca.shape[1] >= 2:
            fig_samples = px.scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                text=matrix_df.index,
                title=f"PCA - Projection des échantillons (PC1: {variance_ratio[0]*100:.1f}%, PC2: {variance_ratio[1]*100:.1f}%)",
                labels={'x': f'PC1 ({variance_ratio[0]*100:.1f}%)', 'y': f'PC2 ({variance_ratio[1]*100:.1f}%)'},
                color_discrete_sequence=DISTINCT_COLORS
            )
            fig_samples.update_traces(textposition="top center", marker=dict(size=12))
            fig_samples.update_layout(height=500)
            st.plotly_chart(fig_samples, use_container_width=True, key=generate_unique_key("pca_samples_markers"))
    
    with col2:
        # Projection des marqueurs avec distinction significatifs/non-significatifs
        if pca.components_.shape[0] >= 2 and n_components >= 2:
            # Créer le graphique des loadings avec TOUS les marqueurs
            fig_loadings = go.Figure()
            
            # Marqueurs non significatifs (gris clair)
            non_significant = loadings_df[~loadings_df['Significatif']]
            if not non_significant.empty:
                fig_loadings.add_trace(go.Scatter(
                    x=non_significant['PC1'],
                    y=non_significant['PC2'],
                    mode='markers',
                    marker=dict(size=4, color='lightgray', opacity=0.5),
                    name='Autres marqueurs',
                    hovertemplate='Feature: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>',
                    text=non_significant.index
                ))
            
            # Marqueurs significatifs (colorés)
            significant = loadings_df[loadings_df['Significatif']]
            if not significant.empty:
                fig_loadings.add_trace(go.Scatter(
                    x=significant['PC1'],
                    y=significant['PC2'],
                    mode='markers',
                    marker=dict(size=8, color='#FF6B6B', opacity=0.8),
                    name='Marqueurs significatifs',
                    hovertemplate='Feature: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>',
                    text=significant.index
                ))
            
            # Ajouter des cercles de contribution
            theta = np.linspace(0, 2*np.pi, 100)
            for radius, label in zip([0.5, 0.8], ['Contribution modérée', 'Forte contribution']):
                x_circle = radius * np.cos(theta)
                y_circle = radius * np.sin(theta)
                fig_loadings.add_trace(go.Scatter(
                    x=x_circle, y=y_circle,
                    mode='lines',
                    line=dict(dash='dash', color='gray', width=1),
                    showlegend=False,
                    hoverinfo='skip',
                    name=label
                ))
            
            fig_loadings.update_layout(
                title=f"Projection de TOUS les marqueurs (Loadings)<br><sub>{len(significant)} significatifs sur {len(loadings_df)} total</sub>",
                xaxis_title=f'PC1 ({variance_ratio[0]*100:.1f}%)',
                yaxis_title=f'PC2 ({variance_ratio[1]*100:.1f}%)',
                xaxis=dict(range=[-1.1, 1.1]),
                yaxis=dict(range=[-1.1, 1.1]),
                height=500
            )
            
            st.plotly_chart(fig_loadings, use_container_width=True, key=generate_unique_key("pca_loadings_markers"))
        else:
            # Affichage 1D si seulement PC1 disponible
            st.info("📊 Seule PC1 disponible - Affichage en 1D")
            
            fig_loadings_1d = px.scatter(
                x=loadings_df['PC1'],
                y=[0] * len(loadings_df),
                color=loadings_df['Significatif'],
                hover_name=loadings_df.index,
                title="Projection 1D des marqueurs (PC1 uniquement)",
                labels={'x': f'PC1 ({variance_ratio[0]*100:.1f}%)', 'y': 'Position'},
                color_discrete_map={True: '#FF6B6B', False: 'lightgray'}
            )
            fig_loadings_1d.update_layout(height=400, yaxis=dict(showticklabels=False))
            st.plotly_chart(fig_loadings_1d, use_container_width=True, key=generate_unique_key("pca_loadings_markers_1d"))
    
    # Tableau des top contributeurs parmi les marqueurs significatifs
    st.subheader("🏆 Top contributeurs parmi les marqueurs significatifs")
    
    significant = loadings_df[loadings_df['Significatif']]
    if not significant.empty:
        top_significant = significant.nlargest(15, 'Distance')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 15 marqueurs significatifs les plus discriminants:**")
            if n_components >= 2:
                display_cols = ['PC1', 'PC2', 'Distance']
            else:
                display_cols = ['PC1', 'Distance']
            st.dataframe(
                top_significant[display_cols].round(4),
                use_container_width=True
            )
        
        with col2:
            # Graphique en barres des contributions
            fig_contrib = px.bar(
                x=top_significant['Distance'].values,
                y=top_significant.index,
                orientation='h',
                title="Contribution des top marqueurs significatifs",
                labels={'x': 'Distance (contribution)', 'y': 'Marqueur'},
                color=top_significant['Distance'],
                color_continuous_scale='Viridis'
            )
            fig_contrib.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_contrib, use_container_width=True, key=generate_unique_key("contrib_significant_markers"))
    
    # Comparaison: significatifs vs non-significatifs
    st.subheader("📊 Comparaison contributions: significatifs vs autres")
    
    non_significant = loadings_df[~loadings_df['Significatif']]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_contrib_sig = significant['Distance'].mean() if not significant.empty else 0
        st.metric("Contribution moyenne - Significatifs", f"{avg_contrib_sig:.3f}")
    
    with col2:
        avg_contrib_other = non_significant['Distance'].mean() if not non_significant.empty else 0
        st.metric("Contribution moyenne - Autres", f"{avg_contrib_other:.3f}")
    
    with col3:
        ratio = avg_contrib_sig / avg_contrib_other if avg_contrib_other > 0 else 0
        st.metric("Ratio (Sig/Autres)", f"{ratio:.2f}x")
    
    # Distribution des contributions
    if not significant.empty and not non_significant.empty:
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(
            x=significant['Distance'],
            name='Marqueurs significatifs',
            opacity=0.7,
            marker_color='#FF6B6B',
            nbinsx=20
        ))
        
        fig_dist.add_trace(go.Histogram(
            x=non_significant['Distance'],
            name='Autres marqueurs',
            opacity=0.7,
            marker_color='lightgray',
            nbinsx=20
        ))
        
        fig_dist.update_layout(
            title="Distribution des contributions (distance à l'origine)",
            xaxis_title="Distance de contribution",
            yaxis_title="Nombre de marqueurs",
            barmode='overlay',
            height=400
        )
        
        st.plotly_chart(fig_dist, use_container_width=True, key=generate_unique_key("distribution_contributions"))

def plot_markers_heatmap(matrix_df, significant_markers_df=None):
    """Heatmap des marqueurs significatifs"""
    st.subheader("🔥 Heatmap des marqueurs significatifs")
    
    if matrix_df is None or matrix_df.empty:
        st.warning("Aucune matrice chargée")
        return
    
    if significant_markers_df is None or significant_markers_df.empty:
        st.info("Aucun marqueur significatif sélectionné. Utilisation des top features par variance.")
        # Utiliser les features avec la plus grande variance
        feature_var = matrix_df.var(axis=0).nlargest(50)
        selected_features = feature_var.index.tolist()
    else:
        selected_features = significant_markers_df['Feature'].tolist()
        selected_features = [f for f in selected_features if f in matrix_df.columns]
    
    if not selected_features:
        st.warning("Aucun marqueur valide trouvé")
        return
    
    # Limiter le nombre de features pour la lisibilité
    max_features = st.slider("Nombre maximum de marqueurs à afficher", 10, 100, 30)
    selected_features = selected_features[:max_features]
    
    # Filtrer la matrice
    markers_matrix = matrix_df[selected_features]
    
    # Normalisation
    scaler = StandardScaler()
    markers_scaled = pd.DataFrame(
        scaler.fit_transform(markers_matrix),
        index=markers_matrix.index,
        columns=markers_matrix.columns
    )
    
    # Créer la heatmap
    fig = px.imshow(
        markers_scaled.T,  # Transposer pour avoir les marqueurs en lignes
        labels=dict(x="Échantillons", y="Marqueurs", color="Intensité normalisée"),
        title=f"Heatmap des {len(selected_features)} marqueurs les plus significatifs",
        aspect="auto",
        color_continuous_scale='RdBu_r'
    )
    
    fig.update_layout(height=max(400, len(selected_features) * 15))
    st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("markers_heatmap"))

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

# CORRECTION DES CALLBACKS MANQUANTS POUR LES FILTRES D'ÉCHANTILLONS

def update_sample_intensity_min():
    """Callback pour sample intensity min"""
    if 'intensity_min_sample_widget' in st.session_state:
        st.session_state.sample_intensity_min = st.session_state.intensity_min_sample_widget

def update_sample_conf_levels():
    """Callback pour sample conf levels"""
    if 'conf_levels_sample_widget' in st.session_state:
        st.session_state.sample_conf_levels = st.session_state.conf_levels_sample_widget

def update_sample_cat_filter():
    """Callback pour sample cat filter"""
    if 'cat_filter_sample_widget' in st.session_state:
        st.session_state.sample_cat_filter = st.session_state.cat_filter_sample_widget

def update_molecules_samples_filter():
    """Callback pour molecules samples filter"""
    if 'samples_filter_molecules_widget' in st.session_state:
        st.session_state.molecules_samples_filter = st.session_state.samples_filter_molecules_widget

def update_molecules_conf_filter():
    """Callback pour molecules conf filter"""
    if 'conf_filter_molecules_widget' in st.session_state:
        st.session_state.molecules_conf_filter = st.session_state.conf_filter_molecules_widget

def update_molecules_sort_by():
    """Callback pour molecules sort by"""
    if 'sort_by_molecules_widget' in st.session_state:
        st.session_state.molecules_sort_by = st.session_state.sort_by_molecules_widget

def update_detection_conf_levels():
    """Callback pour detection conf levels"""
    if 'conf_levels_detection_widget' in st.session_state:
        st.session_state.detection_conf_levels = st.session_state.conf_levels_detection_widget

def update_detection_samples_filter():
    """Callback pour detection samples filter"""
    if 'samples_filter_detection_widget' in st.session_state:
        st.session_state.detection_samples_filter = st.session_state.samples_filter_detection_widget

def update_detection_categories_filter():
    """Callback pour detection categories filter"""
    if 'categories_filter_detection_widget' in st.session_state:
        st.session_state.detection_categories_filter = st.session_state.categories_filter_detection_widget

def update_detection_radar_conf_levels():
    """Callback pour detection radar conf levels"""
    if 'conf_levels_radar_widget' in st.session_state:
        st.session_state.detection_radar_conf_levels = st.session_state.conf_levels_radar_widget

def update_comparison_radar_metrics():
    """Callback pour comparison radar metrics"""
    if 'metrics_radar_comparison_widget' in st.session_state:
        st.session_state.comparison_radar_metrics = st.session_state.metrics_radar_comparison_widget

def update_confidence_selected_level():
    """Callback pour confidence selected level"""
    if 'selected_level_analysis_widget' in st.session_state:
        st.session_state.confidence_selected_level = st.session_state.selected_level_analysis_widget

# Interface principale optimisée avec navigation moderne
def main():
    st.title("🧪 Analyse et visualisation des données de HRMS")
    
    # Initialiser les états de session ET les paramètres de filtres
    if 'features_df' not in st.session_state:
        st.session_state.features_df = None
    if 'matrix_df' not in st.session_state:
        st.session_state.matrix_df = None
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "home"
    
    # Initialiser tous les paramètres de filtres
    init_filter_params()
    
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
            
            # Mettre à jour les listes d'échantillons dans les filtres persistants
            samples_list = list(set([s for samples in features_df['samples'].dropna() 
                                   for s in samples.split(',')]))
            
            # Initialiser les filtres avec les vrais échantillons si ce n'est pas encore fait
            if not st.session_state.molecules_samples_filter:
                st.session_state.molecules_samples_filter = samples_list
            if not st.session_state.detection_samples_filter:
                st.session_state.detection_samples_filter = samples_list
            if not st.session_state.comparison_jaccard_samples:
                st.session_state.comparison_jaccard_samples = samples_list
            
            # Mettre à jour les filtres de confiance avec les vrais niveaux disponibles
            if 'confidence_level' in features_df.columns:
                available_levels = sorted(features_df['confidence_level'].dropna().unique())
                if not st.session_state.molecules_conf_filter:
                    st.session_state.molecules_conf_filter = available_levels
            
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
                            color_discrete_sequence=DISTINCT_COLORS)
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
                        
                        # Filtres avec persistance CORRIGÉE pour l'échantillon
                        with st.expander("Filtres avancés"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                intensity_column = 'sample_specific_intensity' if 'sample_specific_intensity' in aggregated_display.columns else 'total_intensity'
                                intensity_min = st.number_input(
                                    f"Intensité minimale ({intensity_column})",
                                    min_value=0.0,
                                    value=st.session_state.sample_intensity_min,
                                    key="intensity_min_sample_widget",
                                    on_change=update_sample_intensity_min
                                )
                            
                            with col2:
                                if 'confidence_level' in aggregated_display.columns:
                                    conf_levels = st.multiselect(
                                        "Niveaux de confiance",
                                        options=[1, 2, 3, 4, 5],
                                        default=st.session_state.sample_conf_levels,
                                        key="conf_levels_sample_widget",
                                        on_change=update_sample_conf_levels
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
                                        default=st.session_state.sample_cat_filter,
                                        key="cat_filter_sample_widget",
                                        on_change=update_sample_cat_filter
                                    )
                        
                        # Appliquer les filtres
                        filtered_data = aggregated_display[aggregated_display[intensity_column] >= st.session_state.sample_intensity_min]
                        
                        if 'confidence_level' in aggregated_display.columns and st.session_state.sample_conf_levels:
                            filtered_data = filtered_data[filtered_data['confidence_level'].isin(st.session_state.sample_conf_levels)]
                        
                        if st.session_state.sample_cat_filter:
                            filtered_data = filtered_data[
                                filtered_data['categories'].apply(
                                    lambda x: any(cat in x for cat in st.session_state.sample_cat_filter) if isinstance(x, list) else False
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
                # Filtres avec persistance CORRIGÉE
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    samples_list = list(set([s for samples in features_df['samples'].dropna() 
                                           for s in samples.split(',')]))
                    selected_samples_filter = st.multiselect(
                        "Filtrer par échantillons",
                        options=samples_list,
                        default=st.session_state.molecules_samples_filter,
                        key="samples_filter_molecules_widget",
                        on_change=update_molecules_samples_filter
                    )
                
                with col2:
                    if 'confidence_level' in identified_molecules.columns:
                        conf_filter = st.multiselect(
                            "Niveaux de confiance",
                            options=sorted(identified_molecules['confidence_level'].dropna().unique()),
                            default=st.session_state.molecules_conf_filter,
                            key="conf_filter_molecules_widget",
                            on_change=update_molecules_conf_filter
                        )
                
                with col3:
                    sort_by = st.selectbox(
                        "Trier par",
                        ["Nom", "Confiance", "Intensité"],
                        index=["Nom", "Confiance", "Intensité"].index(st.session_state.molecules_sort_by),
                        key="sort_by_molecules_widget",
                        on_change=update_molecules_sort_by
                    )
                
                # Appliquer les filtres
                filtered_molecules = identified_molecules.copy()
                
                if st.session_state.molecules_samples_filter:
                    filtered_molecules = filtered_molecules[
                        filtered_molecules['samples'].apply(
                            lambda x: any(sample in str(x) for sample in st.session_state.molecules_samples_filter) if pd.notna(x) else False
                        )
                    ]
                
                if st.session_state.molecules_conf_filter and 'confidence_level' in filtered_molecules.columns:
                    filtered_molecules = filtered_molecules[
                        filtered_molecules['confidence_level'].isin(st.session_state.molecules_conf_filter)
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
                    default=st.session_state.detection_conf_levels,
                    key="conf_levels_detection_widget",
                    on_change=update_detection_conf_levels
                )
            
            with col2:
                st.info(f"Niveaux sélectionnés : {', '.join(map(str, st.session_state.detection_conf_levels))}")
            
            samples_list = list(set([s for samples in features_df['samples'].dropna() 
                                   for s in samples.split(',')]))
            
            if samples_list and st.session_state.detection_conf_levels:
                detection_factors = calculate_detection_factor(features_df, samples_list, st.session_state.detection_conf_levels)
                
                st.subheader("📡 Graphique radar des facteurs de détection")
                plot_detection_factor_radar(detection_factors)
                
                st.subheader("📊 Tableau détaillé des facteurs de détection")
                
                detection_df_data = []
                for sample, factors in detection_factors.items():
                    for category, factor in factors.items():
                        sample_data = features_df[
                            (features_df['samples'].str.contains(sample, na=False)) & 
                            (features_df['match_name'].notna()) &
                            (features_df['confidence_level'].isin(st.session_state.detection_conf_levels) if 'confidence_level' in features_df.columns else True)
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
                
                # Filtres avec persistance CORRIGÉE
                col1, col2 = st.columns(2)
                with col1:
                    selected_samples = st.multiselect(
                        "Filtrer par échantillons",
                        samples_list,
                        default=st.session_state.detection_samples_filter,
                        key="samples_filter_detection_widget",
                        on_change=update_detection_samples_filter
                    )
                
                with col2:
                    selected_categories = st.multiselect(
                        "Filtrer par catégories",
                        list(DATABASE_CATEGORIES.keys()),
                        default=st.session_state.detection_categories_filter,
                        key="categories_filter_detection_widget",
                        on_change=update_detection_categories_filter
                    )
                
                # Utiliser les valeurs de session_state pour le filtrage
                filtered_detection_df = detection_df[
                    (detection_df['Échantillon'].isin(st.session_state.detection_samples_filter if st.session_state.detection_samples_filter else samples_list)) &
                    (detection_df['Catégorie'].isin(st.session_state.detection_categories_filter if st.session_state.detection_categories_filter else list(DATABASE_CATEGORIES.keys())))
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
                default=st.session_state.detection_radar_conf_levels,
                key="conf_levels_radar_widget",
                on_change=update_detection_radar_conf_levels
            )
            
            plot_category_distribution_radar(features_df, samples_list, st.session_state.detection_radar_conf_levels)
    
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
            
            samples_list = list(set([s for samples in features_df['samples'].dropna() for s in samples.split(',')]))
            
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
                
                # NOUVEAU : Ajout du Volcano Plot dans la section comparaison
                if matrix_df is not None:
                    st.markdown("---")
                    significant_markers = plot_volcano_plot(matrix_df)
                else:
                    significant_markers = None
                
                st.subheader("🎯 Comparaison multi-critères (radar)")
                
                available_metrics = [col for col in stats_df.columns if col not in ['Échantillon'] and stats_df[col].dtype in ['int64', 'float64']]
                selected_metrics = st.multiselect(
                    "Métriques pour le radar",
                    available_metrics,
                    default=st.session_state.comparison_radar_metrics if st.session_state.comparison_radar_metrics else ['Taux_identification_%', 'Niveau_1', 'Niveau_2'][:min(3, len(available_metrics))],
                    key="metrics_radar_comparison_widget",
                    on_change=update_comparison_radar_metrics
                )
                
                if len(st.session_state.comparison_radar_metrics) >= 3:
                    fig_radar = go.Figure()
                    
                    for idx, row in stats_df.iterrows():
                        values = [row[metric] for metric in st.session_state.comparison_radar_metrics]
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=st.session_state.comparison_radar_metrics,
                            fill='toself',
                            name=row['Échantillon'],
                            line=dict(width=2, color=DISTINCT_COLORS[idx % len(DISTINCT_COLORS)]),
                            opacity=0.7
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, max([stats_df[metric].max() for metric in st.session_state.comparison_radar_metrics])],
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
                            default=st.session_state.comparison_jaccard_conf,
                            key="conf_levels_jaccard_widget",
                            on_change=update_comparison_jaccard_conf
                        )
                    else:
                        confidence_levels_jaccard = None

                with col2:
                    selected_samples_jaccard = st.multiselect(
                        "Sélectionner des échantillons spécifiques",
                        options=samples_list,
                        default=st.session_state.comparison_jaccard_samples,
                        key="selected_samples_jaccard_widget",
                        on_change=update_comparison_jaccard_samples
                    )

                plot_hierarchical_clustering(
                    features_df, 
                    samples_list, 
                    confidence_levels=st.session_state.comparison_jaccard_conf, 
                    selected_samples=st.session_state.comparison_jaccard_samples if st.session_state.comparison_jaccard_samples else None
                )
                
            else:
                st.warning("Au moins 2 échantillons sont nécessaires pour la comparaison")
    
    elif st.session_state.active_tab == "statistics":
        # VÉRIFICATION CRITIQUE AU DÉBUT
        if matrix_df is None:
            st.warning("""
            ⚠️ Veuillez charger le fichier **feature_matrix.csv** pour accéder aux analyses statistiques avancées.
            
            Ce fichier doit contenir une matrice avec :
            - Lignes : échantillons
            - Colonnes : features (format : F0001_mz102.9880)
            - Valeurs : intensités
            """)
            st.markdown("""
            <div class="section-header">
                <h2>📈 Analyses statistiques avancées</h2>
                <p>Section non disponible - Chargez votre matrice feature_matrix.csv</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        n_samples = len(matrix_df)
        
        if n_samples < 1:
            st.error("❌ Aucun échantillon détecté dans la matrice")
            return
        
        if n_samples == 1:
            st.markdown("""
            <div class="section-header">
                <h2>📈 Analyses statistiques avancées</h2>
                <p>Section non disponible avec un seul échantillon</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.error("""
            ❌ **Les analyses statistiques ne sont pas disponibles avec un seul échantillon**
            
            Vous avez seulement **1 échantillon** dans votre matrice. Les analyses statistiques nécessitent au moins 2 échantillons pour :
            - Calculer des corrélations
            - Effectuer une PCA
            - Faire du clustering
            - Comparer des groupes
            
            💡 **Solutions :**
            - Ajoutez plus d'échantillons à votre matrice
            - Utilisez les autres sections de l'application (Vue d'ensemble, Analyse par échantillon, etc.)
            """)
            return
        
        # Section disponible pour 2+ échantillons
        st.markdown("""
        <div class="section-header">
            <h2>📈 Analyses statistiques avancées</h2>
            <p>Explorez vos données avec des méthodes statistiques avancées</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Avertissement pour 2 échantillons
        if n_samples == 2:
            st.warning("""
            ⚠️ **Analyses limitées avec 2 échantillons**
            
            Certaines analyses seront restreintes :
            - PCA limitée à 1 composante principale
            - t-SNE non disponible (minimum 3 échantillons requis)
            - Clustering limité à 2 groupes maximum
            - Comparaisons statistiques limitées
            """)
        
        # CORRECTION : Sélecteur du type d'analyse CORRIGÉ avec callback
        analysis_type = st.selectbox(
            "Type d'analyse",
            ["Analyse par individus", "Analyse par marqueurs"],
            index=["Analyse par individus", "Analyse par marqueurs"].index(st.session_state.stats_analysis_type),
            help="**Analyse par individus**: Analyse les échantillons (lignes de la matrice)\n**Analyse par marqueurs**: Analyse les features/marqueurs (colonnes de la matrice)",
            key="analysis_type_widget",
            on_change=update_stats_analysis_type
        )
        
        if analysis_type == "Analyse par marqueurs":
            st.info("🎯 Mode marqueurs : Analyses focalisées sur les features/variables discriminants")
            
            # Vérification spéciale pour l'analyse par marqueurs
            if matrix_df.shape[1] < 2:
                st.error("❌ Analyse par marqueurs impossible : moins de 2 features dans la matrice")
                return
            
            # CORRECTION : PCA complète avec projection de TOUS les marqueurs
            plot_complete_pca_with_loadings(matrix_df, analysis_type)
            
            st.markdown("---")
            
            # PCA des marqueurs significatifs avec projections (si disponibles depuis volcano plot)
            if 'significant_markers' in locals() and significant_markers is not None:
                st.subheader("🎯 PCA des marqueurs significatifs avec projection complète")
                plot_markers_pca_with_projections(matrix_df, significant_markers)
            else:
                st.subheader("🎯 PCA des marqueurs avec projection complète")
                plot_markers_pca_with_projections(matrix_df)
            
            st.markdown("---")
            
            # Heatmap des marqueurs
            st.subheader("🔥 Heatmap des marqueurs")
            if 'significant_markers' in locals() and significant_markers is not None:
                plot_markers_heatmap(matrix_df, significant_markers)
            else:
                plot_markers_heatmap(matrix_df)
        
        else:  # Analyse par individus
            # NOUVEAU : Menu déroulant étendu pour l'analyse par individus avec callback
            stat_section = st.selectbox(
                "Choisir une analyse:",
                ["📊 PCA & t-SNE", "🌋 Volcano Plot", "🔍 Clustering", "📈 Corrélations", "🎨 Heatmaps"],
                index=["📊 PCA & t-SNE", "🌋 Volcano Plot", "🔍 Clustering", "📈 Corrélations", "🎨 Heatmaps"].index(st.session_state.stats_section) if st.session_state.stats_section in ["📊 PCA & t-SNE", "🌋 Volcano Plot", "🔍 Clustering", "📈 Corrélations", "🎨 Heatmaps"] else 0,
                key="stat_navigation_widget",
                on_change=update_stats_section
            )
            
            if stat_section == "📊 PCA & t-SNE":
                st.subheader("📊 Analyse en Composantes Principales (PCA)")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    pca_3d = st.checkbox(
                        "Afficher PCA 3D", 
                        value=st.session_state.stats_pca_3d, 
                        key="pca_3d_checkbox_widget",
                        disabled=n_samples < 3,
                        help="PCA 3D nécessite au moins 3 échantillons" if n_samples < 3 else "Afficher la PCA en 3 dimensions",
                        on_change=update_stats_pca_3d
                    )
                with col2:
                    show_loadings = st.checkbox(
                        "Afficher les loadings", 
                        value=st.session_state.stats_show_loadings, 
                        key="show_loadings_checkbox_widget",
                        on_change=update_stats_show_loadings
                    )
                
                if st.session_state.stats_pca_3d and n_samples >= 3:
                    plot_3d_pca(matrix_df, analysis_type)
                else:
                    if n_samples == 2 and st.session_state.stats_pca_3d:
                        st.info("📊 PCA 3D non disponible avec 2 échantillons. Affichage de la PCA 2D/1D.")
                    perform_pca_analysis(matrix_df, analysis_type)
                
                if st.session_state.stats_show_loadings:
                    st.subheader("🔍 Contribution des features aux composantes principales")
                    
                    pca, X_pca, X_scaled = safe_pca_analysis(matrix_df, n_components=min(10, matrix_df.shape[1], matrix_df.shape[0]), analysis_type=analysis_type)
                    
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
                if n_samples < 3:
                    st.error("❌ t-SNE non disponible : nécessite au moins 3 échantillons")
                    st.info("💡 Utilisez la PCA pour analyser vos données avec 2 échantillons.")
                else:
                    plot_tsne_analysis(matrix_df, analysis_type)
            
            elif stat_section == "🌋 Volcano Plot":
                # NOUVEAU : Section dédiée au Volcano Plot dans l'analyse par individus
                if n_samples < 2:
                    st.error("❌ Volcano Plot non disponible avec moins de 2 échantillons")
                else:
                    plot_volcano_plot(matrix_df)
            
            elif stat_section == "🔍 Clustering":
                st.subheader("🔍 Analyses de clustering")
                
                if n_samples < 2:
                    st.error("❌ Clustering impossible avec moins de 2 échantillons")
                else:
                    st.subheader("🎯 K-means Clustering")
                    perform_kmeans_clustering(matrix_df, analysis_type)
                    
                    st.markdown("---")
                    
                    st.subheader("🌳 Clustering hiérarchique")
                    st.info("Cette analyse utilise les molécules identifiées du fichier features")
                    if features_df is not None:
                        samples_list = list(set([s for samples in features_df['samples'].dropna() 
                                            for s in samples.split(',')]))
                        if len(samples_list) >= 2:
                            plot_hierarchical_clustering(features_df, samples_list)
                        else:
                            st.error("❌ Clustering hiérarchique impossible : moins de 2 échantillons dans les données features")
                    else:
                        st.warning("⚠️ Données features non chargées - clustering hiérarchique non disponible")
            
            elif stat_section == "📈 Corrélations":
                st.subheader("📈 Analyses de corrélation")
                
                if n_samples < 2:
                    st.error("❌ Analyse de corrélation impossible avec moins de 2 échantillons")
                else:
                    corr_matrix = plot_correlation_heatmap(matrix_df, analysis_type)
                    
                    if corr_matrix is not None:
                        st.subheader("📊 Statistiques de corrélation")
                        
                        if analysis_type == "Analyse par marqueurs":
                            # Pour les marqueurs, analyser toutes les corrélations
                            corr_values = []
                            for i in range(len(corr_matrix)):
                                for j in range(i+1, len(corr_matrix)):
                                    corr_values.append(corr_matrix.iloc[i, j])
                            title_suffix = " entre marqueurs"
                        else:
                            # Pour les individus
                            corr_values = []
                            for i in range(len(corr_matrix)):
                                for j in range(i+1, len(corr_matrix)):
                                    corr_values.append(corr_matrix.iloc[i, j])
                            title_suffix = " entre échantillons"
                        
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
                            
                            if len(corr_values) > 1:  # Au moins 2 valeurs pour faire un histogramme
                                fig_corr_dist = px.histogram(
                                    x=corr_values,
                                    nbins=min(30, len(corr_values)),
                                    title=f"Distribution des corrélations{title_suffix}",
                                    labels={'x': 'Coefficient de corrélation', 'y': 'Fréquence'},
                                    color_discrete_sequence=DISTINCT_COLORS
                                )
                                st.plotly_chart(fig_corr_dist, use_container_width=True, key=generate_unique_key("correlation_distribution"))
                            else:
                                st.info(f"📊 Une seule valeur de corrélation disponible : {corr_values[0]:.3f}")
            
            elif stat_section == "🎨 Heatmaps":
                st.subheader("🎨 Heatmaps avancées")
                
                if n_samples < 2:
                    st.error("❌ Heatmaps impossibles avec moins de 2 échantillons")
                else:
                    st.subheader("🔥 Heatmap des intensités")
                    
                    transform_option = st.selectbox(
                        "Transformation des données",
                        ["Aucune", "Log10", "Z-score", "Min-Max"],
                        index=["Aucune", "Log10", "Z-score", "Min-Max"].index(st.session_state.stats_heatmap_transform),
                        key="heatmap_transform_widget",
                        on_change=update_stats_heatmap_transform
                    )
                    
                    try:
                        if st.session_state.stats_heatmap_transform == "Log10":
                            matrix_transformed = np.log10(matrix_df + 1)
                            title_suffix = "(échelle log)"
                        elif st.session_state.stats_heatmap_transform == "Z-score":
                            scaler = StandardScaler()
                            matrix_transformed = pd.DataFrame(
                                scaler.fit_transform(matrix_df.T).T,
                                index=matrix_df.index,
                                columns=matrix_df.columns
                            )
                            title_suffix = "(Z-score)"
                        elif st.session_state.stats_heatmap_transform == "Min-Max":
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
                            10, max_features, 
                            min(st.session_state.stats_heatmap_features, max_features),
                            key="heatmap_features_widget",
                            on_change=update_stats_heatmap_features
                        )
                        
                        feature_var = matrix_transformed.var(axis=0).nlargest(st.session_state.stats_heatmap_features)
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
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la transformation des données : {str(e)}")
                    
                    st.subheader("📏 Heatmap des distances entre échantillons")
                    
                    if n_samples >= 2:
                        distance_metric = st.selectbox(
                            "Métrique de distance",
                            ["Euclidienne", "Cosinus"],
                            index=["Euclidienne", "Cosinus"].index(st.session_state.stats_distance_metric),
                            key="distance_metric_widget",
                            on_change=update_stats_distance_metric
                        )
                        
                        try:
                            from scipy.spatial.distance import pdist, squareform
                            
                            if st.session_state.stats_distance_metric == "Euclidienne":
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
                                title=f"Matrice de distance {st.session_state.stats_distance_metric.lower()} entre échantillons",
                                color_continuous_scale='Plasma'
                            )
                            
                            fig_dist.update_layout(height=600)
                            st.plotly_chart(fig_dist, use_container_width=True, key=generate_unique_key("distance_heatmap"))
                            
                        except Exception as e:
                            st.error(f"❌ Erreur lors du calcul des distances : {str(e)}")
                    else:
                        st.info("📊 Heatmap de distances non disponible avec moins de 2 échantillons")
    
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
            
            # CORRECTION : Ajout des callbacks manquants pour les widgets d'export
            def update_reports_identified_only():
                if 'export_identified_only_widget' in st.session_state:
                    st.session_state.reports_identified_only = st.session_state.export_identified_only_widget
            
            def update_reports_aggregate():
                if 'export_aggregate_widget' in st.session_state:
                    st.session_state.reports_aggregate = st.session_state.export_aggregate_widget
            
            def update_reports_conf_levels():
                if 'export_conf_levels_widget' in st.session_state:
                    st.session_state.reports_conf_levels = st.session_state.export_conf_levels_widget
            
            def update_reports_samples():
                if 'export_samples_widget' in st.session_state:
                    st.session_state.reports_samples = st.session_state.export_samples_widget
            
            def update_reports_columns():
                if 'export_columns_widget' in st.session_state:
                    st.session_state.reports_columns = st.session_state.export_columns_widget
            
            def update_reports_include_stats():
                if 'include_stats_widget' in st.session_state:
                    st.session_state.reports_include_stats = st.session_state.include_stats_widget
            
            def update_reports_include_summary():
                if 'include_summary_widget' in st.session_state:
                    st.session_state.reports_include_summary = st.session_state.include_summary_widget
            
            col1, col2 = st.columns(2)
            
            with col1:
                export_identified_only = st.checkbox(
                    "Exporter uniquement les identifiées", 
                    value=st.session_state.reports_identified_only, 
                    key="export_identified_only_widget",
                    on_change=update_reports_identified_only
                )
                
                export_aggregate = st.checkbox(
                    "Agréger par molécule (grouper adduits)", 
                    value=st.session_state.reports_aggregate, 
                    key="export_aggregate_widget",
                    on_change=update_reports_aggregate
                )
                
                if 'confidence_level' in features_df.columns:
                    export_conf_levels = st.multiselect(
                        "Niveaux de confiance à exporter",
                        options=[1, 2, 3, 4, 5],
                        default=st.session_state.reports_conf_levels,
                        key="export_conf_levels_widget",
                        on_change=update_reports_conf_levels
                    )
                
                samples_list = list(set([s for samples in features_df['samples'].dropna() 
                                       for s in samples.split(',')]))
                export_samples = st.multiselect(
                    "Échantillons à exporter",
                    options=samples_list,
                    default=st.session_state.reports_samples if st.session_state.reports_samples else samples_list,
                    key="export_samples_widget",
                    on_change=update_reports_samples
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
                
                if st.session_state.reports_aggregate:
                    default_export_cols.insert(4, 'total_intensity')
                    default_export_cols.insert(5, 'sample_specific_intensity')
                
                export_columns = st.multiselect(
                    "Colonnes à exporter", 
                    options=all_columns + (['total_intensity', 'sample_specific_intensity'] if st.session_state.reports_aggregate else []),
                    default=st.session_state.reports_columns if st.session_state.reports_columns else [col for col in default_export_cols if col in all_columns + (['total_intensity', 'sample_specific_intensity'] if st.session_state.reports_aggregate else [])],
                    key="export_columns_widget",
                    on_change=update_reports_columns
                )
                
                include_stats = st.checkbox(
                    "Inclure les statistiques par échantillon", 
                    value=st.session_state.reports_include_stats, 
                    key="include_stats_widget",
                    on_change=update_reports_include_stats
                )
                
                include_summary = st.checkbox(
                    "Inclure un résumé en en-tête", 
                    value=st.session_state.reports_include_summary, 
                    key="include_summary_widget",
                    on_change=update_reports_include_summary
                )
            
            # Préparer les données
            export_df = features_df.copy()
            
            if st.session_state.reports_identified_only:
                export_df = export_df[export_df['match_name'].notna()]
            
            if 'confidence_level' in export_df.columns and st.session_state.reports_conf_levels:
                export_df = export_df[export_df['confidence_level'].isin(st.session_state.reports_conf_levels)]
            
            if st.session_state.reports_samples:
                export_df = export_df[
                    export_df['samples'].apply(
                        lambda x: any(sample in str(x) for sample in st.session_state.reports_samples) if pd.notna(x) else False
                    )
                ]
            
            if st.session_state.reports_aggregate and st.session_state.reports_identified_only:
                aggregated_data = []
                for sample in st.session_state.reports_samples:
                    sample_agg = aggregate_molecules_by_name_enhanced(export_df, sample)
                    if not sample_agg.empty:
                        aggregated_data.append(sample_agg)
                
                if aggregated_data:
                    export_df = pd.concat(aggregated_data, ignore_index=True)
                    if 'match_adduct' in export_df.columns:
                        export_df['match_adduct'] = export_df['match_adduct'].apply(
                            lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                        )
            
            if st.session_state.reports_columns:
                available_cols = [col for col in st.session_state.reports_columns if col in export_df.columns]
                export_df = export_df[available_cols]
            
            st.subheader("Aperçu des données à exporter")
            info_text = f"Nombre de lignes à exporter : {len(export_df)}"
            if st.session_state.reports_aggregate and st.session_state.reports_identified_only:
                info_text += " (molécules uniques avec adduits groupés et intensités spécifiques)"
            st.info(info_text)
            st.dataframe(export_df.head(10), use_container_width=True)
            
            # Boutons d'export
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if not export_df.empty:
                    csv_data = export_df.to_csv(index=False)
                    
                    if st.session_state.reports_include_summary:
                        summary = f"""# Résumé de l'export
# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Nombre de lignes: {len(export_df)}
# Agrégé par molécule: {st.session_state.reports_aggregate and st.session_state.reports_identified_only}
# Intensités spécifiques incluses: {st.session_state.reports_aggregate and st.session_state.reports_identified_only}
# Filtres appliqués:
# - Identifiées uniquement: {st.session_state.reports_identified_only}
# - Niveaux de confiance: {st.session_state.reports_conf_levels if 'confidence_level' in features_df.columns else 'Tous'}
# - Échantillons: {', '.join(st.session_state.reports_samples)}
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
                        
                        if st.session_state.reports_include_stats:
                            stats_data = []
                            for sample in st.session_state.reports_samples:
                                if st.session_state.reports_aggregate:
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
                    index=available_levels.index(st.session_state.confidence_selected_level) if st.session_state.confidence_selected_level in available_levels else 0,
                    key="selected_level_analysis_widget",
                    on_change=update_confidence_selected_level
                )
                
                level_data = features_df[features_df['confidence_level'] == st.session_state.confidence_selected_level]
                
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
                
                st.subheader(f"Top 10 molécules uniques - Niveau {st.session_state.confidence_selected_level}")
                
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
                
                st.subheader(f"📈 Analyse des critères - Niveau {st.session_state.confidence_selected_level}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'mz_error_ppm' in level_data.columns and not level_data.empty:
                        fig_error = px.histogram(
                            level_data,
                            x='mz_error_ppm',
                            nbins=20,
                            title=f"Distribution erreurs m/z - Niveau {st.session_state.confidence_selected_level}",
                            labels={'mz_error_ppm': 'Erreur m/z (ppm)'},
                            color_discrete_sequence=DISTINCT_COLORS
                        )
                        fig_error.add_vline(x=5, line_dash="dash", line_color=DISTINCT_COLORS[1])
                        st.plotly_chart(fig_error, use_container_width=True, key=generate_unique_key(f"mz_error_dist_level_{st.session_state.confidence_selected_level}"))
                
                with col2:
                    if 'ms2_similarity_score' in level_data.columns and not level_data.empty:
                        ms2_level_data = level_data[level_data['ms2_similarity_score'] > 0]
                        if not ms2_level_data.empty:
                            fig_ms2 = px.histogram(
                                ms2_level_data,
                                x='ms2_similarity_score',
                                nbins=20,
                                title=f"Distribution scores MS2 - Niveau {st.session_state.confidence_selected_level}",
                                labels={'ms2_similarity_score': 'Score MS2'},
                                color_discrete_sequence=DISTINCT_COLORS
                            )
                            fig_ms2.add_vline(x=0.7, line_dash="dash", line_color=DISTINCT_COLORS[2])
                            fig_ms2.add_vline(x=0.4, line_dash="dash", line_color=DISTINCT_COLORS[3])
                            st.plotly_chart(fig_ms2, use_container_width=True, key=generate_unique_key(f"ms2_score_dist_level_{st.session_state.confidence_selected_level}"))
        else:
            show_confidence_levels_table()

if __name__ == "__main__":
    main()