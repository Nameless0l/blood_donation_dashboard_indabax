import streamlit as st
import sys
import os

# S'assurer que le dossier 'pages' est dans le chemin
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importer les pages d'analyse
from pages.page_analyse_geographique import map as page_analyse_geographique
from pages.page_analyse_eligibilite import analyse_eligibilite as page_analyse_eligibilite

# Configuration de la page
st.set_page_config(
    page_title="Tableau de Bord des Dons de Sang",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour appliquer le style CSS personnalisé
def local_css():
    st.markdown("""
    <style>
        .main-header {
            color: #e74c3c;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .page-description {
            font-size: 18px;
            margin-bottom: 30px;
            color: #555;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #f8f9fa;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 16px;
            height: 60px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #e74c3c;
            color: white;
            font-weight: bold;
        }
        
        /* Style pour les métriques */
        div[data-testid="stMetricValue"] {
            font-size: 28px;
            font-weight: bold;
            color: #e74c3c;
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 16px;
            color: #333;
        }
        
        /* Style pour les cartes */
        div.stCard {
            border-left: 4px solid #e74c3c;
            padding-left: 15px;
            margin-bottom: 15px;
        }
        
        /* Style pour les filtres dans la sidebar */
        .sidebar-filter-title {
            font-weight: bold;
            margin-bottom: 5px;
            color: #333;
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Appliquer le style CSS
    local_css()
    
    # En-tête de l'application
    st.markdown('<div class="main-header">Tableau de Bord des Campagnes de Don de Sang</div>', unsafe_allow_html=True)
    
    # Menu latéral pour la navigation entre les pages
    st.sidebar.title("Navigation")
    
    # Options de pages (mise à jour avec la nouvelle page d'analyse d'éligibilité)
    pages = {
        "Vue d'ensemble": None,  # Sera développée ultérieurement
        "Analyse Géographique": page_analyse_geographique,
        "Analyse d'Éligibilité et Santé": page_analyse_eligibilite,
        "Analyse des Donneurs": None,  # Sera développée ultérieurement
        "Performance des Campagnes": None  # Sera développée ultérieurement
    }
    
    # Sélection de la page active
    selection = st.sidebar.radio("Aller à", list(pages.keys()))
    
    # Afficher la page sélectionnée
    if pages[selection] is not None:
        pages[selection]()
    else:
        st.info(f"La page '{selection}' est en cours de développement et sera disponible prochainement.")
        
        # Afficher un message temporaire pour les pages non encore implémentées
        st.markdown(f"""
        <div class="page-description">
            Cette section du tableau de bord permettra d'explorer:
            
            - Pour "Vue d'ensemble": Les KPIs principaux, tendances et progression vs objectifs
            - Pour "Analyse des Donneurs": Profil démographique, comportement, satisfaction et segmentation
            - Pour "Performance des Campagnes": Efficacité, canaux de communication et analyses temporelles
        </div>
        """, unsafe_allow_html=True)
    
    # Pied de page
    st.sidebar.markdown("---")
    st.sidebar.info("Développé pour le suivi des campagnes de don de sang")
    
if __name__ == "__main__":
    main()