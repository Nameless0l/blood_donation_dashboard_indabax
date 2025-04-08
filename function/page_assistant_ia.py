import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
import json
import re
import base64
from PIL import Image
import io

# Tentative d'importer Google Generative AI avec une gestion d'erreur
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

def get_gemini_api_key():
    """
    Obtient la clé API Gemini depuis différentes sources possibles:
    1. Variables de session Streamlit
    2. Variables d'environnement
    3. Saisie manuelle de l'utilisateur
    """
    # Vérifier si une clé existe déjà dans la session
    if "GEMINI_API_KEY" in st.session_state and st.session_state["GEMINI_API_KEY"]:
        return st.session_state["GEMINI_API_KEY"]
    
    # Tenter de lire depuis les secrets Streamlit (avec gestion d'erreur)
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except:
        pass
    
    # Si pas dans les secrets, essayer depuis les variables d'environnement
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    
    # Si toujours pas de clé, demander à l'utilisateur
    if not api_key:
        st.warning("⚠️ Clé API Gemini non configurée. Veuillez entrer votre clé API ci-dessous.")
        api_key = st.text_input("Clé API Gemini", type="password", 
                             help="Obtenez votre clé sur https://ai.google.dev/")
        if api_key:
            st.session_state["GEMINI_API_KEY"] = api_key
    
    return api_key

# Fonction pour capturer une image d'un graphique Plotly
def fig_to_base64(fig):
    """Convertit une figure Plotly en image base64."""
    img_bytes = fig.to_image(format="png", scale=2)
    return base64.b64encode(img_bytes).decode('utf-8')

def generate_dashboard_context(df):
    """Génère un contexte basique sur les données du dashboard pour l'IA."""
    # Vérifier si df est None ou vide
    if df is None:
        return {
            "total_donneurs": "Non disponible",
            "nombre_donneurs_eligibles": "Non disponible",
            "taux_eligibilite": "Non disponible",
            "distribution_genre": "Non disponible",
            "groupes_sanguins": "Non disponible",
        }
    
    context = {
        "total_donneurs": len(df),
        "nombre_donneurs_eligibles": df[df['eligibilite_code'] == 1].shape[0] if 'eligibilite_code' in df.columns else "Non disponible",
        "taux_eligibilite": f"{df[df['eligibilite_code'] == 1].shape[0] / len(df) * 100:.1f}%" if 'eligibilite_code' in df.columns else "Non disponible",
        "distribution_genre": df['Genre'].value_counts().to_dict() if 'Genre' in df.columns else "Non disponible",
        "groupes_sanguins": df['Groupe_sanguin'].value_counts().to_dict() if 'Groupe_sanguin' in df.columns else "Non disponible",
    }
    
    if 'arrondissement_clean' in df.columns:
        context["donneurs_par_arrondissement"] = df['arrondissement_clean'].value_counts().to_dict()
    
    if 'groupe_age' in df.columns:
        context["donneurs_par_age"] = df['groupe_age'].value_counts().to_dict()
    
    return context

def generate_system_prompt(df):
    """Génère un prompt système pour guider l'IA dans ses réponses."""
    context = generate_dashboard_context(df)
    
    return f"""Tu es Dr. Hemo, un assistant médical spécialisé dans l'analyse des campagnes de don de sang. 
    Tu aides les médecins et responsables de campagnes à interpréter les données et à optimiser leurs futures initiatives.
    
    Voici le contexte actuel des données de la campagne:
    - Nombre total de donneurs: {context['total_donneurs']}
    - Nombre de donneurs éligibles: {context['nombre_donneurs_eligibles']}
    - Taux d'éligibilité: {context['taux_eligibilite']}
    
    Réponds de manière précise, professionnelle mais accessible.
    Si tu ne connais pas la réponse à une question, indique-le honnêtement.
    Lorsqu'on te demande des recommandations, base-toi sur les meilleures pratiques médicales.
    
    N'hésite pas à suggérer des analyses complémentaires pertinentes lorsque c'est approprié.
    
    Ton objectif est d'aider à sauver des vies en optimisant les campagnes de don de sang.
    """

def generate_blood_fact():
    """Génère un fait aléatoire sur le don de sang."""
    facts = [
        "Un don de sang peut sauver jusqu'à trois vies.",
        "Le corps humain contient environ 5 litres de sang.",
        "Le sang contient des globules rouges, des globules blancs, des plaquettes et du plasma.",
        "Le don de sang prend généralement entre 8 et 10 minutes.",
        "Toutes les 2 secondes, quelqu'un a besoin de sang.",
        "Moins de 10% de la population mondiale éligible donne du sang.",
        "Le groupe sanguin O négatif est considéré comme donneur universel.",
        "Les personnes du groupe AB sont receveurs universels.",
        "Le sang a une durée de conservation limitée: 42 jours pour les globules rouges.",
        "Les plaquettes ne peuvent être conservées que 5 jours.",
        "Les hommes peuvent donner du sang tous les 3 mois, les femmes tous les 4 mois.",
        "Le volume sanguin représente environ 7% du poids corporel humain.",
        "Un adulte en bonne santé peut donner environ 470 ml de sang lors d'un don standard.",
        "Le don de plasma peut être effectué plus fréquemment que le don de sang total.",
        "Le fer est essentiel à la formation de l'hémoglobine dans les globules rouges."
    ]
    return np.random.choice(facts)

def process_gemini_response(prompt, chat_history=None, image_base64=None, temperature=0.7):
    """Traite la requête avec l'API Gemini et retourne la réponse."""
    # Vérifier si Gemini est disponible
    if not GEMINI_AVAILABLE:
        return ("Je suis désolé, mais l'API Google Generative AI n'est pas disponible. "
                "Veuillez installer le package avec `pip install google-generativeai`.", None)
    
    # Obtenir la clé API
    api_key = get_gemini_api_key()
    
    if not api_key:
        return ("Je ne peux pas traiter votre demande car aucune clé API Gemini n'est configurée. "
                "Veuillez fournir une clé API pour activer mes fonctionnalités complètes.", None)
    
    try:
        # Configurer Gemini avec la clé
        genai.configure(api_key=api_key)
        
        # Créer le modèle
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"temperature": temperature}
        )
        
        # Deux approches différentes selon que nous avons un historique ou non
        if chat_history:
            # Créer une conversation sans utiliser de rôle système
            # (Nous injecterons le contexte dans le premier message)
            chat = model.start_chat()
            
            # Si une image est fournie
            if image_base64:
                # Convertir base64 en image
                img_data = base64.b64decode(image_base64)
                image_parts = [
                    {
                        "mime_type": "image/png",
                        "data": img_data
                    },
                    prompt
                ]
                response = chat.send_message(image_parts)
            else:
                # Texte uniquement
                response = chat.send_message(prompt)
            
            # Mettre à jour l'historique de conversation
            updated_history = chat.history
            
            return response.text, updated_history
        else:
            # Pour la première interaction, inclure le contexte système dans la requête
            system_context = generate_system_prompt(None)  # Générer sans données spécifiques
            enhanced_prompt = f"{system_context}\n\nQuestion de l'utilisateur: {prompt}"
            
            # Si une image est fournie
            if image_base64:
                # Convertir base64 en image
                img_data = base64.b64decode(image_base64)
                image_parts = [
                    {
                        "mime_type": "image/png",
                        "data": img_data
                    },
                    enhanced_prompt
                ]
                response = model.generate_content(image_parts)
            else:
                # Texte uniquement
                response = model.generate_content(enhanced_prompt)
            
            return response.text, None  # Pas d'historique à cette étape
            
    except Exception as e:
        error_msg = str(e)
        if "API key not valid" in error_msg.lower():
            return "Erreur: La clé API fournie n'est pas valide. Veuillez vérifier votre clé.", None
        return f"Je suis désolé, j'ai rencontré une erreur: {str(e)}", None

def display_chat_message(role, content, avatar=None):
    """Affiche un message de chat avec style."""
    if role == "assistant":
        with st.chat_message(role, avatar="🩸"):
            st.write(content)
    else:
        with st.chat_message(role, avatar="👨‍⚕️"):
            st.write(content)

def create_visualization_for_question(df, question):
    """Crée une visualisation adaptée à la question posée."""
    if df is None or len(df) == 0:
        st.warning("Aucune donnée disponible pour générer des visualisations.")
        return None
    fig = None
    
    # Identifier les mots-clés pour déterminer le type de visualisation
    question_lower = question.lower()
    
    # Analyse par groupe sanguin
    if "groupe sanguin" in question_lower or "blood type" in question_lower:
        if 'Groupe_sanguin' in df.columns:
            blood_counts = df['Groupe_sanguin'].value_counts().reset_index()
            blood_counts.columns = ['Groupe sanguin', 'Nombre de donneurs']
            
            # Choisir entre camembert ou barres selon la question
            if "proportion" in question_lower or "pourcentage" in question_lower or "répartition" in question_lower:
                fig = px.pie(
                    blood_counts, 
                    values='Nombre de donneurs',
                    names='Groupe sanguin',
                    title="Répartition des donneurs par groupe sanguin",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
            else:
                fig = px.bar(
                    blood_counts,
                    x='Groupe sanguin',
                    y='Nombre de donneurs',
                    title="Nombre de donneurs par groupe sanguin",
                    color='Nombre de donneurs',
                    color_continuous_scale='Reds'
                )
    
    # Analyse par âge
    elif "âge" in question_lower or "age" in question_lower or "tranche" in question_lower:
        if 'groupe_age' in df.columns:
            age_counts = df['groupe_age'].value_counts().reset_index()
            age_counts.columns = ['Groupe d\'âge', 'Nombre de donneurs']
            
            # Trier les groupes d'âge
            try:
                def extract_first_num(x):
                    if str(x) == 'nan' or pd.isna(x):
                        return 999
                    try:
                        if '-' in str(x):
                            return int(str(x).split('-')[0])
                        else:
                            return int(''.join(filter(str.isdigit, str(x)[:2])))
                    except:
                        return 998
                
                age_counts['ordre'] = age_counts['Groupe d\'âge'].apply(extract_first_num)
                age_counts = age_counts.sort_values('ordre')
                age_counts = age_counts.drop('ordre', axis=1)
            except Exception as e:
                st.warning(f"Impossible de trier les groupes d'âge: {e}")
            
            fig = px.bar(
                age_counts,
                x='Groupe d\'âge',
                y='Nombre de donneurs',
                title="Répartition des donneurs par groupe d'âge",
                color='Nombre de donneurs',
                color_continuous_scale='Blues'
            )
    
    # Analyse géographique
    elif "arrondissement" in question_lower or "zone" in question_lower or "quartier" in question_lower or "géographique" in question_lower:
        if 'arrondissement_clean' in df.columns:
            arrond_counts = df['arrondissement_clean'].value_counts().reset_index()
            arrond_counts.columns = ['Arrondissement', 'Nombre de donneurs']
            
            fig = px.bar(
                arrond_counts,
                x='Arrondissement',
                y='Nombre de donneurs',
                title="Répartition des donneurs par arrondissement",
                color='Nombre de donneurs',
                color_continuous_scale='Reds'
            )
    
    # Analyse d'éligibilité
    elif "éligibilité" in question_lower or "eligible" in question_lower or "non-éligible" in question_lower:
        if 'ÉLIGIBILITÉ AU DON.' in df.columns:
            eligibility_counts = df['ÉLIGIBILITÉ AU DON.'].value_counts().reset_index()
            eligibility_counts.columns = ['Statut', 'Nombre']
            
            fig = px.pie(
                eligibility_counts,
                values='Nombre',
                names='Statut',
                title="Répartition de l'éligibilité",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
    
    # Analyse par genre
    elif "genre" in question_lower or "homme" in question_lower or "femme" in question_lower or "sexe" in question_lower:
        if 'Genre' in df.columns:
            gender_counts = df['Genre'].value_counts().reset_index()
            gender_counts.columns = ['Genre', 'Nombre de donneurs']
            
            fig = px.pie(
                gender_counts,
                values='Nombre de donneurs',
                names='Genre',
                title="Répartition des donneurs par genre",
                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
            )
    
    # Tendances temporelles
    elif "tendance" in question_lower or "temps" in question_lower or "mois" in question_lower or "année" in question_lower or "evolution" in question_lower:
        if 'Date_don' in df.columns:
            # Convertir en datetime si ce n'est pas déjà fait
            df['Date_don'] = pd.to_datetime(df['Date_don'], errors='coerce')
            
            # Aggrégation par mois
            df['mois'] = df['Date_don'].dt.to_period('M')
            monthly_counts = df.groupby('mois').size().reset_index(name='nombre_dons')
            monthly_counts['mois_str'] = monthly_counts['mois'].astype(str)
            
            fig = px.line(
                monthly_counts,
                x='mois_str',
                y='nombre_dons',
                title="Évolution mensuelle des dons",
                markers=True
            )
    
    # Analyse des conditions de santé
    elif "santé" in question_lower or "conditions" in question_lower or "médicale" in question_lower:
        health_cols = [col for col in df.columns if '_indicateur' in col]
        
        if health_cols:
            health_data = []
            for col in health_cols:
                condition_name = col.replace('_indicateur', '')
                count = df[col].sum()
                health_data.append({
                    'Condition': condition_name,
                    'Nombre de cas': count
                })
            
            health_df = pd.DataFrame(health_data)
            health_df = health_df.sort_values('Nombre de cas', ascending=False)
            
            fig = px.bar(
                health_df,
                x='Condition',
                y='Nombre de cas',
                title="Conditions médicales les plus fréquentes",
                color='Nombre de cas',
                color_continuous_scale='Reds'
            )
    
    # Si aucune visualisation spécifique n'a été créée, créer une visualisation générale
    if fig is None:
        # Créer une visualisation par défaut sur l'éligibilité
        if 'ÉLIGIBILITÉ AU DON.' in df.columns:
            eligibility_counts = df['ÉLIGIBILITÉ AU DON.'].value_counts().reset_index()
            eligibility_counts.columns = ['Statut', 'Nombre']
            
            fig = px.pie(
                eligibility_counts,
                values='Nombre',
                names='Statut',
                title="Répartition globale de l'éligibilité au don",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
        elif 'Genre' in df.columns:
            # Si pas d'éligibilité, utiliser le genre comme fallback
            gender_counts = df['Genre'].value_counts().reset_index()
            gender_counts.columns = ['Genre', 'Nombre de donneurs']
            
            fig = px.pie(
                gender_counts,
                values='Nombre de donneurs',
                names='Genre',
                title="Répartition des donneurs par genre",
                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
            )
    
    return fig


# Mode de démonstration quand Gemini n'est pas disponible
def demo_response(question):
    """Génère une réponse de démonstration basée sur des règles simples."""
    question_lower = question.lower()
    
    # Réponses prédéfinies basées sur des mots-clés
    if "groupe sanguin" in question_lower:
        return "D'après nos données, le groupe sanguin O+ est le plus répandu parmi nos donneurs (environ 45%), suivi par A+ (35%). Les groupes rhésus négatifs sont plus rares mais très demandés, notamment O- qui est le donneur universel. Une stratégie ciblée pour attirer plus de donneurs avec des groupes sanguins rares pourrait être bénéfique."
    
    elif "arrondissement" in question_lower:
        return "L'analyse montre que Douala 3 a le plus grand nombre de donneurs (35%), suivi par Douala 5 (25%). Douala 4 est l'arrondissement avec le moins de donneurs (8%). Cette disparité pourrait s'expliquer par la densité de population, l'accessibilité des centres de don, ou l'efficacité des campagnes locales de sensibilisation."
    
    elif "âge" in question_lower or "age" in question_lower:
        return "La tranche d'âge 26-35 ans représente le groupe le plus important de donneurs, avec également le meilleur taux d'éligibilité (82%). Les jeunes de 18-25 ans ont un bon taux de participation mais légèrement moins d'éligibilité (75%). Nous observons une diminution progressive de l'éligibilité avec l'âge, tombant à environ 45% pour les 56-65 ans."
    
    elif "genre" in question_lower or "homme" in question_lower or "femme" in question_lower:
        return "Les données montrent une participation plus élevée des hommes (60%) par rapport aux femmes (40%). Le taux d'éligibilité est aussi légèrement plus élevé chez les hommes (78% contre 72%), principalement en raison des contraintes spécifiques comme la grossesse, l'allaitement et les taux d'hémoglobine différents. Des stratégies ciblées pour encourager le don féminin pourraient être envisagées."
    
    elif "éligibilité" in question_lower:
        return "Le taux d'éligibilité global est de 75%. Les principales raisons d'inéligibilité temporaire sont le taux d'hémoglobine bas, un don récent (<3 mois), et des infections récentes. Les causes d'inéligibilité définitive incluent principalement les porteurs de VIH/hépatite, l'hypertension non contrôlée et les troubles cardiaques graves."
    
    elif "tendance" in question_lower or "évolution" in question_lower:
        return "L'analyse des tendances montre une augmentation globale des dons de 15% sur l'année, avec des pics notables en avril (+20%) et décembre (+25%). Ces pics coïncident avec les grandes campagnes nationales. Nous observons également une baisse significative en août (-15%), suggérant un impact des vacances d'été."
    
    elif "recommandation" in question_lower or "stratégie" in question_lower or "améliorer" in question_lower:
        return "Basé sur l'analyse des données, je recommande:\n\n1. Intensifier les campagnes dans l'arrondissement Douala 4 où la participation est la plus faible\n2. Cibler particulièrement la tranche d'âge 26-35 ans qui présente le meilleur taux d'éligibilité\n3. Développer des incitations spécifiques pour encourager le retour des donneurs (programme de fidélité)\n4. Organiser des sessions d'information préalables pour réduire le taux d'inéligibilité temporaire\n5. Créer des campagnes ciblées pour les groupes sanguins rares comme O- et B-"
    
    elif "image" in question_lower or "analyser" in question_lower:
        return "Je suis capable d'analyser des images ou des graphiques liés aux campagnes de don de sang. Pour me montrer une image, utilisez la fonction de téléchargement d'image dans l'interface ou posez-moi une question sur un graphique généré par l'assistant."
    
    else:
        return "Je comprends votre question sur les données de don de sang. Dans le mode de démonstration, je peux répondre de façon basique aux questions sur la répartition par groupe sanguin, âge, genre, arrondissement, ainsi que sur l'éligibilité, les tendances et les recommandations. Pour des analyses plus précises, vous devrez configurer l'API Gemini."

def assistant_ia(df):
    """Interface principale pour l'assistant IA."""
    st.title("💬 Assistant IA d'Analyse des Dons de Sang")
    if df is None:
        st.warning("Les données n'ont pas pu être chargées correctement.")
        df = pd.DataFrame() 
    # Vérifier si Gemini est disponible
    if not GEMINI_AVAILABLE:
        st.warning("""
        ⚠️ Le package Google Generative AI n'est pas installé. 
        L'assistant fonctionnera en mode de démonstration limité.
        
        Pour installer le package et activer toutes les fonctionnalités, exécutez:
        ```
        pip install google-generativeai
        ```
        """)
    
    st.markdown("""
    Posez-moi vos questions sur les données de la campagne de don de sang. 
    Je peux vous aider à interpréter les graphiques, à identifier des tendances, 
    ou à formuler des recommandations basées sur les données.
    """)
    
    # Configuration de la clé API si besoin
    if GEMINI_AVAILABLE:
        api_key = get_gemini_api_key()
        
        # Si toujours pas de clé après demande à l'utilisateur, activer le mode démo
        if not api_key:
            st.info("ℹ️ L'assistant fonctionne actuellement en mode démonstration limité.")
    
    # Initialiser la session state pour le chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
        # Message de bienvenue initial
        welcome_message = f"""
        👋 Bonjour! Je suis Dr. Hemo, votre assistant IA pour l'analyse des données de don de sang.
        
        Je peux vous aider à comprendre les tendances, analyser les profils des donneurs, 
        et proposer des stratégies pour améliorer vos futures campagnes.
        
        **Le saviez-vous?** {generate_blood_fact()}
        
        Comment puis-je vous aider aujourd'hui?
        """
        
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})
    
    # Ne plus stocker l'historique de chat avec le rôle 'system'
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialiser un dictionnaire pour stocker les visualisations générées
    if "generated_visualizations" not in st.session_state:
        st.session_state.generated_visualizations = {}
    
    # Afficher l'historique des messages
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])
    
    # Option de téléchargement d'image
    uploaded_file = st.file_uploader("Télécharger une image à analyser (optionnel)", type=["jpg", "jpeg", "png"])
    image_base64 = None

    if uploaded_file is not None:
        # Convertir l'image en base64
        image_bytes = uploaded_file.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Afficher l'image
        st.image(uploaded_file, caption="Image téléchargée", use_container_width=True)

    # Entrée utilisateur
    prompt = st.chat_input("Posez votre question sur les données de don de sang...")

    if prompt:
        # Afficher la question de l'utilisateur
        display_chat_message("user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Si une image a été téléchargée, ajuster le prompt
        if image_base64:
            enhanced_prompt = f"Voici une image liée au don de sang. {prompt}"
            should_generate_viz = False
        else:
            # Vérifier si la question concerne une visualisation
            should_generate_viz = any(keyword in prompt.lower() for keyword in 
                                    ["montrer", "graphique", "visualiser", "afficher", "graphe", 
                                     "voir", "courbe", "diagramme", "camembert", "tendance", 
                                     "évolution", "répartition", "distribution"])
            enhanced_prompt = prompt
        
        # Générer une visualisation si nécessaire et qu'aucune image n'a été téléchargée
        generated_fig = None
        
        if should_generate_viz and not image_base64:
            with st.spinner("Génération d'une visualisation..."):
                generated_fig = create_visualization_for_question(df, prompt)
                
                if generated_fig:
                    # Convertir le graphique en image base64 pour Gemini Vision (si disponible)
                    if GEMINI_AVAILABLE and get_gemini_api_key():
                        viz_base64 = fig_to_base64(generated_fig)
                        image_base64 = viz_base64  # Utiliser le graphique comme image
                        enhanced_prompt = f"""
                        Question: {prompt}
                        
                        J'ai généré un graphique en réponse à cette question. Décris ce que tu vois sur ce graphique,
                        explique les tendances ou les patterns, et fournis des insights pertinents basés sur cette visualisation.
                        """
                    
                    # Stocker la figure pour l'affichage
                    viz_id = f"viz_{len(st.session_state.generated_visualizations)}"
                    st.session_state.generated_visualizations[viz_id] = generated_fig
        
        # Traiter la réponse avec Gemini ou la réponse de démonstration
        with st.spinner("Analyse de votre question..."):
            if GEMINI_AVAILABLE and get_gemini_api_key():
                # Obtenir la réponse de l'API (sans utiliser l'historique pour le moment)
                # Cette approche contourne le problème du rôle "system"
                response, _ = process_gemini_response(
                    enhanced_prompt, 
                    None,  # Pas d'historique pour contourner le problème
                    image_base64
                )
            else:
                # Utiliser la fonction de démonstration si Gemini n'est pas disponible
                response = demo_response(prompt)
        
        # Afficher la visualisation si générée
        if generated_fig:
            response_with_viz = f"{response}\n\n"
            display_chat_message("assistant", response_with_viz)
            st.plotly_chart(generated_fig, use_container_width=True)
        else:
            display_chat_message("assistant", response)
        
        # Sauvegarder la réponse dans l'historique
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar avec des exemples de questions
    with st.sidebar:
        st.subheader("Exemples de questions")
        example_questions = [
            "Quelle est la répartition des donneurs par groupe sanguin?",
            "Quels sont les arrondissements avec le plus de donneurs?",
            "Comment se répartit l'éligibilité au don par tranche d'âge?",
            "Quels sont les facteurs qui influencent le plus l'éligibilité?",
            "Quelles recommandations pour améliorer le taux de don chez les jeunes?",
            "Montre-moi l'évolution des dons au fil du temps.",
            "Quelle est la distribution des donneurs par genre?"
        ]
        
        st.markdown("**Cliquez pour poser une question:**")
        for question in example_questions:
            if st.button(question, key=f"ex_{question[:20]}"):
                # Utiliser la session state pour injecter la question
                st.session_state.user_question = question
                st.rerun()
        
        # Utiliser la session state pour injecter la question
        if "user_question" in st.session_state:
            st.session_state.messages.append({"role": "user", "content": st.session_state.user_question})
            
            # Générer la réponse pour cette question
            response = demo_response(st.session_state.user_question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Nettoyer après utilisation
            del st.session_state.user_question
            st.rerun()
        
        st.divider()
        st.subheader("Saviez-vous?")
        st.info(generate_blood_fact())
        
        # Options avancées
        with st.expander("Options avancées"):
            if st.button("Réinitialiser la conversation"):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.session_state.generated_visualizations = {}
                if "GEMINI_API_KEY" in st.session_state:
                    del st.session_state["GEMINI_API_KEY"]
                st.rerun()

if __name__ == "__main__":
    # Charger des données fictives pour les tests
    df = pd.read_csv('data/processed_data/dataset_don_sang_enrichi.csv', encoding='utf-8')
    assistant_ia(df)