import pandas as pd
import numpy as np
import streamlit as st
import random
from datetime import datetime, timedelta

def format_number(n):
    """
    Formate un nombre pour l'affichage (par exemple, 1000 -> 1 000).
    
    Args:
        n (int): Nombre à formater
        
    Returns:
        str: Nombre formaté
    """
    return f"{n:,}".replace(",", " ")

def create_kpi_card(title, value, delta=None, delta_color="normal"):
    """
    Crée une carte KPI pour l'affichage dans Streamlit.
    
    Args:
        title (str): Titre du KPI
        value (str): Valeur principale
        delta (str, optional): Variation. Defaults to None.
        delta_color (str, optional): Couleur de la variation ('normal', 'inverse', 'off'). Defaults to "normal".
        
    Returns:
        st.metric: Métrique Streamlit
    """
    return st.metric(title, value, delta, delta_color=delta_color)