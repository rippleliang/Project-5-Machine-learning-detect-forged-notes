import streamlit as st
import pandas as pd
import joblib as jl
from sklearn import preprocessing
import plotly.express as px

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


st.markdown(
    """
    <style>
    .main {
    background-color: #F5F5F5;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

background_color = '#F5F5F5'

with header:
	st.title('Projet 10: Détectez des faux billets avec Python')
	st.text('OpenClassrooms -  Data Analyst 2021-2022 - Xiuting LIANG')

with dataset:
	st.header('Le contexte du projet de data analyse')
	st.markdown("L'Organisation nationale de lutte contre le faux-monnayage, ou ONCFM, est une organisation publique ayant pour objectif de mettre en place des méthodes d’identification des contrefaçons des billets en euros.")
	st.markdown("L'Notre mission est construire un algorithme qui, à partir des caractéristiques géométriques d’un billet, serait capable de définir si ce dernier est un vrai ou un faux billet.")
    
with features:
	st.header('Introduction des modèles')
	st.markdown("Nous ultilisons les 5 informations géométriques sur un billet: length, height_left, height_right, margin_up, margin_low.")
	st.markdown("Un algorithme de la régression logistique classique est ultilisé pour identifier les billets vrais et faux avec présentations des résultats et des probabilités.")
                
with model_training:
	st.header('Analyser votre fichier csv')

# load model
logreg = jl.load(r"C:\Users\xiuti\OneDrive\OpenClassrooms\P10\P10_detectez_des_faux_billets_LIANG_Xiuting\LIANG_Xiuting_3_modele_logreg_code_082022.pkl")


df_production = st.file_uploader("Téléchargez votre fichier csv")


if df_production is not None:
    

  df = pd.read_csv(df_production, index_col = 'id',
                   usecols = ['id','height_left', 'height_right', 'margin_low', 'margin_up', 'length'])
  df_norm_pro = df.copy()
  # Normalisation des données
  numeric_range_pro = ['height_left', 'height_right', 'margin_low', 'margin_up', 'length']
  scaled_org_pro = preprocessing.StandardScaler().fit_transform(df_norm_pro[numeric_range_pro])
  df_norm_pro[numeric_range_pro] = scaled_org_pro

  # Importation le modèle
  logreg.predict(df_norm_pro)

  # Créer le dataframe
  df['resultat_logreg'] = logreg.predict(df_norm_pro)
  df['probabilité'] = logreg.predict_proba(df_norm_pro).round(2)[:,1:2]
  df_result = df[['resultat_logreg', 'probabilité']]
  df_result = df_result.reset_index()

  st.dataframe(df_result)
  
  df['resultat_logreg'] = df['resultat_logreg'].astype(str)
  df1 = df.groupby('resultat_logreg')['resultat_logreg'].count()
  df1 = pd.DataFrame(df1)

  fig = px.bar(df1, x='resultat_logreg', y = df1.index)
  st.write(fig)
