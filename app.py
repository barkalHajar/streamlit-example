import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import base64
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import streamlit as st
import xgboost as xgb
from openpyxl import reader, load_workbook, Workbook
import pickle

# Load the compressed Jupyter notebook file
with open('model.pkl', 'rb') as file:
    global model
    model = pickle.load(file)


st.write('''
# L'intelligence artificielle à votre service
PrediX
''')

st.write('''
data avant nettoyage
''')
global y_test
global y_pred

#
# model = xgb.Booster()
# model.load_model(r"Local.xgb")


def afficher_graphique():
    x = np.arange(len(y_test))
    fig, ax = plt.subplots()  # Créer une figure et un axe
    # Prédictions

    ax.plot(x, y_test, label='Données réelles')
    ax.plot(x, y_pred, label='Prédictions')
    ax.legend()
    st.pyplot(fig)  # Utiliser st.pyplot() pour afficher le graphique

def perform_predictions(data):
    # Prétraitement des données (adapté à votre pipeline de prétraitement)
    # ...

    # Convertir les données en une matrice DMatrix pour XGBoost
    dmatrix = xgb.DMatrix(data)

    # Effectuer les prédictions
    predictions = model.predict(dmatrix)

    # Retourner les prédictions arrondies
    return np.round(predictions)

def main():
    # ... Code de configuration Streamlit ...
    os.chdir("C:\\Users\\LENOVO\\OneDrive\\Bureau\\pythonProject11")
    dataset = pd.read_excel("Local.xlsx")
    # Affichage du DataFrame
    st.write(dataset)

    # Suppression des colonnes spécifiées
    columns_to_drop = ['Qté_Récep.', 'Unité_Récep.', 'Fournisseur', 'CC(O/N)', 'Jour', 'MT total', 'Nom_fournisseur',
                       'Désignation', 'N° BC', 'N° BL', 'Qté', 'Unité', 'Montant', 'Type', 'Coût_unitaire_moyen',
                       'Réglement', 'Année', 'Prix_unitaire', 'Unite_de_prix', 'Code_Nature',
                       'Trimestre']

    dataset = dataset.drop(columns_to_drop, axis=1)

    # Encodage des variables catégorielles
    encoder = LabelEncoder()

    dataset['Unité_Commande'] = encoder.fit_transform(dataset['Unité_Commande'])
    dataset['Article'] = encoder.fit_transform(dataset['Article'])

    def m(data):
        mois = data[0]
        if mois == 'janvier':
            return 1
        if mois == 'février':
            return 2
        if mois == 'mars':
            return 3
        if mois == 'avril':
            return 4
        if mois == 'mai':
            return 5
        if mois == 'juin':
            return 6
        if mois == 'juillet':
            return 7
        if mois == 'août':
            return 8
        if mois == 'septembre':
            return 9
        if mois == 'octobre':
            return 10
        if mois == 'novembre':
            return 11
        if mois == 'décembre':
            return 12

    dataset['Mois'] = dataset[['Mois']].apply(m, axis=1)

    global moyenne
    moyenne = dataset['Tps_d_appro'].mean()

    def moy(data):
        Tps_Réappro = data[0]
        if pd.isnull(Tps_Réappro):
            return moyenne
        else:
            return Tps_Réappro

    dataset['Tps_d_appro'] = dataset[['Tps_d_appro']].apply(moy, axis=1)

    st.write('''
    data après nettoyage
    ''')
    st.write(dataset)

    X = dataset.iloc[:, [0, 1, 2, 4]]
    y = dataset.iloc[:, [3]]
    global y_test
    global y_pred
    global model
    # Utiliser la méthode MinMaxScaler pour normaliser les données
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Préparer les données pour XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Définir les paramètres du modèle de gradient boosting
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.2,  # Taux d'apprentissage
        'max_depth': 5,  # Profondeur maximale de l'arbre
        'subsample': 0.8,  # Proportion d'échantillons utilisés pour la construction de chaque arbre
        'colsample_bytree': 0.8,  # Proportion de caractéristiques utilisées pour la construction de chaque arbre
        'seed': 42  # Graine aléatoire pour la reproductibilité
    }
    global model
    # Entraîner le modèle sur les données d'entraînement
    num_rounds = 100
    model = xgb.train(params, dtrain, num_rounds)

    st.write("Prédiction de X_test")
    # Prédire sur les données de test
    y_pred = model.predict(dtest)
    y_pred = np.round(y_pred).astype(int)
    st.write(y_pred)

    # Évaluer les performances du modèle en utilisant le RMSE, MAE et R²
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write("Performance sur l'ensemble de test :")
    st.write("rmse :", rmse)
    st.write("r2 :", r2)
    st.write("mae :", mae)

    # Données réelles
    st.write('Réel vs prédiction')

    # Bouton pour afficher le graphique
    if st.button("Afficher le graphique"):
        afficher_graphique()

    # Interface Streamlit
    st.title("Prédictions à partir d'un fichier Excel")

    # Importer un fichier Excel

    # Vérifier si un fichier a été importé
    # Remplacez "model.xgb" par le chemin vers votre modèle pré-entraîné

    # Fonction pour effectuer les prédictions
    # Importer un fichier Excel
    file = st.file_uploader("Importer un fichier Excel", type=["xlsx"], key="file_uploader")

    columns_to_drop1 = ['Qté_Récep.', 'Unité_Récep.', 'Fournisseur', 'CC(O/N)', 'Jour', 'MT total', 'Nom_fournisseur',
                       'Désignation', 'N° BC', 'N° BL', 'Qté', 'Unité', 'Montant', 'Type', 'Coût_unitaire_moyen',
                       'Réglement', 'Année', 'Prix_unitaire', 'Unite_de_prix', 'Code_Nature',
                       'Trimestre', 'Qté_Commandé']




    # Vérifier si un fichier a été importé
    if file is not None:
        # Lire le fichier Excel en tant que DataFrame
        df = pd.read_excel(file)

        # Afficher le DataFrame importé
        st.write("Données importées :")
        st.write(df)

        # Bouton pour effectuer les prédictions
        if st.button("Effectuer les prédictions",  key="predict_button"):
            columns_to_drop1 = [col for col in columns_to_drop1 if col in df.columns]
            # Nettoyage des données
            df_cleaned = df.drop(columns_to_drop1, axis=1)
            encoder = LabelEncoder()
            encoder.fit(df_cleaned['Unité_Commande'])
            df_cleaned['Unité_Commande'] = encoder.transform(df_cleaned['Unité_Commande'])
            encoder.fit(df_cleaned['Article'])
            df_cleaned['Article'] = encoder.transform(df_cleaned['Article'])
            df_cleaned['Mois'] = df_cleaned[['Mois']].apply(m, axis=1)
            df_cleaned['Tps_d_appro'] = df_cleaned[['Tps_d_appro']].apply(moy, axis=1)
            scaler = MinMaxScaler()
            df_cleaned1 = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df_cleaned.columns)

            # Effectuer les prédictions
            dmatrix_pred = xgb.DMatrix(df_cleaned1)
            predictions = model.predict(dmatrix_pred)
            rounded_predictions = np.round(predictions).astype(int)

            # Ajouter les prédictions au DataFrame
            df_cleaned['Prédictions'] = rounded_predictions

            # Afficher les résultats
            st.write("Résultats des prédictions :")
            st.write(df_cleaned)
            # Convert the DataFrame to Excel file
            excel_path = "predictions.xlsx"
            df_cleaned.to_excel(excel_path, index=False)

            # Encode the Excel file data to Base64
            with open(excel_path, "rb") as file:
                excel_data = file.read()
            b64 = base64.b64encode(excel_data).decode()

            # Generate the download link
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="predictions.xlsx">Télécharger le fichier Excel</a>'
            st.markdown(href, unsafe_allow_html=True)


# Appeler la fonction principale
if __name__ == "__main__":
    main()
