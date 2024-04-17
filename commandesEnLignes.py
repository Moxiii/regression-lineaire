import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# Charger les données
data = pd.read_csv('onlinefoods.csv')

# Supprimer la colonne "Unnamed: 12"
data.drop(columns=['Unnamed: 12'], inplace=True)

# Remplacer les valeurs "Below Rs.10000" par 5000 dans la colonne des revenus mensuels
data['Monthly Income'] = data['Monthly Income'].replace('Below Rs.10000', 5000)

# Remplacer les valeurs "More than 50000" par 60000 dans la colonne des revenus mensuels
data['Monthly Income'] = data['Monthly Income'].replace('More than 50000', 60000)

# Extraire la moyenne de la plage "10001 to 25000" et la remplacer dans la colonne des revenus mensuels
data['Monthly Income'] = data['Monthly Income'].replace('10001 to 25000', (10001 + 25000) / 2)

# Extraire la moyenne de la plage "25001 to 50000" et la remplacer dans la colonne des revenus mensuels
data['Monthly Income'] = data['Monthly Income'].replace('25001 to 50000', (25001 + 50000) / 2)

# Remplacer les valeurs "No Income" par 0 dans la colonne des revenus mensuels
data['Monthly Income'] = data['Monthly Income'].replace('No Income', 0)

# Convertir la colonne des revenus mensuels en type numérique
data['Monthly Income'] = pd.to_numeric(data['Monthly Income'], errors='coerce')

# Supprimer les lignes avec des valeurs non numériques
data = data.dropna(subset=['Monthly Income'])

# Mapper les valeurs "Yes" à 1 et "No" à 0 dans la colonne "Output"
data['Output'] = data['Output'].map({'Yes': 1, 'No': 0})

# Définir les colonnes catégorielles pour l'encodage à chaud
categorical_cols = ['Gender', 'Marital Status', 'Occupation', 'Educational Qualifications', 'Family size']

# Encodage à chaud des variables catégorielles
ct = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'
)

# Sélectionner les colonnes nécessaires pour la modélisation
X_encoded = ct.fit_transform(data.drop(columns=['Output']))

# Effectuer la régression linéaire
X = X_encoded
y = data['Output'].values.reshape(-1, 1)

# Visualiser les données pour chaque âge spécifié
def visualize_data(ages):
    for age in ages:
        # Filtrer les données en fonction de l'âge
        filtered_data = data[data['Age'] == age]

        # Sélectionner les colonnes pour l'encodage à chaud
        X_filtered = filtered_data.drop(columns=['Age', 'Output'])

        # Encodage à chaud des variables catégorielles pour les données filtrées
        X_encoded_filtered = ct.transform(X_filtered)

        # Effectuer la régression linéaire sur les données filtrées
        X_filtered = X_encoded_filtered
        y_filtered = filtered_data['Output'].values.reshape(-1, 1)

        model = LinearRegression()
        model.fit(X_filtered, y_filtered)

        # Afficher les coefficients du modèle
        print(f"Les coefficients du modèle de régression linéaire pour l'âge {age} sont:\n")
        for i, col in enumerate(ct.get_feature_names_out()):
            print(f"{col}: {model.coef_[0][i]}")

        # Faire une prédiction sur les données filtrées
        y_pred = model.predict(X_filtered)

        # Créer un graphique de dispersion des valeurs réelles par rapport aux valeurs prédites
        plt.scatter(y_filtered, y_pred)
        plt.xlabel('Valeurs Réelles')
        plt.ylabel('Valeurs Prédites')
        plt.title(f'Prédiction de Régression Linéaire pour l\'âge {age}')
        plt.show()

# Utilisation : spécifiez les âges en ligne de commande
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python program.py age1 age2 ...")
        sys.exit(1)
    ages = [int(age) for age in sys.argv[1:]]
    visualize_data(ages)
