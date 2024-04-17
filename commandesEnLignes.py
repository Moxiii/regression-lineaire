import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Charger les données
data = pd.read_csv('onlinefoods.csv')

# Supprimer la colonne "Unnamed: 12"
data.drop(columns=['Unnamed: 12'], inplace=True)

# Gestion des variables catégorielles
categorical_cols = ['Gender', 'Marital Status', 'Occupation', 'Educational Qualifications', 'Family size']
ct = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_cols)], remainder='passthrough')
X_encoded = ct.fit_transform(data.drop(columns=['Output']))

# Créer une fenêtre tkinter
root = tk.Tk()
root.title("Régression Linéaire")

# Fonction pour effectuer la régression linéaire
def perform_linear_regression():
    try:
        X = X_encoded
        y = data['Output'].values.reshape(-1, 1)

        model = LinearRegression()
        model.fit(X, y)

        # Afficher un message avec les coefficients du modèle
        messagebox.showinfo("Coefficients du modèle", f"Les coefficients du modèle de régression linéaire sont: {model.coef_}")

    except Exception as e:
        messagebox.showerror("Erreur", str(e))

# Fonction pour afficher le graphique
def show_graph():
    try:
        # Calculer les counts des commandes en ligne en fonction de l'âge
        age_counts = data['Age'].value_counts().sort_index()

        # Créer une fenêtre tkinter pour le graphique
        graph_window = tk.Toplevel(root)
        graph_window.title('Distribution of Ages')

        # Créer une figure matplotlib pour le graphique
        fig = Figure(figsize=(6, 4), facecolor='skyblue')
        ax = fig.add_subplot(111, facecolor='yellow')
        ax.bar(age_counts.index, age_counts.values, color='green', edgecolor='black')

        # Ajouter des titres et des labels d'axes
        ax.set_title('Distribution of Ages', fontsize=16)
        ax.set_xlabel('Ages', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)

        # Créer un canevas tkinter pour afficher la figure
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    except Exception as e:
        messagebox.showerror("Erreur", str(e))

# Bouton pour effectuer la régression linéaire
linear_regression_button = tk.Button(root, text="Régression Linéaire", command=perform_linear_regression)
linear_regression_button.pack()

# Bouton pour afficher le graphique
graph_button = tk.Button(root, text="Afficher le Graphique", command=show_graph)
graph_button.pack()

# Afficher la fenêtre tkinter
root.mainloop()
