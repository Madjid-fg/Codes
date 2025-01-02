import pandas as pd
import matplotlib.pyplot as plt

# === Paramètres de la base de données ===
file_path = "c:/Users/Madjid/Desktop/Mes Cours/Mémoire/Codes/AEP_hourly.csv"  # Chemin vers le fichier CSV
date_column = "Datetime"  # Nom de la colonne des horodatages
value_column = "AEP_MW"  # Nom de la colonne des valeurs

# === Plage de valeurs valides ===
min_value = 1000  # Valeur minimale acceptable
max_value = 50000  # Valeur maximale acceptable

# === Charger les données ===
data = pd.read_csv(file_path)

# Convertir la colonne des dates en type datetime
data[date_column] = pd.to_datetime(data[date_column])

# Calculer les intervalles entre les horodatages
data['interval'] = data[date_column].diff().dt.total_seconds() / 3600  # Intervalle en heures

# === Détection automatique de l'intervalle attendu ===
# Calculer la médiane des intervalles (plus robuste aux anomalies)
expected_interval = data['interval'].median()

# Afficher l'intervalle détecté
print(f"Intervalle attendu (détecté automatiquement) : {expected_interval} heures")

# === Détection des anomalies ===
# Ajouter une colonne pour l'année
data['year'] = data[date_column].dt.year

# Points normaux
normal_points = data[(data['interval'] == expected_interval) | (data.index == 0)]

# Points manquants : intervalle supérieur à l'intervalle attendu
missing_points = data[data['interval'] > expected_interval]

# Points redondants : doublons d'horodatages
duplicated_points = data[data[date_column].duplicated(keep=False)]

# Points retardés : intervalles négatifs
delayed_points = data[data['interval'] < 0]

# === Détection des valeurs anormales ===
# Détecter les valeurs en dehors des plages valides
abnormal_values = data[(data[value_column] < min_value) | (data[value_column] > max_value)]

# === Compter les points par année ===
points_by_year = data.groupby('year').agg(
    Normaux=('interval', lambda x: (x == expected_interval).sum()),  # Nombre de points normaux
    Manquants=('interval', lambda x: (x > expected_interval).sum()),  # Nombre de points manquants
    Redondants=(date_column, lambda x: x.duplicated(keep=False).sum()),  # Nombre de points redondants
    Retardés=('interval', lambda x: (x < 0).sum()),  # Nombre de points retardés
    Anormaux=('AEP_MW', lambda x: ((x < min_value) | (x > max_value)).sum())  # Nombre de valeurs anormales
)

# Ajouter une colonne "Total" qui est la somme des cinq types de points pour chaque année
points_by_year['Total'] = points_by_year.sum(axis=1)

# === Afficher les résultats textuellement ===
print("=== Résumé des Anomalies ===")
print(points_by_year)

# === Visualisation de l'histogramme par année ===

# Tracer l'histogramme
ax = points_by_year.drop(columns='Total').plot(kind='bar', figsize=(14, 8), color=['green', 'orange', 'red', 'purple', 'blue'])

# Ajouter des titres et des labels
plt.title('Histogramme des Types de Points Par Année', fontsize=16)
plt.xlabel('Année', fontsize=14)
plt.ylabel('Nombre de Points', fontsize=14)
plt.xticks(rotation=45)

# Appliquer une échelle logarithmique sur l'axe des ordonnées pour mieux visualiser les petites valeurs
ax.set_yscale('log')

# Afficher la légende
plt.legend(title='Type de point')

# Afficher le graphique
plt.tight_layout()
plt.show()
