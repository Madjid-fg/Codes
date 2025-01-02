import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Charger les données
def load_data(file_path, time_column, value_column):
    df = pd.read_csv(file_path)
    df[time_column] = pd.to_datetime(df[time_column])  # Convertir en datetime
    df = df.sort_values(by=time_column)
    df = df[[time_column, value_column]].dropna(subset=[time_column])  # Supprimer les lignes sans timestamp
    df.rename(columns={time_column: 'time', value_column: 'value'}, inplace=True)
    return df

# Étape 1: Identifier et corriger les erreurs structurelles
def fix_structure(df):
    df = df.drop_duplicates(subset=['time'], keep='first')
    full_time_range = pd.date_range(start=df['time'].min(), end=df['time'].max(), freq='h')
    df = df.set_index('time').reindex(full_time_range).reset_index()
    df.rename(columns={'index': 'time'}, inplace=True)
    return df

# Fonction pour gérer les valeurs manquantes avant la détection des anomalies
def handle_missing_values(df, method='mean'):
    """
    Gère les valeurs manquantes en fonction de la méthode choisie.
    - `mean` : Impute les valeurs manquantes par la moyenne.
    - `interpolation` : Interpolation linéaire pour remplir les manquantes.
    - `ffill` : Remplissage en avant avec la dernière valeur valide.
    - `bfill` : Remplissage en arrière avec la prochaine valeur valide.
    """
    if method == 'mean':
        return df.fillna(df.mean())
    elif method == 'interpolation':
        return df.interpolate(method='linear', limit_direction='both')
    elif method == 'ffill':
        return df.ffill()
    elif method == 'bfill':
        return df.bfill()
    else:
        raise ValueError(f"Méthode {method} non reconnue")

# Étape 4: Détection et gestion des anomalies - Inspiré du package anomalize
def detect_and_handle_anomalies(df, threshold=3):
    """
    Détecte les anomalies dans une série temporelle en utilisant la décomposition STL
    (saison, tendance, résidus), puis détecte les anomalies dans les résidus.
    """
    # Gérer les valeurs manquantes avant de procéder à la décomposition
    df['value'] = handle_missing_values(df['value'], method='mean')  # Imputer les valeurs manquantes
    
    # Décomposer la série temporelle en saisonnalité, tendance et erreur (résidus)
    decomposition = sm.tsa.seasonal_decompose(df['value'], model='additive', period=365)
    residuals = decomposition.resid  # Les résidus (erreurs)
    
    # Calculer les seuils pour les anomalies en utilisant l'IQR
    Q1 = np.percentile(residuals.dropna(), 25)
    Q3 = np.percentile(residuals.dropna(), 75)
    IQR = Q3 - Q1
    
    # Définir les bornes pour identifier les anomalies
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identifier les anomalies
    anomalies = (residuals < lower_bound) | (residuals > upper_bound)
    
    # Ajouter les anomalies au DataFrame
    df['is_anomaly'] = anomalies
    anomaly_count = anomalies.sum()

    return df, anomaly_count

# Fonction pour afficher les points manquants, normaux et anormaux par année
def display_missing_and_anomalies_by_year(df):
    """
    Affiche le total des points manquants, des points normaux et des points anormaux par année.
    """
    # Extraire l'année à partir des dates
    df['year'] = df['time'].dt.year
    
    # Calculer le nombre de valeurs manquantes pour chaque année
    missing_counts = df.groupby('year')['value'].apply(lambda x: x.isna().sum())
    
    # Calculer le nombre de valeurs anormales pour chaque année
    anomaly_counts = df.groupby('year')['is_anomaly'].sum()
    
    # Calculer le nombre de valeurs normales pour chaque année
    normal_counts = df.groupby('year')['is_anomaly'].apply(lambda x: (~x).sum())
    
    # Afficher les résultats
    result = pd.DataFrame({
        'Missing Points': missing_counts,
        'Normal Points': normal_counts,
        'Anomaly Points': anomaly_counts
    })
    
    print("\n### Totaux des points manquants, normaux et anormaux par année ###")
    print(result)
    
    return result

# Fonction pour afficher l'histogramme des points manquants et anormaux par année
def plot_histogram_missing_and_anomalies(df):
    """
    Affiche un histogramme représentant les points manquants et anormaux par année.
    """
    # Créer l'histogramme
    result = display_missing_and_anomalies_by_year(df)
    
    # Créer un histogramme avec des colonnes distinctes pour les points manquants et anormaux
    result[['Missing Points', 'Anomaly Points']].plot(kind='bar', figsize=(12, 6), color=['lightblue', 'salmon'])

    # Ajouter des titres et des labels
    plt.title("Histogramme des points manquants et anormaux par année")
    plt.xlabel("Année")
    plt.ylabel("Nombre de points")
    plt.legend(title="Type de point", labels=["Manquants", "Anormaux"])
    plt.tight_layout()
    plt.show()

# Implémentation complète
file_path = 'C:/Users/Madjid/Desktop/Mes Cours/Mémoire/Codes/AEP_hourly.csv'  # Remplacer par le chemin du dataset
time_column = 'Datetime'  # Remplacer par le nom de la colonne temporelle
value_column = 'AEP_MW'  # Remplacer par le nom de la colonne des valeurs

# Pipeline
original_data = load_data(file_path, time_column, value_column)
structured_data = fix_structure(original_data)

# Simuler les données manquantes
missing_data = introduce_missing_values(structured_data, missing_percent=10)

# Comparer les méthodes d'imputation
imputed_data, imputation_results = impute_missing_values_testbench(missing_data)

# Détecter et corriger les anomalies (inspiré d'anomalize)
cleaned_data, anomaly_count = detect_and_handle_anomalies(structured_data)

# Afficher les totaux des points manquants, normaux et anormaux par année
display_missing_and_anomalies_by_year(cleaned_data)

# Afficher l'histogramme des points manquants et anormaux par année
plot_histogram_missing_and_anomalies(cleaned_data)
