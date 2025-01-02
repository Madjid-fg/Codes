import pandas as pd
import numpy as np

# Charger le dataset
dataset = pd.read_csv("C:/Users/Madjid/Desktop/Mes Cours/Mémoire/Codes/AEP_hourly.csv")
dataset['Datetime'] = pd.to_datetime(dataset['Datetime'])  # Conversion de la colonne 'Datetime'

# Fonctions pour les métriques et leurs solutions

# 1. Conformité (Conformity)
def check_names(df, reference_names):
    correct_names = sum(df.columns == reference_names)
    return correct_names / len(reference_names)

def solve_names(df, reference_names):
    df.columns = reference_names
    return df

def check_formats(df, reference_formats):
    correct_formats = sum([df[col].dtype.name == ref for col, ref in zip(df.columns, reference_formats)])
    return correct_formats / len(reference_formats)

def solve_formats(df, reference_formats):
    for col, expected_format in zip(df.columns, reference_formats):
        if expected_format == "datetime64[ns]":
            df[col] = pd.to_datetime(df[col], errors='coerce')
        elif expected_format == "float64":
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# 2. Unicité temporelle (Time Uniqueness)
def time_uniqueness(time_column):
    unique_timestamps = len(time_column.unique())
    total_timestamps = len(time_column)
    return unique_timestamps / total_timestamps

def solve_time_uniqueness(df, time_column):
    return df.drop_duplicates(subset=[time_column])

# 3. Ponctualité (Timeliness)
def timeliness(time_column, max_time_diff):
    time_diffs = time_column.diff().dt.total_seconds().dropna()
    within_limit = sum(time_diffs <= max_time_diff)
    return within_limit / len(time_diffs)

def solve_timeliness(df, time_column, max_time_diff):
    time_index = pd.date_range(
        start=df[time_column].min(), 
        end=df[time_column].max(), 
        freq=f"{int(max_time_diff / 60)}min"
    )
    df = df.set_index(time_column).reindex(time_index).reset_index()
    df.rename(columns={"index": time_column}, inplace=True)
    return df

# 4. Complétude (Completeness)
def completeness(df):
    total_values = df.size
    known_values = df.notnull().sum().sum()
    return known_values / total_values

def solve_completeness(df):
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
    return df

# 5. Précision (Accuracy)
def range_accuracy(series, min_value, max_value):
    in_range = ((series >= min_value) & (series <= max_value)).sum()
    total_values = len(series)
    return in_range / total_values

def solve_range_accuracy(df, column, min_value, max_value):
    df[column] = np.clip(df[column], min_value, max_value)
    return df

# Application des métriques et solutions
reference_names = ["Datetime", "AEP_MW"]
reference_formats = ["datetime64[ns]", "float64"]

# Conformité
names_score = check_names(dataset, reference_names)
if names_score < 1:
    dataset = solve_names(dataset, reference_names)

formats_score = check_formats(dataset, reference_formats)
if formats_score < 1:
    dataset = solve_formats(dataset, reference_formats)

# Unicité temporelle
time_uniqueness_score = time_uniqueness(dataset['Datetime'])
if time_uniqueness_score < 1:
    dataset = solve_time_uniqueness(dataset, "Datetime")

# Ponctualité
timeliness_score = timeliness(dataset['Datetime'], max_time_diff=3600)
if timeliness_score < 1:
    dataset = solve_timeliness(dataset, "Datetime", max_time_diff=3600)

# Complétude
completeness_score = completeness(dataset)
if completeness_score < 1:
    dataset = solve_completeness(dataset)

# Précision
range_score = range_accuracy(dataset['AEP_MW'], min_value=0, max_value=30000)
if range_score < 1:
    dataset = solve_range_accuracy(dataset, "AEP_MW", min_value=0, max_value=30000)

# Résultats finaux
metrics = {
    "Names Conformity": names_score,
    "Formats Conformity": formats_score,
    "Time Uniqueness": time_uniqueness_score,
    "Timeliness": timeliness_score,
    "Completeness": completeness_score,
    "Range Accuracy": range_score,
}

print("Scores des Métriques de Qualité des Données (avant correction) :")
for metric, score in metrics.items():
    print(f"{metric}: {score:.2f}")

print("\nDataset corrigé :")
print(dataset.head())
