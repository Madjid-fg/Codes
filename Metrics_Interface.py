import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
import matplotlib.pyplot as plt

# === Load Dataset ===
file_path = "c:/Users/Madjid/Desktop/Mes Cours/Mémoire/Codes/AEP_hourly.csv"
time_column = "Datetime"
value_column = "AEP_MW"

data = pd.read_csv(file_path)
data[time_column] = pd.to_datetime(data[time_column])
data = data.sort_values(by=time_column)

# === Document 1 Metrics ===
def calculate_document1_metrics(data):
    """Implements metrics from Document 1."""
    # Completeness
    completeness = data[value_column].notna().sum() / len(data)

    # Conformity
    conformity_names = 1  # Assuming variable names are correct
    conformity_format = 1  # Assuming formats are correct
    conformity = (conformity_names + conformity_format) / 2

    # Timeliness
    data['time_diff'] = data[time_column].diff().dt.total_seconds() / 3600
    timeliness = 1 - (data['time_diff'].dropna() > 1).sum() / len(data)

    # Uniqueness
    uniqueness = float(data[time_column].is_unique)  # Convert Boolean to float for visualization

    # Accuracy
    anomalies = ((data[value_column] < 1000) | (data[value_column] > 50000)).sum()
    accuracy = 1 - anomalies / len(data)

    return {
        "Completeness": completeness,
        "Conformity": conformity,
        "Timeliness": timeliness,
        "Uniqueness": uniqueness,
        "Accuracy": accuracy,
    }

# === Document 2 Metrics ===
def calculate_document2_metrics(data):
    """Implements metrics from Document 2."""
    # Completeness
    completeness = data[value_column].notna().sum() / len(data)

    # Consistency
    duplicates = data.duplicated(subset=[time_column]).sum()
    consistency = 1 - duplicates / len(data)

    # Anomalies
    anomalies = ((data[value_column] < 1000) | (data[value_column] > 50000)).sum()

    return {
        "Completeness": completeness,
        "Consistency": consistency,
        "Anomalies": anomalies,
    }

# === Document 3 Metrics ===
def calculate_document3_metrics(data):
    """Implements metrics from Document 3."""
    # Completeness
    completeness = data[value_column].notna().sum() / len(data)

    # Consistency
    duplicates = data.duplicated(subset=[time_column]).sum()
    consistency = 1 - duplicates / len(data)

    # Timeliness
    data['time_diff'] = data[time_column].diff().dt.total_seconds() / 3600
    timeliness = 1 - (data['time_diff'].dropna() > 1).sum() / len(data)

    # Validity
    validity = ((data[value_column] >= 1000) & (data[value_column] <= 50000)).sum() / len(data)

    return {
        "Completeness": completeness,
        "Consistency": consistency,
        "Timeliness": timeliness,
        "Validity": validity,
    }

# === Calculate Metrics for Each Document ===
metrics_doc1 = calculate_document1_metrics(data.copy())
metrics_doc2 = calculate_document2_metrics(data.copy())
metrics_doc3 = calculate_document3_metrics(data.copy())

# === Combine Metrics into a DataFrame for Visualization ===
combined_metrics = pd.DataFrame({
    "Streaming Time Series": metrics_doc1,
    "Cleanits": metrics_doc2,
    "TsQuality": metrics_doc3,
}).fillna(0)  # Ensure no missing values

# === Visualize Metrics ===
def visualize_metrics(metrics_df):
    """Plot metrics comparison with metrics on x-axis and documents as bars."""
    metrics_df = metrics_df.fillna(0)
    ax = metrics_df.plot(kind='bar', figsize=(12, 6), width=0.6)
    
    plt.title("Comparison of Metrics Across Documents", fontsize=16)
    plt.xlabel("Documents", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.legend(title="Documents", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

visualize_metrics(combined_metrics.T)
