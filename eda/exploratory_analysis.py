import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Assumption: Data is already cleaned and passed as a DataFrame

def plot_target_distribution(df: pd.DataFrame, target_col: str = 'y'):
    """Plot the distribution of the target variable (assume binary classification)."""
    plt.figure(figsize=(6,4))
    sns.countplot(x=target_col, data=df)
    plt.title('Target Variable Distribution')
    plt.show()

def plot_missing_values(df: pd.DataFrame, max_features: int = 50):
    """
    Plot the percentage of missing values for each feature (top N by missingness).
    Assumption: -999 and 'Unknown' are treated as missing for numeric and object columns respectively.
    """
    missing = (df.isin([-999, 'Unknown']).sum() / len(df)).sort_values(ascending=False)
    missing = missing[missing > 0][:max_features]
    plt.figure(figsize=(10,6))
    missing.plot(kind='bar')
    plt.title('Top Features by Missing Value Percentage')
    plt.ylabel('Fraction Missing')
    plt.show()

def plot_feature_distributions(df: pd.DataFrame, features: list, target_col: str = 'y', bins: int = 30):
    """
    Plot histograms for selected features, colored by target variable.
    """
    for feature in features:
        plt.figure(figsize=(6,4))
        sns.histplot(data=df, x=feature, hue=target_col, bins=bins, kde=True, element='step')
        plt.title(f'Distribution of {feature} by Target')
        plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, features: list = None):
    """
    Plot a correlation heatmap for selected features (or all numeric if None).
    """
    if features is None:
        features = df.select_dtypes(include=['number']).columns.tolist()
    corr = df[features].corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.show()

def plot_new_feature_analysis(df: pd.DataFrame, target_col: str = 'y'):
    """
    Plot distributions and boxplots for new features from add_event and add_trans.
    Assumption: Features like event_count_user, total_trans_amt_user, merchant_category_match, recency_last_event, recency_last_trans exist.
    """
    new_features = [
        'event_count_user', 'event_count_offer', 'event_count_user_offer',
        'recency_last_event', 'total_trans_amt_user', 'avg_trans_amt_user',
        'trans_count_user', 'trans_count_before_offer', 'recency_last_trans',
        'merchant_category_match'
    ]
    for feature in new_features:
        if feature in df.columns:
            plt.figure(figsize=(6,4))
            sns.boxplot(x=target_col, y=feature, data=df)
            plt.title(f'{feature} by Target')
            plt.show()
            plt.figure(figsize=(6,4))
            sns.histplot(data=df, x=feature, hue=target_col, bins=30, kde=True, element='step')
            plt.title(f'Distribution of {feature} by Target')
            plt.show()

def plot_feature_importances(importances, feature_names, top_n=20):
    """
    Plot feature importances (for feature selection, not final model necessarily).
    Assumption: importances is a 1D array, feature_names is a list.
    """
    idx = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10,6))
    sns.barplot(x=np.array(feature_names)[idx], y=np.array(importances)[idx])
    plt.title('Top Feature Importances')
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.xticks(rotation=90)
    plt.show()
