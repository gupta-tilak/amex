import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def create_interaction_features(df):
    """
    Create interaction features between customer behavior and offer characteristics
    """
    print("Creating interaction features...")
    
    # Customer-Offer Interaction Features
    interaction_features = []
    
    # CTR-based interactions
    if 'f28' in df.columns and 'f29' in df.columns:
        df['ctr_oet_offer'] = df['f29'] / (df['f28'] + 1e-8)
        interaction_features.append('ctr_oet_offer')
    
    if 'f30' in df.columns and 'f31' in df.columns:
        df['ctr_merchant_offer'] = df['f31'] / (df['f30'] + 1e-8)
        interaction_features.append('ctr_merchant_offer')
    
    # Spending pattern interactions
    spending_features = [f'f{i}' for i in range(152, 174)]  # Last 30 days spending
    if all(col in df.columns for col in spending_features):
        df['total_spending_30d'] = df[spending_features].sum(axis=1)
        interaction_features.append('total_spending_30d')
    
    # Offer value interactions
    if all(col in df.columns for col in ['f217', 'f219']):
        df['offer_value_ratio'] = df['f219'] / (df['f217'] + 1e-8)
        interaction_features.append('offer_value_ratio')
    
    print(f"Created {len(interaction_features)} interaction features")
    return df, interaction_features

def create_temporal_features(df):
    """
    Create temporal features from timestamps
    """
    print("Creating temporal features...")
    
    temporal_features = []
    
    if 'id4' in df.columns:
        df['hour'] = pd.to_datetime(df['id4']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['id4']).dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        temporal_features.extend(['hour', 'day_of_week', 'is_weekend', 'is_business_hours'])
    
    print(f"Created {len(temporal_features)} temporal features")
    return df, temporal_features

def create_aggregated_features(df):
    """
    Create customer-level aggregated features
    """
    print("Creating aggregated features...")
    
    aggregated_features = []
    
    # Customer activity aggregations
    customer_aggs = df.groupby('id2').agg({
        'f28': ['mean', 'sum', 'std'],  # Impressions
        'f29': ['mean', 'sum', 'std'],  # Clicks
        'f217': ['mean', 'min', 'max'],  # Min spend
        'f219': ['mean', 'min', 'max']   # Discount value
    }).reset_index()
    
    # Flatten column names
    customer_aggs.columns = ['id2'] + [f'customer_{col[0]}_{col[1]}' for col in customer_aggs.columns[1:]]
    
    # Merge back to main dataset
    df = df.merge(customer_aggs, on='id2', how='left')
    
    aggregated_features = [col for col in customer_aggs.columns if col != 'id2']
    
    print(f"Created {len(aggregated_features)} aggregated features")
    return df, aggregated_features

def feature_selection_by_importance(df, target_col='y', top_k=100):
    """
    Select top features based on mutual information
    """
    print("Performing feature selection...")
    
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder
    
    # Prepare features for selection
    feature_cols = [col for col in df.columns if col.startswith('f') or col in ['hour', 'day_of_week', 'is_weekend', 'is_business_hours']]
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].copy()
    y = df[target_col] if target_col in df.columns else None
    
    if y is None:
        print("Target column not found, returning all features")
        return df, feature_cols
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Calculate mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Get top features
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': mi_scores
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance.head(top_k)['feature'].tolist()
    
    print(f"Selected top {len(top_features)} features")
    print("Top 10 features:")
    print(feature_importance.head(10))
    
    return df, top_features

def create_full_feature_set_advanced(df):
    """
    Create comprehensive feature set with advanced engineering
    """
    print("Starting advanced feature engineering...")
    
    # Create interaction features
    df, interaction_features = create_interaction_features(df)
    
    # Create temporal features
    df, temporal_features = create_temporal_features(df)
    
    # Create aggregated features
    df, aggregated_features = create_aggregated_features(df)
    
    # Feature selection
    df, selected_features = feature_selection_by_importance(df)
    
    print(f"Advanced feature engineering completed.")
    print(f"Total features created: {len(interaction_features + temporal_features + aggregated_features)}")
    print(f"Selected features: {len(selected_features)}")
    
    return df, selected_features
