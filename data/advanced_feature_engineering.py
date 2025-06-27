import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

def safe_divide(numerator, denominator, fill_value=0):
    """Safe division that handles division by zero and missing values"""
    # Handle missing value indicators
    num_mask = (numerator == -999) | (numerator.isna())
    den_mask = (denominator == -999) | (denominator.isna()) | (denominator == 0)
    
    # Perform safe division
    result = np.where(den_mask | num_mask, fill_value, numerator / denominator)
    return result

def safe_datetime_conversion(series, feature_name):
    """Safe datetime conversion with error handling"""
    try:
        dt_series = pd.to_datetime(series, errors='coerce')
        if feature_name == 'hour':
            return dt_series.dt.hour.fillna(12)  # Default to noon
        elif feature_name == 'day_of_week':
            return dt_series.dt.dayofweek.fillna(1)  # Default to Tuesday
        else:
            return dt_series
    except Exception as e:
        print(f"Warning: Datetime conversion failed for {feature_name}: {e}")
        if feature_name == 'hour':
            return pd.Series([12] * len(series), index=series.index)
        elif feature_name == 'day_of_week':
            return pd.Series([1] * len(series), index=series.index)
        else:
            return series

def validate_data_quality(df, stage_name):
    """Comprehensive data quality validation"""
    print(f"\n=== Data Quality Check: {stage_name} ===")
    
    # Check for NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"ERROR: Found {nan_count} NaN values")
        nan_cols = df.columns[df.isna().any()].tolist()
        print(f"Columns with NaN: {nan_cols[:10]}...")  # Show first 10
        return False
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    if inf_count > 0:
        print(f"ERROR: Found {inf_count} infinite values")
        return False
    
    # Check for very large values that could cause numerical instability
    large_value_threshold = 1e10
    large_values = (df[numeric_cols].abs() > large_value_threshold).sum().sum()
    if large_values > 0:
        print(f"WARNING: Found {large_values} very large values (>{large_value_threshold})")
    
    # Memory usage check
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")
    
    print(f"✓ Data quality check passed for {stage_name}")
    return True

def create_interaction_features(df):
    """Create interaction features with safe operations"""
    print("Creating interaction features...")
    interaction_features = []
    
    # Safe CTR-based interactions
    if 'f28' in df.columns and 'f29' in df.columns:
        df['ctr_oet_offer'] = safe_divide(df['f29'], df['f28'], fill_value=0)
        interaction_features.append('ctr_oet_offer')
    
    if 'f30' in df.columns and 'f31' in df.columns:
        df['ctr_merchant_offer'] = safe_divide(df['f31'], df['f30'], fill_value=0)
        interaction_features.append('ctr_merchant_offer')
    
    # Safe spending pattern interactions
    spending_features = [f'f{i}' for i in range(152, 174) if f'f{i}' in df.columns]
    if len(spending_features) > 0:
        # Replace -999 with 0 for summation
        spending_data = df[spending_features].replace(-999, 0)
        df['total_spending_30d'] = spending_data.sum(axis=1)
        interaction_features.append('total_spending_30d')
    
    # Safe offer value interactions
    if 'f217' in df.columns and 'f219' in df.columns:
        df['offer_value_ratio'] = safe_divide(df['f219'], df['f217'], fill_value=0)
        interaction_features.append('offer_value_ratio')
    
    print(f"Created {len(interaction_features)} interaction features")
    return df, interaction_features

def create_temporal_features(df):
    """Create temporal features with safe datetime handling"""
    print("Creating temporal features...")
    temporal_features = []
    
    if 'id4' in df.columns:
        # Safe datetime conversion
        df['hour'] = safe_datetime_conversion(df['id4'], 'hour')
        df['day_of_week'] = safe_datetime_conversion(df['id4'], 'day_of_week')
        
        # Create derived features with safe operations
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        temporal_features.extend(['hour', 'day_of_week', 'is_weekend', 'is_business_hours'])
    
    print(f"Created {len(temporal_features)} temporal features")
    return df, temporal_features

def create_aggregated_features(df):
    """Create customer-level aggregated features with safe operations"""
    print("Creating aggregated features...")
    aggregated_features = []
    
    if 'id2' not in df.columns:
        print("Warning: id2 column not found, skipping aggregated features")
        return df, aggregated_features
    
    # Safe aggregation with missing value handling
    agg_columns = ['f28', 'f29', 'f217', 'f219']
    available_columns = [col for col in agg_columns if col in df.columns]
    
    if len(available_columns) > 0:
        # Replace -999 with NaN for proper aggregation, then fill with 0
        agg_data = df[['id2'] + available_columns].replace(-999, np.nan)
        
        customer_aggs = agg_data.groupby('id2').agg({
            col: ['mean', 'sum', 'std'] for col in available_columns
        }).reset_index()
        
        # Flatten column names
        customer_aggs.columns = ['id2'] + [f'customer_{col[0]}_{col[1]}' for col in customer_aggs.columns[1:]]
        
        # Fill NaN values in aggregated features
        numeric_agg_cols = customer_aggs.select_dtypes(include=[np.number]).columns
        customer_aggs[numeric_agg_cols] = customer_aggs[numeric_agg_cols].fillna(0)
        
        # Merge back to main dataset
        df = df.merge(customer_aggs, on='id2', how='left')
        aggregated_features = [col for col in customer_aggs.columns if col != 'id2']
    
    print(f"Created {len(aggregated_features)} aggregated features")
    return df, aggregated_features

def emergency_nan_cleanup(df):
    """Emergency cleanup for any remaining NaN values"""
    print("Performing emergency NaN cleanup...")
    
    # Check for NaN values
    nan_cols = df.columns[df.isna().any()].tolist()
    if len(nan_cols) > 0:
        print(f"Found NaN values in {len(nan_cols)} columns, cleaning...")
        
        for col in nan_cols:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # Use median for numerical columns, fallback to 0
                median_val = df[col].median()
                fill_val = median_val if pd.notna(median_val) else 0
                df[col] = df[col].fillna(fill_val)
            else:
                # Use mode for categorical columns, fallback to 'Unknown'
                mode_val = df[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
                df[col] = df[col].fillna(fill_val)
    
    # Final validation
    remaining_nans = df.isna().sum().sum()
    if remaining_nans > 0:
        print(f"ERROR: Still found {remaining_nans} NaN values after cleanup")
        # Force fill any remaining NaNs
        df = df.fillna(0)
    
    return df

def feature_selection_by_importance(df, target_col='y', top_k=100):
    """Select top features with comprehensive error handling"""
    print("Performing feature selection...")
    
    # Validate input data
    if not validate_data_quality(df, "Feature Selection Input"):
        print("Data quality issues detected, performing emergency cleanup...")
        df = emergency_nan_cleanup(df)
    
    # Prepare features for selection
    feature_cols = [col for col in df.columns if col.startswith('f') or 
                   col in ['hour', 'day_of_week', 'is_weekend', 'is_business_hours', 
                          'ctr_oet_offer', 'ctr_merchant_offer', 'total_spending_30d', 'offer_value_ratio']]
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    if len(feature_cols) == 0:
        print("No feature columns found, returning empty selection")
        return df, []
    
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
    
    # Final data validation before mutual information
    X = emergency_nan_cleanup(X)
    
    # Ensure no infinite values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], 0)
    
    try:
        # Calculate mutual information with error handling
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
        
    except Exception as e:
        print(f"Feature selection failed: {e}")
        print("Returning all available features")
        return df, feature_cols

def create_full_feature_set_advanced(df):
    """Create comprehensive feature set with robust error handling"""
    print("Starting advanced feature engineering...")
    
    # Initial data validation
    if not validate_data_quality(df, "Initial Input"):
        print("Input data has quality issues, performing initial cleanup...")
        df = emergency_nan_cleanup(df)
    
    # Create interaction features
    df, interaction_features = create_interaction_features(df)
    validate_data_quality(df, "After Interaction Features")
    
    # Create temporal features
    df, temporal_features = create_temporal_features(df)
    validate_data_quality(df, "After Temporal Features")
    
    # Create aggregated features
    df, aggregated_features = create_aggregated_features(df)
    validate_data_quality(df, "After Aggregated Features")
    
    # Emergency cleanup before feature selection
    df = emergency_nan_cleanup(df)
    
    # Feature selection with robust error handling
    df, selected_features = feature_selection_by_importance(df)
    
    # Final validation
    validate_data_quality(df, "Final Output")
    
    print(f"Advanced feature engineering completed.")
    print(f"Total features created: {len(interaction_features + temporal_features + aggregated_features)}")
    print(f"Selected features: {len(selected_features)}")
    
    return df, selected_features
