import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def robust_datetime_conversion(series):
    """Robust datetime conversion with multiple fallback strategies"""
    try:
        # Try standard conversion first
        result = pd.to_datetime(series, errors='coerce')
        
        # If too many NaT values, try different formats
        if result.isna().sum() > len(series) * 0.5:
            # Try Unix timestamp conversion
            numeric_series = pd.to_numeric(series, errors='coerce')
            if not numeric_series.isna().all():
                result = pd.to_datetime(numeric_series, unit='s', errors='coerce')
        
        return result
    except Exception as e:
        print(f"Warning: Datetime conversion failed: {e}")
        return pd.to_datetime('2023-01-01')  # Default date

def safe_aggregation(group, column, agg_func):
    """Safe aggregation that handles missing values and edge cases"""
    try:
        # Filter out missing value indicators
        valid_data = group[group[column] != -999][column]
        if len(valid_data) == 0:
            return 0 if agg_func in ['mean', 'sum', 'std'] else 'Unknown'
        
        if agg_func == 'mean':
            return valid_data.mean()
        elif agg_func == 'sum':
            return valid_data.sum()
        elif agg_func == 'std':
            return valid_data.std() if len(valid_data) > 1 else 0
        elif agg_func == 'count':
            return len(valid_data)
        elif agg_func == 'nunique':
            return valid_data.nunique()
        elif agg_func == 'mode':
            mode_result = valid_data.mode()
            return mode_result.iloc[0] if len(mode_result) > 0 else 'Unknown'
        else:
            return 0
    except Exception as e:
        print(f"Warning: Aggregation failed for {column} with {agg_func}: {e}")
        return 0 if agg_func in ['mean', 'sum', 'std', 'count', 'nunique'] else 'Unknown'

def create_customer_behavioral_features(train_data, add_event, add_trans, offer_metadata):
    """Create comprehensive customer behavioral profiles with robust error handling"""
    print("Creating customer behavioral features...")
    
    try:
        # Customer event patterns with safe aggregation
        customer_event_features = pd.DataFrame({'id2': train_data['id2'].unique()})
        
        if not add_event.empty and 'id2' in add_event.columns:
            event_aggs = add_event.groupby('id2').agg({
                'id3': lambda x: safe_aggregation(add_event[add_event['id2'].isin([x.name])], 'id3', 'count'),
                'id4': lambda x: safe_aggregation(add_event[add_event['id2'].isin([x.name])], 'id4', 'count')
            }).reset_index()
            
            event_aggs.columns = ['id2', 'event_count', 'event_diversity']
            customer_event_features = customer_event_features.merge(event_aggs, on='id2', how='left')
        
        # Fill missing values with defaults
        customer_event_features['event_count'] = customer_event_features.get('event_count', 0).fillna(0)
        customer_event_features['event_diversity'] = customer_event_features.get('event_diversity', 0).fillna(0)
        customer_event_features['event_recency'] = 999  # Default recency
        customer_event_features['preferred_category'] = 'Unknown'
        
    except Exception as e:
        print(f"Warning: Event feature creation failed: {e}")
        customer_event_features = pd.DataFrame({
            'id2': train_data['id2'].unique(),
            'event_count': 0,
            'event_diversity': 0,
            'event_recency': 999,
            'preferred_category': 'Unknown'
        })
    
    try:
        # Customer transaction patterns with safe aggregation
        customer_trans_features = pd.DataFrame({'id2': train_data['id2'].unique()})
        
        if not add_trans.empty and 'id2' in add_trans.columns:
            # Use first available numeric column for transaction amounts
            amount_col = None
            for col in add_trans.columns:
                if col.startswith('f') and add_trans[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    amount_col = col
                    break
            
            if amount_col:
                trans_aggs = add_trans.groupby('id2').agg({
                    amount_col: ['mean', 'sum', 'std', 'count']
                }).reset_index()
                
                trans_aggs.columns = ['id2', 'avg_trans_amount', 'total_trans_amount', 'trans_amount_std', 'trans_count']
                customer_trans_features = customer_trans_features.merge(trans_aggs, on='id2', how='left')
        
        # Fill missing values with defaults
        for col in ['avg_trans_amount', 'total_trans_amount', 'trans_amount_std', 'trans_count']:
            customer_trans_features[col] = customer_trans_features.get(col, 0).fillna(0)
        
        customer_trans_features['preferred_merchant'] = 'Unknown'
        customer_trans_features['trans_recency'] = 999
        
    except Exception as e:
        print(f"Warning: Transaction feature creation failed: {e}")
        customer_trans_features = pd.DataFrame({
            'id2': train_data['id2'].unique(),
            'avg_trans_amount': 0,
            'total_trans_amount': 0,
            'trans_amount_std': 0,
            'trans_count': 0,
            'preferred_merchant': 'Unknown',
            'trans_recency': 999
        })
    
    try:
        # Offer-based features with safe aggregation
        offer_features = pd.DataFrame({'id3': train_data['id3'].unique()})
        
        if not offer_metadata.empty and 'id3' in offer_metadata.columns:
            # Find discount rate column
            discount_col = None
            for col in ['f376', 'discount_rate', 'rate']:
                if col in offer_metadata.columns:
                    discount_col = col
                    break
            
            if discount_col:
                offer_aggs = offer_metadata.groupby('id3').agg({
                    discount_col: 'mean'
                }).reset_index()
                offer_aggs.columns = ['id3', 'avg_discount_rate']
                offer_features = offer_features.merge(offer_aggs, on='id3', how='left')
        
        # Fill missing values with defaults
        offer_features['avg_discount_rate'] = offer_features.get('avg_discount_rate', 0.1).fillna(0.1)
        offer_features['offer_type_mode'] = 2
        offer_features['offer_category_mode'] = 'Unknown'
        
    except Exception as e:
        print(f"Warning: Offer feature creation failed: {e}")
        offer_features = pd.DataFrame({
            'id3': train_data['id3'].unique(),
            'avg_discount_rate': 0.1,
            'offer_type_mode': 2,
            'offer_category_mode': 'Unknown'
        })
    
    return customer_event_features, customer_trans_features, offer_features

def create_customer_segments_robust(train_data, customer_event_features, customer_trans_features):
    """Create customer segments with robust error handling"""
    print("Creating customer segments with robust clustering...")
    
    try:
        # Merge behavioral features with main dataset
        enriched_data = train_data.merge(customer_event_features, on='id2', how='left')
        enriched_data = enriched_data.merge(customer_trans_features, on='id2', how='left')
        
        # Fill missing behavioral features with defaults
        behavioral_columns = [
            'event_count', 'event_diversity', 'event_recency',
            'avg_trans_amount', 'total_trans_amount', 'trans_count', 'trans_recency'
        ]
        
        for col in behavioral_columns:
            if col in enriched_data.columns:
                enriched_data[col] = enriched_data[col].fillna(0)
            else:
                enriched_data[col] = 0
        
        # Prepare features for clustering
        clustering_features = behavioral_columns
        clustering_data = enriched_data[clustering_features].copy()
        
        # Check for sufficient variation
        variation_check = clustering_data.std().sum()
        if variation_check < 1e-6:
            print("Warning: Insufficient variation in behavioral data. Using simple segmentation.")
            # Create simple segments based on transaction count
            enriched_data['customer_segment'] = pd.cut(
                enriched_data['trans_count'], 
                bins=3, 
                labels=[0, 1, 2]
            ).fillna(0).astype(int)
            return enriched_data, None, None, clustering_features
        
        # Standardize features for clustering
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(clustering_data)
        
        # Use MiniBatchKMeans for efficiency
        n_clusters = min(5, max(3, len(scaled_features) // 50000))
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=min(1000, len(scaled_features) // 10),
            max_iter=100,
            n_init=3
        )
        
        customer_segments = kmeans.fit_predict(scaled_features)
        enriched_data['customer_segment'] = customer_segments
        
        print(f"Created {n_clusters} customer segments successfully")
        return enriched_data, kmeans, scaler, clustering_features
        
    except Exception as e:
        print(f"Warning: Clustering failed: {e}. Using random segmentation.")
        enriched_data['customer_segment'] = np.random.randint(0, 3, size=len(enriched_data))
        return enriched_data, None, None, []

def robust_imputation_strategy(data_with_offers, remaining_features):
    """Robust imputation with multiple fallback strategies"""
    print("Applying robust imputation strategy...")
    
    for feature in remaining_features[:30]:  # Limit for performance
        if feature not in data_with_offers.columns:
            continue
            
        try:
            missing_mask = (data_with_offers[feature] == -999) | (data_with_offers[feature].isna())
            missing_count = missing_mask.sum()
            
            if missing_count == 0:
                continue
                
            print(f"Imputing {missing_count} missing values in {feature}")
            
            # Strategy 1: Segment-based imputation
            if 'customer_segment' in data_with_offers.columns:
                for segment in data_with_offers['customer_segment'].unique():
                    segment_mask = (data_with_offers['customer_segment'] == segment) & missing_mask
                    if segment_mask.sum() > 0:
                        segment_data = data_with_offers[
                            (data_with_offers['customer_segment'] == segment) & 
                            (~missing_mask)
                        ][feature]
                        
                        if len(segment_data) > 0:
                            fill_value = segment_data.median()
                            if pd.notna(fill_value):
                                data_with_offers.loc[segment_mask, feature] = fill_value
                                missing_mask = missing_mask & (~segment_mask)
            
            # Strategy 2: Global median for remaining values
            remaining_missing = missing_mask.sum()
            if remaining_missing > 0:
                valid_data = data_with_offers[~missing_mask][feature]
                if len(valid_data) > 0:
                    global_fill = valid_data.median()
                    if pd.notna(global_fill):
                        data_with_offers.loc[missing_mask, feature] = global_fill
                    else:
                        data_with_offers.loc[missing_mask, feature] = 0
                else:
                    data_with_offers.loc[missing_mask, feature] = 0
                    
        except Exception as e:
            print(f"Warning: Imputation failed for {feature}: {e}")
            # Emergency fallback
            missing_mask = (data_with_offers[feature] == -999) | (data_with_offers[feature].isna())
            data_with_offers.loc[missing_mask, feature] = 0
    
    return data_with_offers

def advanced_segment_based_imputation_robust(data, offer_features):
    """Robust imputation strategy with comprehensive error handling"""
    print("Performing robust advanced segment-based imputation...")
    
    try:
        # Merge offer features safely
        data_with_offers = data.merge(offer_features, on='id3', how='left')
    except Exception as e:
        print(f"Warning: Offer merge failed: {e}")
        data_with_offers = data.copy()
        # Add default offer features
        data_with_offers['avg_discount_rate'] = 0.1
        data_with_offers['offer_type_mode'] = 2
        data_with_offers['offer_category_mode'] = 'Unknown'
    
    # Define feature groups for targeted imputation
    offer_related_features = [col for col in ['f217', 'f219', 'f220', 'f221', 'f222'] 
                             if col in data_with_offers.columns]
    behavioral_features = [col for col in ['f28', 'f29', 'f30', 'f31'] 
                          if col in data_with_offers.columns]
    
    # Get remaining numerical features (limited for performance)
    numerical_features = data_with_offers.select_dtypes(include=[np.number]).columns.tolist()
    remaining_features = [col for col in numerical_features
                         if col.startswith('f') and 
                         col not in offer_related_features + behavioral_features][:40]
    
    # Phase 1: Offer-based imputation
    print("Phase 1: Offer-based imputation...")
    try:
        for feature in offer_related_features:
            if 'offer_type_mode' in data_with_offers.columns:
                for offer_type in data_with_offers['offer_type_mode'].unique():
                    if pd.notna(offer_type):
                        type_mask = (data_with_offers['offer_type_mode'] == offer_type)
                        missing_mask = (data_with_offers[feature] == -999)
                        target_mask = type_mask & missing_mask
                        
                        if target_mask.sum() > 0:
                            source_data = data_with_offers[type_mask & (~missing_mask)][feature]
                            if len(source_data) > 0:
                                fill_value = source_data.median()
                                if pd.notna(fill_value):
                                    data_with_offers.loc[target_mask, feature] = fill_value
    except Exception as e:
        print(f"Warning: Offer-based imputation failed: {e}")
    
    # Phase 2: Behavioral features imputation
    print("Phase 2: Behavioral features imputation...")
    try:
        for feature in behavioral_features:
            missing_mask = (data_with_offers[feature] == -999)
            if missing_mask.sum() > 0:
                valid_data = data_with_offers[~missing_mask][feature]
                if len(valid_data) > 0:
                    fill_value = valid_data.median()
                    if pd.notna(fill_value):
                        data_with_offers.loc[missing_mask, feature] = fill_value
                    else:
                        data_with_offers.loc[missing_mask, feature] = 0
                else:
                    data_with_offers.loc[missing_mask, feature] = 0
    except Exception as e:
        print(f"Warning: Behavioral imputation failed: {e}")
    
    # Phase 3: Remaining features with robust strategy
    print("Phase 3: Robust imputation for remaining features...")
    data_with_offers = robust_imputation_strategy(data_with_offers, remaining_features)
    
    # Phase 4: Categorical features
    print("Phase 4: Categorical feature imputation...")
    try:
        categorical_features = data_with_offers.select_dtypes(include=['object']).columns.tolist()
        for feature in categorical_features:
            if feature in data_with_offers.columns:
                missing_mask = (data_with_offers[feature] == 'Unknown') | (data_with_offers[feature].isna())
                if missing_mask.sum() > 0:
                    valid_data = data_with_offers[~missing_mask][feature]
                    if len(valid_data) > 0:
                        mode_val = valid_data.mode()
                        fill_value = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
                        data_with_offers.loc[missing_mask, feature] = fill_value
    except Exception as e:
        print(f"Warning: Categorical imputation failed: {e}")
    
    return data_with_offers

def clean_train_data_advanced_robust(df, add_event, add_trans, offer_metadata):
    """Robust advanced cleaning with comprehensive error handling"""
    print("Starting robust advanced data cleaning...")
    
    try:
        df = df.copy()
        
        # Safe datetime conversion
        if 'id4' in df.columns:
            df['id4'] = robust_datetime_conversion(df['id4'])
        if 'id5' in df.columns:
            df['id5'] = robust_datetime_conversion(df['id5'])
        
        # Remove duplicates safely
        initial_shape = df.shape
        df = df.drop_duplicates()
        print(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")
        
        # Memory optimization
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype(np.float32)
        
        # Safe missing value handling
        df[numeric_cols] = df[numeric_cols].fillna(-999)
        object_cols = df.select_dtypes(include=['object']).columns
        df[object_cols] = df[object_cols].fillna('Unknown')
        
        # Create behavioral features with error handling
        customer_event_features, customer_trans_features, offer_features = create_customer_behavioral_features(
            df, add_event, add_trans, offer_metadata
        )
        
        # Create customer segments with robust clustering
        enriched_data, kmeans_model, scaler, clustering_features = create_customer_segments_robust(
            df, customer_event_features, customer_trans_features
        )
        
        # Perform robust advanced imputation
        final_data = advanced_segment_based_imputation_robust(enriched_data, offer_features)
        
        # Clean up temporary columns
        columns_to_remove = [
            'event_count', 'event_diversity', 'event_recency', 'preferred_category',
            'avg_trans_amount', 'total_trans_amount', 'trans_amount_std',
            'trans_count', 'preferred_merchant', 'trans_recency',
            'avg_discount_rate', 'offer_type_mode', 'offer_category_mode'
        ]
        final_data = final_data.drop(columns=[col for col in columns_to_remove if col in final_data.columns], errors='ignore')
        
        # Final validation and cleanup
        numeric_cols = final_data.select_dtypes(include=[np.number]).columns
        final_data[numeric_cols] = final_data[numeric_cols].replace([np.inf, -np.inf], 0)
        final_data = final_data.fillna(0)  # Emergency fallback
        
        print(f"Robust advanced cleaning completed. Final shape: {final_data.shape}")
        return final_data, kmeans_model, scaler, clustering_features
        
    except Exception as e:
        print(f"Critical error in data cleaning: {e}")
        print("Falling back to basic cleaning...")
        
        # Emergency fallback to basic cleaning
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        object_cols = df.select_dtypes(include=['object']).columns
        df[object_cols] = df[object_cols].fillna('Unknown')
        
        return df, None, None, []

# Clean helper functions with error handling
def clean_offer_metadata(df):
    """Clean the offer metadata data with error handling."""
    try:
        df = df.copy()
        
        # Safe datetime conversion
        for col in ['id12', 'id13']:
            if col in df.columns:
                df[col] = robust_datetime_conversion(df[col])
        
        df = df.drop_duplicates()
        
        # Memory optimization
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].astype(np.float32)
            df[numeric_cols] = df[numeric_cols].fillna(-999)
        
        # Safe categorical filling
        object_cols = df.select_dtypes(include=['object']).columns
        df[object_cols] = df[object_cols].fillna('Unknown')
        
        return df
    except Exception as e:
        print(f"Warning: Offer metadata cleaning failed: {e}")
        return df.fillna(0) if not df.empty else df

def clean_add_event(df):
    """Clean the add_event data with error handling."""
    try:
        df = df.copy()
        
        if 'id4' in df.columns:
            df['id4'] = robust_datetime_conversion(df['id4'])
        
        df = df.drop_duplicates()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].astype(np.float32)
            df[numeric_cols] = df[numeric_cols].fillna(-999)
        
        return df
    except Exception as e:
        print(f"Warning: Add event cleaning failed: {e}")
        return df.fillna(0) if not df.empty else df

def clean_add_trans(df):
    """Clean the add_trans data with error handling."""
    try:
        df = df.copy()
        
        # Find datetime columns
        for col in df.columns:
            if 'f370' in col or 'date' in col.lower() or 'time' in col.lower():
                df[col] = robust_datetime_conversion(df[col])
        
        df = df.drop_duplicates()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].astype(np.float32)
            df[numeric_cols] = df[numeric_cols].fillna(-999)
        
        return df
    except Exception as e:
        print(f"Warning: Add trans cleaning failed: {e}")
        return df.fillna(0) if not df.empty else df

def clean_all_data_advanced(data):
    """Clean all loaded dataframes with robust error handling"""
    print("Starting robust advanced data cleaning pipeline...")
    
    try:
        # Clean additional datasets with error handling
        add_event_clean = clean_add_event(data.get('add_event', pd.DataFrame()))
        add_trans_clean = clean_add_trans(data.get('add_trans', pd.DataFrame()))
        offer_metadata_clean = clean_offer_metadata(data.get('offer_metadata', pd.DataFrame()))
        
        # Robust advanced cleaning for train data
        train_clean, kmeans_model, scaler, clustering_features = clean_train_data_advanced_robust(
            data['train'], add_event_clean, add_trans_clean, offer_metadata_clean
        )
        
        # Clean test data using the same robust strategy
        test_clean, _, _, _ = clean_train_data_advanced_robust(
            data['test'], add_event_clean, add_trans_clean, offer_metadata_clean
        )
        
        return {
            'train': train_clean,
            'test': test_clean,
            'offer_metadata': offer_metadata_clean,
            'add_event': add_event_clean,
            'add_trans': add_trans_clean,
            'data_dictionary': data.get('data_dictionary', pd.DataFrame()),
            'imputation_models': {
                'kmeans': kmeans_model,
                'scaler': scaler,
                'clustering_features': clustering_features
            }
        }
        
    except Exception as e:
        print(f"Critical error in pipeline: {e}")
        print("Falling back to emergency cleaning...")
        
        # Emergency fallback
        train_emergency = data['train'].fillna(0)
        test_emergency = data['test'].fillna(0)
        
        return {
            'train': train_emergency,
            'test': test_emergency,
            'offer_metadata': data.get('offer_metadata', pd.DataFrame()).fillna(0),
            'add_event': data.get('add_event', pd.DataFrame()).fillna(0),
            'add_trans': data.get('add_trans', pd.DataFrame()).fillna(0),
            'data_dictionary': data.get('data_dictionary', pd.DataFrame()),
            'imputation_models': {'kmeans': None, 'scaler': None, 'clustering_features': []}
        }
