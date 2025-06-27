import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def create_customer_behavioral_features(train_data, add_event, add_trans, offer_metadata):
    """
    Create comprehensive customer behavioral profiles from additional datasets
    """
    print("Creating customer behavioral features...")
    
    # Customer event patterns
    customer_event_features = add_event.groupby('id2').agg({
        'id3': ['count', 'nunique'],  # Event frequency and diversity
        'id4': lambda x: (pd.to_datetime('2023-11-05') - pd.to_datetime(x).max()).days if len(x) > 0 else 999,  # Recency
        'id6': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown'  # Preferred category
    }).reset_index()
    
    # Flatten column names
    customer_event_features.columns = ['id2', 'event_count', 'event_diversity', 'event_recency', 'preferred_category']
    
    # Customer transaction patterns
    customer_trans_features = add_trans.groupby('id2').agg({
        'f367': ['mean', 'sum', 'std', 'count'],  # Transaction amount statistics
        'f368': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown',  # Most common merchant type
        'f370': lambda x: (pd.to_datetime('2023-11-05') - pd.to_datetime(x).max()).days if len(x) > 0 else 999  # Transaction recency
    }).reset_index()
    
    # Flatten column names
    customer_trans_features.columns = ['id2', 'avg_trans_amount', 'total_trans_amount', 'trans_amount_std', 
                                     'trans_count', 'preferred_merchant', 'trans_recency']
    
    # Offer-based features
    offer_features = offer_metadata.groupby('id3').agg({
        'f376': 'mean',  # Average discount rate
        'f375': lambda x: x.mode().iloc[0] if not x.empty else 2,  # Most common offer type
        'id9': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown'  # Most common offer category
    }).reset_index()
    
    offer_features.columns = ['id3', 'avg_discount_rate', 'offer_type_mode', 'offer_category_mode']
    
    return customer_event_features, customer_trans_features, offer_features

def create_customer_segments(train_data, customer_event_features, customer_trans_features):
    """
    Create customer segments using behavioral clustering with proper error handling
    """
    print("Creating customer segments...")
    
    # Merge behavioral features with main dataset
    enriched_data = train_data.merge(customer_event_features, on='id2', how='left')
    enriched_data = enriched_data.merge(customer_trans_features, on='id2', how='left')
    
    # Fill missing behavioral features with defaults
    enriched_data['event_count'] = enriched_data['event_count'].fillna(0)
    enriched_data['event_diversity'] = enriched_data['event_diversity'].fillna(0)
    enriched_data['event_recency'] = enriched_data['event_recency'].fillna(999)
    enriched_data['avg_trans_amount'] = enriched_data['avg_trans_amount'].fillna(0)
    enriched_data['total_trans_amount'] = enriched_data['total_trans_amount'].fillna(0)
    enriched_data['trans_count'] = enriched_data['trans_count'].fillna(0)
    enriched_data['trans_recency'] = enriched_data['trans_recency'].fillna(999)
    
    # Prepare features for clustering
    clustering_features = [
        'event_count', 'event_diversity', 'event_recency',
        'avg_trans_amount', 'total_trans_amount', 'trans_count', 'trans_recency'
    ]
    
    # Handle any remaining NaN values
    clustering_data = enriched_data[clustering_features].fillna(0)
    
    # Check if we have enough variation in the data for clustering
    if clustering_data.nunique().sum() < 2:
        print("Warning: Not enough variation in data for meaningful clustering. Using single segment.")
        enriched_data['customer_segment'] = 0
        return enriched_data, None, None, clustering_features
    
    # Standardize features for clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_data)
    
    # Check if all features are identical (would result in single cluster)
    if np.all(scaled_features == scaled_features[0]):
        print("Warning: All customers have identical behavioral patterns. Using single segment.")
        enriched_data['customer_segment'] = 0
        return enriched_data, None, scaler, clustering_features
    
    # Determine optimal number of clusters using elbow method with error handling
    best_k = 5
    best_score = -1
    
    # Start from k=2 since silhouette score requires at least 2 clusters
    for k in range(2, min(11, len(scaled_features))):  # Ensure k doesn't exceed sample size
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scaled_features)
            
            # Check if we actually got k different clusters
            n_unique_labels = len(np.unique(labels))
            if n_unique_labels < 2:
                continue
                
            score = silhouette_score(scaled_features, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except ValueError as e:
            print(f"Warning: Could not compute silhouette score for k={k}: {e}")
            continue
    
    # If no valid clustering was found, use a simple approach
    if best_score == -1:
        print("Warning: Could not find optimal clustering. Using 3 clusters as default.")
        best_k = min(3, len(scaled_features) - 1)
    
    print(f"Optimal number of clusters: {best_k} (silhouette score: {best_score:.3f})")
    
    # Final clustering
    try:
        final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        customer_segments = final_kmeans.fit_predict(scaled_features)
        
        # Verify we got multiple clusters
        if len(np.unique(customer_segments)) < 2:
            print("Warning: Clustering resulted in single cluster. Using random segmentation.")
            customer_segments = np.random.randint(0, 3, size=len(scaled_features))
            
    except Exception as e:
        print(f"Warning: Clustering failed: {e}. Using random segmentation.")
        customer_segments = np.random.randint(0, 3, size=len(scaled_features))
        final_kmeans = None
    
    enriched_data['customer_segment'] = customer_segments
    
    return enriched_data, final_kmeans, scaler, clustering_features

def advanced_segment_based_imputation(data, offer_features):
    """
    Perform advanced imputation using customer segments and offer information
    """
    print("Performing advanced segment-based imputation...")
    
    # Merge offer features
    data_with_offers = data.merge(offer_features, on='id3', how='left')
    
    # Define feature groups for different imputation strategies
    offer_related_features = ['f217', 'f219', 'f220', 'f221', 'f222']  # Offer-specific features
    behavioral_features = ['f28', 'f29', 'f30', 'f31']  # Impression/click features
    transaction_features = [f'f{i}' for i in range(152, 198)]  # Transaction amount features
    
    # Get all numerical features
    numerical_features = data_with_offers.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features = [col for col in numerical_features if col.startswith('f') and col not in ['f367', 'f368', 'f370']]
    
    # Phase 1: Offer-based imputation for offer-related features
    print("Phase 1: Offer-based imputation...")
    for feature in offer_related_features:
        if feature in data_with_offers.columns:
            # Use offer type and category for imputation
            for offer_type in data_with_offers['offer_type_mode'].unique():
                if pd.notna(offer_type):
                    mask = (data_with_offers['offer_type_mode'] == offer_type) & (data_with_offers[feature] == -999)
                    if mask.sum() > 0:
                        fill_value = data_with_offers[
                            (data_with_offers['offer_type_mode'] == offer_type) & 
                            (data_with_offers[feature] != -999)
                        ][feature].median()
                        
                        if pd.notna(fill_value):
                            data_with_offers.loc[mask, feature] = fill_value
    
    # Phase 2: Segment-based imputation for behavioral features
    print("Phase 2: Segment-based imputation...")
    for feature in behavioral_features:
        if feature in data_with_offers.columns:
            # Check if we have valid segments
            if 'customer_segment' in data_with_offers.columns:
                segment_stats = data_with_offers[data_with_offers[feature] != -999].groupby('customer_segment')[feature].median()
                
                for segment in segment_stats.index:
                    mask = (data_with_offers['customer_segment'] == segment) & (data_with_offers[feature] == -999)
                    if mask.sum() > 0:
                        data_with_offers.loc[mask, feature] = segment_stats[segment]
            else:
                # Fallback to global median if no segments
                global_median = data_with_offers[data_with_offers[feature] != -999][feature].median()
                if pd.notna(global_median):
                    mask = data_with_offers[feature] == -999
                    data_with_offers.loc[mask, feature] = global_median
    
    # Phase 3: KNN imputation for remaining numerical features
    print("Phase 3: KNN imputation for remaining features...")
    remaining_features = [col for col in numerical_features 
                         if col not in offer_related_features + behavioral_features]
    
    if remaining_features:
        # Prepare data for KNN imputation
        knn_features = remaining_features[:50]  # Limit features for KNN to avoid memory issues
        if 'customer_segment' in data_with_offers.columns:
            knn_features.append('customer_segment')
        
        knn_data = data_with_offers[knn_features].copy()
        knn_data = knn_data.replace(-999, np.nan)
        
        # Use KNN imputation with reduced neighbors for large datasets
        n_neighbors = min(5, len(knn_data) // 10)
        if n_neighbors < 1:
            n_neighbors = 1
            
        try:
            knn_imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
            knn_imputed = knn_imputer.fit_transform(knn_data)
            
            # Update the original data
            for i, feature in enumerate(knn_features):
                if feature != 'customer_segment':
                    data_with_offers[feature] = knn_imputed[:, i]
        except Exception as e:
            print(f"Warning: KNN imputation failed: {e}. Using median imputation.")
            for feature in remaining_features:
                if feature in data_with_offers.columns:
                    median_val = data_with_offers[data_with_offers[feature] != -999][feature].median()
                    if pd.notna(median_val):
                        mask = data_with_offers[feature] == -999
                        data_with_offers.loc[mask, feature] = median_val
    
    # Phase 4: Handle categorical features
    print("Phase 4: Categorical feature imputation...")
    categorical_features = data_with_offers.select_dtypes(include=['object']).columns.tolist()
    categorical_features = [col for col in categorical_features if col not in ['preferred_category', 'preferred_merchant']]
    
    for feature in categorical_features:
        if feature in data_with_offers.columns:
            if 'customer_segment' in data_with_offers.columns:
                segment_modes = data_with_offers[data_with_offers[feature] != 'Unknown'].groupby('customer_segment')[feature].apply(
                    lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
                )
                
                for segment in segment_modes.index:
                    mask = (data_with_offers['customer_segment'] == segment) & (data_with_offers[feature] == 'Unknown')
                    if mask.sum() > 0:
                        data_with_offers.loc[mask, feature] = segment_modes[segment]
            else:
                # Fallback to global mode
                global_mode = data_with_offers[data_with_offers[feature] != 'Unknown'][feature].mode()
                if not global_mode.empty:
                    mask = data_with_offers[feature] == 'Unknown'
                    data_with_offers.loc[mask, feature] = global_mode.iloc[0]
    
    return data_with_offers

def clean_train_data_advanced(df, add_event, add_trans, offer_metadata):
    """
    Advanced cleaning with clustering-based imputation
    """
    print("Starting advanced data cleaning...")
    df = df.copy()
    
    # Convert date columns
    df['id4'] = pd.to_datetime(df['id4'], errors='coerce')
    df['id5'] = pd.to_datetime(df['id5'], errors='coerce')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Cast numeric columns to float32
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].astype(np.float32)
    
    # Replace NaN with -999 for numerical and 'Unknown' for categorical (temporarily)
    df[num_cols] = df[num_cols].fillna(-999)
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].fillna('Unknown')
    
    # Create behavioral features
    customer_event_features, customer_trans_features, offer_features = create_customer_behavioral_features(
        df, add_event, add_trans, offer_metadata
    )
    
    # Create customer segments
    enriched_data, kmeans_model, scaler, clustering_features = create_customer_segments(
        df, customer_event_features, customer_trans_features
    )
    
    # Perform advanced imputation
    final_data = advanced_segment_based_imputation(enriched_data, offer_features)
    
    # Remove temporary behavioral features that were added for clustering
    columns_to_remove = ['event_count', 'event_diversity', 'event_recency', 'preferred_category',
                        'avg_trans_amount', 'total_trans_amount', 'trans_amount_std', 
                        'trans_count', 'preferred_merchant', 'trans_recency',
                        'avg_discount_rate', 'offer_type_mode', 'offer_category_mode']
    
    final_data = final_data.drop(columns=[col for col in columns_to_remove if col in final_data.columns], errors='ignore')
    
    print(f"Advanced cleaning completed. Final shape: {final_data.shape}")
    
    return final_data, kmeans_model, scaler, clustering_features

# Keep the rest of your existing functions unchanged
def clean_offer_metadata(df):
    """Clean the offer metadata data."""
    df = df.copy()
    df['id12'] = pd.to_datetime(df['id12'], errors='coerce')
    df['id13'] = pd.to_datetime(df['id13'], errors='coerce')
    df = df.drop_duplicates()
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].astype(np.float32)
    df = df.fillna({'f376': -999, 'f377': -999, 'f378': 'Unknown', 'f374': 'Unknown'})
    df[num_cols] = df[num_cols].fillna(-999)
    
    return df

def clean_add_event(df):
    """Clean the add_event data."""
    df = df.copy()
    df['id4'] = pd.to_datetime(df['id4'], errors='coerce')
    df = df.drop_duplicates()
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].astype(np.float32)
    df[num_cols] = df[num_cols].fillna(-999)
    
    return df

def clean_add_trans(df):
    """Clean the add_trans data."""
    df = df.copy()
    df['f370'] = pd.to_datetime(df['f370'], errors='coerce')
    df = df.drop_duplicates()
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].astype(np.float32)
    df[num_cols] = df[num_cols].fillna(-999)
    
    return df

def clean_all_data_advanced(data):
    """
    Clean all loaded dataframes with advanced imputation strategy
    """
    # Clean additional datasets first
    add_event_clean = clean_add_event(data['add_event'])
    add_trans_clean = clean_add_trans(data['add_trans'])
    offer_metadata_clean = clean_offer_metadata(data['offer_metadata'])
    
    # Advanced cleaning for train data
    train_clean, kmeans_model, scaler, clustering_features = clean_train_data_advanced(
        data['train'], add_event_clean, add_trans_clean, offer_metadata_clean
    )
    
    # Clean test data using the same strategy
    test_clean, _, _, _ = clean_train_data_advanced(
        data['test'], add_event_clean, add_trans_clean, offer_metadata_clean
    )
    
    return {
        'train': train_clean,
        'test': test_clean,
        'offer_metadata': offer_metadata_clean,
        'add_event': add_event_clean,
        'add_trans': add_trans_clean,
        'data_dictionary': data['data_dictionary'],
        'imputation_models': {
            'kmeans': kmeans_model,
            'scaler': scaler,
            'clustering_features': clustering_features
        }
    }
