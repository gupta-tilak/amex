#!/usr/bin/env python3
"""
Test script for the optimized pipeline with limited feature scope
"""

import pandas as pd
import numpy as np
from data.data_loader import load_all_data
from data.data_cleaning import clean_all_data_advanced

def test_optimized_pipeline():
    """Test the optimized pipeline with limited feature scope"""
    print("Testing optimized pipeline with limited feature scope...")
    
    try:
        # Load data
        print("1. Loading data...")
        data = load_all_data()
        print(f"   - Train data: {data['train'].shape}")
        print(f"   - Test data: {data['test'].shape}")
        print(f"   - Add event: {data['add_event'].shape}")
        print(f"   - Add trans: {data['add_trans'].shape}")
        print(f"   - Offer metadata: {data['offer_metadata'].shape}")
        
        # Test the optimized cleaning pipeline
        print("\n2. Running optimized cleaning pipeline...")
        cleaned_data = clean_all_data_advanced(data)
        
        print(f"\n3. Results:")
        print(f"   - Cleaned train: {cleaned_data['train'].shape}")
        print(f"   - Cleaned test: {cleaned_data['test'].shape}")
        
        # Check for priority features
        priority_features = ['f28', 'f29', 'f217', 'f219', 'f152', 'f157', 'f169']
        available_priority = [col for col in priority_features if col in cleaned_data['train'].columns]
        print(f"   - Available priority features: {len(available_priority)}/{len(priority_features)}")
        print(f"   - Priority features found: {available_priority}")
        
        # Check for customer segment column
        if 'customer_segment' in cleaned_data['train'].columns:
            print(f"   - Customer segments created: {cleaned_data['train']['customer_segment'].nunique()}")
        else:
            print("   - Customer segments: Not created")
        
        # Check for imputation models
        if 'imputation_models' in cleaned_data:
            models = cleaned_data['imputation_models']
            print(f"   - KMeans model: {'Created' if models.get('kmeans') else 'Not created'}")
            print(f"   - Scaler: {'Created' if models.get('scaler') else 'Not created'}")
            print(f"   - Clustering features: {len(models.get('clustering_features', []))}")
        
        # Check memory usage
        train_memory = cleaned_data['train'].memory_usage(deep=True).sum() / 1024**2
        test_memory = cleaned_data['test'].memory_usage(deep=True).sum() / 1024**2
        print(f"   - Train memory usage: {train_memory:.2f} MB")
        print(f"   - Test memory usage: {test_memory:.2f} MB")
        
        # Check for missing values in priority features
        print(f"\n4. Missing values in priority features:")
        for feature in available_priority:
            missing_count = (cleaned_data['train'][feature] == -999).sum()
            total_count = len(cleaned_data['train'])
            missing_pct = (missing_count / total_count) * 100
            print(f"   - {feature}: {missing_count}/{total_count} ({missing_pct:.2f}%)")
        
        print("\n✅ Optimized pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_optimized_pipeline() 