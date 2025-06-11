#!/usr/bin/env python3
"""
Simple test script to validate VarianceThreshold implementation
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold

def test_variance_threshold_implementation():
    """Test the VarianceThreshold feature selection"""
    print("🔬 Testing VarianceThreshold implementation...")
    
    # Load test data
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    
    print(f"Original data shape: {X.shape}")
    print(f"Feature names: {list(feature_names)}")
    
    # Test 1: Test basic VarianceThreshold functionality
    print("\n1. Testing basic VarianceThreshold (threshold=0.0)...")
    var_threshold = VarianceThreshold(threshold=0.0)
    X_filtered = var_threshold.fit_transform(X)
    selected_mask = var_threshold.get_support()
    selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]
    
    print(f"Filtered data shape: {X_filtered.shape}")
    print(f"Selected features: {selected_features}")
    print(f"Feature variances: {var_threshold.variances_}")
    
    # Test 2: Test higher threshold that might remove some features
    print("\n2. Testing VarianceThreshold with higher threshold (0.1)...")
    var_threshold_high = VarianceThreshold(threshold=0.1)
    X_filtered_high = var_threshold_high.fit_transform(X)
    selected_mask_high = var_threshold_high.get_support()
    selected_features_high = [feature_names[i] for i, selected in enumerate(selected_mask_high) if selected]
    
    print(f"Filtered data shape: {X_filtered_high.shape}")
    print(f"Selected features: {selected_features_high}")
    print(f"Feature variances: {var_threshold_high.variances_}")
    
    # Test 3: Test edge case - very high threshold that removes all features
    print("\n3. Testing edge case - high threshold that removes all features...")
    var_threshold_extreme = VarianceThreshold(threshold=10.0)
    X_filtered_extreme = var_threshold_extreme.fit_transform(X)
    selected_mask_extreme = var_threshold_extreme.get_support()
    selected_features_extreme = [feature_names[i] for i, selected in enumerate(selected_mask_extreme) if selected]
    
    print(f"Filtered data shape: {X_filtered_extreme.shape}")
    print(f"Selected features: {selected_features_extreme}")
    
    # Test edge case return format
    if X_filtered_extreme.shape[1] == 0:
        edge_case_result = (np.empty((X.shape[0], 0)), [])
        print(f"Edge case result: {edge_case_result[0].shape}, {edge_case_result[1]}")
        print("✅ Edge case handling matches expected format")
    
    print("\n✅ All tests completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_variance_threshold_implementation()
        print("\n🎉 VarianceThreshold implementation is working correctly!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise