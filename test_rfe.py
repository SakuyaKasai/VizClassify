#!/usr/bin/env python3
"""
Test script to verify RFE implementation works correctly
"""

import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def test_rfe_implementation():
    """Test the RFE implementation logic"""
    print("Testing RFE implementation...")
    
    # Load test data
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    print(f"Original dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Feature names: {list(feature_names)}")
    
    # Test different estimators
    estimators = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', random_state=42)
    }
    
    n_features_to_select = 2
    
    for estimator_name, estimator in estimators.items():
        print(f"\n--- Testing with {estimator_name} ---")
        
        # Apply standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply RFE
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
        X_rfe = rfe.fit_transform(X_scaled, y)
        
        # Get selected features
        selected_mask = rfe.support_
        selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]
        
        print(f"Selected {X_rfe.shape[1]} features: {selected_features}")
        print(f"Feature ranking: {rfe.ranking_}")
        
        assert X_rfe.shape[1] == n_features_to_select, f"Expected {n_features_to_select} features, got {X_rfe.shape[1]}"
        assert len(selected_features) == n_features_to_select, f"Feature names mismatch"
    
    print("\n✅ All RFE tests passed!")

if __name__ == "__main__":
    test_rfe_implementation()