import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Import utility functions (to be created)
# from utils.data_utils import load_dataset, apply_feature_engineering
# from utils.model_utils import get_model, train_model
# from utils.visualization_utils import plot_decision_boundary, plot_metrics
# from utils.validation_utils import perform_cross_validation

def main():
    st.set_page_config(
        page_title="VizClassify - Classification Algorithm Visualizer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🔬 VizClassify - Classification Algorithm Visualizer")
    st.markdown("Interactive tool for visualizing and understanding classification algorithms")
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Sidebar for main controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # 1. Dataset Selection
        st.subheader("📊 Dataset Selection")
        dataset_name = st.selectbox(
            "Choose a dataset:",
            ["Iris", "Breast Cancer", "Wine"],
            help="Select from scikit-learn's built-in datasets"
        )
        
        # Load selected dataset
        X, y, feature_names, target_names = load_selected_dataset(dataset_name)
        
        # 2. Feature Selection & Engineering
        st.subheader("🔧 Feature Engineering")
        
        # Feature selection method
        feature_method = st.radio(
            "Feature selection method:",
            ["Manual", "PCA (Automatic)"]
        )
        
        if feature_method == "Manual":
            # Manual feature selection
            selected_features = st.multiselect(
                "Select features:",
                feature_names,
                default=feature_names[:2],
                help="Choose features for training"
            )
            
            # Show live count and warning
            n_features = len(selected_features)
            st.write(f"**Features selected:** {n_features}")
            if n_features > 2:
                st.warning("⚠️ More than 2 features selected. 2D visualization not available.")
            elif n_features < 2:
                st.error("❌ At least 2 features required.")
        
        else:  # PCA
            n_components = st.slider(
                "Number of PCA components:",
                min_value=2,
                max_value=min(len(feature_names), 10),
                value=2,
                help="Dimensionality reduction via PCA"
            )
            if n_components > 2:
                st.warning("⚠️ More than 2 components selected. 2D visualization not available.")
        
        # Standardization option
        standardize = st.checkbox(
            "Standardize features",
            value=True,
            help="Apply StandardScaler to features"
        )
        
        # 3. Cross-validation Strategy
        st.subheader("🔄 Cross-validation")
        cv_method = st.selectbox(
            "CV method:",
            ["Stratified K-Fold", "K-Fold"]
        )
        
        k_folds = st.number_input(
            "Number of folds (k):",
            min_value=2,
            max_value=20,
            value=5,
            help="Number of cross-validation folds"
        )
        
        # 4. Model Selection
        st.subheader("🤖 Classification Model")
        model_name = st.selectbox(
            "Choose model:",
            ["Logistic Regression", "SVM", "Kernel SVM", "Random Forest"]
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📈 Dataset Overview")
        
        # Display dataset info
        st.write(f"**Dataset:** {dataset_name}")
        st.write(f"**Samples:** {X.shape[0]}")
        st.write(f"**Features:** {X.shape[1]}")
        st.write(f"**Classes:** {len(target_names)}")
        
        # Show feature statistics
        if feature_method == "Manual" and selected_features:
            df_display = pd.DataFrame(X, columns=feature_names)[selected_features]
            st.write("**Selected features statistics:**")
            st.dataframe(df_display.describe())
        
        # 5. Hyperparameter Tuning Section
        st.subheader("🎛️ Hyperparameter Tuning")
        hyperparams = get_hyperparameter_controls(model_name)
    
    with col2:
        st.subheader("🎯 Model Training & Results")
        
        # Training button
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                # Prepare data
                X_processed, feature_labels = prepare_data(
                    X, feature_names, feature_method, 
                    selected_features if feature_method == "Manual" else None,
                    n_components if feature_method == "PCA" else None,
                    standardize
                )
                
                # Check if data is valid
                if X_processed is None:
                    st.error("❌ Please select valid features.")
                else:
                    # Train model
                    model = get_model(model_name, hyperparams)
                    
                    # Perform cross-validation
                    cv_scores = perform_cross_validation(
                        model, X_processed, y, cv_method, k_folds
                    )
                    
                    # Store results
                    st.session_state.results = {
                        'model': model,
                        'X_processed': X_processed,
                        'y': y,
                        'feature_labels': feature_labels,
                        'cv_scores': cv_scores,
                        'hyperparams': hyperparams,
                        'model_name': model_name,
                        'can_visualize': X_processed.shape[1] == 2
                    }
                    st.session_state.model_trained = True
        
        # Display results if available
        if st.session_state.model_trained and st.session_state.results:
            display_results(st.session_state.results)

def load_selected_dataset(dataset_name):
    """Load the selected dataset and return X, y, feature names, target names"""
    if dataset_name == "Iris":
        data = load_iris()
    elif dataset_name == "Breast Cancer":
        data = load_breast_cancer()
    elif dataset_name == "Wine":
        data = load_wine()
    
    return data.data, data.target, data.feature_names, data.target_names

def prepare_data(X, feature_names, method, selected_features, n_components, standardize):
    """Prepare data based on feature selection method and preprocessing options"""
    if method == "Manual":
        if not selected_features or len(selected_features) < 2:
            return None, None
        
        # Select features
        feature_indices = [list(feature_names).index(f) for f in selected_features]
        X_processed = X[:, feature_indices]
        feature_labels = selected_features
    
    else:  # PCA
        # Apply PCA
        if n_components is None:
            n_components = 2  # Default to 2 components if not specified
            
        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        pca = PCA(n_components=n_components)
        X_processed = pca.fit_transform(X_scaled)
        feature_labels = [f"PC{i+1}" for i in range(n_components)]
    
    # Apply standardization if requested (for manual selection)
    if method == "Manual" and standardize:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_processed)
    
    return X_processed, feature_labels

def get_hyperparameter_controls(model_name):
    """Create hyperparameter input controls based on selected model"""
    hyperparams = {}
    
    if model_name == "Logistic Regression":
        hyperparams['C'] = st.number_input("Regularization (C):", 0.01, 100.0, 1.0, step=0.01)
        hyperparams['max_iter'] = st.number_input("Max iterations:", 100, 2000, 1000, step=100)
    
    elif model_name == "SVM":
        hyperparams['C'] = st.number_input("Regularization (C):", 0.01, 100.0, 1.0, step=0.01)
        hyperparams['kernel'] = 'linear'  # Fixed for linear SVM
    
    elif model_name == "Kernel SVM":
        hyperparams['C'] = st.number_input("Regularization (C):", 0.01, 100.0, 1.0, step=0.01)
        hyperparams['kernel'] = st.selectbox("Kernel:", ["rbf", "poly", "sigmoid"])
        if hyperparams['kernel'] == 'rbf':
            hyperparams['gamma'] = st.selectbox("Gamma:", ["scale", "auto"])
        elif hyperparams['kernel'] == 'poly':
            hyperparams['degree'] = st.number_input("Polynomial degree:", 2, 8, 3)
    
    elif model_name == "Random Forest":
        hyperparams['n_estimators'] = st.number_input("Number of trees:", 10, 500, 100, step=10)
        hyperparams['max_depth'] = st.selectbox("Max depth:", [None, 3, 5, 10, 20])
        hyperparams['min_samples_split'] = st.number_input("Min samples split:", 2, 20, 2)
    
    return hyperparams

def get_model(model_name, hyperparams):
    """Create and return the selected model with hyperparameters"""
    if model_name == "Logistic Regression":
        return LogisticRegression(**hyperparams, random_state=42)
    elif model_name == "SVM":
        return SVC(**hyperparams, random_state=42)
    elif model_name == "Kernel SVM":
        return SVC(**hyperparams, random_state=42)
    elif model_name == "Random Forest":
        return RandomForestClassifier(**hyperparams, random_state=42)

def perform_cross_validation(model, X, y, cv_method, k_folds):
    """Perform cross-validation and return scores"""
    if cv_method == "Stratified K-Fold":
        cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores

def display_results(results):
    """Display model results including metrics and visualizations"""
    st.success("✅ Model trained successfully!")
    
    # Display CV scores
    cv_scores = results['cv_scores']
    st.write(f"**Cross-validation Accuracy:** {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
    
    # Calculate additional metrics on full dataset
    model = results['model']
    X_processed = results['X_processed']
    y = results['y']
    
    # Fit model on full data to get predictions for metrics
    model.fit(X_processed, y)
    y_pred = model.predict(X_processed)
    
    # Calculate and display additional metrics
    f1 = f1_score(y, y_pred, average='weighted')
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    accuracy = accuracy_score(y, y_pred)
    
    st.write("**Performance Metrics:**")
    st.write(f"- Accuracy: {accuracy:.3f}")
    st.write(f"- F1 Score: {f1:.3f}")
    st.write(f"- Precision: {precision:.3f}")
    st.write(f"- Recall: {recall:.3f}")
    
    # Display hyperparameters
    st.write("**Hyperparameters:**")
    for param, value in results['hyperparams'].items():
        st.write(f"- {param}: {value}")
    
    # Visualization section
    if results['can_visualize']:
        st.subheader("📊 2D Visualization")
        
        # Fit model on full data for visualization
        model = results['model']
        X_processed = results['X_processed']
        y = results['y']
        
        model.fit(X_processed, y)
        
        # Create decision boundary plot
        fig = create_decision_boundary_plot(model, X_processed, y, results['feature_labels'])
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("⚠️ 2D visualization not available (more than 2 features/components selected)")
        
        # Show feature importance if applicable
        if hasattr(results['model'], 'feature_importances_'):
            st.subheader("📊 Feature Importance")
            model = results['model']
            model.fit(results['X_processed'], results['y'])
            
            importance_df = pd.DataFrame({
                'Feature': results['feature_labels'],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig_imp, use_container_width=True)

def create_decision_boundary_plot(model, X, y, feature_labels):
    """Create a 2D decision boundary plot using Plotly"""
    h = 0.02  # Step size in mesh
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Create plot
    fig = go.Figure()
    
    # Add decision boundary
    fig.add_trace(go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=Z,
        showscale=False,
        opacity=0.3,
        name="Decision Boundary"
    ))
    
    # Add data points
    colors = px.colors.qualitative.Set1
    for i, class_val in enumerate(np.unique(y)):
        mask = y == class_val
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            marker=dict(color=colors[i], size=8),
            name=f'Class {class_val}'
        ))
    
    fig.update_layout(
        title="Decision Boundary Visualization",
        xaxis_title=feature_labels[0],
        yaxis_title=feature_labels[1],
        showlegend=True
    )
    
    return fig

if __name__ == "__main__":
    main()