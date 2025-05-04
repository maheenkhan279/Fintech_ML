import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    confusion_matrix, 
    classification_report,
    silhouette_score,
    calinski_harabasz_score,
    roc_curve, 
    auc
)

def train_model(X_train, y_train, model_type="Linear Regression", params=None):
    """
    Train a machine learning model on the training data.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Feature matrix for training.
    y_train : numpy.ndarray
        Target vector for training.
    model_type : str
        Type of model to train: 'Linear Regression', 'Logistic Regression', or 'K-Means Clustering'.
    params : dict
        Dictionary of parameters for the model.
        
    Returns:
    --------
    tuple
        (trained_model, training_results)
    """
    if X_train is None or y_train is None:
        st.error("No training data provided.")
        return None, None
    
    if params is None:
        params = {}
    
    try:
        if model_type == "Linear Regression":
            return train_linear_regression(X_train, y_train, params)
        
        elif model_type == "Logistic Regression":
            return train_logistic_regression(X_train, y_train, params)
        
        elif model_type == "K-Means Clustering":
            return train_kmeans_clustering(X_train, params)
        
        else:
            st.error(f"Unknown model type: {model_type}")
            return None, None
    
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None

def train_linear_regression(X_train, y_train, params):
    """
    Train a Linear Regression model.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Feature matrix for training.
    y_train : numpy.ndarray
        Target vector for training.
    params : dict
        Dictionary of parameters for the model.
        
    Returns:
    --------
    tuple
        (trained_model, training_results)
    """
    # Extract parameters
    fit_intercept = params.get('fit_intercept', True)
    
    # normalize parameter has been deprecated in scikit-learn
    # Create and train the model without the deprecated parameter
    model = LinearRegression(fit_intercept=fit_intercept)
    
    # If normalization is needed, it should be done in the preprocessing step
    model.fit(X_train, y_train)
    
    # Make predictions on training data
    y_train_pred = model.predict(X_train)
    
    # Calculate training metrics
    r2_train = r2_score(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    
    # Store training results
    training_results = {
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'r2_train': r2_train,
        'mse_train': mse_train,
        'mae_train': mae_train
    }
    
    return model, training_results

def train_logistic_regression(X_train, y_train, params):
    """
    Train a Logistic Regression model.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Feature matrix for training.
    y_train : numpy.ndarray
        Target vector for training.
    params : dict
        Dictionary of parameters for the model.
        
    Returns:
    --------
    tuple
        (trained_model, training_results)
    """
    # Extract parameters
    C = params.get('C', 1.0)
    max_iter = params.get('max_iter', 100)
    
    # Create and train the model
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions on training data
    y_train_pred = model.predict(X_train)
    
    # Calculate training metrics
    accuracy_train = accuracy_score(y_train, y_train_pred)
    
    # Multi-class case
    if len(np.unique(y_train)) > 2:
        precision_train = precision_score(y_train, y_train_pred, average='weighted')
        recall_train = recall_score(y_train, y_train_pred, average='weighted')
    else:
        precision_train = precision_score(y_train, y_train_pred)
        recall_train = recall_score(y_train, y_train_pred)
    
    # Store training results
    training_results = {
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'accuracy_train': accuracy_train,
        'precision_train': precision_train,
        'recall_train': recall_train
    }
    
    return model, training_results

def train_kmeans_clustering(X_train, params):
    """
    Train a K-Means Clustering model.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Feature matrix for training.
    params : dict
        Dictionary of parameters for the model.
        
    Returns:
    --------
    tuple
        (trained_model, training_results)
    """
    # Extract parameters
    n_clusters = params.get('n_clusters', 3)
    max_iter = params.get('max_iter', 300)
    
    # Create and train the model
    model = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
    cluster_labels = model.fit_predict(X_train)
    
    # Calculate metrics
    inertia = model.inertia_
    
    # Calculate silhouette score if there's more than one cluster and more than one sample
    silhouette = None
    if n_clusters > 1 and X_train.shape[0] > n_clusters:
        try:
            silhouette = silhouette_score(X_train, cluster_labels)
        except:
            silhouette = "N/A"  # If silhouette calculation fails
    
    # Store training results
    training_results = {
        'cluster_centers': model.cluster_centers_,
        'inertia': inertia,
        'silhouette': silhouette,
        'cluster_labels': cluster_labels
    }
    
    return model, training_results

def evaluate_model(model, X_test, y_test, model_type="Linear Regression"):
    """
    Evaluate a trained model on testing data.
    
    Parameters:
    -----------
    model : sklearn model
        Trained machine learning model.
    X_test : numpy.ndarray
        Feature matrix for testing.
    y_test : numpy.ndarray
        Target vector for testing.
    model_type : str
        Type of model: 'Linear Regression', 'Logistic Regression', or 'K-Means Clustering'.
        
    Returns:
    --------
    tuple
        (evaluation_metrics, predictions)
    """
    if model is None:
        st.error("No trained model provided.")
        return None, None
    
    try:
        if model_type == "Linear Regression":
            return evaluate_linear_regression(model, X_test, y_test)
        
        elif model_type == "Logistic Regression":
            return evaluate_logistic_regression(model, X_test, y_test)
        
        elif model_type == "K-Means Clustering":
            return evaluate_kmeans_clustering(model, X_test)
        
        else:
            st.error(f"Unknown model type: {model_type}")
            return None, None
    
    except Exception as e:
        st.error(f"Error evaluating model: {e}")
        return None, None

def evaluate_linear_regression(model, X_test, y_test):
    """
    Evaluate a Linear Regression model.
    
    Parameters:
    -----------
    model : LinearRegression
        Trained Linear Regression model.
    X_test : numpy.ndarray
        Feature matrix for testing.
    y_test : numpy.ndarray
        Target vector for testing.
        
    Returns:
    --------
    tuple
        (evaluation_metrics, predictions)
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Percent error
    percent_error = np.abs((y_test - y_pred) / y_test) * 100
    mean_percent_error = np.mean(percent_error)
    
    evaluation_metrics = {
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mean_percent_error': mean_percent_error
    }
    
    return evaluation_metrics, y_pred

def evaluate_logistic_regression(model, X_test, y_test):
    """
    Evaluate a Logistic Regression model.
    
    Parameters:
    -----------
    model : LogisticRegression
        Trained Logistic Regression model.
    X_test : numpy.ndarray
        Feature matrix for testing.
    y_test : numpy.ndarray
        Target vector for testing.
        
    Returns:
    --------
    tuple
        (evaluation_metrics, predictions)
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Handle multi-class case
    if len(np.unique(y_test)) > 2:
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        # For multi-class, pass y_score as an array with probability for each class
        y_proba = model.predict_proba(X_test)
        
        # For simplicity, we're not doing ROC curve for multi-class
        roc_curve_data = None
    else:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # ROC Curve data
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        roc_curve_data = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report as string
    report = classification_report(y_test, y_pred)
    
    evaluation_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    if roc_curve_data:
        evaluation_metrics['roc_curve'] = roc_curve_data
    
    return evaluation_metrics, y_pred

def evaluate_kmeans_clustering(model, X_test):
    """
    Evaluate a K-Means Clustering model.
    
    Parameters:
    -----------
    model : KMeans
        Trained K-Means model.
    X_test : numpy.ndarray
        Feature matrix for testing.
        
    Returns:
    --------
    tuple
        (evaluation_metrics, predictions)
    """
    # Make predictions (cluster assignments)
    cluster_labels = model.predict(X_test)
    
    # Calculate metrics
    inertia = model.inertia_
    
    # Calculate silhouette score if there's more than one cluster
    silhouette = None
    if model.n_clusters > 1 and X_test.shape[0] > model.n_clusters:
        try:
            silhouette = silhouette_score(X_test, cluster_labels)
        except:
            silhouette = "N/A"
    
    # Calculate Calinski-Harabasz index if there's more than one cluster
    calinski_harabasz = None
    if model.n_clusters > 1 and X_test.shape[0] > model.n_clusters:
        try:
            calinski_harabasz = calinski_harabasz_score(X_test, cluster_labels)
        except:
            calinski_harabasz = "N/A"
    
    evaluation_metrics = {
        'inertia': inertia,
        'silhouette': silhouette,
        'calinski_harabasz': calinski_harabasz
    }
    
    return evaluation_metrics, cluster_labels
