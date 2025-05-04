import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

def plot_correlation_matrix(df):
    """
    Plot a correlation matrix for the features in the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure containing the correlation matrix.
    """
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title="Feature Correlation Matrix"
    )
    
    fig.update_layout(
        width=800,
        height=800
    )
    
    return fig

def plot_feature_importance(importance_df):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    importance_df : pandas.DataFrame
        DataFrame containing feature importance values.
        Must have columns 'Feature' and 'Importance'.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure containing the feature importance plot.
    """
    # Sort by importance
    df_sorted = importance_df.sort_values('Importance')
    
    # Create horizontal bar chart
    fig = px.bar(
        df_sorted,
        y='Feature',
        x='Importance',
        orientation='h',
        title="Feature Importance",
        color='Importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(400, len(df_sorted) * 20),  # Adjust height based on number of features
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def plot_predictions(y_true, y_pred):
    """
    Plot actual vs predicted values.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Actual values.
    y_pred : numpy.ndarray
        Predicted values.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure containing the actual vs predicted plot.
    """
    # Create DataFrame
    df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    
    # Calculate perfect prediction line limits
    min_val = min(df['Actual'].min(), df['Predicted'].min())
    max_val = max(df['Actual'].max(), df['Predicted'].max())
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='Actual',
        y='Predicted',
        title="Actual vs Predicted Values",
        opacity=0.6
    )
    
    # Add perfect prediction line
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        )
    )
    
    fig.update_layout(
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        height=600,
        width=800
    )
    
    return fig

def plot_clusters(X, cluster_labels):
    """
    Plot clusters using PCA for dimensionality reduction if needed.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix.
    cluster_labels : numpy.ndarray
        Cluster assignments.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure containing the cluster plot.
    """
    # Determine dimensionality
    n_dims = X.shape[1]
    
    # If more than 2 dimensions, use PCA to reduce to 2D for visualization
    if n_dims > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create DataFrame with PCA results
        df = pd.DataFrame({
            'PCA1': X_pca[:, 0],
            'PCA2': X_pca[:, 1],
            'Cluster': cluster_labels
        })
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x='PCA1',
            y='PCA2',
            color='Cluster',
            title="Cluster Visualization (PCA)",
            color_continuous_scale=px.colors.qualitative.G10
        )
    else:
        # Create DataFrame with original features
        if n_dims == 1:
            df = pd.DataFrame({
                'Feature1': X[:, 0],
                'Feature2': np.zeros(X.shape[0]),  # Add dummy second dimension
                'Cluster': cluster_labels
            })
        else:  # n_dims == 2
            df = pd.DataFrame({
                'Feature1': X[:, 0],
                'Feature2': X[:, 1],
                'Cluster': cluster_labels
            })
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x='Feature1',
            y='Feature2',
            color='Cluster',
            title="Cluster Visualization",
            color_continuous_scale=px.colors.qualitative.G10
        )
    
    fig.update_layout(
        height=600,
        width=800
    )
    
    return fig

def plot_roc_curve(fpr, tpr, auc_value):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    fpr : numpy.ndarray
        False positive rates.
    tpr : numpy.ndarray
        True positive rates.
    auc_value : float
        Area under the curve.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure containing the ROC curve.
    """
    # Create ROC curve
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr, 
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_value:.3f})',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add diagonal line (random classifier)
    fig.add_trace(
        go.Scatter(
            x=[0, 1], 
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        )
    )
    
    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=700,
        height=500,
        showlegend=True
    )
    
    return fig

def plot_learning_curve(train_sizes, train_scores, test_scores):
    """
    Plot learning curve.
    
    Parameters:
    -----------
    train_sizes : numpy.ndarray
        The training set sizes.
    train_scores : numpy.ndarray
        The training scores for each training size.
    test_scores : numpy.ndarray
        The test scores for each training size.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure containing the learning curve.
    """
    # Calculate mean and std for train/test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Create learning curve
    fig = go.Figure()
    
    # Add training score
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines+markers',
            name='Training Score',
            line=dict(color='blue'),
            marker=dict(color='blue', size=8)
        )
    )
    
    # Add training score error band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.1)',
            line=dict(color='rgba(0, 0, 255, 0)'),
            showlegend=False
        )
    )
    
    # Add test score
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=test_mean,
            mode='lines+markers',
            name='Cross-validation Score',
            line=dict(color='green'),
            marker=dict(color='green', size=8)
        )
    )
    
    # Add test score error band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([test_mean + test_std, (test_mean - test_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.1)',
            line=dict(color='rgba(0, 255, 0, 0)'),
            showlegend=False
        )
    )
    
    fig.update_layout(
        title="Learning Curve",
        xaxis_title="Training Examples",
        yaxis_title="Score",
        width=700,
        height=500
    )
    
    return fig

def plot_residuals(y_true, y_pred):
    """
    Plot residuals.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Actual values.
    y_pred : numpy.ndarray
        Predicted values.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure containing the residual plots.
    """
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Create subplots: residuals vs predicted and residual distribution
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Residuals vs Predicted", "Residual Distribution")
    )
    
    # Add residuals vs predicted scatter plot
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(color='blue', opacity=0.6),
            name="Residuals"
        ),
        row=1, col=1
    )
    
    # Add horizontal line at y=0
    fig.add_trace(
        go.Scatter(
            x=[min(y_pred), max(y_pred)],
            y=[0, 0],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name="Zero Line"
        ),
        row=1, col=1
    )
    
    # Add residual histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            marker=dict(color='blue', opacity=0.6),
            name="Residual Distribution"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Residual Analysis",
        height=400,
        width=900,
        showlegend=False
    )
    
    # Update x and y axis labels
    fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Residual Value", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    return fig
