import streamlit as st

# Set page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="FinTech ML Pipeline",
    page_icon="💹",
    layout="wide"
)

import pandas as pd
import numpy as np
import time
import os
import plotly.express as px
from data_loader import load_kragle_dataset
from data_processor import preprocess_data, engineer_features, split_data, convert_text_to_numeric
from model_trainer import train_model, evaluate_model
from visualizer import plot_correlation_matrix, plot_feature_importance, plot_predictions, plot_clusters
from utils import display_notification, load_finance_gif, to_excel

# Load custom CSS
with open(os.path.join(".streamlit", "style.css")) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 38px;
        color: #0078ff;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px #00000030;
    }
    
    .step-header {
        font-size: 26px;
        color: #0078ff;
        margin-bottom: 10px;
        border-bottom: 2px solid #0078ff;
        padding-bottom: 5px;
    }
    
    .card {
        border-radius: 5px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .highlight-text {
        background: linear-gradient(120deg, rgba(132, 250, 176, 0.2), rgba(143, 211, 244, 0.2));
        padding: 0.2em 0.5em;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .model-equation {
        font-family: monospace;
        background-color: #f7f7f7;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #1976D2;
        overflow-x: auto;
    }
    
    .interpretation-card {
        background: linear-gradient(120deg, #f8fbff, #ffffff);
        border-left: 5px solid #1976D2;
        margin-bottom: 25px;
        padding: 15px;
        border-radius: 5px;
    }
    
    .step-icon {
        font-size: 24px;
        margin-right: 10px;
    }
    
    .stButton>button {
        background-color: #0078ff;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #666;
        font-size: 14px;
    }
    
    .highlight {
        background-color: rgba(0, 120, 255, 0.1);
        padding: 5px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Define the steps in the ML pipeline with icons
STEPS = [
    {"name": "Welcome", "icon": "👋"},
    {"name": "Data Loading", "icon": "📊"},
    {"name": "Data Preprocessing", "icon": "🧹"},
    {"name": "Feature Engineering", "icon": "⚙️"},
    {"name": "Model Selection", "icon": "🔍"},
    {"name": "Model Training", "icon": "🧠"},
    {"name": "Model Evaluation", "icon": "📏"},
    {"name": "Results Visualization", "icon": "📈"}
]

# Initialize session state variables if they don't exist
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'text_encoders' not in st.session_state:
    st.session_state.text_encoders = None
if 'preprocessing_done' not in st.session_state:
    st.session_state.preprocessing_done = False
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Data Overview"
if 'need_rerun' not in st.session_state:
    st.session_state.need_rerun = False
    
# Handle rerun request after callback 
# This fixes the "Calling st.rerun() within a callback is a no-op" warning
if st.session_state.need_rerun:
    st.session_state.need_rerun = False
    st.rerun()

def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step -= 1

def reset_pipeline():
    """Reset the entire ML pipeline."""
    # Set flag to trigger rerun after callback completes
    if 'need_rerun' not in st.session_state:
        st.session_state.need_rerun = False
    
    st.session_state.step = 0
    st.session_state.data = None
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.model = None
    st.session_state.predictions = None
    st.session_state.metrics = None
    st.session_state.features = None
    st.session_state.target = None
    st.session_state.processed_data = None
    st.session_state.feature_importance = None
    st.session_state.model_type = None
    st.session_state.text_encoders = None
    st.session_state.preprocessing_done = False
    st.session_state.active_tab = "Data Overview"
    st.session_state.need_rerun = True

# Define a function to display an interactive progress indicator
def display_progress_indicator():
    current_step = st.session_state.step
    total_steps = len(STEPS)
    
    # Progress bar styling with escaped HTML/CSS to avoid display issues
    progress_css = """
    <style>
    .progress-container {
        width: 100%;
        margin: 20px 0;
        position: relative;
    }
    .progress-bar-bg {
        height: 10px;
        background-color: #f0f0f0;
        border-radius: 10px;
        position: relative;
    }
    .progress-bar-fill {
        position: absolute;
        top: 0;
        left: 0;
        height: 10px;
        background: linear-gradient(90deg, #1E88E5, #4CAF50);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    .step-indicators {
        display: flex;
        justify-content: space-between;
        margin-top: -5px;
    }
    .step-indicator {
        width: 20px;
        height: 20px;
        background-color: #f0f0f0;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        z-index: 1;
        transition: all 0.3s ease;
        font-size: 12px;
    }
    .step-indicator.completed {
        background-color: #4CAF50;
        box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
    }
    .step-indicator.active {
        background-color: #1E88E5;
        box-shadow: 0 0 8px rgba(30, 136, 229, 0.7);
        transform: scale(1.2);
    }
    .step-label {
        position: absolute;
        top: 25px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 12px;
        white-space: nowrap;
        color: #888;
        font-weight: 500;
    }
    .step-label.active {
        color: #1E88E5;
        font-weight: 700;
    }
    .step-label.completed {
        color: #4CAF50;
    }
    </style>
    """
    st.markdown(progress_css, unsafe_allow_html=True)
    
    # Calculate progress percentage
    progress_percent = (current_step / (total_steps - 1)) * 100 if total_steps > 1 else 0
    
    # Create a cleaner approach without showing HTML elements directly
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width: {progress_percent}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for each step
    cols = st.columns(len(STEPS))
    
    # Add step indicators to columns instead of using raw HTML
    for i, (col, step) in enumerate(zip(cols, STEPS)):
        with col:
            if i < current_step:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="width: 30px; height: 30px; background-color: #4CAF50; border-radius: 50%; 
                         display: inline-flex; align-items: center; justify-content: center; color: white; 
                         margin: 0 auto; font-weight: bold; box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);">✓</div>
                    <div style="font-size: 11px; margin-top: 5px; color: #4CAF50; font-weight: 500;">{step['name']}</div>
                </div>
                """, unsafe_allow_html=True)
            elif i == current_step:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="width: 30px; height: 30px; background-color: #1E88E5; border-radius: 50%; 
                         display: inline-flex; align-items: center; justify-content: center; color: white; 
                         margin: 0 auto; font-weight: bold; box-shadow: 0 0 8px rgba(30, 136, 229, 0.7);
                         transform: scale(1.2);">{step['icon']}</div>
                    <div style="font-size: 11px; margin-top: 5px; color: #1E88E5; font-weight: 700;">{step['name']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="width: 30px; height: 30px; background-color: #f0f0f0; border-radius: 50%; 
                         display: inline-flex; align-items: center; justify-content: center; color: #888; 
                         margin: 0 auto; font-weight: bold;">{i+1}</div>
                    <div style="font-size: 11px; margin-top: 5px; color: #888; font-weight: 500;">{step['name']}</div>
                </div>
                """, unsafe_allow_html=True)
                
    # Add space after progress indicator
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# Display the interactive progress indicator
display_progress_indicator()

# Display the current step with styled header and icon
current_step = STEPS[st.session_state.step]
st.markdown(f'<h2 class="step-header"><span class="step-icon">{current_step["icon"]}</span> Step {st.session_state.step + 1}/{len(STEPS)}: {current_step["name"]}</h2>', unsafe_allow_html=True)

# Step 1: Welcome
if st.session_state.step == 0:
    st.markdown('<h1 class="main-header">Welcome to FinTech ML Pipeline</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="card">
        This application guides you through a complete machine learning pipeline for financial data analysis.
        You will be able to:
        
        - Load financial data from Kragle datasets
        - Preprocess and engineer features from the data
        - Train machine learning models (Linear Regression, Logistic Regression, or K-Means Clustering)
        - Evaluate model performance
        - Visualize results with interactive charts
        
        Click 'Next' to start the journey!
        </div>
        """, unsafe_allow_html=True)
        
        # Additional themed image
        st.image("assets/images/ml_finance.svg")
    
    with col2:
        # Display finance-themed GIF or SVG
        finance_gif = load_finance_gif()
        st.image(finance_gif, use_container_width=True, caption="Financial Analysis Visualization")
        
        # Additional description
        st.markdown("""
        <div class="card">
        <h3>Machine Learning for Financial Analysis</h3>
        <p>Use advanced ML techniques to analyze financial data, predict trends, and make data-driven decisions.</p>
        <p class="highlight">Features include data preprocessing, feature engineering, model training, and results visualization.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.button("Next", on_click=next_step, key="welcome_next")

# Step 2: Data Loading
elif st.session_state.step == 1:
    # Enhanced header for data loading section
    st.markdown("""
    <div class="card" style="background: linear-gradient(120deg, #f8fbff, #ffffff); border-left: 5px solid #1976D2; margin-bottom: 25px;">
        <h3 style="color: #1976D2;">Kragle Financial Datasets</h3>
        <p>Access pre-curated financial datasets for machine learning analysis.</p>
        <p><strong>Available datasets:</strong> Stock Market Data, Financial Indicators, Economic Data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Only use Kragle Dataset option now
    st.subheader("Select a Kragle Dataset Type")
    
    kragle_options = {
        "Stock Market Data": "stock_market",
        "Financial Indicators": "financial_indicators",
        "Economic Data": "economic_data"
    }
    
    dataset_name = st.selectbox("Select Dataset", list(kragle_options.keys()))
    
    if st.button("Load Kragle Dataset"):
        with st.spinner("Loading data from Kragle..."):
            try:
                data = load_kragle_dataset(kragle_options[dataset_name])
                if data is not None and not data.empty:
                    st.session_state.data = data
                    display_notification("Data loaded successfully!", type="success")
                    st.dataframe(data.head())
                    st.write(f"Data shape: {data.shape}")
                else:
                    st.error("Failed to load data. Please try again.")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    if st.session_state.data is not None:
        st.button("Next", on_click=next_step, key="data_next")
    
    st.button("Reset", on_click=reset_pipeline, key="data_reset")

# Step 3: Data Preprocessing
elif st.session_state.step == 2:
    # Enhanced header for data preprocessing section
    st.markdown("""
    <div class="card" style="background: linear-gradient(120deg, #f8fbff, #ffffff); border-left: 5px solid #1976D2; margin-bottom: 25px;">
        <h3 style="color: #1976D2;">Data Cleaning and Normalization</h3>
        <p>Prepare your financial data for ML models by handling missing values, normalizing features, and removing outliers.</p>
        <p><strong>Why it matters:</strong> Clean data is essential for accurate ML predictions in financial analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.error("No data available. Please go back to the Data Loading step.")
        st.button("Previous", on_click=prev_step)
    else:
        # Create tabs for different sections of preprocessing
        tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🧹 Preprocessing Options", "📈 Results"])
        
        # Check if we need to automatically switch to results tab after successful preprocessing
        if "preprocessing_done" in st.session_state and st.session_state.preprocessing_done:
            # Set active tab to Results (index 2)
            st.session_state.preprocessing_done = False  # Reset flag
            # In Streamlit, you can't directly control which tab is active, but we can
            # use a trick to make the Results tab (tab3) stand out with a notification
            with tab3:
                st.info("✅ Preprocessing completed successfully! Showing results below.")
                st.session_state.active_tab = "Results"
        
        with tab1:
            # Data overview layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Original Data Sample")
                
                # Display the data in a scrollable container with custom styling
                st.markdown('<div class="table-container" style="max-height: 350px; overflow-y: auto;">', unsafe_allow_html=True)
                st.dataframe(st.session_state.data.head(10), height=300)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show data summary metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Records", f"{len(st.session_state.data):,}")
                with col_b:
                    st.metric("Total Columns", f"{st.session_state.data.shape[1]}")
                with col_c:
                    missing_count = st.session_state.data.isnull().sum().sum()
                    st.metric("Missing Values", f"{missing_count:,}")
            
            with col2:
                st.subheader("Data Insights")
                st.image("assets/images/data_analytics.svg", use_container_width=True)
                
                # Data quality assessment
                st.markdown("""
                <div class="card" style="margin-top: 15px; background: #f5f9ff;">
                    <h4 style="color: #1976D2; margin-top: 0;">Data Quality Check</h4>
                    <ul style="padding-left: 20px; margin-top: 10px;">
                """, unsafe_allow_html=True)
                
                # Calculate data quality metrics
                data = st.session_state.data
                has_missing = data.isnull().sum().sum() > 0
                has_duplicates = data.duplicated().sum() > 0
                
                # Check for outliers in numeric columns
                has_outliers = False
                for col in data.select_dtypes(include=['number']).columns:
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    iqr = q3 - q1
                    outliers = ((data[col] < (q1 - 1.5 * iqr)) | (data[col] > (q3 + 1.5 * iqr))).sum()
                    if outliers > 0:
                        has_outliers = True
                        break
                
                # Display data quality results with icons
                st.markdown(f"""
                    <li>{"⚠️ Missing values detected" if has_missing else "✅ No missing values"}</li>
                    <li>{"⚠️ Duplicate records detected" if has_duplicates else "✅ No duplicate records"}</li>
                    <li>{"⚠️ Outliers detected" if has_outliers else "✅ No significant outliers"}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Show column types distribution in an expandable section
                with st.expander("Column Types Distribution"):
                    dtypes = data.dtypes.value_counts()
                    st.write(f"Numeric columns: {len(data.select_dtypes(include=['number']).columns)}")
                    st.write(f"Categorical columns: {len(data.select_dtypes(include=['object']).columns)}")
                    st.write(f"Date columns: {len(data.select_dtypes(include=['datetime']).columns)}")
        
        with tab2:
            st.subheader("Select Preprocessing Options")
            
            # Create tabs for different preprocessing categories
            preprocess_tab1, preprocess_tab2 = st.tabs(["Basic Preprocessing", "Text Conversion"])
            
            with preprocess_tab1:
                # Create 3 columns for basic preprocessing options with card styling
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="card" style="height: 280px;">', unsafe_allow_html=True)
                    st.markdown("#### Missing Values")
                    handle_missing = st.checkbox("Handle Missing Values", value=True)
                    missing_strategy = None
                    if handle_missing:
                        missing_strategy = st.selectbox(
                            "Strategy for Missing Values",
                            ["drop", "mean", "median", "zero", "forward_fill"],
                            help="Method to handle missing values in the dataset"
                        )
                        
                        # Help text with more details
                        strategy_help = {
                            "drop": "Removes rows with missing values",
                            "mean": "Fills missing values with column means",
                            "median": "Fills missing values with column medians",
                            "zero": "Fills missing values with zeros",
                            "forward_fill": "Fills missing values with the previous value"
                        }
                        
                        st.markdown(f"<div style='background-color: #e3f2fd; padding: 8px; border-radius: 4px; font-size: 13px;'>{strategy_help.get(missing_strategy, '')}</div>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col2:
                    st.markdown('<div class="card" style="height: 280px;">', unsafe_allow_html=True)
                    st.markdown("#### Normalization")
                    normalize_data = st.checkbox("Normalize Data", value=True)
                    normalization_method = None
                    if normalize_data:
                        normalization_method = st.selectbox(
                            "Normalization Method",
                            ["min_max", "standard", "robust"],
                            help="Method to scale numeric features"
                        )
                        
                        # Help text with more details
                        method_help = {
                            "min_max": "Scales values to range [0,1]",
                            "standard": "Standardizes data to mean=0, std=1",
                            "robust": "Uses median and IQR, robust to outliers"
                        }
                        
                        st.markdown(f"<div style='background-color: #e3f2fd; padding: 8px; border-radius: 4px; font-size: 13px;'>{method_help.get(normalization_method, '')}</div>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col3:
                    st.markdown('<div class="card" style="height: 280px;">', unsafe_allow_html=True)
                    st.markdown("#### Outliers")
                    handle_outliers = st.checkbox("Handle Outliers", value=True)
                    outlier_method = None
                    if handle_outliers:
                        outlier_method = st.selectbox(
                            "Outlier Handling Method",
                            ["clip", "iqr", "z_score"],
                            help="Method to handle outliers in the data"
                        )
                        
                        # Help text with more details
                        outlier_help = {
                            "clip": "Clips values to a specified range",
                            "iqr": "Uses Interquartile Range to identify and handle outliers",
                            "z_score": "Uses Z-score to identify and handle outliers"
                        }
                        
                        st.markdown(f"<div style='background-color: #e3f2fd; padding: 8px; border-radius: 4px; font-size: 13px;'>{outlier_help.get(outlier_method, '')}</div>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with preprocess_tab2:
                st.markdown("""
                <div class="card" style="background: linear-gradient(120deg, #f0f7ff, #ffffff); border-left: 5px solid #2196F3; margin-bottom: 20px; padding: 15px;">
                    <h4 style="color: #2196F3; margin-top: 0;">Text to Numeric Conversion</h4>
                    <p>Convert textual or categorical data to numeric format for machine learning models.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detect text columns in the dataset
                text_columns = []
                if st.session_state.data is not None:
                    text_columns = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
                
                if text_columns:
                    st.write(f"Detected {len(text_columns)} text/categorical columns in your dataset.")
                    
                    # Option to enable text conversion
                    convert_text = st.checkbox("Convert Text to Numeric", value=True if text_columns else False)
                    
                    if convert_text:
                        # Select columns to convert
                        selected_text_columns = st.multiselect(
                            "Select Text Columns to Convert",
                            text_columns,
                            default=text_columns
                        )
                        
                        # Select conversion method
                        text_method = st.radio(
                            "Conversion Method",
                            ["label_encoding", "one_hot", "ordinal", "count_vectorizer", "tfidf"],
                            horizontal=True
                        )
                        
                        # Help text with more details based on the selected method
                        method_descriptions = {
                            "label_encoding": "Converts categories to integer labels (0, 1, 2, ...). Best for ordinal data or when order matters.",
                            "one_hot": "Creates binary columns for each category. Best for nominal data with few unique values.",
                            "ordinal": "Similar to label encoding but allows custom ordering of categories.",
                            "count_vectorizer": "Converts text to word frequency counts. Best for text data like reviews or descriptions.",
                            "tfidf": "Term Frequency-Inverse Document Frequency. Weighs words by their importance in documents."
                        }
                        
                        st.markdown(f"""
                        <div style="background-color: #e3f2fd; padding: 12px; border-radius: 4px; margin-top: 10px;">
                            <strong>About {text_method}:</strong><br>
                            {method_descriptions.get(text_method, '')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional options for text vectorization methods
                        max_features = 100
                        if text_method in ["count_vectorizer", "tfidf"]:
                            max_features = st.slider("Maximum Features", 10, 500, 100, 
                                                    help="Maximum number of words/tokens to keep in the vocabulary")
                            
                            st.info("For large text columns, using a smaller number of maximum features improves performance.")
                    else:
                        selected_text_columns = []
                        text_method = "label_encoding"
                        max_features = 100
                else:
                    st.info("No text/categorical columns detected in your dataset.")
                    convert_text = False
                    selected_text_columns = []
                    text_method = "label_encoding"
                    max_features = 100
            
            # Add a divider before action buttons
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Process button in larger size and centered with animation
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                preprocess_button = st.button("🔄 Preprocess Data", use_container_width=True, 
                                            help="Click to apply the selected preprocessing options")
                
                if preprocess_button:
                    with st.spinner("Preprocessing data..."):
                        try:
                            # First apply basic preprocessing
                            processed_data = preprocess_data(
                                st.session_state.data,
                                handle_missing=handle_missing,
                                missing_strategy=missing_strategy if missing_strategy else "drop",
                                normalize=normalize_data,
                                normalization_method=normalization_method if normalization_method else "standard",
                                handle_outliers=handle_outliers,
                                outlier_method=outlier_method if outlier_method else "iqr"
                            )
                            
                            # Then apply text conversion if enabled
                            if processed_data is not None and 'convert_text' in locals() and convert_text and selected_text_columns:
                                # Additional status message
                                st.info(f"Converting {len(selected_text_columns)} text columns to numeric format...")
                                
                                processed_data, encoders = convert_text_to_numeric(
                                    processed_data,
                                    text_columns=selected_text_columns,
                                    method=text_method,
                                    max_features=max_features
                                )
                                
                                # Save encoders for later use (like when processing new data)
                                st.session_state.text_encoders = encoders
                            
                            if processed_data is not None:
                                st.session_state.processed_data = processed_data
                                # Set a flag to indicate successful preprocessing
                                st.session_state.preprocessing_done = True
                                display_notification("Data preprocessed successfully!", type="success")
                                # Instead of using st.rerun() directly, we'll use a session state flag to switch tabs
                            else:
                                st.error("Failed to preprocess data. Please try different options.")
                        except Exception as e:
                            st.error(f"Error during preprocessing: {str(e)}")
        
        with tab3:
            if "processed_data" in st.session_state and st.session_state.processed_data is not None:
                processed_data = st.session_state.processed_data
                original_data = st.session_state.data
                
                st.subheader("Preprocessing Results")
                
                # Summary metrics of preprocessing results
                st.markdown("<div class='card' style='margin-bottom: 20px;'>", unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    missing_before = original_data.isnull().sum().sum()
                    missing_after = processed_data.isnull().sum().sum()
                    st.metric("Missing Values", f"{missing_after:,}", 
                              delta=f"-{missing_before - missing_after:,}" if missing_before > missing_after else "0",
                              delta_color="normal")
                
                with col2:
                    orig_shape = original_data.shape
                    proc_shape = processed_data.shape
                    st.metric("Data Rows", f"{proc_shape[0]:,}", 
                              delta=f"{proc_shape[0] - orig_shape[0]:,}", delta_color="normal")
                
                with col3:
                    # Count numeric columns that were normalized
                    numeric_cols = len(original_data.select_dtypes(include=['number']).columns)
                    st.metric("Processed Columns", f"{numeric_cols:,}")
                
                with col4:
                    # Count outliers (estimate)
                    outliers = 0
                    if handle_outliers and outlier_method:
                        for col in original_data.select_dtypes(include=['number']).columns:
                            q1 = original_data[col].quantile(0.25)
                            q3 = original_data[col].quantile(0.75)
                            iqr = q3 - q1
                            outliers += ((original_data[col] < (q1 - 1.5 * iqr)) | 
                                       (original_data[col] > (q3 + 1.5 * iqr))).sum()
                    st.metric("Outliers Handled", f"{outliers:,}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Before/After data comparison with better styling
                st.subheader("Data Before/After")
                comparison_tabs = st.tabs(["Side-by-Side View", "Statistics Comparison"])
                
                with comparison_tabs[0]:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown("<h4 style='color: #1976D2;'>Before Preprocessing</h4>", unsafe_allow_html=True)
                        st.dataframe(original_data.head(5), height=200)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown("<h4 style='color: #4CAF50;'>After Preprocessing</h4>", unsafe_allow_html=True)
                        st.dataframe(processed_data.head(5), height=200)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                with comparison_tabs[1]:
                    stats_tab1, stats_tab2 = st.tabs(["Before", "After"])
                    
                    with stats_tab1:
                        stats_before = original_data.describe()
                        st.dataframe(stats_before.style.background_gradient(cmap='Blues'), height=300)
                    
                    with stats_tab2:
                        stats_after = processed_data.describe()
                        st.dataframe(stats_after.style.background_gradient(cmap='Greens'), height=300)
                
                # Data visualization section
                st.subheader("Data Distribution Visualization")
                
                # Select column to visualize
                numeric_cols = processed_data.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    selected_col = st.selectbox("Select column to visualize:", numeric_cols)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if selected_col in original_data.columns:
                            fig = px.histogram(original_data, x=selected_col, 
                                              title=f"Distribution Before",
                                              color_discrete_sequence=['#1976D2'])
                            fig.update_layout(
                                title_font=dict(size=16),
                                plot_bgcolor='rgba(240,250,255,0.8)',
                                xaxis_title_font=dict(size=14),
                                yaxis_title_font=dict(size=14)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if selected_col in processed_data.columns:
                            fig = px.histogram(processed_data, x=selected_col, 
                                              title=f"Distribution After",
                                              color_discrete_sequence=['#4CAF50'])
                            fig.update_layout(
                                title_font=dict(size=16),
                                plot_bgcolor='rgba(240,255,240,0.8)',
                                xaxis_title_font=dict(size=14),
                                yaxis_title_font=dict(size=14)
                            )
                            st.plotly_chart(fig, use_container_width=True)
            else:
                # Display a message when no processed data is available
                st.info("Please preprocess the data using the options in the 'Preprocessing Options' tab.")
                st.markdown(
                    """
                    <div style="display: flex; justify-content: center; margin: 30px;">
                        <img src="assets/images/data_analytics.svg" width="200">
                    </div>
                    <div style="text-align: center; color: #666;">
                        <p>No processed data available. Go to the "Preprocessing Options" tab to clean and prepare your data.</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        # Navigation buttons with better styling
        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.button("⬅️ Previous", on_click=prev_step, key="preprocess_prev", 
                    help="Go back to the Data Loading step")
        
        with col3:
            if "processed_data" in st.session_state and st.session_state.processed_data is not None:
                st.button("Next ➡️", on_click=next_step, key="preprocess_next", 
                        help="Proceed to Feature Engineering")
        
        with col2:
            # Center the reset button
            st.button("🔄 Reset Pipeline", on_click=reset_pipeline, key="preprocess_reset", 
                    help="Reset the entire machine learning pipeline")

# Step 4: Feature Engineering
elif st.session_state.step == 3:
    col1, col2 = st.columns([3, 1])
    with col2:
        st.image("assets/images/data_analytics.svg")
    
    if st.session_state.processed_data is None:
        st.error("No processed data available. Please go back to the Data Preprocessing step.")
        st.button("Previous", on_click=prev_step)
    else:
        st.write("Processed Data Sample:")
        st.dataframe(st.session_state.processed_data.head())
        
        st.subheader("Select Features and Target")
        
        # Get all columns for selection
        all_columns = st.session_state.processed_data.columns.tolist()
        
        # Select target variable
        target_variable = st.selectbox("Select Target Variable", all_columns)
        
        # Select features
        remaining_columns = [col for col in all_columns if col != target_variable]
        selected_features = st.multiselect(
            "Select Features",
            remaining_columns,
            default=remaining_columns
        )
        
        st.subheader("Feature Engineering Options")
        
        create_polynomial = st.checkbox("Create Polynomial Features", value=False)
        poly_degree = 2
        if create_polynomial:
            poly_degree = st.slider("Polynomial Degree", min_value=2, max_value=5, value=2)
        
        create_interaction = st.checkbox("Create Interaction Features", value=False)
        
        add_lag_features = st.checkbox("Add Lag Features (for Time Series)", value=False)
        lag_periods = [1]
        if add_lag_features:
            lag_periods_input = st.text_input("Lag Periods (comma-separated)", "1, 2, 3")
            try:
                lag_periods = [int(x.strip()) for x in lag_periods_input.split(",")]
            except:
                st.warning("Invalid lag periods. Using default value of 1.")
                lag_periods = [1]
        
        if st.button("Engineer Features"):
            with st.spinner("Engineering features..."):
                try:
                    if not selected_features:
                        st.error("Please select at least one feature.")
                    else:
                        X, y, feature_names = engineer_features(
                            st.session_state.processed_data,
                            selected_features,
                            target_variable,
                            create_polynomial=create_polynomial,
                            poly_degree=poly_degree,
                            create_interaction=create_interaction,
                            add_lag_features=add_lag_features,
                            lag_periods=lag_periods
                        )
                        
                        if X is not None and y is not None:
                            st.session_state.features = X
                            st.session_state.target = y
                            st.session_state.feature_names = feature_names
                            
                            display_notification("Features engineered successfully!", type="success")
                            
                            st.subheader("Engineered Features Sample")
                            feature_df = pd.DataFrame(X, columns=feature_names)
                            st.dataframe(feature_df.head())
                            st.write(f"Features shape: {X.shape}")
                            
                            # Show correlation matrix
                            st.subheader("Feature Correlation Matrix")
                            fig = plot_correlation_matrix(feature_df)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Failed to engineer features. Please try different options.")
                except Exception as e:
                    st.error(f"Error during feature engineering: {str(e)}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.button("Previous", on_click=prev_step, key="feature_prev")
        with col2:
            if "features" in st.session_state and st.session_state.features is not None:
                st.button("Next", on_click=next_step, key="feature_next")
        
        st.button("Reset", on_click=reset_pipeline, key="feature_reset")

# Step 5: Model Selection
elif st.session_state.step == 4:
    col1, col2 = st.columns([3, 1])
    with col2:
        st.image("assets/images/model_training.svg")
    
    if st.session_state.features is None or st.session_state.target is None:
        st.error("No features or target available. Please go back to the Feature Engineering step.")
        st.button("Previous", on_click=prev_step)
    else:
        st.write(f"Selected features shape: {st.session_state.features.shape}")
        
        st.subheader("Select Model Type")
        
        model_type = st.radio(
            "Model Type",
            ["Linear Regression", "Logistic Regression", "K-Means Clustering"],
            horizontal=True
        )
        
        # Handle model-specific parameters
        if model_type == "Linear Regression":
            st.subheader("Linear Regression Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                fit_intercept = st.checkbox("Fit Intercept", value=True)
                positive = st.checkbox("Enforce Positive Coefficients", value=False)
            
            with col2:
                st.markdown("""
                <div class="card">
                <p><b>Note:</b> The 'normalize' parameter is deprecated. Data normalization 
                should be done separately in the preprocessing step.</p>
                </div>
                """, unsafe_allow_html=True)
                
                solver = st.selectbox(
                    "Solver Algorithm", 
                    ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                    help="Algorithm to use in the computational routines"
                )
            
            st.session_state.model_params = {
                "fit_intercept": fit_intercept,
                "positive": positive,
                "solver": solver
            }
            
        elif model_type == "Logistic Regression":
            st.subheader("Logistic Regression Parameters")
            
            # Check if the target is binary
            unique_targets = np.unique(st.session_state.target)
            if len(unique_targets) > 2:
                st.warning(f"Target has {len(unique_targets)} classes. Logistic Regression works best with binary classification.")
            
            col1, col2 = st.columns(2)
            with col1:
                C = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0, 
                             help="Smaller values specify stronger regularization")
                max_iter = st.slider("Maximum Iterations", 100, 1000, 100)
                solver = st.selectbox(
                    "Solver Algorithm", 
                    ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                    help="Algorithm to use in the optimization problem"
                )
            
            with col2:
                st.markdown("""
                <div class="card">
                <h4>About Logistic Regression</h4>
                <p>Logistic Regression is used for classification problems. It works by estimating probabilities 
                using a logistic/sigmoid function.</p>
                <p>Common applications in finance:</p>
                <ul>
                    <li>Credit scoring</li>
                    <li>Fraud detection</li>
                    <li>Market direction prediction</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                penalty = st.selectbox(
                    "Penalty", 
                    ["l2", "l1", "elasticnet", "none"],
                    help="Specify the norm used in the penalization"
                )
                
                tol = st.text_input("Tolerance for stopping criteria", "0.0001")
                try:
                    tol = float(tol)
                except:
                    tol = 0.0001
            
            st.session_state.model_params = {
                "C": C,
                "max_iter": max_iter,
                "solver": solver,
                "penalty": penalty,
                "tol": tol
            }
            
        else:  # K-Means Clustering
            st.subheader("K-Means Clustering Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                max_iter = st.slider("Maximum Iterations", 100, 1000, 300)
                n_init = st.slider("Number of Initializations", 1, 20, 10,
                                 help="Number of times the k-means algorithm will run with different centroid seeds")
                random_state = st.slider("Random State", 0, 100, 42)
            
            with col2:
                st.markdown("""
                <div class="card">
                <h4>About K-Means Clustering</h4>
                <p>K-Means clustering is an unsupervised learning algorithm that groups similar data points 
                into clusters based on their features.</p>
                <p>Common applications in finance:</p>
                <ul>
                    <li>Customer segmentation</li>
                    <li>Stock market segmentation</li>
                    <li>Risk assessment grouping</li>
                    <li>Anomaly detection</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                algorithm = st.selectbox(
                    "Algorithm", 
                    ["auto", "full", "elkan"],
                    help="K-means algorithm to use"
                )
                
                tol = st.text_input("Convergence Tolerance", "0.0001", 
                                   help="Relative tolerance with regards to Frobenius norm of the difference in the cluster centers")
                try:
                    tol = float(tol)
                except:
                    tol = 0.0001
            
            st.session_state.model_params = {
                "n_clusters": n_clusters,
                "max_iter": max_iter,
                "n_init": n_init,
                "random_state": random_state,
                "algorithm": algorithm,
                "tol": tol
            }
        
        # Train/Test Split Parameters
        st.subheader("Train/Test Split Parameters")
        
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        random_state = st.slider("Random State", 0, 100, 42)
        
        if st.button("Split Data and Prepare Model"):
            with st.spinner("Splitting data and preparing model..."):
                try:
                    X_train, X_test, y_train, y_test = split_data(
                        st.session_state.features,
                        st.session_state.target,
                        test_size=test_size,
                        random_state=random_state
                    )
                    
                    if X_train is not None and X_test is not None:
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.model_type = model_type
                        
                        display_notification("Data split successfully!", type="success")
                        
                        # Show train/test set information
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Training Set:")
                            st.write(f"X_train shape: {X_train.shape}")
                            st.write(f"y_train shape: {y_train.shape}")
                        with col2:
                            st.write("Testing Set:")
                            st.write(f"X_test shape: {X_test.shape}")
                            st.write(f"y_test shape: {y_test.shape}")
                    else:
                        st.error("Failed to split data. Please try different parameters.")
                except Exception as e:
                    st.error(f"Error during data splitting: {str(e)}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.button("Previous", on_click=prev_step, key="model_sel_prev")
        with col2:
            if "X_train" in st.session_state and st.session_state.X_train is not None:
                st.button("Next", on_click=next_step, key="model_sel_next")
        
        st.button("Reset", on_click=reset_pipeline, key="model_sel_reset")

# Step 6: Model Training
elif st.session_state.step == 5:
    col1, col2 = st.columns([3, 1])
    with col2:
        st.image("assets/images/model_training.svg")
    
    if (st.session_state.X_train is None or 
        st.session_state.X_test is None or 
        st.session_state.model_type is None):
        st.error("Missing training data or model selection. Please go back to previous steps.")
        st.button("Previous", on_click=prev_step)
    else:
        st.subheader(f"Training {st.session_state.model_type}")
        
        st.write(f"X_train shape: {st.session_state.X_train.shape}")
        st.write(f"y_train shape: {st.session_state.y_train.shape}")
        
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)
        
        if st.button("Train Model"):
            start_time = time.time()
            
            # Simulate progress during training
            for i in range(101):
                progress_bar.progress(i)
                time.sleep(0.01)
            
            with st.spinner("Training model..."):
                try:
                    model, training_results = train_model(
                        st.session_state.X_train,
                        st.session_state.y_train,
                        model_type=st.session_state.model_type,
                        params=st.session_state.model_params
                    )
                    
                    if model is not None:
                        st.session_state.model = model
                        st.session_state.training_results = training_results
                        
                        end_time = time.time()
                        training_time = end_time - start_time
                        
                        display_notification(f"Model trained successfully in {training_time:.2f} seconds!", type="success")
                        
                        # Display training results
                        st.subheader("Training Results")
                        
                        if st.session_state.model_type == "Linear Regression":
                            st.write(f"Model Coefficients: {training_results.get('coefficients')}")
                            st.write(f"Model Intercept: {training_results.get('intercept')}")
                            st.write(f"Training R²: {training_results.get('r2_train'):.4f}")
                            
                        elif st.session_state.model_type == "Logistic Regression":
                            st.write(f"Model Coefficients: {training_results.get('coefficients')}")
                            st.write(f"Model Intercept: {training_results.get('intercept')}")
                            st.write(f"Training Accuracy: {training_results.get('accuracy_train'):.4f}")
                            
                        else:  # K-Means
                            st.write(f"Cluster Centers: {training_results.get('cluster_centers')}")
                            st.write(f"Inertia: {training_results.get('inertia'):.4f}")
                    else:
                        st.error("Failed to train model. Please try different parameters.")
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.button("Previous", on_click=prev_step, key="train_prev")
        with col2:
            if "model" in st.session_state and st.session_state.model is not None:
                st.button("Next", on_click=next_step, key="train_next")
        
        st.button("Reset", on_click=reset_pipeline, key="train_reset")

# Step 7: Model Evaluation
elif st.session_state.step == 6:
    col1, col2 = st.columns([3, 1])
    with col2:
        st.image("assets/images/results_visualization.svg")
    
    if st.session_state.model is None:
        st.error("No trained model available. Please go back to the Model Training step.")
        st.button("Previous", on_click=prev_step)
    else:
        st.subheader(f"Evaluating {st.session_state.model_type}")
        
        st.write(f"X_test shape: {st.session_state.X_test.shape}")
        st.write(f"y_test shape: {st.session_state.y_test.shape}")
        
        if st.button("Evaluate Model"):
            with st.spinner("Evaluating model..."):
                try:
                    evaluation_results, predictions = evaluate_model(
                        st.session_state.model,
                        st.session_state.X_test,
                        st.session_state.y_test,
                        model_type=st.session_state.model_type
                    )
                    
                    if evaluation_results is not None:
                        st.session_state.metrics = evaluation_results
                        st.session_state.predictions = predictions
                        
                        display_notification("Model evaluated successfully!", type="success")
                        
                        # Display evaluation metrics
                        st.subheader("Evaluation Metrics")
                        
                        if st.session_state.model_type == "Linear Regression":
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("R² Score", f"{evaluation_results.get('r2'):.4f}")
                            with col2:
                                st.metric("Mean Absolute Error", f"{evaluation_results.get('mae'):.4f}")
                            with col3:
                                st.metric("Root Mean Squared Error", f"{evaluation_results.get('rmse'):.4f}")
                            
                            # Plot actual vs predicted
                            st.subheader("Actual vs Predicted Values")
                            fig = plot_predictions(
                                st.session_state.y_test,
                                predictions
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif st.session_state.model_type == "Logistic Regression":
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Accuracy", f"{evaluation_results.get('accuracy'):.4f}")
                            with col2:
                                st.metric("Precision", f"{evaluation_results.get('precision'):.4f}")
                            with col3:
                                st.metric("Recall", f"{evaluation_results.get('recall'):.4f}")
                            
                            # Show confusion matrix
                            st.subheader("Confusion Matrix")
                            st.table(evaluation_results.get('confusion_matrix'))
                            
                        else:  # K-Means
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Silhouette Score", f"{evaluation_results.get('silhouette'):.4f}")
                            with col2:
                                st.metric("Calinski-Harabasz Score", f"{evaluation_results.get('calinski_harabasz'):.4f}")
                            
                            # Plot clusters
                            st.subheader("Cluster Visualization")
                            if st.session_state.X_test.shape[1] > 2:
                                # If more than 2 dimensions, use PCA to reduce to 2D for visualization
                                st.info("Using PCA to reduce dimensions for visualization.")
                            
                            fig = plot_clusters(
                                st.session_state.X_test,
                                predictions
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Failed to evaluate model.")
                except Exception as e:
                    st.error(f"Error during model evaluation: {str(e)}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.button("Previous", on_click=prev_step, key="eval_prev")
        with col2:
            if "metrics" in st.session_state and st.session_state.metrics is not None:
                st.button("Next", on_click=next_step, key="eval_next")
        
        st.button("Reset", on_click=reset_pipeline, key="eval_reset")

# Step 8: Results Visualization
elif st.session_state.step == 7:
    col1, col2 = st.columns([3, 1])
    with col2:
        st.image("assets/images/results_visualization.svg")
    
    if (st.session_state.predictions is None or 
        st.session_state.metrics is None):
        st.error("No evaluation results available. Please go back to the Model Evaluation step.")
        st.button("Previous", on_click=prev_step)
    else:
        st.subheader("Model Performance Visualization")
        
        # Create tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["Performance Metrics", "Feature Importance", "Model Interpretation", "Additional Insights"])
        
        with viz_tab1:
            if st.session_state.model_type == "Linear Regression":
                # Residual plot
                st.subheader("Residual Analysis")
                
                y_test = st.session_state.y_test
                predictions = st.session_state.predictions
                residuals = y_test - predictions
                
                # Create residual plot
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                fig = make_subplots(rows=1, cols=2, 
                                    subplot_titles=("Residuals vs Predicted", "Residual Distribution"))
                
                # Residuals vs Predicted
                fig.add_trace(
                    go.Scatter(x=predictions, y=residuals, mode='markers',
                               marker=dict(color='blue', opacity=0.6),
                               name="Residuals"),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=[predictions.min(), predictions.max()], y=[0, 0],
                               mode='lines', line=dict(color='red', dash='dash'),
                               name="Zero Line"),
                    row=1, col=1
                )
                
                # Residual distribution
                fig.add_trace(
                    go.Histogram(x=residuals, marker=dict(color='blue', opacity=0.6),
                                 name="Residual Distribution"),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics table
                st.subheader("Performance Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': ['R² Score', 'Mean Absolute Error', 'Root Mean Squared Error'],
                    'Value': [
                        f"{st.session_state.metrics.get('r2'):.4f}",
                        f"{st.session_state.metrics.get('mae'):.4f}",
                        f"{st.session_state.metrics.get('rmse'):.4f}"
                    ]
                })
                st.table(metrics_df)
                
            elif st.session_state.model_type == "Logistic Regression":
                # ROC Curve
                st.subheader("ROC Curve")
                
                if 'roc_curve' in st.session_state.metrics:
                    fpr = st.session_state.metrics['roc_curve']['fpr']
                    tpr = st.session_state.metrics['roc_curve']['tpr']
                    auc = st.session_state.metrics['roc_curve']['auc']
                    
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(x=fpr, y=tpr, mode='lines',
                                  line=dict(color='blue', width=2),
                                  name=f'ROC Curve (AUC = {auc:.4f})')
                    )
                    fig.add_trace(
                        go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                  line=dict(color='red', width=2, dash='dash'),
                                  name='Random Classifier')
                    )
                    fig.update_layout(
                        title='Receiver Operating Characteristic (ROC) Curve',
                        xaxis=dict(title='False Positive Rate'),
                        yaxis=dict(title='True Positive Rate'),
                        legend=dict(x=0.1, y=0.9),
                        width=700,
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Classification report
                st.subheader("Classification Report")
                if 'classification_report' in st.session_state.metrics:
                    st.text(st.session_state.metrics['classification_report'])
                
            else:  # K-Means
                # 3D cluster visualization if possible
                st.subheader("3D Cluster Visualization")
                
                if st.session_state.X_test.shape[1] >= 3:
                    # Use first 3 dimensions or PCA
                    from sklearn.decomposition import PCA
                    
                    if st.session_state.X_test.shape[1] > 3:
                        pca = PCA(n_components=3)
                        X_pca = pca.fit_transform(st.session_state.X_test)
                        st.info(f"Using PCA to reduce from {st.session_state.X_test.shape[1]} to 3 dimensions for visualization.")
                    else:
                        X_pca = st.session_state.X_test
                    
                    import plotly.express as px
                    
                    df = pd.DataFrame({
                        'x': X_pca[:, 0],
                        'y': X_pca[:, 1],
                        'z': X_pca[:, 2],
                        'cluster': st.session_state.predictions
                    })
                    
                    fig = px.scatter_3d(df, x='x', y='y', z='z', color='cluster',
                                        color_continuous_scale=px.colors.qualitative.G10)
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough dimensions for 3D visualization.")
                
                # Silhouette analysis
                st.subheader("Clustering Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': ['Silhouette Score', 'Calinski-Harabasz Score'],
                    'Value': [
                        f"{st.session_state.metrics.get('silhouette', 'N/A')}",
                        f"{st.session_state.metrics.get('calinski_harabasz', 'N/A')}"
                    ]
                })
                st.table(metrics_df)
        
        with viz_tab2:
            st.subheader("Feature Importance Analysis")
            
            if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
                # Get feature importance
                if hasattr(st.session_state.model, 'coef_'):
                    if st.session_state.model_type == "Logistic Regression" and len(st.session_state.model.classes_) > 2:
                        st.warning("Feature importance visualization is simplified for multi-class logistic regression.")
                    
                    coefs = st.session_state.model.coef_
                    if coefs.ndim > 1:
                        # For multi-class, use the mean absolute coefficient values
                        importance = np.mean(np.abs(coefs), axis=0)
                    else:
                        importance = np.abs(coefs)
                    
                    feature_names = st.session_state.feature_names
                    
                    # Create importance dataframe
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importance
                    }).sort_values('Importance', ascending=False)
                    
                    # Plot feature importance
                    fig = plot_feature_importance(importance_df)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display importance table
                    st.dataframe(importance_df)
                else:
                    st.warning("Feature importance not available for this model.")
            else:
                st.info("Feature importance analysis not applicable for K-Means clustering.")
        
        with viz_tab3:
            st.subheader("Model Interpretation")
            
            if st.session_state.model_type == "Linear Regression":
                # Linear Regression Interpretation
                st.markdown("""
                <div class="interpretation-card">
                    <h4 style="color: #1976D2;">Understanding Linear Regression Results</h4>
                    <p>Linear regression models the relationship between your target variable and features using a linear equation.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Equation of the line
                if hasattr(st.session_state.model, 'coef_') and hasattr(st.session_state.model, 'intercept_'):
                    st.subheader("Model Equation")
                    
                    coefs = st.session_state.model.coef_
                    intercept = st.session_state.model.intercept_
                    feature_names = st.session_state.feature_names
                    
                    equation = f"y = {intercept:.4f}"
                    for i, coef in enumerate(coefs):
                        sign = "+" if coef >= 0 else ""
                        equation += f" {sign} {coef:.4f} × {feature_names[i]}"
                    
                    st.markdown(f"""
                    <div class="model-equation">
                        {equation}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Interpretation of coefficients
                    st.subheader("Coefficient Interpretation")
                    
                    interpretation_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': coefs,
                        'Interpretation': [
                            f"For each unit increase in {feature}, the target increases by {coef:.4f} units, holding all other features constant."
                            if coef > 0 else
                            f"For each unit increase in {feature}, the target decreases by {abs(coef):.4f} units, holding all other features constant."
                            for feature, coef in zip(feature_names, coefs)
                        ]
                    })
                    
                    st.dataframe(interpretation_df, height=300)
                    
                    # R-squared interpretation
                    if 'r2' in st.session_state.metrics:
                        r2 = st.session_state.metrics['r2']
                        st.subheader("Model Fit Quality")
                        
                        r2_interpretation = ""
                        if r2 < 0.4:
                            r2_interpretation = "The model explains a small portion of the variance in the data, suggesting that linear regression might not be the best model for this problem or that important predictors might be missing."
                        elif r2 < 0.7:
                            r2_interpretation = "The model explains a moderate portion of the variance in the data. This may be acceptable for some applications, but there could be room for improvement."
                        else:
                            r2_interpretation = "The model explains a large portion of the variance in the data, indicating a good fit. The selected features capture the patterns in the target variable well."
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric("R² Score", f"{r2:.4f}")
                        with col2:
                            st.markdown(f"""
                            <div style="padding: 10px; border-radius: 5px; background-color: #e3f2fd;">
                                <p style="margin: 0;">{r2_interpretation}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Prediction Analysis
                st.subheader("Prediction Analysis")
                
                y_test = st.session_state.y_test
                predictions = st.session_state.predictions
                
                # Show some example predictions
                sample_size = min(5, len(y_test))
                sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
                
                samples_df = pd.DataFrame({
                    'Actual': y_test[sample_indices],
                    'Predicted': predictions[sample_indices],
                    'Error': y_test[sample_indices] - predictions[sample_indices],
                    'Error (%)': (y_test[sample_indices] - predictions[sample_indices]) / y_test[sample_indices] * 100
                })
                
                st.write("Sample Predictions:")
                st.dataframe(samples_df)
                
                # Overall error analysis
                errors = y_test - predictions
                
                st.markdown("""
                <div style="background-color: #f0f7ff; padding: 15px; border-radius: 5px; margin-top: 15px;">
                    <h5 style="margin-top: 0;">What to look for in residuals:</h5>
                    <ul>
                        <li><strong>Mean near zero:</strong> Indicates unbiased predictions</li>
                        <li><strong>Symmetrical distribution:</strong> Suggests errors are random</li>
                        <li><strong>No patterns:</strong> Confirms linear model is appropriate</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            elif st.session_state.model_type == "Logistic Regression":
                # Logistic Regression Interpretation
                st.markdown("""
                <div class="interpretation-card">
                    <h4 style="color: #1976D2;">Understanding Logistic Regression Results</h4>
                    <p>Logistic regression models the probability of a binary outcome based on your features.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Coefficient interpretation for Logistic Regression
                if hasattr(st.session_state.model, 'coef_'):
                    st.subheader("Coefficient Interpretation")
                    
                    coefs = st.session_state.model.coef_[0] if st.session_state.model.coef_.ndim > 1 else st.session_state.model.coef_
                    intercept = st.session_state.model.intercept_[0] if hasattr(st.session_state.model.intercept_, '__iter__') else st.session_state.model.intercept_
                    odds_ratios = np.exp(coefs)
                    feature_names = st.session_state.feature_names
                    
                    interpretation_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': coefs,
                        'Odds Ratio': odds_ratios,
                        'Interpretation': [
                            f"For each unit increase in {feature}, the odds of the positive class increase by {(odds_ratio-1)*100:.1f}%."
                            if odds_ratio > 1 else
                            f"For each unit increase in {feature}, the odds of the positive class decrease by {(1-odds_ratio)*100:.1f}%."
                            for feature, odds_ratio in zip(feature_names, odds_ratios)
                        ]
                    })
                    
                    st.dataframe(interpretation_df, height=300)
                    
                    # Log-odds equation
                    st.subheader("Model Equation (Log-odds)")
                    
                    equation = f"ln(p/(1-p)) = {intercept:.4f}"
                    for i, coef in enumerate(coefs):
                        sign = "+" if coef >= 0 else ""
                        equation += f" {sign} {coef:.4f} × {feature_names[i]}"
                    
                    st.markdown(f"""
                    <div class="model-equation">
                        {equation}
                    </div>
                    <p style="font-size: 14px; color: #666; margin-top: 5px;">p = probability of the positive class</p>
                    """, unsafe_allow_html=True)
                
                # Confusion Matrix Interpretation
                if 'confusion_matrix' in st.session_state.metrics:
                    st.subheader("Confusion Matrix Interpretation")
                    
                    cm = st.session_state.metrics['confusion_matrix']
                    
                    # Calculate metrics from confusion matrix
                    tn, fp, fn, tp = cm.ravel()
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    # Display metrics in a 2x2 grid
                    col1, col2 = st.columns(2)
                    col3, col4 = st.columns(2)
                    
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                        st.markdown("Proportion of all predictions that are correct")
                        
                    with col2:
                        st.metric("Precision", f"{precision:.4f}")
                        st.markdown("Proportion of positive predictions that are correct")
                        
                    with col3:
                        st.metric("Recall", f"{recall:.4f}")
                        st.markdown("Proportion of actual positives that are predicted positive")
                        
                    with col4:
                        st.metric("F1 Score", f"{f1:.4f}")
                        st.markdown("Harmonic mean of precision and recall")
                    
                    # Interpretation guidance
                    st.markdown("""
                    <div style="background-color: #f0f7ff; padding: 15px; border-radius: 5px; margin-top: 15px;">
                        <h5 style="margin-top: 0;">What these metrics mean for your model:</h5>
                        <ul>
                            <li><strong>High precision:</strong> Few false positives - good for situations where false positives are costly</li>
                            <li><strong>High recall:</strong> Few false negatives - good for situations where missing positives is costly</li>
                            <li><strong>High F1 score:</strong> Good balance between precision and recall</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
            else:  # K-Means
                # K-Means Interpretation
                st.markdown("""
                <div class="interpretation-card">
                    <h4 style="color: #1976D2;">Understanding K-Means Clustering Results</h4>
                    <p>K-Means groups similar data points into clusters based on their features.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Cluster profile analysis
                if hasattr(st.session_state.model, 'cluster_centers_'):
                    st.subheader("Cluster Profiles")
                    
                    centers = st.session_state.model.cluster_centers_
                    n_clusters = centers.shape[0]
                    feature_names = st.session_state.feature_names
                    
                    # Create radar chart for cluster profiles
                    fig = go.Figure()
                    
                    for i in range(n_clusters):
                        fig.add_trace(go.Scatterpolar(
                            r=centers[i],
                            theta=feature_names,
                            fill='toself',
                            name=f'Cluster {i}'
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                            )
                        ),
                        title="Cluster Characteristic Profiles",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster insights - find distinguishing features for each cluster
                    st.subheader("Cluster Insights")
                    
                    # Display top distinctive features for each cluster
                    cluster_insights = []
                    
                    for i in range(n_clusters):
                        # Find features where this cluster has highest or lowest values
                        highest_features = []
                        lowest_features = []
                        
                        for j, feature in enumerate(feature_names):
                            is_highest = True
                            is_lowest = True
                            
                            for k in range(n_clusters):
                                if k != i:
                                    if centers[i, j] < centers[k, j]:
                                        is_highest = False
                                    if centers[i, j] > centers[k, j]:
                                        is_lowest = False
                            
                            if is_highest:
                                highest_features.append(feature)
                            if is_lowest:
                                lowest_features.append(feature)
                        
                        # Create insight
                        insight = f"**Cluster {i}**: "
                        
                        if highest_features:
                            insight += f"Highest values in {', '.join(highest_features)}. "
                        
                        if lowest_features:
                            insight += f"Lowest values in {', '.join(lowest_features)}."
                        
                        if not highest_features and not lowest_features:
                            insight += "No distinguishing extreme values."
                        
                        cluster_insights.append(insight)
                    
                    # Display insights
                    for insight in cluster_insights:
                        st.markdown(insight)
                    
                    # Provide interpretation guidance
                    st.markdown("""
                    <div style="background-color: #f0f7ff; padding: 15px; border-radius: 5px; margin-top: 15px;">
                        <h5 style="margin-top: 0;">How to interpret these clusters:</h5>
                        <ul>
                            <li>Each cluster represents a segment with similar characteristics</li>
                            <li>The radar chart shows the average values of each feature in each cluster</li>
                            <li>Look for distinctive patterns in the feature values to understand what makes each cluster unique</li>
                            <li>Consider how these segments might represent different types in your domain (e.g., customer segments, risk profiles)</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
        
        with viz_tab4:
            st.subheader("Additional Insights")
            
            if st.session_state.model_type == "Linear Regression":
                # Prediction interval
                st.write("Prediction Distribution")
                
                import scipy.stats as stats
                
                # Calculate prediction interval (simplified approach)
                residuals = st.session_state.y_test - st.session_state.predictions
                residual_std = np.std(residuals)
                
                # Plot histogram of predictions with confidence intervals
                fig = go.Figure()
                
                # Add histogram of predictions
                fig.add_trace(go.Histogram(
                    x=st.session_state.predictions,
                    nbinsx=30,
                    marker_color='blue',
                    opacity=0.7,
                    name='Predictions'
                ))
                
                fig.update_layout(
                    title="Distribution of Predictions",
                    xaxis_title="Predicted Value",
                    yaxis_title="Count",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("Prediction Intervals (95% confidence)")
                
                # Create a sample of predictions to show intervals
                sample_size = min(10, len(st.session_state.predictions))
                sample_indices = np.random.choice(len(st.session_state.predictions), sample_size, replace=False)
                
                interval_df = pd.DataFrame({
                    'Actual': st.session_state.y_test[sample_indices],
                    'Predicted': st.session_state.predictions[sample_indices],
                    'Lower Bound': st.session_state.predictions[sample_indices] - 1.96 * residual_std,
                    'Upper Bound': st.session_state.predictions[sample_indices] + 1.96 * residual_std
                })
                
                st.dataframe(interval_df)
                
            elif st.session_state.model_type == "Logistic Regression":
                # Probability distribution
                st.write("Prediction Probability Distribution")
                
                if hasattr(st.session_state.model, 'predict_proba'):
                    # Get probabilities
                    probas = st.session_state.model.predict_proba(st.session_state.X_test)
                    
                    if probas.shape[1] == 2:  # Binary classification
                        # Plot histogram of class 1 probabilities
                        fig = go.Figure()
                        
                        fig.add_trace(go.Histogram(
                            x=probas[:, 1],
                            nbinsx=30,
                            marker_color='blue',
                            opacity=0.7,
                            name='Probability of Class 1'
                        ))
                        
                        fig.update_layout(
                            title="Distribution of Prediction Probabilities",
                            xaxis_title="Probability",
                            yaxis_title="Count",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show examples with highest uncertainty
                        st.write("Most Uncertain Predictions (Probabilities closest to 0.5)")
                        
                        # Calculate uncertainty as distance from 0.5
                        uncertainty = np.abs(probas[:, 1] - 0.5)
                        most_uncertain_idx = np.argsort(uncertainty)[:10]
                        
                        uncertain_df = pd.DataFrame({
                            'Actual Class': st.session_state.y_test[most_uncertain_idx],
                            'Predicted Class': st.session_state.predictions[most_uncertain_idx],
                            'Probability Class 1': probas[most_uncertain_idx, 1]
                        })
                        
                        st.dataframe(uncertain_df)
                    else:
                        st.info("Multi-class probability visualization not shown for simplicity.")
                
            else:  # K-Means
                # Cluster analysis
                st.write("Cluster Distribution")
                
                # Count number of samples in each cluster
                cluster_counts = np.bincount(st.session_state.predictions.astype(int))
                
                cluster_df = pd.DataFrame({
                    'Cluster': range(len(cluster_counts)),
                    'Count': cluster_counts
                })
                
                # Create bar chart
                fig = go.Figure(
                    go.Bar(
                        x=cluster_df['Cluster'],
                        y=cluster_df['Count'],
                        marker_color='blue'
                    )
                )
                
                fig.update_layout(
                    title="Number of Samples in Each Cluster",
                    xaxis_title="Cluster",
                    yaxis_title="Count",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show cluster centers
                if hasattr(st.session_state.model, 'cluster_centers_'):
                    st.write("Cluster Centers")
                    
                    centers = st.session_state.model.cluster_centers_
                    center_df = pd.DataFrame(
                        centers,
                        columns=st.session_state.feature_names
                    )
                    
                    center_df.index.name = 'Cluster'
                    center_df.reset_index(inplace=True)
                    
                    st.dataframe(center_df)
        
        # Download results button
        @st.cache_data
        def get_results_csv():
            if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
                results_df = pd.DataFrame({
                    'Actual': st.session_state.y_test,
                    'Predicted': st.session_state.predictions,
                    'Error': st.session_state.y_test - st.session_state.predictions
                })
            else:  # K-Means
                # For K-means, provide cluster assignments
                results_df = pd.DataFrame({
                    'Cluster': st.session_state.predictions
                })
                
                # Add the original features
                feature_df = pd.DataFrame(
                    st.session_state.X_test,
                    columns=st.session_state.feature_names
                )
                
                results_df = pd.concat([results_df, feature_df], axis=1)
            
            return results_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            "Download Results as CSV",
            get_results_csv(),
            "model_results.csv",
            "text/csv",
            key="download_results"
        )
        
        # Final summary and completion message
        st.success("🎉 Congratulations! You have successfully completed the FinTech ML Pipeline!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.button("Previous", on_click=prev_step, key="viz_prev")
        with col2:
            st.button("Start Over", on_click=reset_pipeline, key="viz_reset")
