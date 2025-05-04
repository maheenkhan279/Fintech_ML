# FinTech ML Pipeline

A sophisticated Streamlit-based financial machine learning application that provides an end-to-end workflow for data analysis, model training, and interactive result visualization.

## Features

- Load financial data from Kragle datasets
- Preprocess and engineer features from the data
- Train machine learning models (Linear Regression, Logistic Regression, or K-Means Clustering)
- Evaluate model performance
- Visualize results with interactive charts

## Deployment Instructions

### Local Deployment

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fintech-ml-pipeline.git
cd fintech-ml-pipeline
```

2. Install the required packages:
```bash
pip install -r deployment_requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

### Deployment on Streamlit Cloud

1. Fork this repository to your GitHub account.

2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in with your GitHub account.

3. Click "New app" and select this repository.

4. In the deployment settings, set:
   - Main file path: `app.py`
   - Python version: 3.9 (or the version you prefer)

5. Click "Deploy" and your app will be live in a few minutes!

## Project Structure

- `app.py` - Main Streamlit application with UI components and workflow
- `data_loader.py` - Handles loading financial data from Kragle datasets
- `data_processor.py` - Contains data preprocessing and feature engineering logic
- `model_trainer.py` - Implements ML model training functionality
- `visualizer.py` - Handles data and results visualization
- `utils.py` - Utility functions for the application
- `.streamlit/` - Streamlit configuration and styling

## Requirements

The application requires the following Python packages:
- streamlit
- numpy
- pandas
- matplotlib
- plotly
- scikit-learn
- trafilatura
- xlsxwriter
- scipy

See `deployment_requirements.txt` for specific version requirements.