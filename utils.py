import streamlit as st
import time
import pandas as pd
import numpy as np
import base64
import os
from io import BytesIO
import matplotlib.pyplot as plt

def display_notification(message, type="info"):
    """
    Display a notification message with the specified type.
    
    Parameters:
    -----------
    message : str
        The message to display.
    type : str
        The type of notification: 'info', 'success', 'warning', or 'error'.
    """
    if type == "info":
        st.info(message)
    elif type == "success":
        st.success(message)
    elif type == "warning":
        st.warning(message)
    elif type == "error":
        st.error(message)
    else:
        st.write(message)

def load_finance_gif():
    """
    Load a finance-themed SVG animation.
    
    Returns:
    --------
    str
        Path to the finance-themed GIF or SVG.
    """
    # Return the path to the SVG file
    svg_path = "assets/fin_gif.svg"
    
    # Check if the file exists
    if not os.path.exists(svg_path):
        st.warning(f"SVG file {svg_path} not found. Using placeholder image.")
        return "https://via.placeholder.com/800x500.png?text=Finance+Analytics"
    
    return svg_path

def to_excel(df):
    """
    Convert a DataFrame to an Excel file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to convert.
        
    Returns:
    --------
    bytes
        Excel file as bytes.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    
    return output.getvalue()

def get_html_table(df, max_rows=10):
    """
    Convert a DataFrame to an HTML table.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to convert.
    max_rows : int
        Maximum number of rows to include.
        
    Returns:
    --------
    str
        HTML table as a string.
    """
    return df.head(max_rows).to_html(classes='dataframe')

def time_function(func):
    """
    Decorator to time a function execution.
    
    Parameters:
    -----------
    func : function
        Function to time.
        
    Returns:
    --------
    function
        Wrapped function that prints execution time.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

def format_large_number(num):
    """
    Format large numbers with suffixes like K, M, B.
    
    Parameters:
    -----------
    num : float
        Number to format.
        
    Returns:
    --------
    str
        Formatted number as a string.
    """
    for suffix in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000:
            return f"{num:.1f}{suffix}"
        num /= 1000
    return f"{num:.1f}P"  # Quadrillion

def create_color_palette(n_colors):
    """
    Create a color palette with the specified number of colors.
    
    Parameters:
    -----------
    n_colors : int
        Number of colors to generate.
        
    Returns:
    --------
    list
        List of colors in hex format.
    """
    # Create a colormap
    cmap = plt.cm.get_cmap('viridis', n_colors)
    
    # Convert colors to hex
    colors = []
    for i in range(n_colors):
        rgba = cmap(i)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgba[0] * 255), 
            int(rgba[1] * 255), 
            int(rgba[2] * 255)
        )
        colors.append(hex_color)
    
    return colors

def get_stock_ticker_examples():
    """
    Get a list of example stock tickers.
    
    Returns:
    --------
    list
        List of example stock tickers.
    """
    return [
        'AAPL',  # Apple
        'MSFT',  # Microsoft
        'GOOGL', # Alphabet (Google)
        'AMZN',  # Amazon
        'META',  # Meta Platforms (Facebook)
        'TSLA',  # Tesla
        'NVDA',  # NVIDIA
        'JPM',   # JPMorgan Chase
        'V',     # Visa
        'PYPL',  # PayPal
        'SQ',    # Block (Square)
        'COIN',  # Coinbase
        'DIS',   # Disney
        'NFLX',  # Netflix
        'UBER',  # Uber
        'ABNB',  # Airbnb
    ]

def get_economic_indicator_examples():
    """
    Get a list of example economic indicators.
    
    Returns:
    --------
    list
        List of example economic indicators.
    """
    return [
        'GDP Growth Rate',
        'Inflation Rate',
        'Unemployment Rate',
        'Interest Rate',
        'Consumer Price Index',
        'Producer Price Index',
        'Retail Sales',
        'Industrial Production',
        'Consumer Confidence',
        'Housing Starts',
        'Building Permits',
        'Existing Home Sales',
        'Durable Goods Orders',
        'Factory Orders',
        'Trade Balance',
        'Nonfarm Payrolls',
    ]
