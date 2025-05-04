import pandas as pd
import numpy as np
import streamlit as st
import os
import random
from datetime import datetime, timedelta

def load_kragle_dataset(dataset_type):
    """
    Load financial dataset from Kragle.
    Since we don't have actual access to Kragle, this function simulates 
    different financial datasets based on the dataset_type.
    
    Parameters:
    -----------
    dataset_type : str
        Type of financial dataset to load.
        Options: 'stock_market', 'financial_indicators', 'economic_data'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with financial data.
    """
    try:
        # Simulating Kragle datasets with different financial data
        if dataset_type == "stock_market":
            # Generate a stock market dataset with multiple stocks
            return generate_stock_market_data()
            
        elif dataset_type == "financial_indicators":
            # Generate financial indicators dataset
            return generate_financial_indicators_data()
            
        elif dataset_type == "economic_data":
            # Generate economic indicators dataset
            return generate_economic_data()
            
        else:
            st.error(f"Unknown dataset type: {dataset_type}")
            return None
    
    except Exception as e:
        st.error(f"Error loading Kragle dataset: {e}")
        return None

def generate_stock_market_data():
    """Generate simulated stock market data for multiple companies."""
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate dates for the past 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    business_days = pd.bdate_range(start=start_date, end=end_date)
    
    # List of simulated companies
    companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Start with an empty DataFrame
    market_data = pd.DataFrame()
    
    for company in companies:
        # Initial stock price
        initial_price = np.random.randint(50, 500)
        
        # Generate random price movements with some trends
        daily_returns = np.random.normal(0.0005, 0.015, len(business_days))
        
        # Add a few market shocks
        for i in range(3):
            shock_idx = np.random.randint(0, len(daily_returns))
            daily_returns[shock_idx] = np.random.choice([-0.08, 0.08])
        
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + daily_returns)
        prices = initial_price * cum_returns
        
        # Create volume data
        volume = np.random.randint(100000, 10000000, len(business_days))
        
        # Create company data
        company_data = pd.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in business_days],
            'Company': company,
            'Open': prices * np.random.uniform(0.99, 1.0, len(prices)),
            'High': prices * np.random.uniform(1.0, 1.05, len(prices)),
            'Low': prices * np.random.uniform(0.95, 1.0, len(prices)),
            'Close': prices,
            'Volume': volume,
            'MarketCap': prices * np.random.randint(1000000, 10000000, len(prices)),
            'PE_Ratio': np.random.uniform(10, 30, len(prices)),
            'Dividend_Yield': np.random.uniform(0.5, 3.0, len(prices)) / 100,
        })
        
        market_data = pd.concat([market_data, company_data])
    
    # Reset index
    market_data.reset_index(drop=True, inplace=True)
    
    return market_data

def generate_financial_indicators_data():
    """Generate simulated financial indicators data for companies."""
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate quarterly data for 5 years
    quarters = pd.date_range(start='2018-01-01', end='2023-01-01', freq='Q')
    
    # List of simulated companies
    companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'TSLA']
    sectors = ['Technology', 'Technology', 'Technology', 'Consumer Cyclical', 
              'Communication Services', 'Communication Services', 'Automotive']
    
    # Start with an empty DataFrame
    financial_data = pd.DataFrame()
    
    for i, company in enumerate(companies):
        # Generate financials with some trends
        revenue_base = np.random.randint(1000, 10000)
        revenue_growth = np.random.uniform(0.02, 0.1)
        revenue = [revenue_base * (1 + revenue_growth) ** i for i in range(len(quarters))]
        
        # Add seasonal variations
        for j in range(len(revenue)):
            quarter = j % 4
            if quarter == 0:  # Q1
                revenue[j] *= np.random.uniform(0.9, 0.95)
            elif quarter == 3:  # Q4
                revenue[j] *= np.random.uniform(1.1, 1.2)
        
        # Create profit margins that fluctuate
        profit_margin_base = np.random.uniform(0.1, 0.3)
        profit_margins = [profit_margin_base + np.random.uniform(-0.05, 0.05) for _ in range(len(quarters))]
        
        # Calculate net income based on revenue and profit margin
        net_income = [revenue[j] * profit_margins[j] for j in range(len(quarters))]
        
        # Create other financial metrics
        total_assets = [revenue[j] * np.random.uniform(2.5, 4.0) for j in range(len(quarters))]
        total_debt = [total_assets[j] * np.random.uniform(0.2, 0.5) for j in range(len(quarters))]
        equity = [total_assets[j] - total_debt[j] for j in range(len(quarters))]
        
        # Create company data
        company_data = pd.DataFrame({
            'Date': [q.strftime('%Y-%m-%d') for q in quarters],
            'Company': company,
            'Sector': sectors[i],
            'Revenue': revenue,
            'NetIncome': net_income,
            'TotalAssets': total_assets,
            'TotalDebt': total_debt,
            'Equity': equity,
            'ROE': [net_income[j] / equity[j] for j in range(len(quarters))],
            'ROA': [net_income[j] / total_assets[j] for j in range(len(quarters))],
            'DebtToEquity': [total_debt[j] / equity[j] for j in range(len(quarters))],
            'ProfitMargin': profit_margins
        })
        
        financial_data = pd.concat([financial_data, company_data])
    
    # Reset index
    financial_data.reset_index(drop=True, inplace=True)
    
    return financial_data

def generate_economic_data():
    """Generate simulated economic indicators data."""
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate monthly data for 10 years
    months = pd.date_range(start='2013-01-01', end='2023-01-01', freq='M')
    
    # Start with base values for various economic indicators
    gdp_base = 20000  # Billions
    inflation_base = 0.02  # 2%
    unemployment_base = 0.06  # 6%
    interest_rate_base = 0.025  # 2.5%
    
    # Generate indicators with trends and cycles
    data = {
        'Date': [m.strftime('%Y-%m-%d') for m in months],
        'Country': 'US',
        'GDP': [gdp_base * (1 + 0.005) ** i + np.random.normal(0, 100) for i in range(len(months))],
        'Inflation': [inflation_base + 0.001 * np.sin(i/12) + np.random.normal(0, 0.002) for i in range(len(months))],
        'Unemployment': [unemployment_base - 0.0002 * i + 0.01 * np.sin(i/24) + np.random.normal(0, 0.003) for i in range(len(months))],
        'InterestRate': [interest_rate_base + 0.0005 * i + np.random.normal(0, 0.001) for i in range(len(months))],
        'ConsumerSentiment': [75 + 10 * np.sin(i/12) + np.random.normal(0, 3) for i in range(len(months))],
        'RetailSales': [450 + i + 20 * np.sin(i/6) + np.random.normal(0, 10) for i in range(len(months))],
        'HousingStarts': [1500 + 500 * np.sin(i/12) + np.random.normal(0, 100) for i in range(len(months))],
    }
    
    # Create DataFrame
    economic_data = pd.DataFrame(data)
    
    # Add recession indicator (True for a period in the middle)
    recession_start = len(months) // 3
    recession_end = recession_start + 12  # 12 month recession
    economic_data['Recession'] = False
    economic_data.loc[recession_start:recession_end, 'Recession'] = True
    
    # During recession, worsen some indicators
    economic_data.loc[recession_start:recession_end, 'GDP'] *= 0.98
    economic_data.loc[recession_start:recession_end, 'Unemployment'] *= 1.3
    economic_data.loc[recession_start:recession_end, 'ConsumerSentiment'] *= 0.85
    economic_data.loc[recession_start:recession_end, 'RetailSales'] *= 0.9
    
    # Add more countries with correlated but different data
    countries = ['UK', 'Germany', 'Japan', 'China']
    
    for country in countries:
        # Create copy of US data with variations
        country_data = economic_data[economic_data['Country'] == 'US'].copy()
        country_data['Country'] = country
        
        # Add country-specific variations
        gdp_factor = np.random.uniform(0.2, 1.5)
        country_data['GDP'] = country_data['GDP'] * gdp_factor
        
        inflation_offset = np.random.uniform(-0.01, 0.02)
        country_data['Inflation'] = country_data['Inflation'] + inflation_offset
        
        unemployment_factor = np.random.uniform(0.7, 1.5)
        country_data['Unemployment'] = country_data['Unemployment'] * unemployment_factor
        
        interest_offset = np.random.uniform(-0.01, 0.02)
        country_data['InterestRate'] = country_data['InterestRate'] + interest_offset
        
        # Add noise to all metrics
        for col in ['GDP', 'Inflation', 'Unemployment', 'InterestRate', 'ConsumerSentiment', 'RetailSales', 'HousingStarts']:
            noise_factor = np.random.uniform(0.02, 0.1)
            noise = np.random.normal(0, noise_factor * country_data[col].mean(), len(country_data))
            country_data[col] = country_data[col] + noise
        
        # Concatenate with main dataset
        economic_data = pd.concat([economic_data, country_data])
    
    # Reset index
    economic_data.reset_index(drop=True, inplace=True)
    
    return economic_data
