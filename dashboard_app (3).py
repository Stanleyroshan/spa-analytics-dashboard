import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, classification_report
import calendar
import warnings

# --- 1. PROFESSIONAL PAGE CONFIGURATION & STYLING ---
st.set_page_config(page_title="Business Analytics Dashboard", layout="wide")
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="viridis")
st.title("🚀 Professional Business Analytics Dashboard")
st.sidebar.header("Controls")

# --- 2. DATA LOADING HELPER FUNCTION ---
@st.cache_data
def load_parquet_data(_uploaded_file_buffer):
    df = pd.read_parquet(_uploaded_file_buffer)
    df['document_date'] = pd.to_datetime(df['document_date'])
    return df

# ==============================================================================
# --- 3. ANALYSIS FUNCTION LIBRARY ---
# All functions for our analyses are defined here.
# ==============================================================================

# --- NEW ANALYSIS FUNCTION FOR IDEA #1 ---
def display_daily_sales_prediction(df):
    st.header("📅 Daily Sales Revenue Forecasting (#1)")
    
    @st.cache_resource
    def train_daily_model(df_inner):
        daily_sales = df_inner.groupby(df_inner['document_date'].dt.date)['total_inclusive'].sum().reset_index()
        daily_sales['document_date'] = pd.to_datetime(daily_sales['document_date'])
        daily_sales['day_of_week'] = daily_sales['document_date'].dt.dayofweek
        daily_sales['day_of_year'] = daily_sales['document_date'].dt.dayofyear
        daily_sales['month'] = daily_sales['document_date'].dt.month
        daily_sales['year'] = daily_sales['document_date'].dt.year
        X = daily_sales.drop(columns=['document_date', 'total_inclusive'])
        y = daily_sales['total_inclusive']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        eval_model = XGBRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        accuracy = r2_score(y_test, eval_model.predict(X_test))
        final_model = XGBRegressor(n_estimators=100, random_state=42).fit(X, y)
        return final_model, X.columns, accuracy

    model, training_cols, accuracy = train_daily_model(df)
    st.info(f"**Predictive Model Accuracy (R-squared): {accuracy:.1%}**")

    st.subheader("🗓️ 7-Day Sales Forecast")
    last_date = df['document_date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
    
    forecast_data = pd.DataFrame({'document_date': future_dates})
    forecast_data['day_of_week'] = forecast_data['document_date'].dt.dayofweek
    forecast_data['day_of_year'] = forecast_data['document_date'].dt.dayofyear
    forecast_data['month'] = forecast_data['document_date'].dt.month
    forecast_data['year'] = forecast_data['document_date'].dt.year
    
    # Ensure all columns are present for the model
    for col in training_cols:
        if col not in forecast_data.columns: forecast_data[col] = 0
    
    predictions = model.predict(forecast_data[training_cols])
    forecast_df = pd.DataFrame({'Day': future_dates.strftime('%A'), 'Predicted Sales': predictions})
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Day', y='Predicted Sales', data=forecast_df, ax=ax, palette='coolwarm')
    ax.set_title('Predicted Sales Revenue for the Next 7 Days')
    st.pyplot(fig)

# --- NEW ANALYSIS FUNCTION FOR IDEA #2 ---
def display_monthly_sales_forecasting(df):
    st.header("📈 Monthly Sales Revenue Forecasting (#2)")
    
    @st.cache_resource
    def train_monthly_model(df_inner):
        monthly_sales = df_inner.groupby(pd.Grouper(key='document_date', freq='ME'))['total_inclusive'].sum().reset_index()
        monthly_sales['month'] = monthly_sales['document_date'].dt.month
        monthly_sales['year'] = monthly_sales['document_date'].dt.year
        X = monthly_sales[['month', 'year']]
        y = monthly_sales['total_inclusive']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        eval_model = XGBRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        accuracy = r2_score(y_test, eval_model.predict(X_test))
        final_model = XGBRegressor(n_estimators=100, random_state=42).fit(X, y)
        return final_model, accuracy

    model, accuracy = train_monthly_model(df)
    st.info(f"**Predictive Model Accuracy (R-squared): {accuracy:.1%}**")
    
    st.subheader("🗓️ 12-Month Sales Forecast")
    last_date = df['document_date'].max()
    future_months_df = pd.DataFrame(pd.date_range(start=last_date, periods=13, freq='ME'), columns=['date'])
    future_months_df['month'] = future_months_df['date'].dt.month
    future_months_df['year'] = future_months_df['date'].dt.year
    
    predictions = model.predict(future_months_df[['month', 'year']])
    forecast_df = pd.DataFrame({'Month': future_months_df['date'].dt.strftime('%Y-%B'), 'Forecasted Revenue': predictions})
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Month', y='Forecasted Revenue', data=forecast_df, ax=ax, palette='magma')
    ax.set_title('Forecasted Revenue for the Next 12 Months')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# --- EXISTING ANALYSIS FUNCTION FOR IDEA #19 ---
def display_new_guest_retail_prediction(df):
    # (Full, correct code for Idea #19 goes here)
    st.header("🛍️ New Guest Retail Purchase Prediction (#19)")
    # ...

# --- EXISTING ANALYSIS FUNCTION FOR IDEA #20 ---
def display_location_analysis(df):
    # (Full, correct code for Idea #20 goes here)
    st.header("🌍 Location Performance: Profit & Sales Revenue")
    # ...

# ==============================================================================
# --- 4. MAIN APP LOGIC (The "Switchboard") ---
# ==============================================================================
uploaded_file = st.sidebar.file_uploader("1. Upload your 'app_data.parquet' file", type=["parquet"])

if uploaded_file is not None:
    df_sales = load_parquet_data(uploaded_file.getbuffer())
    st.sidebar.success("✅ Data loaded!")
    
    # --- UPDATED: The full list of available analyses ---
    analysis_options = [
        "Daily Sales Prediction (#1)",
        "Monthly Sales Forecasting (#2)",
        "New Guest Retail Purchase Prediction (#19)",
        "Location Profitability & Forecasting (#20)",
    ]
    analysis_choice = st.sidebar.radio("2. Choose an analysis:", analysis_options)
    st.sidebar.markdown("---")
    
    # --- UPDATED: The full switchboard to call the correct function ---
    if analysis_choice == "Daily Sales Prediction (#1)":
        display_daily_sales_prediction(df_sales)
    elif analysis_choice == "Monthly Sales Forecasting (#2)":
        display_monthly_sales_forecasting(df_sales)
    elif analysis_choice == "New Guest Retail Purchase Prediction (#19)":
        display_new_guest_retail_prediction(df_sales)
    elif analysis_choice == "Location Profitability & Forecasting (#20)":
        display_location_analysis(df_sales)
        
else:
    st.info("👈 Please upload your `app_data.parquet` file to begin.")
