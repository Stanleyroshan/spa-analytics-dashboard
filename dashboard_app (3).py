import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import calendar
import warnings

st.set_page_config(
    page_title="Business Analytics Dashboard",
    page_icon="🚀",
    layout="wide"
)
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="viridis")
st.title("🚀 Professional Business Analytics Dashboard")
st.sidebar.header("Controls")

@st.cache_data
def load_parquet_data(_uploaded_file_buffer):
    """Reads the uploaded Parquet file into a pandas DataFrame."""
    df = pd.read_parquet(_uploaded_file_buffer)
    df['document_date'] = pd.to_datetime(df['document_date'])
    return df

def display_location_analysis(df_sales):
    st.header("🌍 Location Performance: Profit & Sales Revenue")
    tab1, tab2 = st.tabs(["📊 Past Performance Dashboard", "🔮 Future Profit Forecasting"])

    with tab1:
        st.subheader("Historical Performance Analysis")
        
        location_performance = df_sales.groupby('location_name').agg(
            total_profit=('profit', 'sum'),
            total_revenue=('total_inclusive', 'sum')
        ).reset_index()
        
        st.markdown("---")
        st.subheader("🥇 Top Performer Finder")
        col1, col2 = st.columns(2)
        if col1.button("Find Top Location by SALES REVENUE", use_container_width=True):
            revenue_leader = location_performance.sort_values(by='total_revenue', ascending=False).iloc[0]
            st.success(f"**Top Sales Location:** {revenue_leader['location_name']}")
            st.metric(label="Total Revenue Generated", value=f"{revenue_leader['total_revenue']:,.0f}")
        if col2.button("Find Top Location by PROFIT", use_container_width=True):
            profit_leader = location_performance.sort_values(by='total_profit', ascending=False).iloc[0]
            st.success(f"**Top Profit Location:** {profit_leader['location_name']}")
            st.metric(label="Total Profit Generated", value=f"{profit_leader['total_profit']:,.0f}")
        st.markdown("---")
        
        st.subheader("Performance Visualizations")
        vis_col1, vis_col2 = st.columns(2)
        with vis_col1:
            fig1, ax1 = plt.subplots()
            sns.barplot(x='total_revenue', y='location_name', data=location_performance.sort_values('total_revenue', ascending=False), ax=ax1, palette='plasma')
            ax1.set_title("Total Sales Revenue by Location")
            st.pyplot(fig1)
        with vis_col2:
            fig2, ax2 = plt.subplots()
            sns.barplot(x='total_profit', y='location_name', data=location_performance.sort_values('total_profit', ascending=False), ax=ax2, palette='Greens_r')
            ax2.set_title("Total Profit by Location")
            st.pyplot(fig2)

    with tab2:
        st.subheader("Interactive Profit Forecasting")
        
        @st.cache_resource
        def train_profit_model(df):
            monthly_profit = df.groupby(['location_name', pd.Grouper(key='document_date', freq='ME')]).agg(monthly_profit=('profit', 'sum')).reset_index()
            monthly_profit['month'], monthly_profit['year'] = monthly_profit['document_date'].dt.month, monthly_profit['document_date'].dt.year
            model_data = pd.get_dummies(monthly_profit, columns=['location_name'])
            X = model_data.drop(columns=['document_date', 'monthly_profit'])
            y = model_data['monthly_profit']
            if len(X) < 10: return None, None, "Not enough data to train a reliable model."
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            eval_model = XGBRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
            accuracy = r2_score(y_test, eval_model.predict(X_test))
            final_model = XGBRegressor(n_estimators=100, random_state=42).fit(X, y)
            return final_model, X.columns, accuracy

        model, training_cols, accuracy = train_profit_model(df_sales)
        
        if model:
            st.info(f"**Predictive Model Accuracy (R-squared): {accuracy:.1%}**")
            all_locations = sorted(df_sales['location_name'].unique())
            selected_location = st.selectbox("Select a Location to Forecast:", all_locations)
            
            if selected_location:
                forecast_year = df_sales['document_date'].dt.year.max() + 1
                forecasts = []
                for month in range(1, 13):
                    future_data = pd.DataFrame([{'month': month, 'year': forecast_year}])
                    for col in [c for c in training_cols if 'location_name_' in c]:
                        future_data[col] = (col == f'location_name_{selected_location}')
                    for col in training_cols:
                        if col not in future_data.columns: future_data[col] = 0
                    prediction = model.predict(future_data[training_cols])[0]
                    forecasts.append({'Month': calendar.month_name[month], 'Forecasted Profit': prediction})
                forecast_df = pd.DataFrame(forecasts)
                
                st.subheader(f"12-Month Profit Forecast for {selected_location}")
                fig, ax = plt.subplots(figsize=(12, 6)); sns.barplot(x='Month', y='Forecasted Profit', data=forecast_df, ax=ax, palette='cividis'); plt.xticks(rotation=45); st.pyplot(fig)
                
                peak_month = forecast_df.loc[forecast_df['Forecasted Profit'].idxmax()]
                st.success(f"**Automated Recommendation:** The model predicts **{peak_month['Month']}** will be the most profitable month for '{selected_location}'.")
        else:
            st.warning("Could not train the predictive model. More monthly data is needed.")

uploaded_file = st.sidebar.file_uploader("1. Upload your 'app_data.parquet' file", type=["parquet"])
if uploaded_file is not None:
    df_sales = load_parquet_data(uploaded_file)
    st.sidebar.success("✅ Data loaded!")
    analysis_options = ["Location Profitability & Forecasting (#20)"]
    analysis_choice = st.sidebar.radio("2. Choose an analysis:", analysis_options)
    
    if analysis_choice == "Location Profitability & Forecasting (#20)":
        display_location_analysis(df_sales)
else:
    st.info("👈 Please upload your `app_data.parquet` file to begin.")
