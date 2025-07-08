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

st.set_page_config(page_title="Business Analytics Dashboard", layout="wide")
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
            total_profit=('profit', 'sum'), total_revenue=('total_inclusive', 'sum')).reset_index()
        st.markdown("---")
        st.subheader("🥇 Top Performer Finder")
        col1, col2 = st.columns(2)
        if col1.button("Find Top Location by SALES REVENUE", use_container_width=True, key='loc_sales_btn'):
            revenue_leader = location_performance.sort_values(by='total_revenue', ascending=False).iloc[0]
            st.success(f"**Top Sales Location:** {revenue_leader['location_name']}")
            st.metric(label="Total Revenue Generated", value=f"{revenue_leader['total_revenue']:,.0f}")
        if col2.button("Find Top Location by PROFIT", use_container_width=True, key='loc_profit_btn'):
            profit_leader = location_performance.sort_values(by='total_profit', ascending=False).iloc[0]
            st.success(f"**Top Profit Location:** {profit_leader['location_name']}")
            st.metric(label="Total Profit Generated", value=f"{profit_leader['total_profit']:,.0f}")
        st.markdown("---")
        st.subheader("Performance Visualizations")
        vis_col1, vis_col2 = st.columns(2)
        with vis_col1:
            fig1, ax1 = plt.subplots(); sns.barplot(x='total_revenue', y='location_name', data=location_performance.sort_values(by='total_revenue', ascending=False), ax=ax1, palette='plasma'); ax1.set_title("Total Sales Revenue"); st.pyplot(fig1)
        with vis_col2:
            fig2, ax2 = plt.subplots(); sns.barplot(x='total_profit', y='location_name', data=location_performance.sort_values(by='total_profit', ascending=False), ax=ax2, palette='Greens_r'); ax2.set_title("Total Profit"); st.pyplot(fig2)

    with tab2:
        st.subheader("Interactive Profit Forecasting")
        @st.cache_resource
        def train_profit_model(df):
            monthly_profit = df.groupby(['location_name', pd.Grouper(key='document_date', freq='ME')]).agg(monthly_profit=('profit', 'sum')).reset_index()
            monthly_profit['month'], monthly_profit['year'] = monthly_profit['document_date'].dt.month, monthly_profit['document_date'].dt.year
            model_data = pd.get_dummies(monthly_profit, columns=['location_name'])
            X = model_data.drop(columns=['document_date', 'monthly_profit'])
            y = model_data['monthly_profit']
            if len(X) < 10: return None, None, "Not enough data for a reliable model."
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            eval_model = XGBRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
            accuracy = r2_score(y_test, eval_model.predict(X_test))
            final_model = XGBRegressor(n_estimators=100, random_state=42).fit(X, y)
            return final_model, X.columns, accuracy
        model, training_cols, accuracy = train_profit_model(df_sales)
        if model:
            st.info(f"**Predictive Model Accuracy (R-squared): {accuracy:.1%}**")
            all_locations = sorted(df_sales['location_name'].unique())
            selected_location = st.selectbox("Select a Location to Forecast:", all_locations, key='loc_forecast_select')
            if selected_location:
                forecast_year = df_sales['document_date'].dt.year.max() + 1
                forecasts = [{'Month': calendar.month_name[m], 'Forecasted Profit': model.predict(pd.DataFrame([{'month': m, 'year': forecast_year, **{c: (c == f'location_name_{selected_location}') for c in training_cols if 'location_name_' in c}}])[training_cols])[0]} for m in range(1, 13)]
                forecast_df = pd.DataFrame(forecasts)
                st.subheader(f"12-Month Profit Forecast for {selected_location}")
                fig, ax = plt.subplots(figsize=(12, 6)); sns.barplot(x='Month', y='Forecasted Profit', data=forecast_df, ax=ax, palette='cividis'); plt.xticks(rotation=45); st.pyplot(fig)
        else:
            st.warning(accuracy)
def display_new_guest_retail_prediction(df_sales):
    st.header("🛍️ New Guest Retail Purchase Prediction")
    tab1, tab2 = st.tabs(["📊 Model Performance & Insights", "🔮 Live Prediction Tool"])
    @st.cache_resource
    def train_retail_model(df):
        df['visit_date'] = df['document_date'].dt.date
        first_visit_dates = df.groupby('client_id')['visit_date'].min().reset_index()
        first_visit_df = pd.merge(df, first_visit_dates, on=['client_id', 'visit_date'])
        retail_buyers = df[df['item_type'] == 'Retail']['client_id'].unique()
        first_visit_features = first_visit_df.groupby('client_id').agg(
            first_visit_spend=('total_inclusive', 'sum'), first_visit_items=('quantity', 'sum'),
            first_service_category=('category', 'first'), location=('location_name', 'first')).reset_index()
        first_visit_features['bought_retail'] = first_visit_features['client_id'].isin(retail_buyers).astype(int)
        model_df = first_visit_features.dropna(subset=['first_service_category', 'location'])
        model_data = pd.get_dummies(model_df, columns=['first_service_category', 'location'])
        X = model_data.drop(columns=['client_id', 'bought_retail'])
        y = model_data['bought_retail']
        if len(y.unique()) < 2: return None, None, None, "Not enough data with both buyers and non-buyers."
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        eval_model = XGBClassifier(random_state=42).fit(X_train, y_train)
        accuracy = accuracy_score(y_test, eval_model.predict(X_test))
        report = classification_report(y_test, eval_model.predict(X_test), output_dict=True, zero_division=0)
        final_model = XGBClassifier(random_state=42).fit(X, y)
        return final_model, X.columns, accuracy, report
    model, training_cols, accuracy, report = train_retail_model(df_sales)
    with tab1:
        st.subheader("Model Performance Evaluation")
        if model:
            st.info(f"**Predictive Model Accuracy: {accuracy:.1%}**")
            st.json({"Precision_Will_Buy": f"{report.get('1', {}).get('precision', 0):.1%}", "Recall_Will_Buy": f"{report.get('1', {}).get('recall', 0):.1%}"})
            st.caption("Recall: Of all guests who actually bought retail, how many did we correctly identify?")
        else:
            st.warning(report)
    with tab2:
        st.subheader("Live Prediction Tool")
        if model:
            col1, col2 = st.columns(2)
            with col1:
                spend = st.slider("First Visit Spend", 0, 5000, 1000, key='retail_spend')
                items = st.number_input("Items on First Visit", 1, 10, 1, key='retail_items')
            with col2:
                category = st.selectbox("Category of First Service", sorted(df_sales['category'].unique()), key='retail_cat')
                location = st.selectbox("Location of Visit", sorted(df_sales['location_name'].unique()), key='retail_loc')
            if st.button("Predict Likelihood", use_container_width=True, key='retail_predict_btn'):
                future_data = pd.DataFrame([{'first_visit_spend': spend, 'first_visit_items': items, 'first_service_category': category, 'location': location}])
                future_data_encoded = pd.get_dummies(future_data)
                for col in training_cols:
                    if col not in future_data_encoded.columns: future_data_encoded[col] = 0
                prediction_proba = model.predict_proba(future_data_encoded[training_cols])[0][1]
                st.metric("Likelihood of Buying Retail", f"{prediction_proba:.1%}")
                if prediction_proba > 0.6: st.success("💡 Recommendation: High-potential customer. Confidently recommend products.")
                else: st.info("💡 Recommendation: Lower potential. Focus on the service experience first.")
        else:
            st.warning("Prediction tool disabled because the model could not be trained.")
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

uploaded_file = st.sidebar.file_uploader("1. Upload your 'app_data.parquet' file", type=["parquet"])

if uploaded_file is not None and not st.session_state.data_loaded:
    with st.spinner("Processing data... Please wait."):
        st.session_state.df_sales = load_parquet_data(uploaded_file)
    st.session_state.data_loaded = True
    st.rerun() 

if st.session_state.data_loaded:
    st.sidebar.success("✅ Data prepared successfully!")
    analysis_options = [
        "Location Profitability & Forecasting (#20)",
        "New Guest Retail Purchase Prediction (#19)"
    ]
    analysis_choice = st.sidebar.radio("2. Choose an analysis:", analysis_options)
    st.sidebar.markdown("---")
    
    if analysis_choice == "Location Profitability & Forecasting (#20)":
        display_location_analysis(st.session_state.df_sales)
    elif analysis_choice == "New Guest Retail Purchase Prediction (#19)":
        display_new_guest_retail_prediction(st.session_state.df_sales)
else:
    st.info("👈 Please upload your `app_data.parquet` file to begin.")
