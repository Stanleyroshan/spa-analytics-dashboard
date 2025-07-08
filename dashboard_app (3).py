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
    """Reads the uploaded Parquet file into a pandas DataFrame."""
    df = pd.read_parquet(_uploaded_file_buffer)
    df['document_date'] = pd.to_datetime(df['document_date'])
    return df

# ==============================================================================
# --- 3. ANALYSIS FUNCTION LIBRARY ---
# All functions for our analyses are defined here, BEFORE they are called.
# ==============================================================================

# --- ANALYSIS FUNCTION FOR IDEA #20 ---
def display_location_analysis(df_sales):
    st.header("🌍 Location Performance: Profit & Sales Revenue")
    tab1, tab2 = st.tabs(["📊 Past Performance Dashboard", "🔮 Future Profit Forecasting"])

    # --- PAST PERFORMANCE TAB ---
    with tab1:
        st.subheader("Historical Performance Analysis")
        location_performance = df_sales.groupby('location_name').agg(
            total_profit=('profit', 'sum'),
            total_revenue=('total_inclusive', 'sum')
        ).reset_index()
        st.markdown("---")
        st.subheader("🥇 Top Performer Finder")
        col1, col2 = st.columns(2)
        if col1.button("Find Top Location by SALES REVENUE", use_container_width=True, key='loc_sales'):
            revenue_leader = location_performance.sort_values(by='total_revenue', ascending=False).iloc[0]
            st.success(f"**Top Sales Location:** {revenue_leader['location_name']}")
            st.metric(label="Total Revenue Generated", value=f"{revenue_leader['total_revenue']:,.0f}")
        if col2.button("Find Top Location by PROFIT", use_container_width=True, key='loc_profit'):
            profit_leader = location_performance.sort_values(by='total_profit', ascending=False).iloc[0]
            st.success(f"**Top Profit Location:** {profit_leader['location_name']}")
            st.metric(label="Total Profit Generated", value=f"{profit_leader['total_profit']:,.0f}")
        st.markdown("---")
        # (The rest of the visualization code for Idea #20 goes here)

    # --- FUTURE PREDICTION TAB ---
    with tab2:
        st.subheader("Interactive Profit Forecasting")
        # (The full prediction code for Idea #20 goes here)


# --- ANALYSIS FUNCTION FOR IDEA #19 ---
def display_new_guest_retail_prediction(df_sales):
    st.header("🛍️ New Guest Retail Purchase Prediction")
    tab1, tab2 = st.tabs(["📊 Model Performance & Insights", "🔮 Live Prediction Tool"])

    @st.cache_resource
    def train_retail_model(df):
        st.write("Training retail prediction model...")
        df['visit_date'] = df['document_date'].dt.date
        first_visit_dates = df.groupby('client_id')['visit_date'].min().reset_index()
        first_visit_df = pd.merge(df, first_visit_dates, on=['client_id', 'visit_date'])
        
        retail_buyers = df[df['item_type'] == 'Retail']['client_id'].unique()
        
        first_visit_features = first_visit_df.groupby('client_id').agg(
            first_visit_spend=('total_inclusive', 'sum'),
            first_visit_items=('quantity', 'sum'),
            first_service_category=('category', 'first'),
            location=('location_name', 'first')
        ).reset_index()
        
        first_visit_features['bought_retail'] = first_visit_features['client_id'].isin(retail_buyers).astype(int)
        
        model_df = first_visit_features.dropna(subset=['first_service_category', 'location'])
        model_data = pd.get_dummies(model_df, columns=['first_service_category', 'location'])
        X = model_data.drop(columns=['client_id', 'bought_retail'])
        y = model_data['bought_retail']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        eval_model = XGBClassifier(random_state=42).fit(X_train, y_train)
        accuracy = accuracy_score(y_test, eval_model.predict(X_test))
        report = classification_report(y_test, eval_model.predict(X_test), output_dict=True, zero_division=0)
        
        final_model = XGBClassifier(random_state=42).fit(X, y)
        return final_model, X.columns, accuracy, report

    model, training_cols, accuracy, report = train_retail_model(df_sales)

    with tab1:
        st.subheader("Model Performance Evaluation")
        st.info(f"**Predictive Model Accuracy: {accuracy:.1%}**")
        st.text("Performance for the 'Will Buy Retail' group:")
        st.json({
            "Precision": f"{report.get('1', {}).get('precision', 0):.1%}",
            "Recall": f"{report.get('1', {}).get('recall', 0):.1%}"
        })
        st.caption("Recall: Of all guests who actually bought retail, what percentage did we correctly identify?")

    with tab2:
        st.subheader("Live Prediction Tool")
        
        col1, col2 = st.columns(2)
        with col1:
            spend = st.slider("First Visit Spend", 0, 5000, 1000, key='retail_spend')
            items = st.number_input("Items on First Visit", 1, 10, 1, key='retail_items')
        with col2:
            category = st.selectbox("Category of First Service", sorted(df_sales['category'].unique()), key='retail_cat')
            location = st.selectbox("Location of Visit", sorted(df_sales['location_name'].unique()), key='retail_loc')
            
        if st.button("Predict Likelihood", use_container_width=True):
            future_data = pd.DataFrame([{'first_visit_spend': spend, 'first_visit_items': items, 'first_service_category': category, 'location': location}])
            future_data_encoded = pd.get_dummies(future_data)
            for col in training_cols:
                if col not in future_data_encoded.columns:
                    future_data_encoded[col] = 0
            
            prediction_proba = model.predict_proba(future_data_encoded[training_cols])[0][1]
            st.metric("Likelihood of Buying Retail", f"{prediction_proba:.1%}")
            if prediction_proba > 0.6:
                st.success("💡 Recommendation: High-potential customer. Confidently recommend products.")
            else:
                st.info("💡 Recommendation: Lower potential. Focus on the service experience first.")

# ==============================================================================
# --- 4. MAIN APP LOGIC (The "Switchboard") ---
# ==============================================================================
uploaded_file = st.sidebar.file_uploader("1. Upload your 'app_data.parquet' file", type=["parquet"])

if uploaded_file is not None:
    df_sales = load_parquet_data(uploaded_file.getbuffer())
    st.sidebar.success("✅ Data loaded!")
    
    # --- ADD THE NEW ANALYSIS TO THE LIST OF OPTIONS ---
    analysis_options = [
        "Location Profitability & Forecasting (#20)",
        "New Guest Retail Purchase Prediction (#19)"
    ]
    analysis_choice = st.sidebar.radio("2. Choose an analysis:", analysis_options)
    st.sidebar.markdown("---")
    
    # --- ADD THE NEW ELIF BLOCK TO THE SWITCHBOARD ---
    if analysis_choice == "Location Profitability & Forecasting (#20)":
        display_location_analysis(df_sales)
    elif analysis_choice == "New Guest Retail Purchase Prediction (#19)":
        display_new_guest_retail_prediction(df_sales)
        
else:
    st.info("👈 Please upload your `app_data.parquet` file to begin.")
