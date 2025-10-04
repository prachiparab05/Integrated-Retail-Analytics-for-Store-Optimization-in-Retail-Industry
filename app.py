import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Retail Sales Forecaster",
    page_icon="🛒",
    layout="wide"
)

# --- LOAD RESOURCES (Cached for performance) ---

@st.cache_resource
def load_model():
    """Load the pre-trained sales forecasting model."""
    try:
        model = joblib.load('sales_forecast_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file 'sales_forecast_model.joblib' not found. Please ensure it's in the same directory.")
        return None

@st.cache_data
def load_store_data():
    """Load the store dataset to get 'Type' and 'Size'."""
    try:
        stores_df = pd.read_csv('stores data-set.csv')
        return stores_df
    except FileNotFoundError:
        st.error("Store data 'stores data-set.csv' not found. Please ensure it's in the same directory.")
        return None

model = load_model()
stores_df = load_store_data()

# --- HEADER ---
st.title('🛒 Retail Sales Forecasting App')
st.write(
    "This application uses a Random Forest model to predict **Weekly Sales**. "
)
st.markdown("---")


# --- MAIN INPUT SECTION ---
st.header("1. Core Information")
col1, col2 = st.columns(2)

with col1:
    if stores_df is not None:
        valid_stores = sorted(stores_df['Store'].unique())
        store = st.selectbox('**Store Number**', valid_stores)
        
        # Display store details automatically
        if store:
            store_details = stores_df[stores_df['Store'] == store].iloc[0]
            store_size = store_details['Size']
            store_type = store_details['Type']
            st.info(f"**Store Details:** Type `{store_type}`, Size: `{store_size:,}` sq. ft.")
    else:
        store = st.number_input('**Store Number**', min_value=1, max_value=45, value=1)
        store_size = 150000 # Placeholder
        store_type = 'A'     # Placeholder

    dept = st.number_input('**Department Number**', min_value=1, max_value=99, value=1, help="Enter the department ID (1-99).")

with col2:
    date = st.date_input('**Date**', value=datetime(2022, 12, 2))
    is_holiday = st.selectbox('**Is it a Holiday Week?**', [True, False])


# --- EXPANDER FOR SECONDARY INPUTS ---
st.header("2. External Factors (Optional)")
with st.expander("Adjust Economic & Promotional Factors"):
    eco_col1, eco_col2 = st.columns(2)
    
    with eco_col1:
        st.subheader("Economic Factors")
        temperature = st.slider('Temperature (°F)', -10.0, 110.0, 65.0, 0.5)
        fuel_price = st.slider('Fuel Price ($)', 2.00, 5.00, 3.50, 0.01)
        cpi = st.slider('Consumer Price Index (CPI)', 120.0, 230.0, 195.0, 0.1)
        unemployment = st.slider('Unemployment Rate (%)', 3.0, 15.0, 7.5, 0.1)

    with eco_col2:
        st.subheader("Promotional Markdowns ($)")
        markdown1 = st.number_input('Markdown 1', value=0.0, step=100.0, help="e.g., Seasonal clearance")
        markdown2 = st.number_input('Markdown 2', value=0.0, step=100.0, help="e.g., Holiday event")
        markdown3 = st.number_input('Markdown 3', value=0.0, step=100.0, help="e.g., Category discount")
        markdown4 = st.number_input('Markdown 4', value=0.0, step=100.0, help="e.g., 'BOGO' deals")
        markdown5 = st.number_input('Markdown 5', value=0.0, step=100.0, help="e.g., Flyer coupon")


# --- PREDICTION BUTTON & RESULTS ---
st.markdown("---")
if st.button('Predict Weekly Sales', use_container_width=True, type="primary"):
    if model is None or stores_df is None:
        st.error("Cannot predict because required model or data files are missing.")
    else:
        with st.spinner('Analyzing data and making a prediction...'):
            # 1. Create a dictionary with all the features
            input_data = {
                'Store': store, 'Dept': dept, 'IsHoliday': 1 if is_holiday else 0,
                'Size': store_size, 'Temperature': temperature, 'Fuel_Price': fuel_price,
                'MarkDown1': markdown1, 'MarkDown2': markdown2, 'MarkDown3': markdown3,
                'MarkDown4': markdown4, 'MarkDown5': markdown5, 'CPI': cpi,
                'Unemployment': unemployment, 'Month': date.month, 'Year': date.year,
                'Type_A': 1 if store_type == 'A' else 0,
                'Type_B': 1 if store_type == 'B' else 0,
                'Type_C': 1 if store_type == 'C' else 0,
            }

            # 2. Convert to DataFrame with the exact column order from your notebook
            column_order = [
                'Store', 'Dept', 'IsHoliday', 'Size', 'Temperature', 'Fuel_Price',
                'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
                'CPI', 'Unemployment', 'Month', 'Year', 'Type_A', 'Type_B', 'Type_C'
            ]
            input_df = pd.DataFrame([input_data])[column_order]

            # 3. Make a prediction
            prediction = model.predict(input_df)

        # 4. Display the result
        st.subheader('Prediction Result')
        st.metric(label="Predicted Weekly Sales", value=f"${prediction[0]:,.2f}")
        
        with st.expander("See the full data sent to the model"):
            st.dataframe(input_df)
        
        st.success("Prediction complete!")

st.markdown("---")
st.info("This app uses a pre-trained Random Forest model. The accuracy is based on historical data and should be used as a strategic planning tool.")

