import streamlit as st

import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.tree import DecisionTreeClassifier

import base64


# Page settings

st.set_page_config(page_title="Supermarket Sales Analysis", layout="wide")



# Custom Styling with CSS

st.markdown("""

    <style>

    /* Background image */

    .stApp {

        background-color: #f0f0f0;

        font-size: 20px !important;  /* Base font size */

    }


    /* Buttons */

    div.stButton > button {

        background-color: #c0afe3;

        color: #252526;

        border: None;

        padding: 0.6em 1.5em;

        font-size: 28px !important;  /* Button font size */

        border-radius: 8px;

        font-weight: bold;

    }


    div.stButton > button:hover {

        background-color: #c0afe3;

        color: #252526;

        transition: 0.3s;

    }


    /* Headings */

    h1 {

        font-size: 50px !important;

        color: #252526;

    }

    h2 {

        font-size: 36px !important;

        color: #252526;

    }

    </style>

""", unsafe_allow_html=True)



# Session state init

if 'page' not in st.session_state:

    st.session_state.page = 'home'



@st.cache_data

def load_data():

    df = pd.read_csv(r"C:\\path\\to\\your\\supermarket_sales.csv")

    df['Total'] = df['Unit price'] * df['Quantity']  # Deriving total sales

    return df



df = load_data()


# Encoding categorical variables

product_line_encoder = LabelEncoder()

df['Product line_encoded'] = product_line_encoder.fit_transform(df['Product line'])



# Pages

def home_page():

    st.title("🛒 Supermarket Sales Analysis")

    if st.button("👉 Get Started"):

        st.session_state.page = "menu"



def menu_page():

    st.title("📋 Choose What You Want to Do")


    col1, col2 = st.columns(2)

    with col1:

        if st.button("📊 Product Line Classification"):

            st.session_state.page = "classification"

        if st.button("📈 Total Sales Regression"):

            st.session_state.page = "regression"

    with col2:

        if st.button("👥 Customer Clustering"):

            st.session_state.page = "clustering"



def classification_page():

    st.title("📊 Classify Product Line")

    gender = st.selectbox("Gender", df['Gender'].unique())

    unit_price = st.number_input("Unit Price", min_value=0.0)

    quantity = st.number_input("Quantity", min_value=1)


    if st.button("🧠 Predict Product Line"):

        user_input = pd.DataFrame([{

            'Gender': gender,

            'Unit price': unit_price,

            'Quantity': quantity

        }])


        # Prepare data for classification

        X = df[['Gender', 'Unit price', 'Quantity']]

        y = df['Product line_encoded']

        model = DecisionTreeClassifier()

        model.fit(X, y)

        prediction = model.predict(user_input)[0]

        predicted_product_line = product_line_encoder.inverse_transform([prediction])[0]

        st.success(f"Predicted Product Line: {predicted_product_line}")


    if st.button("⬅️ Back to Menu"):

        st.session_state.page = "menu"



def regression_page():

    st.title("📈 Predict Total Sales")

    unit_price = st.number_input("Unit Price", min_value=0.0)

    quantity = st.number_input("Quantity", min_value=1)


    if st.button("Calculate Total Sales"):

        total_sales = unit_price * quantity

        st.success(f"Total Sales: {total_sales:.2f}")


    if st.button("⬅️ Back to Menu"):

        st.session_state.page = "menu"



def clustering_page():

    st.title("👥 Customer Clustering")

    st.markdown("Enter customer details to find similar customers:")


    with st.form("clustering_form"):

        gender = st.selectbox("Gender", df['Gender'].unique())

        unit_price = st.number_input("Unit Price", min_value=0.0)

        quantity = st.number_input("Quantity", min_value=1)

        submitted = st.form_submit_button("🔍 Find Similar Customers")


    if submitted:

        user_input = pd.DataFrame([{

            'Gender': gender 'Unit price': unit_price,

            'Quantity': quantity

        }])


        features = ['Gender', 'Unit price', 'Quantity']

        data = df[features].dropna()


        scaler = StandardScaler()

        scaled_data = scaler.fit_transform(data)

        scaled_user = scaler.transform(user_input[features])


        distances = np.linalg.norm(scaled_data - scaled_user, axis=1)

        similarity_threshold = 1.5

        similar_indices = np.where(distances <= similarity_threshold)[0]


        if len(similar_indices) > 0:

            similar_rows = data.iloc[similar_indices]

            st.success(f"Found {len(similar_rows)} similar customers!")

            st.dataframe(similar_rows.reset_index(drop=True))

        else:

            st.warning("No similar customers found. Try adjusting your input slightly.")


    if st.button("⬅️ Back to Menu"):

        st.session_state.page = "menu"



# Navigation Controller

if st.session_state.page == "home":

    home_page()

elif st.session_state.page == "menu":

    menu_page()

elif st.session_state.page == "classification":

    classification_page()

elif st.session_state.page == "regression":

    regression_page()

elif st.session_state.page == "clustering":

    clustering_page() 