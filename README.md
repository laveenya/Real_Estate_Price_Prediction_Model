# Real_Estate_Price_Prediction_Model
!pip install streamlit
from google.colab import files
uploaded = files.upload()

import pandas as pd
df = pd.read_csv('bengaluru_real_estate_dummy.csv')  # Use the exact name of the file you uploaded

!pip install streamlit scikit-learn pandas

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
import pickle

# Upload the dataset
from google.colab import files
uploaded = files.upload()

df = pd.read_csv('bengaluru_real_estate_dummy.csv')
df = df[df['Sale_Price'].notnull()]

# Features and Target
features = ['Area_sqft', 'Bedrooms', 'Bathrooms', 'Age', 'Amenities_Count', 'Furnish_Status', 'Locality', 'Property_Type']
X = df[features]
y = df['Sale_Price']

# Label Encoding
le_furnish = LabelEncoder()
le_locality = LabelEncoder()
le_ptype = LabelEncoder()
X['Furnish_Status'] = le_furnish.fit_transform(X['Furnish_Status'])
X['Locality'] = le_locality.fit_transform(X['Locality'])
X['Property_Type'] = le_ptype.fit_transform(X['Property_Type'])

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=200, random_state=101)
model.fit(X_train, y_train)

# Save encoders and scaler
with open('le_furnish.pkl', 'wb') as f:
    pickle.dump(le_furnish, f)
with open('le_locality.pkl', 'wb') as f:
    pickle.dump(le_locality, f)
with open('le_ptype.pkl', 'wb') as f:
    pickle.dump(le_ptype, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

%%writefile app.py

import streamlit as st
import pickle
import numpy as np

# Load encoders and model
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le_furnish = pickle.load(open('le_furnish.pkl', 'rb'))
le_locality = pickle.load(open('le_locality.pkl', 'rb'))
le_ptype = pickle.load(open('le_ptype.pkl', 'rb'))

st.header("🔑 Bengaluru Real Estate Sale Price Predictor")

# Form Inputs
area = st.number_input("Area (sqft):", min_value=1, value=1000)
bedrooms = st.number_input("Bedrooms:", min_value=1, max_value=10, value=2)
bathrooms = st.number_input("Bathrooms:", min_value=1, max_value=10, value=2)
age = st.number_input("Age (years):", min_value=0, max_value=100, value=5)
amenities = st.number_input("Amenities Count:", min_value=0, max_value=10, value=2)

furnish_status = st.selectbox("Furnish Status:", le_furnish.classes_)
locality = st.selectbox("Locality:", le_locality.classes_)
ptype = st.selectbox("Property Type:", le_ptype.classes_)

if st.button("Predict Price"):
    input_data = np.array([
        area,
        bedrooms,
        bathrooms,
        age,
        amenities,
        le_furnish.transform([furnish_status])[0],
        le_locality.transform([locality])[0],
        le_ptype.transform([ptype])[0]
    ]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)

    # pred is array, so use .item() to get scalar before rounding
    pred_val = pred.item()
    rounded_pred = round(pred_val, 2)

    st.success(f"🏠 Predicted Sale Price: ₹ {round(pred.item(), 2):,}")

st.caption("Powered by AI | DeepEstate (G6, VIT Pune)")

!pip install pyngrok
from pyngrok import ngrok

# Start ngrok tunnel on localhost port 8501 (default Streamlit port)
public_url = ngrok.connect(addr="8501")
print(public_url)
ngrok.set_auth_token("YOUR AUTH_TOKEN")

# Import ngrok, start Streamlit and open tunnel
from pyngrok import ngrok
import subprocess

# Start streamlit app in background (adjust command if needed)
process = subprocess.Popen(["streamlit", "run", "app.py", "--server.port=8501"])

# Open ngrok tunnel to 8501
public_url = ngrok.connect(addr="8501")
print(f"Streamlit app URL: {public_url}")
