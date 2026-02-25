import streamlit as st
import pickle
import pandas as pd

# Load model data
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
columns = data["columns"]

# Title
st.title("🚗 Car Price Prediction App")

st.write("Enter car details:")

# Inputs
car = st.selectbox(
    "Car Name",
    ["Swift","i20","City","Creta","Baleno","Verna","Innova","Alto","Ertiga","Polo"]
)

year = st.number_input("Year", 2000, 2025, 2018)

fuel = st.selectbox("Fuel Type", ["Petrol","Diesel"])

km = st.number_input("KM Driven", 0, 200000, 30000)

trans = st.selectbox("Transmission", ["Manual","Automatic"])

owner = st.selectbox("Owner Type", ["First","Second"])


# Predict button
if st.button("Predict Price"):

    # Create input
    input_df = pd.DataFrame(
        [[car, year, fuel, km, trans, owner]],
        columns=["Car_Name","Year","Fuel_Type","KM_Driven","Transmission","Owner_Type"]
    )

    # One-hot encode
    input_df = pd.get_dummies(input_df)

    # Match training columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Predict
    price = model.predict(input_df)[0]

    st.success(f"Estimated Price: ₹ {int(price)}")