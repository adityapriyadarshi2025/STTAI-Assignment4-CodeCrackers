import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="UrbanNest Rent Predictor", page_icon="🏠", layout="centered")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    with open("models/best_rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("models/feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return model, encoders, feature_columns

model, encoders, feature_columns = load_artifacts()

st.title("UrbanNest Rent Predictor")
st.write("Predict monthly rent (INR) for properties in Mumbai, Delhi, Pune & Hisar.")

# ---------- Inputs ----------
st.header("Property Details")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("City", sorted(encoders["city"].classes_))
    location = st.selectbox("Location", sorted(encoders["location"].classes_))
    property_type = st.selectbox("Property Type", sorted(encoders["property_type"].classes_))
    status = st.selectbox("Furnishing Status", sorted(encoders["Status"].classes_))
    size = st.number_input("Size (ft²)", min_value=100, max_value=10000, value=1000, step=50)
    bhk = st.selectbox("BHK (1 = BHK, 0 = RK)", [1, 0])
    rooms_num = st.number_input("Number of Rooms", min_value=1, max_value=10, value=2)

with col2:
    num_bathrooms = st.number_input("Number of Bathrooms", min_value=0, max_value=10, value=2)
    num_balconies = st.number_input("Number of Balconies", min_value=0, max_value=10, value=1)
    is_negotiable = st.selectbox("Is Price Negotiable?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    security_deposit = st.number_input("Security Deposit (INR)", min_value=0, max_value=1000000, value=50000, step=5000)
    latitude = st.number_input("Latitude", value=19.0760, format="%.4f")
    longitude = st.number_input("Longitude", value=72.8777, format="%.4f")
    verification_days = st.number_input("Verification Days (days since posted)", min_value=0.0, max_value=1000.0, value=30.0)

# ---------- Predict ----------
if st.button("Predict Rent", type="primary"):
    # Build raw input dict matching dataset columns
    raw = {
        "location": location,
        "city": city,
        "latitude": latitude,
        "longitude": longitude,
        "numBathrooms": num_bathrooms,
        "numBalconies": num_balconies,
        "isNegotiable": is_negotiable,
        "SecurityDeposit": security_deposit,
        "Status": status,
        "Size_ft²": size,
        "BHK": bhk,
        "rooms_num": rooms_num,
        "property_type": property_type,
        "verification_days": verification_days,
    }

    # Apply label encoders to categorical columns
    for col, le in encoders.items():
        raw[col] = int(le.transform([str(raw[col])])[0])

    # Build DataFrame in the exact feature order the model was trained on
    X = pd.DataFrame([raw])[feature_columns]

    prediction = model.predict(X)[0]
    st.success(f"Predicted Monthly Rent: ₹ {prediction:,.0f}")
    st.caption("Prediction based on Random Forest model tuned via Bayesian/Grid/Random search.")
