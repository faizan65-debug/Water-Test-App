import streamlit as st
import pickle
import numpy as np
import sklearn  

# ----------------------------
# Load the trained model
# ----------------------------
import os
import pickle

model_path = os.path.join(os.path.dirname(__file__), "water-test-model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)


# ----------------------------
# Streamlit App Config
# ----------------------------
st.set_page_config(page_title="Water Potability App", page_icon="💧", layout="centered")
st.title("💧 Water Potability Prediction App")
st.markdown("Check if your water sample is **safe to drink** by entering quality parameters below:")

# ----------------------------
# User Inputs (with ranges shown)
# ----------------------------
st.subheader("🔬 Enter Water Quality Parameters")

ph = st.number_input("pH Value (Range: 0 - 14)", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
hardness = st.number_input("Hardness (mg/L) (Range: 0 - 500)", min_value=0.0, max_value=500.0, value=150.0, step=1.0)
solids = st.number_input("Total Dissolved Solids (ppm) (Range: 0 - 50,000)", min_value=0.0, max_value=50000.0, value=10000.0, step=10.0)
chloramines = st.number_input("Chloramines (mg/L) (Range: 0 - 15)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
sulfate = st.number_input("Sulfate (mg/L) (Range: 0 - 500)", min_value=0.0, max_value=500.0, value=250.0, step=1.0)
conductivity = st.number_input("Conductivity (μS/cm) (Range: 0 - 1000)", min_value=0.0, max_value=1000.0, value=400.0, step=1.0)
organic_carbon = st.number_input("Organic Carbon (mg/L) (Range: 0 - 30)", min_value=0.0, max_value=30.0, value=10.0, step=0.1)
trihalomethanes = st.number_input("Trihalomethanes (μg/L) (Range: 0 - 120)", min_value=0.0, max_value=120.0, value=60.0, step=0.1)
turbidity = st.number_input("Turbidity (NTU) (Range: 0 - 10)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

# Collect inputs into correct feature order
input_data = np.array([[ph, hardness, solids, chloramines, sulfate,
                        conductivity, organic_carbon, trihalomethanes, turbidity]])

# ----------------------------
# Prediction Section
# ----------------------------
if st.button("🚰 Check Potability"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction] * 100

    if prediction == 1:
        st.success(f"✅ The water is **Potable (Safe to Drink)**.\n\n🔹 Model confidence: **{probability:.2f}%**")
    else:
        st.error(f"❌ The water is **Not Potable**.\n\n🔹 Model confidence: **{probability:.2f}%**")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit")
