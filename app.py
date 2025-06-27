import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model and features
model = joblib.load("mushroom_model.pkl")
features = joblib.load("mushroom_features.pkl")

# Define categorical options for selected features
feature_options = {
    "odor": ['almond', 'anise', 'creosote', 'fishy', 'foul', 'musty', 'none', 'pungent', 'spicy'],
    "gill-color": ['black', 'brown', 'buff', 'chocolate', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow'],
    "spore-print-color": ['black', 'brown', 'buff', 'chocolate', 'green', 'orange', 'purple', 'white', 'yellow'],
    "habitat": ['grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste', 'woods'],
    "bruises": ['bruises', 'no'],
    "gill-spacing": ['close', 'crowded']
}

st.title("🍄 Mushroom Edibility Predictor")

# Get user input
user_input = {}
for feature in features:
    user_input[feature] = st.selectbox(f"{feature.replace('-', ' ').title()}", feature_options[feature])

if st.button("Predict"):
    # Encode input using same encoding logic
    df_input = pd.DataFrame([user_input])
    df_encoded = df_input.apply(LabelEncoder().fit_transform)

    # Align to model feature order
    df_encoded = df_encoded.reindex(columns=features, fill_value=0)

    # Predict
    prediction = model.predict(df_encoded)[0]
    result = "🍽️ Edible" if prediction == 0 else "☠️ Poisonous"
    st.success(f"The mushroom is predicted to be: **{result}**")
