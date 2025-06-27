import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("mushroom_model.pkl")
features = joblib.load("mushroom_features.pkl")

# Manual encoding (must match training time)
encoding_maps = {
    "odor": {'almond': 0, 'anise': 1, 'creosote': 2, 'fishy': 3, 'foul': 4, 'musty': 5, 'none': 6, 'pungent': 7, 'spicy': 8},
    "gill-color": {'black': 0, 'brown': 1, 'buff': 2, 'chocolate': 3, 'gray': 4, 'green': 5, 'orange': 6, 'pink': 7, 'purple': 8, 'red': 9, 'white': 10, 'yellow': 11},
    "spore-print-color": {'black': 0, 'brown': 1, 'buff': 2, 'chocolate': 3, 'green': 4, 'orange': 5, 'purple': 6, 'white': 7, 'yellow': 8},
    "habitat": {'grasses': 0, 'leaves': 1, 'meadows': 2, 'paths': 3, 'urban': 4, 'waste': 5, 'woods': 6},
    "bruises": {'bruises': 0, 'no': 1},
    "gill-spacing": {'close': 0, 'crowded': 1}
}

# Define form inputs
st.title("🍄 Mushroom Edibility Predictor")

user_input = {}
for feature in features:
    options = list(encoding_maps[feature].keys())
    user_input[feature] = st.selectbox(f"{feature.replace('-', ' ').title()}", options)

if st.button("Predict"):
    # Manually encode inputs using the encoding map
    encoded_input = {feature: encoding_maps[feature][user_input[feature]] for feature in features}
    
    df_encoded = pd.DataFrame([encoded_input])

    prediction = model.predict(df_encoded)[0]
    result = "🍽️ Edible" if prediction == 0 else "☠️ Poisonous"
    st.success(f"The mushroom is predicted to be: **{result}**")

