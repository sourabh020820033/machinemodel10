import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load and encode data
df = pd.read_csv("mushrooms.csv")
df = df.apply(LabelEncoder().fit_transform)

# Select only key features
selected_features = ["odor", "gill-color", "spore-print-color", "habitat", "bruises", "gill-spacing"]
X = df[selected_features]
y = df["class"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = SVC()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["edible", "poisonous"]))

# Save model and selected feature list
joblib.dump(model, "mushroom_model.pkl")
joblib.dump(selected_features, "mushroom_features.pkl")
