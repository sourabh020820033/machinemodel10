import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("mushrooms.csv")

# Manually encode the class column first: 'e' = 0 (edible), 'p' = 1 (poisonous)
df['class'] = df['class'].map({'e': 0, 'p': 1})

# Encode all other features
le = LabelEncoder()
for col in df.columns:
    if col != 'class':
        df[col] = le.fit_transform(df[col])

# Select features
selected_features = ["odor", "gill-color", "spore-print-color", "habitat", "bruises", "gill-spacing"]
X = df[selected_features]
y = df["class"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = SVC()
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and features
joblib.dump(model, "mushroom_model.pkl")
joblib.dump(selected_features, "mushroom_features.pkl")
