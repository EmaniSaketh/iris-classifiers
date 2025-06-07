import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🌼 Iris Flower Classification using Logistic Regression")

# Load and display data
@st.cache_data
def load_data():
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    df = pd.read_csv("iris.data", header=None, names=column_names)
    df.dropna(inplace=True)
    return df

df = load_data()
st.subheader("📄 Dataset Preview")
st.write(df.head())

# Split into features and label
X = df.iloc[:, :-1]
y = df['species']

# Encode species labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions for evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Show evaluation results
st.subheader("✅ Model Accuracy")
st.write(f"**Accuracy:** {acc:.2f}")

st.subheader("📋 Classification Report")
st.text(classification_report(y_test, y_pred, target_names=le.classes_))

st.subheader("📊 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# ---------------------------------------
# 🔮 User Input Prediction Section
# ---------------------------------------
st.subheader("🔍 Predict Iris Species")

# Get input from user
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]
    predicted_species = le.inverse_transform([prediction])[0]
    st.success(f"🌺 Predicted Iris Species: **{predicted_species}**")
