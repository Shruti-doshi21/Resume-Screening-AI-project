import streamlit as st
import pandas as pd
import PyPDF2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Load dataset
data = pd.read_csv("resume_data.csv")


# Features
X = data["resume"]

# Labels
y = data["label"]


# TF-IDF
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)


# Model
model = LogisticRegression()

model.fit(X_train, y_train)


# Accuracy
accuracy = model.score(X_test, y_test)


# Streamlit UI
st.title("Resume Screening AI")

st.write("Model Accuracy:", accuracy)


# Upload PDF
uploaded_file = st.file_uploader("Upload Resume PDF")


if uploaded_file:

    try:

        reader = PyPDF2.PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            text += page.extract_text()

        st.subheader("Extracted Text")
        st.write(text)

        # Prediction
        new_vector = vectorizer.transform([text])

        prediction = model.predict(new_vector)

        st.subheader("Prediction")

        if prediction[0] == 1:
            st.success("Selected")

        else:
            st.error("Rejected")

    except:
        st.error("Error reading PDF")
