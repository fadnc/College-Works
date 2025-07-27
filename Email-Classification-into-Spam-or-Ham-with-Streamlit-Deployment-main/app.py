import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("SMS Spam Classifier")

user_input = st.text_area("Enter your message:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        label = "Spam" if prediction == 1 else "Ham"
        st.success(f"Prediction: {label}")
