import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load model, vectorizer, and metrics
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
metrics = joblib.load("model_metrics.pkl")

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide",
)

st.title("📧 Email Spam Classifier (TF-IDF + SVM)")
st.markdown(
    """
    A text classification app that predicts whether an email or SMS message is **Spam** or **Ham (Not Spam)**  
    using **TF-IDF** features and a **Support Vector Machine (SVM)** model.
    """
)

st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Choose an option:",
    ["Classify Message", "Model Dashboard", "Experiment Logs", "About"]
)

# ========== 1️⃣ CLASSIFY ==========
if option == "Classify Message":
    st.subheader("🔍 Message Classification")
    user_input = st.text_area("Enter your message:", height=150)

    if st.button("Classify"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a message.")
        else:
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]
            label = "Spam" if prediction == 1 else "Ham"

            if label == "Spam":
                st.error("🚨 This message is likely Spam!")
            else:
                st.success("✅ This message looks clean (Ham).")

# ========== 2️⃣ DASHBOARD ==========
elif option == "Model Dashboard":
    st.subheader("📊 Model Performance Dashboard")

    acc = round(metrics["accuracy"] * 100, 2)
    st.metric("Model Accuracy", f"{acc}%")

    # Confusion matrix
    cm = metrics["confusion_matrix"]
    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

    # Word cloud
    try:
        spam_text = joblib.load("spam_words.pkl")
        st.markdown("### Common Words in Spam Messages")
        wordcloud = WordCloud(width=700, height=400, background_color="white").generate(spam_text)
        st.image(wordcloud.to_array(), use_container_width=True)
    except:
        st.info("⚠️ Word cloud not found. Run training script to generate it.")

# ========== 3️⃣ EXPERIMENT LOGS ==========
elif option == "Experiment Logs":
    st.subheader("🧾 Experiment Tracking Logs")

    try:
        df_logs = pd.read_csv("experiment_log.csv", header=None, names=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
        st.dataframe(df_logs, use_container_width=True)
    except FileNotFoundError:
        st.warning("No experiment logs found. Train the model first.")

# ========== 4️⃣ ABOUT ==========
else:
    st.subheader("About the Project")
    st.markdown(
        """
        **Project Overview:**  
        - Developed an NLP-based spam detection model using **TF-IDF** and **SVM**.  
        - Preprocessing: text cleaning, tokenization, stopword removal, and vectorization.  
        - Achieved ~95% accuracy on real-world SMS datasets.  
        - Includes an interactive dashboard and experiment tracking via Streamlit.

        **Tech Stack:**  
        Python • Scikit-learn • Pandas • Streamlit • Joblib • Matplotlib • Seaborn

        **Developed by:** Fadhil  
        """
    )
