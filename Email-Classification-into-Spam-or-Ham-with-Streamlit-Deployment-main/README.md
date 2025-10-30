##  Email Spam Classification using NLP & SVM

A machine learning project that classifies email or SMS messages as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** and a **Support Vector Machine (SVM)** model.
Deployed as an **interactive Streamlit web app** for real-time message classification.

---

###  Project Overview

This project leverages **TF-IDF vectorization** for text feature extraction and **Support Vector Machines (SVM)** for classification.
It also includes NLP-based preprocessing (tokenization, stopword removal, and lemmatization) to clean text data and improve accuracy.

---

###  Features
 
 Classifies any text message as **Spam** or **Ham**
 Uses **TF-IDF** and **SVM (linear/RBF kernels)** for robust classification
 Implements **GridSearchCV** for hyperparameter tuning
 Provides **confidence score (%)** for predictions in Streamlit
 Visualizes results with a **Confusion Matrix**
 Simple, clean, and deployable interface

---

###  Tech Stack

| Component          | Technology                                              |
| ------------------ | ------------------------------------------------------- |
| Language           | Python                                                  |
| Libraries          | scikit-learn, pandas, nltk, seaborn, matplotlib, joblib |
| Model              | Support Vector Machine (SVM)                            |
| Feature Extraction | TF-IDF Vectorizer                                       |
| Deployment         | Streamlit                                               |

---

###  Setup Instructions

#### 1️. Clone the Repository

```bash
git clone https://github.com/<your-username>/Email-Spam-Classifier.git
cd Email-Spam-Classifier
```

#### 2️. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3️. Train the Model

Ensure your dataset file (`SMSSpamCollection`) is in the correct path, or update it in `train.py`.

Then run:

```bash
python train.py
```

This will generate:

* `spam_classifier_model.pkl`
* `tfidf_vectorizer.pkl`

#### 4️. Launch Streamlit App

```bash
streamlit run app.py
```

Open your browser at:
 **[http://localhost:8501](http://localhost:8501)**

---

###  Example Output

**Input:**

> “Congratulations! You’ve won a $1000 gift card. Claim now!”

**Output:**
 Prediction: **Spam** (98.45% confidence)

---

###  Model Performance

| Metric    | Score                    |
| --------- | ------------------------ |
| Accuracy  | ~95%                     |
| Precision | High for both Spam & Ham |
| Recall    | Balanced across classes  |

A Confusion Matrix heatmap is displayed after training to visualize performance.

---

###  Preprocessing Steps

1. Lowercasing text
2. Removing punctuation and non-alphanumeric characters
3. Removing English stopwords
4. Lemmatization using WordNetLemmatizer
5. TF-IDF Vectorization with `max_features=5000`

---

###  Deployment

You can deploy your Streamlit app for free on [Streamlit Cloud](https://streamlit.io/cloud):

1. Push your repo to GitHub
2. Sign in to Streamlit Cloud
3. Connect your repo and deploy instantly

Your app link will look like:
🔗 `https://fadhil-spam-classifier.streamlit.app`

---

###  Folder Structure

```
Email-Spam-Classifier/
│
├── app.py                     # Streamlit web app
├── train.py                   # Model training & evaluation script
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── spam_classifier_model.pkl   # Trained SVM model (generated)
├── tfidf_vectorizer.pkl        # TF-IDF vectorizer (generated)
└── dataset/
    └── SMSSpamCollection       # Input dataset
```

---

###  Dataset Used

Dataset: **[SMS Spam Collection Dataset (UCI Machine Learning Repository)](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)**
Contains 5,574 labeled messages — “ham” (legitimate) or “spam” (unwanted).

---

###  Results

* Optimized SVM using **GridSearchCV** (`C`, `kernel`, `gamma`)
* Achieved **~95% accuracy**
* Demonstrated stable performance on unseen messages
* Clean and user-friendly Streamlit interface for real-time testing

---

###  Learning Outcomes
- Applied end-to-end NLP workflow from data cleaning to model deployment.  
- Understood TF-IDF vectorization and SVM optimization.  
- Learned model deployment with Streamlit for real-time inference.  
- Enhanced understanding of performance metrics like precision, recall, and F1-score.

###  Future Improvements

* Add multilingual spam detection
* Train on real email datasets
* Integrate Gmail API for live email classification
* Compare with deep learning models (LSTM, BERT)

---
