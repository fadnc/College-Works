# Import libraries for data handling, text processing, modeling, and visualization
import pandas as pd
import numpy as np
import re
import string
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data files (only needs to be done once)
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset (update the path to your dataset accordingly)
df = pd.read_csv('/content/emotions.csv')

# Check for and remove missing values and duplicate tweets
print("Checking for missing or duplicate entries...")
df.dropna(inplace=True)
df.drop_duplicates(subset='text', inplace=True)

print("Distribution of emotion labels:")
print(df['label'].value_counts())

# Define a function to clean the tweets by:
# - lowering case, removing URLs, mentions, hashtags, punctuation, numbers, and stopwords
# - applying stemming to words
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'@\w+|#', '', text)  # Remove mentions and hashtags symbols
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.strip()

    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_tokens]

    return ' '.join(stemmed_words)

# Apply the cleaning function to the tweets
df['processed_text'] = df['text'].apply(clean_tweet)

# Convert categorical emotion labels into numerical values
encoder = LabelEncoder()
df['label_encoded'] = encoder.fit_transform(df['label'])

# Prepare the feature and target variables
X = df['processed_text']
y = df['label_encoded']

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(max_features=20000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Limit training size if desired (optional)
X_train_tfidf = X_train_tfidf[:20000]
y_train = y_train[:20000]

# Function to train and evaluate an SVM classifier with a specified kernel
def train_and_evaluate_svm(kernel):
    print(f"\nTraining SVM with kernel: {kernel}")
    svm_clf = SVC(kernel=kernel, random_state=42)
    svm_clf.fit(X_train_tfidf, y_train)

    predictions = svm_clf.predict(X_test_tfidf)

    labels_names = encoder.classes_

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=labels_names, output_dict=True)

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, target_names=labels_names))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_names, yticklabels=labels_names)
    plt.title(f"Confusion Matrix for SVM ({kernel} kernel)")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    return accuracy, report, svm_clf

# Try multiple SVM kernels and track the best performing one
kernels_to_try = ['linear', 'poly', 'rbf', 'sigmoid']
results_summary = {}
best_acc = 0
best_svm_model = None
best_kernel_used = None

for k in kernels_to_try:
    acc, rep, model = train_and_evaluate_svm(k)
    results_summary[k] = {
        'accuracy': acc,
        'precision': rep['weighted avg']['precision'],
        'recall': rep['weighted avg']['recall'],
        'f1-score': rep['weighted avg']['f1-score']
    }
    if acc > best_acc:
        best_acc = acc
        best_svm_model = model
        best_kernel_used = k

# Display performance comparison
results_df = pd.DataFrame(results_summary).T
print("\nSummary of performance across kernels:")
print(results_df)

results_df.plot.bar(y=['accuracy', 'precision', 'recall', 'f1-score'], figsize=(10,6),
                    title='Comparison of SVM Kernels Performance')
plt.xticks(rotation=0)
plt.ylabel('Score')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

print(f"\nThe best kernel for this task is: {best_kernel_used} with accuracy {best_acc:.4f}")

# Save the best model, the vectorizer, and the label encoder for later use
joblib.dump(best_svm_model, 'svm_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(encoder, 'label_encoder.pkl')

print("\nModel, vectorizer, and label encoder have been saved successfully.")
