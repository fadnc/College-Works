import pandas as pd
import re
import joblib
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load dataset
df = pd.read_csv(r"C:\Users\ncfad\Downloads\SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])

# Step 2: Preprocess the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['clean_message'] = df['message'].apply(clean_text)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['clean_message'], df['label_num'], test_size=0.2, random_state=42)

# Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train Models
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(random_state=42)
}

results = []
metrics_dict = {}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results.append([name, round(acc * 100, 2), round(prec, 2), round(rec, 2), round(f1, 2)])
    metrics_dict[name] = {"accuracy": acc, "confusion_matrix": cm}

# Step 6: Save Best Model (SVM)
best_model = "SVM"
joblib.dump(models[best_model], 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Save Metrics and Spam Text
joblib.dump(metrics_dict, 'model_metrics.pkl')
spam_text = " ".join(df[df['label_num'] == 1]['clean_message'].values)
joblib.dump(spam_text, 'spam_words.pkl')

# Step 7: Log Experiments (CSV)
with open("experiment_log.csv", "a", newline="") as f:
    writer = csv.writer(f)
    for row in results:
        writer.writerow(row)

# Display summary
summary = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
print(summary)
