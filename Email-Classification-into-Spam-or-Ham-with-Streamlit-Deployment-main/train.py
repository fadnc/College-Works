import pandas as pd
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Step 1: Load dataset
df = pd.read_csv(r"C:\Users\ncfad\Downloads\SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])

# Step 2: Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)       # Remove non-alphanumeric
    text = re.sub(r'\s+', ' ', text)      # Remove extra spaces
    return text.strip()

df['clean_message'] = df['message'].apply(clean_text)

# Encode labels: ham = 0, spam = 1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'], df['label_num'], test_size=0.2, random_state=42
)

# Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train SVM model
svm = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm.fit(X_train_tfidf, y_train)

# Step 6: Evaluate model
y_pred = svm.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save model and vectorizer
joblib.dump(svm, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
