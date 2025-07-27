import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\ncfad\Downloads\SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])

# Step 2: Preprocess the data (basic cleaning function)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

df['clean_message'] = df['message'].apply(clean_text)

# Encode labels: 'ham' = 0, 'spam' = 1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'], df['label_num'], test_size=0.2, random_state=42)

# Step 4: Vectorize text data with TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer for later use in Streamlit app
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
