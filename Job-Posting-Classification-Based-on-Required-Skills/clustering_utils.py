from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import joblib

def perform_clustering(skills_list):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(skills_list)
    model = AgglomerativeClustering(n_clusters=5)
    labels = model.fit_predict(X.toarray())
    return labels, model, vectorizer

def save_model(model, vectorizer):
    joblib.dump(model, "models/clustering_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

def load_model():
    model = joblib.load("models/clustering_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

def classify_new_job(skills):
    model, vectorizer = load_model()
    X = vectorizer.transform([skills])
    return model.fit_predict(X.toarray())[0]