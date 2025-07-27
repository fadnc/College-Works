import pandas as pd
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def simple_preprocess(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return ' '.join(word for word in tokens if word not in ENGLISH_STOP_WORDS)

def load_data():
    df = pd.read_csv("dataset_combined.csv")
    text_columns = ['Name', 'Education Qualification', 'Community', 'Religion', 'Income']
    df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)
    df['clean_text'] = df['combined_text'].apply(simple_preprocess)
    return df

def recommend_with_constraints(df, vectorizer, user_query, edu=None, community=None, religion=None, income_bracket=None, top_n=5):
    from sklearn.metrics.pairwise import cosine_similarity
    
    filtered_df = df.copy()
    if edu:
        filtered_df = filtered_df[filtered_df['Education Qualification'].str.contains(edu, case=False, na=False)]
    if community:
        filtered_df = filtered_df[filtered_df['Community'].str.contains(community, case=False, na=False)]
    if religion:
        filtered_df = filtered_df[filtered_df['Religion'].str.contains(religion, case=False, na=False)]
    if income_bracket:
        filtered_df = filtered_df[filtered_df['Income'].str.contains(income_bracket, case=False, na=False)]

    if filtered_df.empty:
        return pd.DataFrame(columns=['Name', 'Education Qualification', 'Community', 'Religion', 'Income'])

    filtered_df = filtered_df.drop_duplicates()
    filtered_tfidf = vectorizer.transform(filtered_df['clean_text'])
    query_vector = vectorizer.transform([simple_preprocess(user_query)])

    scores = cosine_similarity(query_vector, filtered_tfidf).flatten()
    top_indices = scores.argsort()[::-1][:top_n]
    
    return filtered_df.iloc[top_indices][['Name', 'Education Qualification', 'Community', 'Religion', 'Income']], scores[top_indices]
