
import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('movies_metadata.csv', low_memory=False)
df = df[['title','overview']].dropna().head(2000)

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['clean_text'] = df['overview'].apply(clean)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
matrix = tfidf.fit_transform(df['clean_text'])
similarity = cosine_similarity(matrix)

def recommend(title):
    if title not in df['title'].values:
        return ["Movie not found"]
    idx = df[df['title']==title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    return [df.iloc[i[0]].title for i in scores]

st.title("Movie Recommendation System")

movie = st.selectbox("Select a movie", df['title'].values)

if st.button("Recommend"):
    recs = recommend(movie)
    for r in recs:
        st.write(r)
