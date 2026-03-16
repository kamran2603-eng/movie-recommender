import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load only movies data
movies = pickle.load(open('movies.pkl', 'rb'))

# Recreate vectors and similarity
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# Recommendation function


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),
                         reverse=True, key=lambda x: x[1])[1:6]
    recommended = []
    for i in movies_list:
        recommended.append(movies.iloc[i[0]].title)
    return recommended


# UI
st.title('🎬 Movie Recommender System')
st.subheader('Find movies similar to your favorites!')

selected_movie = st.selectbox(
    'Select a movie:',
    movies['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    st.write("### 🎯 Top 5 Recommendations:")
    for i, movie in enumerate(recommendations, 1):
        st.write(f"**{i}.** {movie}")
