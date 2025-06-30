import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
books = pd.read_csv('data/books.csv')
movies = pd.read_csv('data/tmdb_5000_movies.csv')

# Create fake descriptions for books
books = books[['title', 'authors']].dropna().drop_duplicates()
books['description'] = books['title'] + ' by ' + books['authors']

# Clean movies
movies = movies[['title', 'overview']].dropna().drop_duplicates()
movies.rename(columns={'overview': 'description'}, inplace=True)

# Add type info
books['type'] = 'book'
movies['type'] = 'movie'

# Merge
books = books[['title', 'description', 'type']]
movies = movies[['title', 'description', 'type']]
data = pd.concat([books, movies], ignore_index=True)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['description'])

# Similarity
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save
data.to_csv('combined_data.csv', index=False)
np.save('cosine_matrix.npy', cos_sim)
print("Model and data saved.")
