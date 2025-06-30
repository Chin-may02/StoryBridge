from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import requests
from urllib.parse import quote
import os
from functools import lru_cache
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup

app = Flask(__name__)

# API Keys
TMDB_API_KEY = "1963bc0ae87071836d9bb53f8cb86b67"
TMDB_ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIxOTYzYmMwYWU4NzA3MTgzNmQ5YmI1M2Y4Y2I4NmI2NyIsInN1YiI6IjY0YTQ4NjQ2ZDY1OTBiMDBhZTg4NDYyYiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.qs82_WaYkXxD8kOzYxUwTUBxl7mBmQpXN0h7Qz1_MXM"

# Create a session with retry strategy
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
session.mount("https://", HTTPAdapter(max_retries=retry_strategy))

# Load data
try:
    data = pd.read_csv('combined_data.csv')
    cos_sim = np.load('cosine_matrix.npy')
except Exception as e:
    print(f"Error loading data: {str(e)}")
    data = pd.DataFrame()
    cos_sim = np.array([])

def create_placeholder_url(title, item_type):
    """Create a styled placeholder URL with gradient background"""
    safe_title = quote(title.replace(' ', '\n'))  # Add line breaks for longer titles
    
    if item_type.lower() == 'movie':
        # Movie gradient: Dark purple to deep blue
        gradient = "151515,252525"
        icon = "🎬"  # Movie icon
    else:
        # Book gradient: Deep blue to dark teal
        gradient = "1a1a2e,16213e"
        icon = "📚"  # Book icon
        
    # Add icon to title
    display_text = f"{icon}\n{safe_title}"
    
    return f'https://placehold.co/600x900/gradient/{gradient}/ffffff?text={display_text}&font=montserrat'

@lru_cache(maxsize=1000)
def get_movie_image(title):
    try:
        # 1. Try TMDB API with higher resolution
        search_url = "https://api.themoviedb.org/3/search/movie"
        headers = {
            "Authorization": f"Bearer {TMDB_ACCESS_TOKEN}",
            "accept": "application/json"
        }
        params = {
            "query": title,
            "include_adult": False,
            "language": "en-US",
            "page": 1
        }
        
        response = session.get(search_url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                movie_id = results[0].get('id')
                
                # Get movie details for higher quality images
                if movie_id:
                    details_url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
                    details_response = session.get(
                        details_url,
                        headers=headers,
                        timeout=10
                    )
                    
                    if details_response.status_code == 200:
                        images = details_response.json()
                        posters = images.get('posters', [])
                        
                        if posters:
                            # Get the highest resolution poster
                            poster_path = posters[0].get('file_path')
                            if poster_path:
                                return f"https://image.tmdb.org/t/p/original{poster_path}"
                
                # Fallback to standard poster if detailed search fails
                poster_path = results[0].get('poster_path')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/original{poster_path}"

        # 2. Try IMDb through web scraping for high-res images
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        search_url = f"https://www.imdb.com/find?q={quote(title)}&s=tt&ttype=ft"
        response = session.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            poster = soup.find('img', class_='ipc-image')
            if poster:
                src = poster.get('src', '')
                if '@' in src:
                    base_url = src.split('@')[0]
                    return f"{base_url}@.jpg"
                return src

    except Exception as e:
        print(f"Movie image search error for {title}: {str(e)}")
    
    return None

@lru_cache(maxsize=1000)
def get_book_cover(title):
    try:
        # 1. Try Open Library API with larger size
        open_library_url = f"https://openlibrary.org/search.json?title={quote(title)}&limit=1"
        response = session.get(open_library_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('docs') and len(data['docs']) > 0:
                cover_id = data['docs'][0].get('cover_i')
                if cover_id:
                    return f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"

        # 2. Try Google Books API with larger thumbnails
        google_books_url = f"https://www.googleapis.com/books/v1/volumes?q={quote(title)}"
        response = session.get(google_books_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('items') and len(data['items']) > 0:
                volume_info = data['items'][0].get('volumeInfo', {})
                image_links = volume_info.get('imageLinks', {})
                if image_links:
                    for size in ['extraLarge', 'large', 'medium', 'thumbnail']:
                        if size in image_links:
                            img_url = image_links[size].replace('http://', 'https://')
                            img_url = img_url.split('&zoom=')[0]
                            return img_url
    
    except Exception as e:
        print(f"Book cover search error for {title}: {str(e)}")
    
    return None

def get_item_image(title, item_type):
    try:
        image_url = None
        if item_type.lower() == 'movie':
            image_url = get_movie_image(title)
        else:
            image_url = get_book_cover(title)
        
        if not image_url:
            return create_placeholder_url(title, item_type)
        
        return image_url
    except Exception as e:
        print(f"Error getting image for {title}: {str(e)}")
        return create_placeholder_url(title, item_type)

def get_recommendations(title, top_n=5):
    try:
        if title not in data['title'].values:
            return []
        
        idx = data[data['title'] == title].index[0]
        sim_scores = list(enumerate(cos_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        results = []
        
        for i, score in sim_scores:
            item_type = data.iloc[i]['type']
            item_title = data.iloc[i]['title']
            
            # Get image from API
            image_url = get_item_image(item_title, item_type)
                
            results.append({
                'title': item_title,
                'description': data.iloc[i]['description'][:300] + '...' if isinstance(data.iloc[i]['description'], str) else 'No description available.',
                'type': item_type,
                'score': float(score),
                'image_url': image_url
            })
        return results
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        return []

def get_popular_items(type, top_n=5):
    try:
        type_items = data[data['type'] == type]
        if type_items.empty:
            return []
            
        items = type_items.sample(min(top_n, len(type_items)))
        results = []
        
        for _, item in items.iterrows():
            # Get image from API
            image_url = get_item_image(item['title'], type)
            
            results.append({
                'title': item['title'],
                'description': item['description'][:150] + '...' if isinstance(item['description'], str) else 'No description available.',
                'type': type,
                'image_url': image_url
            })
            
        return results
    except Exception as e:
        print(f"Error in get_popular_items: {str(e)}")
        return []

@app.route('/')
def landing():
    try:
        popular_books = get_popular_items('book', 5)
        popular_movies = get_popular_items('movie', 5)
        return render_template('landing.html', 
                             popular_books=popular_books, 
                             popular_movies=popular_movies)
    except Exception as e:
        print(f"Error in landing route: {str(e)}")
        return render_template('landing.html', 
                             popular_books=[], 
                             popular_movies=[])

@app.route('/recommend')
def recommend_page():
    try:
        titles = data['title'].tolist()
        titles = [title for title in titles if title and not pd.isna(title)]
        return render_template('recommend.html', titles=titles)
    except Exception as e:
        print(f"Error in recommend_page route: {str(e)}")
        return render_template('recommend.html', titles=[])

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        req = request.get_json()
        if not req or 'title' not in req:
            return jsonify({'error': 'No title provided'}), 400
        
        title = req.get('title')
        if not title:
            return jsonify({'error': 'Empty title'}), 400
            
        recommendations = get_recommendations(title)
        return jsonify(recommendations)
    except Exception as e:
        print(f"Error in recommend route: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)