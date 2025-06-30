from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load data
data = pd.read_csv('combined_data.csv')
cos_sim = np.load('cosine_matrix.npy')

# Recommendation function
def get_recommendations(title, top_n=5):
    if title not in data['title'].values:
        return []
    idx = data[data['title'] == title].index[0]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    results = []
    for i, score in sim_scores:
        # Handle NaN image_url
        image_url = data.iloc[i].get('image_url', '')
        if pd.isna(image_url) or not image_url:
            image_url = 'https://placehold.co/300x450'
            
        # Handle description
        description = data.iloc[i].get('description', '')
        if pd.isna(description) or not description:
            description = 'No description available.'
        else:
            description = str(description)[:300] + '...'
            
        results.append({
            'title': data.iloc[i]['title'],
            'description': description,
            'type': data.iloc[i]['type'],
            'score': float(score),  # Convert numpy float to Python float
            'image_url': image_url
        })
    return results

# Function to get popular items for slideshow
def get_popular_items(type, top_n=5):
    try:
        # Get items of specified type
        type_items = data[data['type'] == type]
        
        # If no items found, return empty list
        if type_items.empty:
            return []
            
        # Sample items
        items = type_items.sample(min(top_n, len(type_items)))
        
        # Convert to list of dictionaries with proper handling of NaN values
        results = []
        for _, item in items.iterrows():
            image_url = item.get('image_url', '')
            if pd.isna(image_url) or not image_url:
                image_url = 'https://placehold.co/300x450'
                
            description = item.get('description', '')
            if pd.isna(description) or not description:
                description = 'No description available.'
            else:
                description = str(description)[:150] + '...'
                
            results.append({
                'title': item['title'],
                'description': description,
                'type': item['type'],
                'image_url': image_url
            })
            
        return results
    except Exception as e:
        print(f"Error in get_popular_items: {str(e)}")
        return []

# Routes
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
        # Remove any None or NaN values
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
        
        if not recommendations:
            return jsonify([])  # Return empty list instead of error
            
        return jsonify(recommendations)
    except Exception as e:
        print(f"Error in recommend route: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    try:
        print("Loading data...")
        print(f"Total items in dataset: {len(data)}")
        print(f"Books: {len(data[data['type'] == 'book'])}")
        print(f"Movies: {len(data[data['type'] == 'movie'])}")
        print("Starting server...")
        app.run(debug=True)
    except Exception as e:
        print(f"Error starting application: {str(e)}")