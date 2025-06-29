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
        results.append({
            'title': data.iloc[i]['title'],
            'description': data.iloc[i]['description'][:300] + '...',
            'type': data.iloc[i]['type'],
            'score': round(score, 3),
            'image_url': data.iloc[i].get('image_url', '')  # might be NaN for movies
    })

    return results

# Routes
@app.route('/')
def home():
    titles = data['title'].tolist()
    return render_template('index.html', titles=titles)

@app.route('/recommend', methods=['POST'])
def recommend():
    req = request.get_json()
    title = req.get('title')
    recommendations = get_recommendations(title)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)