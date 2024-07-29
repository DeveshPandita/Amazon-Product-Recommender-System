from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from flask_cors import CORS

print("---------------libraries Loaded---------------")

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)

# Load the pre-trained BERT model
bert = SentenceTransformer('bert-base-nli-mean-tokens')

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
bert = bert.to(device)

print("---------------Bert Loaded---------------")

"""
Download preprocesed data from my drive:
sentence_embedds(vector embeddings of product description) -> https://drive.google.com/file/d/1-1UlOZiUofQ8ZwrYwZMo1ncYSTpEpVHc/view?usp=sharing
Meta-data -> https://drive.google.com/file/d/1Vs5C8CeV6CDfG340n4bQ_bsZFDqggJEG/view?usp=sharing
"""
#Load product embeddings
with open('meta_data_vector_embedds.pickle', 'rb') as f:
    sentence_embeddings = pickle.load(f)
print("---------------Embeddings Loaded---------------")

# Load product meta data
with open('meta_data.pickle', 'rb') as f:
    df = pickle.load(f)
print("---------------Meta Data Loaded---------------")

# Function to convert list datatype to string
def get_first_image(images):
    if isinstance(images, list) and images:
        return images[0]
    else:
        return " "

@app.route('/api/recommend', methods=['POST'])

def recommend():
    user_input = request.json.get('input')
    print("success")
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    user_input = re.sub('[^A-Za-z0-9]+', ' ', user_input)
    user_input_embedding = bert.encode([user_input])
    similarities = cosine_similarity(user_input_embedding, sentence_embeddings)
    top_indices = np.argsort(similarities, axis=1)[:, -10:][:, ::-1]
    top_matches = top_indices[0]
    temp = df.iloc[top_matches]
    temp = temp[['asin', 'title', 'imageURLHighRes']]
    temp = pd.DataFrame(temp)
    temp['imageURLHighRes'] = temp['imageURLHighRes'].apply(get_first_image)
    return jsonify(temp.to_dict(orient='records'))
    # recommended_products = df.iloc[top_matches].to_dict(orient='records')
    # return jsonify(recommended_products)

if __name__ == '__main__':
    socketio.run(app, port=5000)
