import re
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the pre-trained BERT model
bert = SentenceTransformer('bert-base-nli-mean-tokens')

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
bert = bert.to(device)

#Load product embeddings
with open('meta_data_vector_embedds.pickle', 'rb') as f:
    sentence_embeddings = pickle.load(f)

# Load product meta data
with open('meta_data.pickle', 'rb') as f:
    df = pickle.load(f)


def recommend(user_input, sentence_embeddings):
    user_input = re.sub('[^A-Za-z0-9]+', ' ', user_input)
    user_input_embedding = bert.encode([user_input])
    similarities = cosine_similarity(user_input_embedding, sentence_embeddings)
    top_indices = np.argsort(similarities, axis=1)[:, -10:][:, ::-1]
    return top_indices[0]

user_input = sys.argv[1]
top_matches = recommend(user_input, sentence_embeddings)
recommended_products = df.iloc[top_matches].to_dict(orient='records')
print(json.dumps(recommended_products))


def get_recommendations(user_input):
    top_matches = recommend(user_input, sentence_embeddings)
    return df.iloc[top_matches].to_dict(orient='records')
