import torch
import numpy as np
import pickle
import pandas as pd
import json
import os
from joblib import load
import torch.nn as nn
import torch.optim as optim

BASE_PATH = "/app/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path_1 = os.path.join(os.path.dirname(__file__), "Music Info.csv")
music_df = pd.read_csv(csv_path_1)
csv_path_2 = os.path.join(os.path.dirname(__file__), "User Listening History.csv")
user_df=pd.read_csv(csv_path_2)

with open(os.path.join(BASE_PATH, "user_to_index.json"), "r") as f:
    user_to_index = json.load(f)
with open(os.path.join(BASE_PATH, "track_to_index.json"), "r") as f:
    track_to_index = json.load(f)
with open(os.path.join(BASE_PATH, "index_to_track.json"), "r") as f:
    index_to_track = json.load(f)
with open(os.path.join(BASE_PATH, "track_to_name.json"), "r") as f:
    track_to_name = json.load(f)


feature_matrix = np.load("feature_matrix.npy")
with open("content_model.pkl", "rb") as f:
    knn_model = load(os.path.join(BASE_PATH, "content_model.pkl"))


class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        x = torch.cat([user_embeds, item_embeds], dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()

num_users = len(user_to_index)
num_items = len(track_to_index)

# Initialize the model
collaborative_model = NCF(num_users, num_items)

# Load saved state dictionary


if os.path.exists("/app/"):  
    BASE_PATH = "/app/"  # Running inside Docker
else:
    BASE_PATH = "E:\\Projects\\Machine Learning\\Music Recommender System\\"  # Running locally

# Load model using adjusted path
model_path = os.path.join(BASE_PATH, "collaborative_model.pth")
state_dict = torch.load(model_path, map_location=torch.device("cpu"))

# Load state dict with adjusted size
collaborative_model.load_state_dict(state_dict, strict=False)
collaborative_model.eval()


def recommend_songs_collaborative(user_id, top_n=10):
    if user_id not in user_to_index:
        return {"error": f"User ID {user_id} not found in dataset."}

    user_idx = user_to_index[user_id]
    user_tensor = torch.tensor([user_idx] * len(track_to_index), dtype=torch.long)
    track_tensor = torch.tensor(list(track_to_index.values()), dtype=torch.long)

    with torch.no_grad():
        predictions = collaborative_model(user_tensor, track_tensor)

    # Get top N recommendations
    _, indices = torch.topk(predictions, top_n)

    # Ensure only valid indices are returned
    valid_indices = [idx.item() for idx in indices if str(idx.item()) in index_to_track]
    print("Raw indices from model:", indices.tolist())
    print("Filtered valid indices:", [idx.item() for idx in indices if str(idx.item()) in index_to_track])


    if not valid_indices:
        return {"error": f"No valid recommendations found for user {user_id}"}
    
    recommended_track_ids = [index_to_track[str(idx)] for idx in valid_indices]
    print("Recommended track ids are:",recommended_track_ids)
    recommended_songs = [track_to_name.get(track_id, "Unknown Song") for track_id in recommended_track_ids]

    return recommended_songs



# Content-Based Filtering (KNN)
def recommend_songs_content(song_name, top_n=10):
    try:
        idx = music_df[music_df["name"] == song_name].index[0]
        distances, indices = knn_model.kneighbors([feature_matrix[idx]])
        recommended_track_ids = music_df.iloc[indices[0][1:]]["track_id"].tolist()
        recommended_songs = music_df[music_df["track_id"].isin(recommended_track_ids)]["name"].tolist()
        return recommended_songs
    except IndexError:
        return f"Song '{song_name}' not found in dataset."


# Hybrid Recommender
def recommend_songs_hybrid(user_id, song_name, top_n=10, alpha=0.3):
    if user_id not in user_to_index:
        content_recommended_songs = recommend_songs_content(song_name, top_n)
        return content_recommended_songs
       

    user_tensor = torch.tensor([user_to_index[user_id]] * len(track_to_index), dtype=torch.long)
    track_tensor = torch.tensor(list(track_to_index.values()), dtype=torch.long)

    with torch.no_grad():
        predictions = collaborative_model(user_tensor, track_tensor)

    _, indices = torch.topk(predictions, top_n)
    collab_recommended_ids = [index_to_track[str(idx.item())] for idx in indices]
    collab_recommended_songs = [track_to_name.get(track_id, "Unknown Song") for track_id in collab_recommended_ids]

    content_recommended_songs = recommend_songs_content(song_name, top_n)

    final_recommendations = {}
    for song in collab_recommended_songs:
        final_recommendations[song] = alpha
    for song in content_recommended_songs:
        if song in final_recommendations:
            final_recommendations[song] += (1 - alpha)
        else:
            final_recommendations[song] = (1 - alpha)

    sorted_songs = sorted(final_recommendations, key=final_recommendations.get, reverse=True)[:top_n]

    return sorted_songs
