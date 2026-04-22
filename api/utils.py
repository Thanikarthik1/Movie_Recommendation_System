import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# 1. Define Model Architecture
class MovieRecommender(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size=50):
        super(MovieRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, user_idx, movie_idx):
        u = self.user_embedding(user_idx)
        m = self.movie_embedding(movie_idx)
        x = torch.cat([u, m], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.output(x)

# 2. Data Loading
# Since this file is inside the /api folder, and your data files 
# are also inside /api, we use the direct filename.
ratings = pd.read_csv('api/ratings.dat', sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
users = pd.read_csv('api/users.dat', sep='::', engine='python', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
movies = pd.read_csv('api/movies.dat', sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], encoding='latin-1')

df = ratings.merge(users, on='UserID').merge(movies, on='MovieID')

user_enc = LabelEncoder()
movie_enc = LabelEncoder()
df['User_idx'] = user_enc.fit_transform(df['UserID'])
df['Movie_idx'] = movie_enc.fit_transform(df['MovieID'])

num_users = df['User_idx'].nunique()
num_movies = df['Movie_idx'].nunique()

# 3. Helper Function
def get_top_n_recommendations(user_id, model, n=10):
    model.eval()
    try:
        user_idx = user_enc.transform([user_id])[0]
    except ValueError:
        return ["User ID not found."]

    watched_movies = df[df['UserID'] == user_id]['Movie_idx'].values
    all_movie_indices = df['Movie_idx'].unique()
    unseen_movies = [m for m in all_movie_indices if m not in watched_movies]
    
    # Handle cases where user has seen all movies
    if not unseen_movies:
        return ["No new movies to recommend."]
    
    user_tensor = torch.LongTensor([user_idx] * len(unseen_movies))
    movie_tensor = torch.LongTensor(unseen_movies)
    
    with torch.no_grad():
        predictions = model(user_tensor, movie_tensor)
        
    top_n_indices = torch.topk(predictions.squeeze(), n).indices
    recommended_indices = [unseen_movies[i] for i in top_n_indices]
    
    titles = [df[df['MovieID'] == movie_enc.inverse_transform([idx])[0]]['Title'].iloc[0] for idx in recommended_indices]
    return titles