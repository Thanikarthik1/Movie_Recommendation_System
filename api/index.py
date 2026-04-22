# api/index.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from .utils import MovieRecommender, get_top_n_recommendations, df, num_users, num_movies
import os

app = FastAPI()

# Add CORS to allow your frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (make sure path is correct)
model = MovieRecommender(num_users, num_movies)
model.load_state_dict(torch.load('api/movie_model.pth', map_location=torch.device('cpu')))
model.eval()

@app.get("/api/recommend/{user_id}")
def recommend(user_id: int):
    recs = get_top_n_recommendations(user_id, model, n=10)
    return {"user_id": user_id, "recommendations": recs}