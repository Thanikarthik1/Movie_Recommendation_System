# app.py
from fastapi import FastAPI
import torch
import uvicorn
from utils import MovieRecommender, get_top_n_recommendations, df

app = FastAPI()

# 1. Initialize and Load Model globally
print("Starting API - Loading model...")
num_users = df['User_idx'].nunique()
num_movies = df['Movie_idx'].nunique()

model = MovieRecommender(num_users, num_movies)
model.load_state_dict(torch.load('movie_model.pth'))
model.eval()

# 2. Define API route
@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    recs = get_top_n_recommendations(user_id, model, n=10)
    return {"user_id": user_id, "recommendations": recs}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)