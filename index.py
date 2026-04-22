from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
# Importing from utils.py in the same folder
from utils import MovieRecommender, get_top_n_recommendations, num_users, num_movies

app = FastAPI()

# Enable CORS so your frontend can communicate with your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (make sure 'movie_model.pth' is in the root directory)
model = MovieRecommender(num_users, num_movies)
model.load_state_dict(torch.load('movie_model.pth', map_location=torch.device('cpu')))
model.eval()

# API Route
@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    recs = get_top_n_recommendations(user_id, model, n=10)
    return {"user_id": user_id, "recommendations": recs}

# Serve the frontend (public folder)
# This mounts the 'public' folder so index.html loads automatically at the root URL
app.mount("/", StaticFiles(directory="public", html=True), name="public")