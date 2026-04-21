import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

# --- 1. DATA INGESTION ---
ratings = pd.read_csv('ratings.dat', sep='::', engine='python', 
                      names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
users = pd.read_csv('users.dat', sep='::', engine='python', 
                    names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
movies = pd.read_csv('movies.dat', sep='::', engine='python', 
                     names=['MovieID', 'Title', 'Genres'], encoding='latin-1')

df = ratings.merge(users, on='UserID').merge(movies, on='MovieID')

# --- 2. ENCODING ---
user_enc = LabelEncoder()
movie_enc = LabelEncoder()
df['User_idx'] = user_enc.fit_transform(df['UserID'])
df['Movie_idx'] = movie_enc.fit_transform(df['MovieID'])

num_users = df['User_idx'].nunique()
num_movies = df['Movie_idx'].nunique()

# --- 3. MODEL ARCHITECTURE ---
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

model = MovieRecommender(num_users, num_movies) 
print("Model created successfully!")

# --- 4. TRAINING LOOP ---
user_tensor = torch.LongTensor(df['User_idx'].values)
movie_tensor = torch.LongTensor(df['Movie_idx'].values)
rating_tensor = torch.FloatTensor(df['Rating'].values)

dataset = TensorDataset(user_tensor, movie_tensor, rating_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
print("\nStarting Training...")

for epoch in range(3):
    total_loss = 0
    for u, m, r in dataloader:
        optimizer.zero_grad()
        predictions = model(u, m)
        loss = criterion(predictions.squeeze(), r)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(dataloader):.4f}")

print("Training finished!")
torch.save(model.state_dict(), 'movie_model.pth')
print("Model saved! You can now use this for production.")

# # --- 5. INFERENCE: Predicting a rating ---
# model.eval() # Switch to evaluation mode (turns off dropout)

# # Let's test User #10 (index 9) for Movie #50 (index 49)
# test_user = torch.LongTensor([9]) 
# test_movie = torch.LongTensor([49])

# with torch.no_grad(): # We don't need gradients for predicting
#     prediction = model(test_user, test_movie)
#     print(f"\nPredicted rating for User 10 on Movie 50: {prediction.item():.2f} stars")


# def get_top_n_recommendations(user_id, model, df, n=10):
#     model.eval()
    
#     # 1. Translate real UserID to our internal User_idx
#     # We use a try-except block just in case the UserID doesn't exist
#     try:
#         user_idx = user_enc.transform([user_id])[0]
#     except ValueError:
#         return "User ID not found in the training data."

#     # 2. Get all movie indices the user has ALREADY watched
#     watched_movies = df[df['UserID'] == user_id]['Movie_idx'].values
    
#     # 3. Get ALL possible movie indices
#     all_movie_indices = df['Movie_idx'].unique()
    
#     # 4. Filter: Keep only movies the user hasn't seen
#     unseen_movies = [m for m in all_movie_indices if m not in watched_movies]
    
#     # 5. Prepare tensors for the model
#     user_tensor = torch.LongTensor([user_idx] * len(unseen_movies))
#     movie_tensor = torch.LongTensor(unseen_movies)
    
#     # 6. Predict all at once
#     with torch.no_grad():
#         predictions = model(user_tensor, movie_tensor)
        
#     # 7. Get the Top N
#     # predictions.squeeze() gives us a list of scores
#     # torch.topk gives us the indices of the highest scores
#     top_n_scores, top_n_indices = torch.topk(predictions.squeeze(), n)
    
#     # 8. Translate back to real Movie Titles
#     recommended_movie_indices = [unseen_movies[i] for i in top_n_indices]
    
#     # Lookup the titles using the inverse transform
#     recommended_titles = []
#     for idx in recommended_movie_indices:
#         # Get the real MovieID from the encoded index
#         real_movie_id = movie_enc.inverse_transform([idx])[0]
#         title = df[df['MovieID'] == real_movie_id]['Title'].iloc[0]
#         recommended_titles.append(title)
        
#     return recommended_titles

# # --- RUN IT ---
# recommendations = get_top_n_recommendations(user_id=10, model=model, df=df, n=10)
# print(f"\nTop 10 Recommendations for User 10:")
# for i, title in enumerate(recommendations, 1):
#     print(f"{i}. {title}")