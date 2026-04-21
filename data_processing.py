import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --- BLOCK 1: Load and Merge ---
ratings = pd.read_csv('ratings.dat', sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
users = pd.read_csv('users.dat', sep='::', engine='python', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
movies = pd.read_csv('movies.dat', sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], encoding='latin-1')

df = ratings.merge(users, on='UserID').merge(movies, on='MovieID')

# --- BLOCK 2: Encoding (This goes immediately below) ---
user_enc = LabelEncoder()
movie_enc = LabelEncoder()

df['User_idx'] = user_enc.fit_transform(df['UserID'])
df['Movie_idx'] = movie_enc.fit_transform(df['MovieID'])

print(df.head())