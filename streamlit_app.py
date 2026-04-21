import streamlit as st
import requests

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 AI Movie Recommender")
st.write("Enter your User ID below to see what our Neural Network suggests for you!")

# Input field for User ID
user_id = st.number_input("User ID", min_value=1, step=1, value=10)

# Button to trigger the recommendation
if st.button("Get Recommendations"):
    # Connect to your FastAPI server
    try:
        api_url = f"http://127.0.0.1:8000/recommend/{user_id}"
        response = requests.get(api_url)
        
        if response.status_code == 200:
            data = response.json()
            st.success(f"Here are the top picks for User {user_id}:")
            
            # Displaying the list
            for i, title in enumerate(data['recommendations'], 1):
                st.write(f"**{i}.** {title}")
        else:
            st.error("Error: Could not connect to the API. Is the server running?")
            
    except Exception as e:
        st.error(f"Something went wrong: {e}")