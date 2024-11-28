import requests
import pandas as pd
import numpy as np

song_list = []

def get_access_token():
     # Your Spotify client credentials
    client_id = "1355fd61aece44608400151a13db54c4"        # Replace with your Client ID
    client_secret = "17c60ee0429b48228810ee5ebaf7a747" # Replace with your Client Secret

    # Spotify token endpoint
    url = "https://accounts.spotify.com/api/token"

    # Set the request headers and body
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }

    # Make the POST request to get the access token
    response = requests.post(url, headers=headers, data=data)

    # Check if the request was successful
    if response.status_code == 200:
        token_info = response.json()
        access_token = token_info["access_token"]
        expires_in = token_info["expires_in"]
        print("Access Token:", access_token)
        print("Expires in (seconds):", expires_in)
        return access_token
    else:
        print("Failed to get access token. Status code:", response.status_code)
        print(response.json())
        return None


def get_song_features(song_url):

    token = get_access_token()

    if token is None:
        return None
     #chill kill by red velvet
    # song_url = "https://open.spotify.com/track/68gQG2HpRMxIRom4pCugMq"
    track_id = song_url.split('/')[-1].split('?')[0]  # Example track ID; replace with your track ID
    print(track_id)
    # Define the endpoint
    url = f"https://api.spotify.com/v1/audio-features/{track_id}"

    # Set up the headers with the authorization token
    headers = {
        "Authorization": f"Bearer {token}"
    }

    # Make the GET request to fetch audio features
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        audio_features = response.json()
        # valence = audio_features.get("valence", None)  # Extract valence

        # # Print the valence score and other audio features
        # print(f"Valence for track ID {track_id}: {valence}")
        # print("Full audio features:", audio_features)
        audio_features_df = pd.DataFrame([audio_features])
        return audio_features_df
    else:
        print(f"Failed to fetch audio features. Status code: {response.status_code}")
        print(response.json())
        return None
