import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split
from spotify_api import get_song_features
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
song_list = dict()


# Function to predict the probabilities for a given image
def predict_image(image_path):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    model = models.resnet18()


    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features  
    num_classes = 6  


    model.fc = nn.Sequential(
        # nn.Dropout(p=0.5),  
        nn.Linear(num_ftrs, num_classes)  # final fully connected layer
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    model = model.to(device)
    model.load_state_dict(torch.load('image_classifier_models/model_epoch_100.pth'))
    model.eval()  
    image = Image.open(image_path)

    image = transform(image).unsqueeze(0).to(device)  
    

    with torch.no_grad():
        outputs = model(image)

    probabilities = F.softmax(outputs, dim=1)  # np array of probs 
    return probabilities.squeeze(0).cpu().numpy()



# image_path = 'Emotion6/images/surprise/115.jpg'  
# probabilities = predict_image(image_path)
# print("Class Probabilities:")
# for i, prob in enumerate(probabilities):
#     print(f"Class {i}: {prob:.2f}")






def predict_song(song_url):


    col_names = ['duration (ms)', 'danceability', 'energy', 'loudness', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']


    song_df = get_song_features(song_url)
    song_df = song_df.rename(columns={'duration_ms': 'duration (ms)'})
    song_df = song_df.loc[:, col_names]
    scaler = StandardScaler()
    song_scaled = scaler.transform(song_df)

    hgb_model = joblib.load('hgb_model.pkl')
    song_emo_prediction = hgb_model.predict(song_scaled)

    emotions = ['Sad', 'Happy', 'Energetic', 'Calm']
    return emotions[song_emo_prediction[0]]


def create_song_list(song_url_list):
    for song_url in song_url_list:
        emotion = predict_song(song_url)
        global song_list
        song_list[song_url] = emotion



def get_emo_match_song_list(image_path,user_own_list=False, list_size=10):
    # returns a list of songs with their spotify track id
    emo_probabilities = predict_image(image_path=image_path)

    sad_weight = 0.2 * np.sum(emo_probabilities[0:3]) + 0.8 * emo_probabilities[4]
    happy_weight = emo_probabilities[3] if emo_probabilities[3] > 0.5 else 0.2
    calm_weight = emo_probabilities[3] if emo_probabilities[3] < 0.5 else 0.2
    energetic_weight = emo_probabilities[5] * 0.7 + emo_probabilities[3] * 0.3
    total_weight = sad_weight + happy_weight + calm_weight + energetic_weight

    sad_prob, happy_prob = round(sad_weight/total_weight,1), round(happy_weight/total_weight,1)
    calm_prob, energetic_prob = round(calm_weight/total_weight,1), round(energetic_weight/total_weight,1)

    
    probs = np.array([sad_prob, happy_prob, calm_prob, energetic_prob])
    probs[-1] = 1 - sum(probs[:-1])


    emo_match_song_list = []
    mus_wt_url_df = pd.read_csv("archive/278k_labelled_uri.csv")
    if not user_own_list:
        num_songs = (probs * list_size).astype(int)
        for i, num in enumerate(num_songs):
            emo_music_subset = mus_wt_url_df[mus_wt_url_df['labels'] == i]
            sampled_songs = emo_music_subset.sample(n=num, random_state=42)
            sampled_songs['uri']
            emo_match_song_list.extend(sampled_songs['uri'].tolist())
        print(emo_match_song_list)
    else:
        ## use user's complete list to extract emotional matched songs
        ## not finished since my spotify api auth has problem
        num_songs = (probs * list_size).astype(int)
        for i, num in enumerate(num_songs):
            emo_music_subset = mus_wt_url_df[mus_wt_url_df['labels'] == i]
            sampled_songs = emo_music_subset.sample(n=num, random_state=42)
            sampled_songs['uri']
            emo_match_song_list.extend(sampled_songs['uri'].tolist())
        print(emo_match_song_list)
    return emo_match_song_list # list of track ids



