#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-27T12:57:50.401Z
"""

# # Data preparation


# Features
# - Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
# - Vocal channel (01 = speech, 02 = song).
# - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
# - Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
# - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
# - Repetition (01 = 1st repetition, 02 = 2nd repetition).
# - Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).


# NB We considered only the speech videos (vocal channel=01) with both audio and video (modality=01)


emotions = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}
emotional_intensity = {1:'normal', 2:'strong'}

import re
import os
import pandas as pd
import cv2
import random
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

path = "/content/drive/MyDrive/Datasets/RAVDESS/"
COMPLETED_ACTORS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14']
filenames = []
feats = []
labels = []
paths = []

for (dirpath, dirnames, fn) in os.walk(path):
    for name in fn:
        filename = name.split('.')[0]
        feat = filename.split('-')[2:]
        label = feat[0]
        filenames.append(filename)
        feats.append(feat)
        labels.append(label)
        paths.append(dirpath + '/' + filename)
        
filenames[:5]

# ## Data Exploration


df = pd.DataFrame(feats, columns = ['emotion', 'emotional intensity', 'statement', 'repetition', 'actor']).astype(int)

df['emotion'] = df['emotion'].map(emotions)
df['emotional intensity'] = df['emotional intensity'].map(emotional_intensity)

df['index'] = filenames
df.set_index('index', inplace=True)

df

# ## Export frames


# - one frame every skip=3 starting from the 21th frame
# - proportional resize to obtain height=224
# - saved as png with and name videoname_iframe


# ### 398x224 normal


def prepare_all_videos(filenames, paths, skip=1, completed_actors_list=COMPLETED_ACTORS):
    nframes_tot = 0
    for count, video in enumerate(zip(filenames, paths)):
        path_parts = video[1].split('/')
        actor_folder = path_parts[-2] # e.g., 'Actor_05'
        actor_id = actor_folder.split('_')[-1] # e.g., '05'

        if completed_actors_list is not None and actor_id in completed_actors_list:
            print(f"Skipping already processed actor: {actor_folder} ({count+1}/{len(paths)})")
            continue # Skip this video and move to the next one
        # Gather all its frames
        save_frames(video[0], video[1], video[1].replace('RAVDESS', 'RAVDESS_frames'), skip)
        print(f"Processed videos {count+1}/{len(paths)}")
    return


def save_frames(filename, input_path, output_path, skip):
    # Initialize video reader
    cap = cv2.VideoCapture(input_path + '.mp4')
    frames = []
    count = 0
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    try:
        # Loop through all frames
        while True:
            # Capture frame
            ret, frame = cap.read()
            if (count % skip == 0 and count > 20):
                #print(frame.shape)
                if not ret:
                    break
                frame = cv2.resize(frame, (398, 224))
                cv2.imwrite(output_path + '/' + f'{filename}_{count}' + '.png', frame)
            count += 1
    finally:
        cap.release()
    return

prepare_all_videos(filenames, paths, skip=3)

# ### 224x224 black background


def prepare_all_videos(filenames, paths, skip=1, completed_actors_list=COMPLETED_ACTORS):
    nframes_tot = 0
    for count, video in enumerate(zip(filenames, paths)):
        path_parts = video[1].split('/')
        actor_folder = path_parts[-2] # e.g., 'Actor_05'
        actor_id = actor_folder.split('_')[-1] # e.g., '05'

        if completed_actors_list is not None and actor_id in completed_actors_list:
            print(f"Skipping already processed actor: {actor_folder} ({count+1}/{len(paths)})")
            continue # Skip this video and move to the next one
        # Gather all its frames
        save_frames(video[0], video[1], video[1].replace('RAVDESS', 'RAVDESS_frames_black'), skip)
        print(f"Processed videos {count+1}/{len(paths)}")
    return


def save_frames(filename, input_path, output_path, skip):
    # Initialize video reader
    cap = cv2.VideoCapture(input_path + '.mp4')
    frames = []
    count = 0
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    try:
        # Loop through all frames
        while True:
            # Capture frame
            ret, frame = cap.read()
            if (count % skip == 0 and count > 20):
                #print(frame.shape)
                if not ret:
                    break
                #####
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                  # background from white to black
                ret, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
                frame[thresh == 255] = 0
                #####
                frame = cv2.resize(frame, (398, 224))
                frame = frame[0:224, 87:311]
                cv2.imwrite(output_path + '/' + f'{filename}_{count}' + '.png', frame)
            count += 1
    finally:
        cap.release()
    return

prepare_all_videos(filenames, paths, skip=3)

# ### 224x224 only faces BW


def prepare_all_videos(filenames, paths, skip=1, completed_actors_list=COMPLETED_ACTORS):
    nframes_tot = 0

    for count, video in enumerate(zip(filenames, paths)):
        path_parts = video[1].split('/')
        actor_folder = path_parts[-2] # e.g., 'Actor_05'
        actor_id = actor_folder.split('_')[-1] # e.g., '05'

        if completed_actors_list is not None and actor_id in completed_actors_list:
            print(f"Skipping already processed actor: {actor_folder} ({count+1}/{len(paths)})")
            continue # Skip this video and move to the next one
        # Gather all its frames
        save_frames(video[0], video[1], video[1].replace('RAVDESS', 'RAVDESS_frames_face_BW'), skip)
        print(f"Processed videos {count+1}/{len(paths)}")
    return


def save_frames(filename, input_path, output_path, skip):
    # Initialize video reader
    cap = cv2.VideoCapture(input_path + '.mp4')
    haar_cascade = cv2.CascadeClassifier('./Other/haarcascade_frontalface_default.xml')
    frames = []
    count = 0
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    try:
        # Loop through all frames
        while True:
            # Capture frame
            ret, frame = cap.read()
            if (count % skip == 0 and count > 20):
                #print(frame.shape)
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = haar_cascade.detectMultiScale(frame, scaleFactor=1.12, minNeighbors=9)
                # if len(faces) != 1:
                    
                if len(faces) == 0:
                    faces = haar_cascade.detectMultiScale(frame, scaleFactor=1.02, minNeighbors=9)
                    if len(faces) == 0:
                        raise Exception(f"Still no faces {len(faces)} {filename}")
                if len(faces) > 1:
                    ex = []
                    print(type(faces))
                    for elem in faces:
                        for (x, y, w, h) in [elem]:
                            ex.append(frame[y:y + h, x:x + w])

                    print(filename)
                    plt.figure()
                    f, axarr = plt.subplots(4,1)
                    axarr[0].imshow(ex[0])
                    axarr[1].imshow(ex[1])
                    plt.show()

                    inp = int(input())
                    faces = [faces[inp]]
                #     raise Exception(f"More than 1 faces detected in {filename}")

                for (x, y, w, h) in faces:
                    face = frame[y:y + h, x:x + w]

                face = cv2.resize(face, (234, 234))
                face = face[5:-5, 5:-5]
                cv2.imwrite(output_path + '/' + f'{filename}_{count}' + '.png', face)
            count += 1
    finally:
        cap.release()
    return


prepare_all_videos(filenames, paths, skip=3, completed_actors_list=COMPLETED_ACTORS)

# ### Mean face


emotions_tras = {1:1, 2:4, 3:5, 4:0, 5:3, 6:2, 7:6}
emotions = {0:'angry', 1:'calm', 2:'disgust', 3:'fear', 4:'happy', 5:'sad', 6:'surprise'}

dataset_path = "content/drive/MyDrive/Datasets/RAVDESS_frames_face_BW/"

height_orig = 224
width_orig = 224
height_targ = 112
width_targ = 112

val_actors = ['19', '20']
test_actors = ['01', '02', '03', '04']

filenames_train = [] # train

for (dirpath, dirnames, fn) in os.walk(dataset_path):
    if fn != []:
        class_temp = int(fn[0].split('-')[2]) - 1
        if class_temp != 0:                                                     # exclude 'neutral' label
            if any(act in dirpath for act in (test_actors+val_actors))==False:  # select only train actors
                path = [os.path.join(dirpath, elem) for elem in fn]
                label = [emotions_tras[class_temp]] * len(fn)                   # emotion transposition
                filenames_train.append(list(zip(path, label)))

def sampling(list, num_frames_desired):
    tot = []
    for elem in list:
        sampled_list = random.sample(elem, num_frames_desired)
        tot += sampled_list
    return(tot)


def compute_mean_face(filenames):
    frames_per_vid = min([len(elem) for elem in filenames])     # number of frames per clip in order to have balanced classes
    print("frames per video:", frames_per_vid) 

    filenames_sampled = sampling(filenames, frames_per_vid)
    random.shuffle(filenames_sampled)

    faces = []

    for path, label in tqdm(filenames_sampled):
        face = cv2.imread(path)
        face = cv2.resize(face, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        faces.append(face)

    faces = np.array(faces)
    mean_face = np.mean(faces, axis=0)
    mean_face = mean_face/255
    mean_face = np.expand_dims(mean_face, axis=2)
    np.save('Other/mean_face.npy', mean_face)

compute_mean_face(filenames_train)