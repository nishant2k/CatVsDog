
"""Creating model for Dog vs Cat classification model"""
#Importing necessery libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = "C:/Users/91811/Anaconda3/Desktop/kagglecatsanddogs_3367a/PetImages" # Path of the directory conatining the cat and dog dataset

CATEGORIES = ["Dog", "Cat"]

IMG_SIZE = 50 #Resizing each images to particular size


"""Creating the training data for the model"""
training_data = []

def create_training_data():
    for category in CATEGORIES:  # Do dogs and cats

        path = os.path.join(DATADIR,category)  # Create path to dogs and cats
        class_num = CATEGORIES.index(category)  # Get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # Iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # Convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # Resize to normalize data size
                training_data.append([new_array, class_num])  # Add this to our training_data
            except Exception as e:  # In the interest in keeping the output clean...
                pass
create_training_data() # calling the function to create our training model

print(len(training_data)) # lets check the length of our training data

import random
random.shuffle(training_data) #shuffle our data 
#Create dataset X containing the encoded images and y will be conatining the index of the image
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

#print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 

import pickle

pickle_out = open("X.pickle","wb") #Using pickle for saving the array and corresponding index
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
"""We can load our model by importing the pickles """
