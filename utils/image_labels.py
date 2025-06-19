import os
import torch
import pandas as pd
import numpy as np
import cv2

dataset = []
IMG_SIZE = 100
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
class_names = pd.read_csv(os.path.join(ROOT_DIR, 'dataset/classnames.csv'))

# 18 classes of images (Characters)
for idx, label in enumerate(class_names['Class']):

    print(f"Processing images for '{label}'")
    character_path = os.path.join(ROOT_DIR, f'dataset/Data/{label}')

    for img in os.listdir(character_path):

        raw_img = cv2.imread(os.path.join(character_path, img))
        resized_img = cv2.resize(raw_img, (IMG_SIZE, IMG_SIZE))
        dataset.append([resized_img, idx])
