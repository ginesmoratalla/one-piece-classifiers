import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

device = "mps"
dataset = []
IMG_SIZE = 150
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATASET_PATH = os.path.join(ROOT_DIR, 'dataset/onepiecedata.pt')
class_names = pd.read_csv(os.path.join(ROOT_DIR, 'dataset/classnames.csv'))

# 18 classes of images (Characters)
for idx, label in enumerate(class_names['Class']):

    print(f"Processing images for '{label}'")
    character_path = os.path.join(ROOT_DIR, f'dataset/Data/{label}')

    for img in os.listdir(character_path):

        raw_img = cv2.imread(os.path.join(character_path, img))
        resized_img = cv2.resize(raw_img, (IMG_SIZE, IMG_SIZE))
        dataset.append([torch.tensor(resized_img), torch.tensor(idx)])

torch.save(dataset, DATASET_PATH)
