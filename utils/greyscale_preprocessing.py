import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import cv2
# import numpy as np

# --- Initial setup ---
device = "mps"
dataset = []
images = []
IMG_SIZE = (180, 180)
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATASET_PATH = os.path.join(ROOT_DIR, 'dataset/onepiecedata_greyscale.pt')
class_names = pd.read_csv(os.path.join(ROOT_DIR, 'dataset/classnames.csv'))
mean = 131.67
stdev = 85.40
class_labels = {
    0: 'Ace',
    1: 'Akainu',
    2: 'Brook',
    3: 'Chopper',
    4: 'Crocodile',
    5: 'Franky',
    6: 'Jinbei',
    7: 'Kurohige',
    8: 'Law',
    9: 'Luffy',
    10: 'Mihawk',
    11: 'Nami',
    12: 'Rayleigh',
    13: 'Robin',
    14: 'Sanji',
    15: 'Shanks',
    16: 'Usopp',
    17: 'Zoro'
}
# ---------------------


# -------- 18 Image Classes --------
for idx, label in tqdm(enumerate(class_names['Class']), desc='Class'):

    """
    Dataset pixel value mean: 131.67
    Dataset pixel value stdev: 85.40
    """

    print(f" Processing images for '{label}'")
    character_path = os.path.join(ROOT_DIR, f'dataset/Data/{label}')

    for img in os.listdir(character_path):

        raw_img = cv2.imread(os.path.join(character_path, img), cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(raw_img, IMG_SIZE)

        image_tensor = torch.from_numpy(resized_img)

        images.append(image_tensor)
        dataset.append((image_tensor, idx))
# ----------------------------------

images = torch.stack(images)
# images = images.reshape(-1).float()
# print(f'Image list stacked {images.shape} mean: {images.mean()}, stdev: {images.std()}')
torch.save(dataset, DATASET_PATH)

cols = 3
rows = (64 + cols - 1) // 3
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
axes = axes.flatten()

for i in range(64):
    idx = random.randint(0, len(dataset))
    tup = dataset[idx]

    real = class_labels[tup[1]]
    raw_img = tup[0]

    temporal = (raw_img - mean) / stdev
    unnormalized_img = (temporal * stdev) + mean
    rescaled_img = (unnormalized_img - unnormalized_img.min()) / (unnormalized_img.max() - unnormalized_img.min())

    axes[i].imshow(rescaled_img, cmap='grey')
    axes[i].set_title(f"True: {real}")
    axes[i].axis('off')

# Turn off extra axes
for j in range(64):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
