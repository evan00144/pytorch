import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
import shutil

# Load the feature data
result = scipy.io.loadmat('features.mat')
print(result)


# Assuming 'gallery_features' is in the format (num_images, feature_dim)
gallery_feature = torch.FloatTensor(result['gallery_features'])
gallery_image = result['gallery_images']

# Use a specific query image, here we are just using the first one from gallery for demonstration
# query_feature = gallery_feature[1094]  # Make query feature 2D

i = 0

# Move to GPU if available
if torch.cuda.is_available():
    gallery_feature = gallery_feature.cuda()
    # query_feature = query_feature.cuda()

# print(query_feature.shape)
# print(gallery_feature.shape)

def sort_img(query_feature, gallery_feature):
    # Ensure query is of shape (1, feature_dim)
    query = query_feature.view(-1, 1)
    score = torch.mm(gallery_feature, query)  # Perform matrix multiplication
    score = score.squeeze(1).cpu()  # Remove singleton dimensions
    score = score.numpy()  # Convert to numpy array
    scoreSorted = np.sort(score)
    index = np.argsort(score)  # Get sorted indices
    index = index[::-1]  # Reverse to get highest scores first
    scoreSorted = scoreSorted[::-1]
    return index, scoreSorted

# Get sorted indices based on the query feature
# index, scoreSorted = sort_img(query_feature, gallery_feature)
# print(index)

index, scoreSorted = sort_img(gallery_feature[34], gallery_feature)

for i in range(len(scoreSorted)):
    print(f"{gallery_image[index[i]]} - {scoreSorted[i]}")
