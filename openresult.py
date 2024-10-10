import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json

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

list = {}
isUsed = set()
# Print the top 10 indices
for i in range(len(gallery_image)):
    query_feature = gallery_feature[i]
    if torch.cuda.is_available():
        query_feature = query_feature.cuda()
    
    if i not in isUsed:
        index, scoreSorted = sort_img(query_feature, gallery_feature)
        isUsed.add(i)
    
    if i not in list:
        list[i] = {}

    for j in range(len(scoreSorted)):
        if scoreSorted[j] > 0.5942655801773071:
            if index[j] not in isUsed:
                isUsed.add(index[j])
                if 'image' not in list[i]:
                    list[i]['image'] = []
                if 'score' not in list[i]:
                    list[i]['score'] = []
                list[i]['image'].append(gallery_image[index[j]])
                list[i]['score'].append(scoreSorted[j])
    

# if list[i] length == 0 then delete it
for i in list.copy():
    if len(list[i]) == 0:
        del list[i]
for i in list:
    if 'score' in list[i]:
        list[i]['score'] = [float(score) for score in list[i]['score']]


print(list)
with open('result.json', 'w') as f:
    json.dump(list, f)


# 0 = > dapet list score [1,2,3,4...,1111]

# ambil yang score nya lebih dari 0.5942655801773071
