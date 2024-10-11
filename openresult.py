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
import mediapipe as mp
import cv2
import gc
from concurrent.futures import ThreadPoolExecutor

# Load the feature data
result = scipy.io.loadmat('features.mat')
print(result)

mp_pose = mp.solutions.pose

# Assuming 'gallery_features' is in the format (num_images, feature_dim)
gallery_feature = torch.FloatTensor(result['gallery_features'])
gallery_image = result['gallery_images']

print(gallery_feature[0])

# Move to GPU if available
if torch.cuda.is_available():
    gallery_feature = gallery_feature.cuda()

def sort_img(query_feature, gallery_feature):
    if torch.cuda.is_available():
        query_feature = query_feature.cuda()
        gallery_feature = torch.FloatTensor(gallery_feature).cuda()
    
    # Ensure query is of shape (1, feature_dim)
    print(gallery_feature.shape)
    print(query_feature.shape)
    query = query_feature.view(-1, 1)
    score = torch.mm(gallery_feature, query)  # Perform matrix multiplication
    score = score.squeeze(1).cpu()  # Remove singleton dimensions
    score = score.numpy()  # Convert to numpy array
    scoreSorted = np.sort(score)
    index = np.argsort(score)  # Get sorted indices
    index = index[::-1]  # Reverse to get highest scores first
    scoreSorted = scoreSorted[::-1]
    return index, scoreSorted

def isPoseExist(img):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        img_input = cv2.imread(img)
        if img_input is None:
            print(f"Failed to load image: {img}")
            return False
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(img_input)

        if not pose_result.pose_landmarks:
            return False
        else:
            return True

def isBlur(img):
    img_input = cv2.imread(img)
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    laplacian_var = cv2.Laplacian(img_input, cv2.CV_64F).var()
    return laplacian_var < 150 

index_pose = []
index_blur = []
index_feature = []
added_feature = []

def isFeatureExist(i):
    numpy_feature = gallery_feature[i].detach().cpu().numpy()
    if len(added_feature) == 0:
        added_feature.append(numpy_feature)
        index_feature.append(i)
        return
    
    print(f"Feature yang ada (index) : {index_feature}")
    print(f"Feature yang lagi di cek (index) : {i}")

    index, scoreSorted = sort_img(gallery_feature[i], added_feature)

    print(f"Score Kemiripan index {i} dengan yang sudah ada : {scoreSorted[0]}")
    if scoreSorted[0] > 0.5:
        return False
    else:
        added_feature.append(numpy_feature)
        index_feature.append(i)
        return True

def process_images_concurrently(image_paths):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(isPoseExist, image_paths))
    return results

# Process images concurrently
image_paths = [f'{img}' for img in gallery_image]
pose_results = process_images_concurrently(image_paths)

# Filter images based on pose detection results
index_pose = [i for i, has_pose in enumerate(pose_results) if has_pose]

for i in index_pose:
    if isBlur(gallery_image[i]):
        print(f'{gallery_image[i]} is blur')
    else:
        index_blur.append(i)

for i in index_blur:
    isFeatureExist(i)

for i in index_feature:
    shutil.copy(gallery_image[i],'imagevalidasi')
    print(f'{gallery_image[i]} gambar yang masuk validas atas')