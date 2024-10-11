import mediapipe as mp
import cv2
import os


mp_pose = mp.solutions.pose


def isPoseExist(img_path):
    # Load the image
    img_input = cv2.imread(img_path)
    
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    if img_input is None:
        print(f"Image not found or cannot be loaded: {img_path}")
        return False
    
    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe
    pose_result = pose.process(img_rgb)

    # Check if pose landmarks exist
    if not pose_result.pose_landmarks:
        print(f'Pose not found in {img_path}')
        return False
    else:
        print(f'Pose detected in {img_path}')
        return True


# Test the specific file
print("Testing single file:")
isPoseExist('gallery_images/193.jpg')

# Iterate through the files in the directory and check for poses
print("\nTesting all files in directory:")
for file in sorted(os.listdir('gallery_images')):
    img_path = os.path.join('gallery_images', file)
    isPoseExist(img_path)