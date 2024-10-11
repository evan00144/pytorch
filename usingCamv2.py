# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from tqdm import tqdm
from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_swinv2, ft_net_efficient, ft_net_NAS, ft_net_convnext, PCB, PCB_test
from utils import fuse_all_conv_bn
import cv2
from ultralytics import YOLO
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import mediapipe as mp
import tensorflow as tf
from skimage.measure import shannon_entropy

version = torch.__version__
# fp16
try:
    from apex.fp16_utils import *
except ImportError:  # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

    # Download latest version of PoseNet model from TensorFlow
    path = '1.tflite'

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=path)
interpreter.allocate_tensors()

# Get input and output details of the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

######################################################################
# Options
# --------
def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
    parser.add_argument('--test_dir', default='../Market/pytorch', type=str, help='./test_data')
    parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
    parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
    parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
    parser.add_argument('--use_dense', action='store_true', help='use densenet121')
    parser.add_argument('--use_efficient', action='store_true', help='use efficient-b4')
    parser.add_argument('--use_hr', action='store_true', help='use hr18 net')
    parser.add_argument('--PCB', action='store_true', help='use PCB')
    parser.add_argument('--multi', action='store_true', help='use multiple query')
    parser.add_argument('--fp16', action='store_true', help='use fp16.')
    parser.add_argument('--ibn', action='store_true', help='use ibn.')
    parser.add_argument('--ms', default='1', type=str, help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

    opt = parser.parse_args()
    ###load config###
    # load the training config
    config_path = os.path.join('./model', opt.name, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)  # for the new pyyaml via 'conda install pyyaml'
    opt.fp16 = config['fp16']
    opt.PCB = config['PCB']
    opt.use_dense = config['use_dense']
    opt.use_NAS = config['use_NAS']
    opt.stride = config['stride']
    if 'use_swin' in config:
        opt.use_swin = config['use_swin']
    if 'use_swinv2' in config:
        opt.use_swinv2 = config['use_swinv2']
    if 'use_convnext' in config:
        opt.use_convnext = config['use_convnext']
    if 'use_efficient' in config:
        opt.use_efficient = config['use_efficient']
    if 'use_hr' in config:
        opt.use_hr = config['use_hr']

    if 'nclasses' in config:  # tp compatible with old config files
        opt.nclasses = config['nclasses']
    else:
        opt.nclasses = 751

    if 'ibn' in config:
        opt.ibn = config['ibn']
    if 'linear_num' in config:
        opt.linear_num = config['linear_num']

    str_ids = opt.gpu_ids.split(',')
    name = opt.name
    test_dir = opt.test_dir

    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    print('We use the scale: %s' % opt.ms)
    str_ms = opt.ms.split(',')
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    ######################################################################
    # Load Data
    # ---------
    if opt.use_swin:
        h, w = 224, 224
    else:
        h, w = 256, 128

    data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if opt.PCB:
        data_transforms = transforms.Compose([
            transforms.Resize((384, 192), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        h, w = 384, 192

    data_dir = test_dir

    if opt.multi:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query', 'multi-query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                      shuffle=False, num_workers=16) for x in ['gallery', 'query', 'multi-query']}
    else:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                      shuffle=False, num_workers=16) for x in ['gallery', 'query']}
    class_names = image_datasets['query'].classes
    use_gpu = torch.cuda.is_available()

    def preprocess_image(image):
        img_resized = cv2.resize(image, (input_details[0]['shape'][2], input_details[0]['shape'][1]))  # Resize the image
        input_data = np.expand_dims(img_resized, axis=0)  # Add batch dimension
        input_data = (input_data.astype(np.float32) / 127.5) - 1.0  # Normalize the image
        return input_data

    ######################################################################
    # Load model
    #---------------------------
    def load_network(network):
        save_path = os.path.join('./model', name, 'net_%s.pth' % opt.which_epoch)
        try:
            network.load_state_dict(torch.load(save_path))
        except:
            if torch.cuda.get_device_capability()[0] > 6 and len(opt.gpu_ids) == 1 and int(version[0]) > 1:  # should be >=7
                print("Compiling model...")
                torch.set_float32_matmul_precision('high')
            network.load_state_dict(torch.load(save_path))

        return network

    ######################################################################
    # Extract feature
    # ----------------------
    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3) - 1, -1, -1, device='cuda').long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def extract_feature(model, image):
        pbar = tqdm()
        if opt.linear_num <= 0:
            if opt.use_swin or opt.use_swinv2 or opt.use_dense or opt.use_convnext:
                opt.linear_num = 1024
            elif opt.use_efficient:
                opt.linear_num = 1792
            elif opt.use_NAS:
                opt.linear_num = 4032
            else:
                opt.linear_num = 2048

        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        transformed_frame = data_transforms(frame_pil).unsqueeze(0).cuda()

        ff = torch.FloatTensor(1, opt.linear_num).zero_().cuda()
        frame = fliplr(transformed_frame)
        input_img = Variable(frame.cuda())
        for scale in ms:
            if scale != 1:
                input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
            outputs = model(input_img)
            ff += outputs
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        print(ff.shape)
        return ff[0]

    def get_id(img_path):
        camera_id = []
        labels = []
        for path, v in img_path:
            filename = os.path.basename(path)
            label = filename[0:4]
            camera = filename.split('c')[1]
            if label[0:2] == '-1':
                labels.append(-1)
            else:
                labels.append(int(label))
            camera_id.append(int(camera[0]))
        return camera_id, labels

    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    if opt.multi:
        mquery_path = image_datasets['multi-query'].imgs
        mquery_cam, mquery_label = get_id(mquery_path)

    ######################################################################
    # Load Collected data Trained model
    print('-------test-----------')
    if opt.use_dense:
        model_structure = ft_net_dense(opt.nclasses, stride=opt.stride, linear_num=opt.linear_num)
    elif opt.use_NAS:
        model_structure = ft_net_NAS(opt.nclasses, linear_num=opt.linear_num)
    elif opt.use_swin:
        model_structure = ft_net_swin(opt.nclasses, linear_num=opt.linear_num)
    elif opt.use_swinv2:
        model_structure = ft_net_swinv2(opt.nclasses, (h, w), linear_num=opt.linear_num)
    elif opt.use_convnext:
        model_structure = ft_net_convnext(opt.nclasses, linear_num=opt.linear_num)
    elif opt.use_efficient:
        model_structure = ft_net_efficient(opt.nclasses, linear_num=opt.linear_num)
    elif opt.use_hr:
        model_structure = ft_net_hr(opt.nclasses, linear_num=opt.linear_num)
    else:
        model_structure = ft_net(opt.nclasses, stride=opt.stride, ibn=opt.ibn, linear_num=opt.linear_num)

    if opt.PCB:
        model_structure = PCB(opt.nclasses)

    model = load_network(model_structure)

    # Remove the final fc layer and classifier layer
    if opt.PCB:
        model = PCB_test(model)
    else:
        model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    print('Here I fuse conv and bn for faster inference, and it does not work for transformers. Comment out this following line if you do not want to fuse conv&bn.')
    model = fuse_all_conv_bn(model)
    print('masu')

    yolo = YOLO('yolov10n.pt')

    gallery_features = []
    gallery_images = []

    def calculate_psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    def calculate_ssim(img1, img2):
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        return ssim(img1_gray, img2_gray)

    def is_blurry_tenengrad(image, threshold=15):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        tenengrad = np.mean(gradient_magnitude)
        print(tenengrad)
        return tenengrad < threshold

    def is_low_entropy(image, threshold=5.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        entropy = shannon_entropy(gray)
        return entropy < threshold

    def is_valid_crop(x1, y1, x2, y2, frame_shape):
        return x1 < x2 and y1 < y2 and x2 <= frame_shape[1] and y2 <= frame_shape[0]

    def calculate_iqa(img1, ssim_threshold=0.7, psnr_threshold=25):
        ssim_value = calculate_ssim(img1, img1)
        psnr_value = calculate_psnr(img1, img1)

        if ssim_value < ssim_threshold or psnr_value < psnr_threshold:
            return False
        return True

    def is_close(box1, box2, threshold=50):
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2

        center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        center2 = ((x1_ + x2_) / 2, (y1_ + y2_) / 2)

        distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        return distance < threshold

    def group_boxes(boxes, threshold=50):
        grouped_boxes = []
        while boxes:
            box = boxes.pop(0)
            group = [box]
            to_remove = []
            for other_box in boxes:
                if is_close(box, other_box, threshold):
                    group.append(other_box)
                    to_remove.append(other_box)
            for item in to_remove:
                boxes.remove(item)
            grouped_boxes.append(group)

        return grouped_boxes

    person_counts = []

    cap = cv2.VideoCapture('test.mp4')

    if not cap.isOpened():
        print("Error: Unable to access the camera.")

    def sort_img(query_feature, gallery_feature):
        if torch.cuda.is_available():
            gallery_feature = torch.FloatTensor(gallery_feature).cuda()

        query = query_feature.view(-1, 1)
        score = torch.mm(gallery_feature, query)
        score = score.squeeze(1).cpu()
        score = score.detach().cpu().numpy()
        scoreSorted = np.sort(score)
        index = np.argsort(score)
        index = index[::-1]
        scoreSorted = scoreSorted[::-1]
        return index, scoreSorted

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame")
            break

        if frame is not None:
            results = yolo(frame)
            boxes = [box.xyxy[0].tolist() for result in results for box in result.boxes if box.cls[0] == 0]

            grouped_boxes = group_boxes(boxes)

            for group in grouped_boxes:
                x1 = max(0, min(box[0] for box in group))
                y1 = max(0, min(box[1] for box in group))
                x2 = min(frame.shape[1], max(box[2] for box in group))
                y2 = min(frame.shape[0], max(box[3] for box in group))

                if not is_valid_crop(x1, y1, x2, y2, frame.shape):
                    print("Invalid bounding box, skipping")
                    continue

                cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]

                if cropped_image.shape[0] < 50 or cropped_image.shape[1] < 50:
                    print("Skipping small image")
                    continue

                if is_blurry_tenengrad(cropped_image):
                    print("Skipping blurry image (Tenengrad)")
                    continue

                if is_low_entropy(cropped_image):
                    print("Skipping low-entropy image")
                    continue

                if not calculate_iqa(cropped_image):
                    print("Skipping low-quality image (SSIM/PSNR check failed)")
                    continue

                features = extract_feature(model, cropped_image)

                if len(gallery_features) > 0:
                    index, scoreSorted = sort_img(features, gallery_features)
                    if scoreSorted[0] > 0.5:
                        print("Skipping image as it is already in the gallery")
                        continue

                pose_results = pose.process(cropped_image)
                pose_exist = False

                if pose_results.pose_landmarks:
                    visible_landmarks = sum(1 for item in pose_results.pose_landmarks.landmark if item.visibility > 0.9)
                    if visible_landmarks > 0:
                        pose_exist = True
                    else:
                        print("Skipping image as not enough landmarks are visible")
                else:
                    print("Skipping image as there is no person by pose")

                if pose_exist:
                    temp = features.detach().cpu().numpy()
                    gallery_features.append(temp)

                    path = f'gallery_images/{len(gallery_images)}.jpg'
                    gallery_images.append(path)
                    cv2.imwrite(path, cropped_image)

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    scipy.io.savemat('features.mat', {'gallery_features': gallery_features, 'gallery_images': gallery_images})

if __name__ == '__main__':
    main()