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
import cv2  # For camera frame capture
from PIL import Image  # For converting OpenCV frames to PIL
from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_swinv2, ft_net_efficient, ft_net_NAS, ft_net_convnext, PCB, PCB_test
from utils import fuse_all_conv_bn

version = torch.__version__

# fp16
try:
    from apex.fp16_utils import *
except ImportError:
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support.')

######################################################################
# Options
def main():
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

    # Load the training config
    config_path = os.path.join('./model', opt.name, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
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

    if 'nclasses' in config:
        opt.nclasses = config['nclasses']
    else:
        opt.nclasses = 751

    if 'ibn' in config:
        opt.ibn = config['ibn']
    if 'linear_num' in config:
        opt.linear_num = config['linear_num']

    str_ids = opt.gpu_ids.split(',')
    gpu_ids = [int(id) for id in str_ids if int(id) >= 0]

    # Set GPU ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    # Set input image size
    if opt.use_swin:
        h, w = 224, 224
    else:
        h, w = 256, 128

    # Transformations
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

    ######################################################################
    # Load model
    def load_network(network):
        save_path = os.path.join('./model', opt.name, 'net_%s.pth' % opt.which_epoch)
        network.load_state_dict(torch.load(save_path))
        return network

    # Load model structure
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

    # Load the network
    model = load_network(model_structure)

    # Remove the final fully connected layer and classifier layer
    if opt.PCB:
        model = PCB_test(model)
    else:
        model.classifier.classifier = nn.Sequential()

    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    model = fuse_all_conv_bn(model)

    ######################################################################
    # Extract features from a single camera frame
    def fliplr(img):

        inv_idx = torch.arange(img.size(3) - 1, -1, -1, device='cuda').long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def extract_feature_from_frame(model, frame, ms):
        ff = torch.FloatTensor(1, opt.linear_num).zero_().cuda()  # Batch size = 1
        for i in range(2):
            if i == 1:
                frame = fliplr(frame)
            input_img = Variable(frame.cuda())
            for scale in ms:
                if scale != 1:
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img)
                ff += outputs
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        return ff

    ######################################################################
    # Save feature matrix
    def save_feature(feature, save_path='features.mat'):
        result = {'features': feature}
        scipy.io.savemat(save_path, result)
        print(f"Features saved to {save_path}")

    ######################################################################
    # Laplacian filter to detect blurry images
    def is_blurry(frame, threshold=100):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold

    ######################################################################
    # Camera capture and process
    def process_camera_frame(model):
        cap = cv2.VideoCapture(0)  # Capture from default camera

        if not cap.isOpened():
            print("Error: Unable to access the camera.")
            return

        ms = [math.sqrt(float(s)) for s in opt.ms.split(',')]
        
        gallery_feature = []
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame from the camera.")
                break

            # Check if the frame is blurry
            if is_blurry(frame):
                print(f"Skipping blurry frame {i}")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            transformed_frame = data_transforms(frame_pil).unsqueeze(0).cuda()  # Add batch dimension

            with torch.no_grad():
                features = extract_feature_from_frame(model, transformed_frame, ms)

            # Save the feature
            gallery_feature.append(features.cpu().numpy())
            
            cv2.imwrite('gallery/frame'+str(i)+'.jpg', frame)
            i += 1

            # Display the captured frame (optional)
            cv2.imshow('Camera Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                save_feature(gallery_feature)
                break

        cap.release()
        cv2.destroyAllWindows()

    # Start processing camera feed
    process_camera_frame(model)

if __name__ == '__main__':
    main()
