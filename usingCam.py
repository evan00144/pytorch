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
version =  torch.__version__
#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

######################################################################
# Options
# --------
def main():
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
    parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
    parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
    parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
    parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
    parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
    parser.add_argument('--use_efficient', action='store_true', help='use efficient-b4' )
    parser.add_argument('--use_hr', action='store_true', help='use hr18 net' )
    parser.add_argument('--PCB', action='store_true', help='use PCB' )
    parser.add_argument('--multi', action='store_true', help='use multiple query' )
    parser.add_argument('--fp16', action='store_true', help='use fp16.' )
    parser.add_argument('--ibn', action='store_true', help='use ibn.' )
    parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

    opt = parser.parse_args()
    ###load config###
    # load the training config
    config_path = os.path.join('./model',opt.name,'opts.yaml')
    with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader) # for the new pyyaml via 'conda install pyyaml'
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

    if 'nclasses' in config: # tp compatible with old config files
        opt.nclasses = config['nclasses']
    else: 
        opt.nclasses = 751 

    if 'ibn' in config:
        opt.ibn = config['ibn']
    if 'linear_num' in config:
        opt.linear_num = config['linear_num']

    str_ids = opt.gpu_ids.split(',')
    #which_epoch = opt.which_epoch
    name = opt.name
    test_dir = opt.test_dir

    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >=0:
            gpu_ids.append(id)

    print('We use the scale: %s'%opt.ms)
    str_ms = opt.ms.split(',')
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))

    # set gpu ids
    if len(gpu_ids)>0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    ######################################################################
    # Load Data
    # ---------
    #
    # We will use torchvision and torch.utils.data packages for loading the
    # data.
    #
    if opt.use_swin:
        h, w = 224, 224
    else:
        h, w = 256, 128

    data_transforms = transforms.Compose([
            transforms.Resize((h, w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ############### Ten Crop        
            #transforms.TenCrop(224),
            #transforms.Lambda(lambda crops: torch.stack(
            #   [transforms.ToTensor()(crop) 
            #      for crop in crops]
            # )),
            #transforms.Lambda(lambda crops: torch.stack(
            #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
            #       for crop in crops]
            # ))
    ])

    if opt.PCB:
        data_transforms = transforms.Compose([
            transforms.Resize((384,192), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
        h, w = 384, 192


    data_dir = test_dir

    if opt.multi:
        image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
    else:
        image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                shuffle=False, num_workers=16) for x in ['gallery','query']}
    class_names = image_datasets['query'].classes
    use_gpu = torch.cuda.is_available()

    ######################################################################
    # Load model
    #---------------------------
    def load_network(network):
        save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
        try:
            network.load_state_dict(torch.load(save_path))
        except: 
            if torch.cuda.get_device_capability()[0]>6 and len(opt.gpu_ids)==1 and int(version[0])>1: # should be >=7
                print("Compiling model...")
                # https://huggingface.co/docs/diffusers/main/en/optimization/torch2.0
                torch.set_float32_matmul_precision('high')
                # network = torch.compile(network, mode="default", dynamic=True) # pytorch 2.0
            network.load_state_dict(torch.load(save_path))

        return network


    ######################################################################
    # Extract feature
    # ----------------------
    #
    # Extract feature from  a trained model.
    #
    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1,-1,-1,device='cuda').long()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip

    def extract_feature(model,image):
        #features = torch.FloatTensor()
        # count = 0
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
            #filename = path.split('/')[-1]
            filename = os.path.basename(path)
            label = filename[0:4]
            camera = filename.split('c')[1]
            if label[0:2]=='-1':
                labels.append(-1)
            else:
                labels.append(int(label))
            camera_id.append(int(camera[0]))
        return camera_id, labels

    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam,gallery_label = get_id(gallery_path)
    query_cam,query_label = get_id(query_path)

    if opt.multi:
        mquery_path = image_datasets['multi-query'].imgs
        mquery_cam,mquery_label = get_id(mquery_path)

    ######################################################################
    # Load Collected data Trained model
    print('-------test-----------')
    if opt.use_dense:
        model_structure = ft_net_dense(opt.nclasses, stride = opt.stride, linear_num=opt.linear_num)
    elif opt.use_NAS:
        model_structure = ft_net_NAS(opt.nclasses, linear_num=opt.linear_num)
    elif opt.use_swin:
        model_structure = ft_net_swin(opt.nclasses, linear_num=opt.linear_num)
    elif opt.use_swinv2:
        model_structure = ft_net_swinv2(opt.nclasses, (h,w),  linear_num=opt.linear_num)
    elif opt.use_convnext:
        model_structure = ft_net_convnext(opt.nclasses, linear_num=opt.linear_num)
    elif opt.use_efficient:
        model_structure = ft_net_efficient(opt.nclasses, linear_num=opt.linear_num)
    elif opt.use_hr:
        model_structure = ft_net_hr(opt.nclasses, linear_num=opt.linear_num)
    else:
        model_structure = ft_net(opt.nclasses, stride = opt.stride, ibn = opt.ibn, linear_num=opt.linear_num)

    if opt.PCB:
        model_structure = PCB(opt.nclasses)

    #if opt.fp16:
    #    model_structure = network_to_half(model_structure)


    model = load_network(model_structure)

    # Remove the final fc layer and classifier layer
    if opt.PCB:
        #if opt.fp16:
        #    model = PCB_test(model[1])
        #else:
            model = PCB_test(model)
    else:
        #if opt.fp16:
            #model[1].model.fc = nn.Sequential()
            #model[1].classifier = nn.Sequential()
        #else:
            model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()


    print('Here I fuse conv and bn for faster inference, and it does not work for transformers. Comment out this following line if you do not want to fuse conv&bn.')
    model = fuse_all_conv_bn(model)

    # We can optionally trace the forward method with PyTorch JIT so it runs faster.
    # To do so, we can call `.trace` on the reparamtrized module with dummy inputs
    # expected by the module.
    # Comment out this following line if you do not want to trace.
    #dummy_forward_input = torch.rand(opt.batchsize, 3, h, w).cuda()
    #model = torch.jit.trace(model, dummy_forward_input)

    yolo = YOLO('yolov10n.pt')

    cap = cv2.VideoCapture('test.mp4')

    gallery_features=[]
    gallery_images=[]

    # Laplacian filter to detect blurry images
    def is_blurry(frame, threshold=100):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = yolo(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls[0] == 0:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # if is_blurry(frame):
                    #     print(f"Skipping blurry frame")
                    #     continue
                    
                    cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
                    # Before feature extract, if there is faces, blur the face
                    faces = faceCascade.detectMultiScale(cropped_image,1.2,4)
                    for (x, y, w, h) in faces:
                        # To make a face blurred
                        ROI = cropped_image[y:y+h, x:x+w]
                        blur = cv2.GaussianBlur(ROI, (91,91),0) 
                        # Insert ROI back into image
                        cropped_image[y:y+h, x:x+w] = blur


                    features = extract_feature(model, cropped_image)

                    temp = features.detach().cpu().numpy()
                    gallery_features.append(temp)

                    path = f'gallery_images/{len(gallery_images)}.jpg'
                    gallery_images.append(path)
                    cv2.imwrite(path, cropped_image)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    scipy.io.savemat('features.mat', {'gallery_features': gallery_features, 'gallery_images': gallery_images})
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
