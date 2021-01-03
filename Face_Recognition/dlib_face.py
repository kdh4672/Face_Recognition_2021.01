# 학습 모델 다운로드
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

import dlib
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import shutil
from torch.optim import lr_scheduler
from ArcMarginProduct import *
import copy
import time
import os
from PIL import Image
from FaceAligner import FaceAligner
from Face_Rate import *


dlib.cuda.set_device(3)

import os
weight = './mmod_human_face_detector.dat'
detector = dlib.cnn_face_detection_model_v1(weight)
predictor = dlib.shape_predictor('./shape_predictor_5_face_landmarks.dat')
ALL = list(range(0, 5))
index = ALL
path = '/database/daehyeon/High_Resolution/non_frontal/'
save_path = '/database/daehyeon/High_Resolution/non_frontal_crop/'
list_ = [x for x in range(0,12800)]
for file in list_:
    print(str(file)+'번째 처리중 ...')
    img_frame = cv.imread(path+str(file)+'.png')
    gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    print("얼굴 갯수:", "{}개".format(len(dets)))
    for face in dets:
        fa = FaceAligner(predictor,desiredLeftEye=(0.3, 0.3), desiredFaceWidth=224)
        faceAligned = fa.align(img_frame,gray,face.rect)
        cv.imwrite(save_path+'{}.png'.format(file),faceAligned)
        # faceAligned = Image.fromarray(faceAligned)




