import cv2
import dlib
import time
import numpy as np
import copy
import os
from FaceAligner import FaceAligner
import torchvision
import argparse
# predictor=dlib.shape_predictor( './shape_predictor_68_face_landmarks2.dat' )
def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--img_path', type=str, required=True,
                        help="source_image_path.")
    args = parser.parse_args()
    return args    
    
args = parse_args()

predictor=dlib.shape_predictor( './shape_predictor_5_face_landmarks.dat' )

weight='./mmod_human_face_detector.dat'

face_detector=dlib.cnn_face_detection_model_v1( weight )
ALL=list( range( 0 , 5 ) )
print("DDDDDDDDDDDD")
print(args.img_path)
print("DDDDDDDDDDDD")
image = cv2.imread(args.img_path)
path = '/home/daehyeon/DepthNets/pipeline'
# image = cv2.resize(image, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
# Create a HOG face detector using the built-in dlib class
# Load the image into an array
start=time.time()
try:
    faces_cnn=face_detector( image , 1 )
except: 
    pass
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# kdkd 원래는 0.375 였음
for face in faces_cnn :
    fa = FaceAligner(predictor,desiredLeftEye=(0.3, 0.3), desiredFaceWidth=256)
    faceAligned = fa.align(image,gray,face.rect)
    cv2.imwrite(path+'/face_alignmented/{}.png'.format("source"), faceAligned)
    # cv2.imshow("Aligned", faceAligned)
    end=time.time()
    cv2.waitKey()
    cv2.destroyAllWindows()
    break







# dir(train_data.root)

# train_data = torchvision.datasets.ImageFolder(root='/home/daehyeon/hdd/deepfake_1st/fake/',)
# count = 0
# path = '/home/daehyeon/DepthNets/pipeline'
# file_list = os.listdir(path)
# print(file_list)
# for file in file_list:
#     print('file_name:',file)
#     image = cv2.imread(path+'/'+file)
#     # image = cv2.resize(image, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
#     # Create a HOG face detector using the built-in dlib class
#     # Load the image into an array
#     start=time.time()
#     try:faces_cnn=face_detector( image , 1 )
#     except: continue

#     count += 1
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     ct = 0
#     # kdkd 원래는 0.375 였음
#     for face in faces_cnn :
#         fa = FaceAligner(predictor,desiredLeftEye=(0.3, 0.3), desiredFaceWidth=256)
#         faceAligned = fa.align(image,gray,face.rect)
#         cv2.imwrite(path+'/face_alignmented/{}.png'.format(file.split('.')[0]), faceAligned)
#         # cv2.imshow("Aligned", faceAligned)
#         end=time.time()
#         print('{}개 중 {}번째 이미지'.format(len(file_list),count), '걸린시간:' , format( end - start , '.2f' ) )
#         cv2.waitKey()
#         cv2.destroyAllWindows()
#         break


