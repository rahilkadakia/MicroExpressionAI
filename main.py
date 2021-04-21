import struct
import os
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
#from PIL import Image
#import pandas as pd
from openpyxl import load_workbook
import imageio


os.chdir("D:\\SAMMDataset\\SAMM")
# print(os.getcwd())
path = "D:\\SAMMDataset\\SAMM"
# print(os.path.join(path))
image_rows, image_columns, image_depth = 64, 64, 96


def get_training_data():
    training_list = []
    sub_dir = os.path.join(path, "SAMM")
    for sub_img in os.listdir(sub_dir):
        print(sub_img)
        for video in os.listdir(os.path.join(sub_dir + "\\" + sub_img)):
            video_path = sub_dir+"\\"+sub_img+"\\"+video
            video = []
            for img in os.listdir(video_path):
                #print(os.path.join(video_path, img))
                img_path = os.path.join(video_path, img)
                #frame = cv2.imread(img_path)
                #print("path ", img_path)
                frame = imageio.imread(img_path, 'jpg')
                # print(frame.shape)
                # imageresize = cv2.resize(
                #     frame, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
                # grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
                video.append(frame)
                # print(os.path.join(sub_dir+img))
                #im = cv2.imread(os.path.join(sub_dir+img))
            training_list.append(video)
    return training_list


# print(struct.calcsize("P")*8)
X = get_training_data()
# print(len(training_list))


def get_labels():
    filename = 'SAMM_Micro_FACS_Codes_v2.xlsx'
    label_path = os.path.join(path + "\\" + filename)
    label_file = load_workbook(label_path)['MICRO_ONLY']
    # print(label_file)
    labels = []
    for i in label_file["J"]:
        labels.append(i.value)
    return labels[14:]


y = get_labels()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# print(len(get_labels()))
