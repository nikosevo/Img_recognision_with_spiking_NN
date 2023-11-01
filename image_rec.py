import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
def photo_input():
    cam = cv.VideoCapture(0)
    cv.namedWindow("test")


    k = -1
    while k==-1:
        data,frame = cam.read()
        if data:
            cv.imshow('test',frame)

        k = cv.waitKey(1)
        if(k != -1):
            img = frame
            break

    cv.destroyWindow("test")
    print("key entered")
    print(k)

    img = rgb2gray(img)

    img = img[40:1040,500:1500]

    for i in range(0,1000):
        for j in range (0,1000):
            if img[i][j] < 130:
                img[i][j] = 1
            else:
                img[i][j] =0


    img = cv.resize(img,(28,28))
    input = torch.zeros(1,28,28)
    input[0] = torch.tensor(img,dtype=torch.float32)


    return input



