import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import glob

imagens = []
for imageName in glob.glob('C:/Users/LPO/Documents/GitHub/Inteceleri/images/*.jpg'):
  img = cv.imread("C:/Users/LPO/Documents/GitHub/Inteceleri/images"+imageName)
  #img = cv.medianBlur(img,5)
  #th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
  cv.imshow("file", img)