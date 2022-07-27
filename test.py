import cv2 as cv
import torch
import numpy as np
from matplotlib import pyplot as plt
import streamlink
from keras.preprocessing import image
import imutils

kernel = np.ones((5,5),np.uint8)
link = 'https://youtu.be/NZF-3bksnf8'

def videofeed(url): # codifica o link em video para a intrpratação
    streams = streamlink.streams(url)
    feed = streams["best"].url
    return feed


cap = cv.VideoCapture(0)
# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")


# Carregando o modelo do yolov5("YoloV5s", "YoloV5m", "YoloV5l", "YoloV5xl", "YoloV5s6") disponível na pasta /wheight
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/3dYOLOv5n6.pt', force_reload=True)
model.conf = 0.5

try:
    model.cuda()
except:
    model.cpu()

#Colour Quantization Using K-Means Clustering and OpenCV
def quantimage(image,k):
    i = np.float32(image).reshape(-1,3)
    condition = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,5,1.0)
    ret,label,center = cv.kmeans(i, k , None, condition,5,cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    return final_img

#capture
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = quantimage(frame,2)

    #model detection
    detect = model(frame)
    detect = detect.pandas().xyxy[0]
    detect = detect.to_numpy()

    #frame draw
    for i in detect:
        xmin, ymin, xmax, ymax, confidence, label = int(i[0]), int(i[1]), int(i[2]), int(i[3]), \
                                                    i[4], i[6]
        cv.putText(frame, str(float("{0:.2f}".format(confidence))), (xmax + 20, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 1, cv.LINE_AA)
        cv.putText(frame, label, (xmax + 20, ymin + 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv.LINE_AA)
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    if ret == True:
        cv.imshow('Frame',frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()