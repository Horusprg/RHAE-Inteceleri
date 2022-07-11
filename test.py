import cv2 as cv
import torch
import numpy as np
from matplotlib import pyplot as plt
import streamlink
from keras.preprocessing import image
import imutils

link = 'https://youtu.be/NZF-3bksnf8'
def videofeed(url): # codifica o link em video para a intrpratação
    streams = streamlink.streams(url)
    feed = streams["best"].url
    return feed


cap = cv.VideoCapture(1)
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

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    th = cv.threshold(gray, 161, 355, cv.THRESH_BINARY_INV)[1]
    """
    img_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img_grey = img_grey.astype(np.uint8)

    th = cv.adaptiveThreshold(img_grey,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
    """
    cnts = cv.findContours(th.copy(), cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv.contourArea)
    # draw the shape of the contour on the output image, compute the
    # bounding box, and display the number of points in the contour
    output = frame.copy()
    #cv.drawContours(output, [c], -1, (0, 255, 0), 3)
    detect = model(frame)
    detect = detect.pandas().xyxy[0]
    detect = detect.to_numpy()
    for i in detect:
        xmin, ymin, xmax, ymax, confidence, label = int(i[0]), int(i[1]), int(i[2]), int(i[3]), \
                                                    i[4], i[6]
        cv.putText(output, str(float("{0:.2f}".format(confidence))), (xmax + 20, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 1, cv.LINE_AA)
        cv.putText(output, label, (xmax + 20, ymin + 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv.LINE_AA)
        cv.rectangle(output, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    if ret == True:
        cv.imshow('Frame',output)
        cv.imshow('Frame2',th)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()