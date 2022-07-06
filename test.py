import cv2 as cv
import torch
import numpy as np
from matplotlib import pyplot as plt
import streamlink
from keras.preprocessing import image

link = 'https://youtu.be/NZF-3bksnf8'
def videofeed(url): # codifica o link em video para a intrpratação
    streams = streamlink.streams(url)
    feed = streams["best"].url
    return feed


cap = cv.VideoCapture(videofeed(link))
# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")


# Carregando o modelo do yolov5("YoloV5s", "YoloV5m", "YoloV5l", "YoloV5xl", "YoloV5s6") disponível na pasta /wheight
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
model.conf = 0.3

try:
    model.cuda()
except:
    model.cpu()

while(cap.isOpened()):
    ret, frame = cap.read()
    img_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img_grey = img_grey.astype(np.uint8)

    th3 = cv.adaptiveThreshold(img_grey,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)

    detect = model(th3)
    detect = detect.pandas().xyxy[0]
    detect = detect.to_numpy()
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