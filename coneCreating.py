import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import cv2 as cv
import imutils

# Equation of ring cyclide
# see https://en.wikipedia.org/wiki/Dupin_cyclide
import numpy as np
a, b, d = 1.32, 1., 0.8
c = a**2 - b**2
u, v = np.mgrid[0:2*np.pi:100j, 0:2*np.pi:100j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)

fig = go.Figure(data=go.Cone(x=[1], y=[1], z=[1], u=[1], v=[1], w=[0],
                  showscale=False))

fig.update_layout()
fig.update_layout(title_text="")
fig.update_layout(showlegend=False)
fig.update_layout(scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(256, 256, 256)",
                         gridcolor="white",
                         showbackground=False,
                         zerolinecolor="white",
                         showticklabels = False,
                         showgrid = False,
                         visible = False),
                    yaxis = dict(
                        backgroundcolor="rgb(230, 200,230)",
                        gridcolor="white",
                        showbackground=False,
                        zerolinecolor="white",
                         showticklabels = False,
                         showgrid = False,
                         visible = False),
                    zaxis = dict(
                        backgroundcolor="rgb(230, 230,200)",
                        gridcolor="white",
                        showbackground=False,
                        zerolinecolor="white",
                         showticklabels = False,
                         showgrid = False,
                         visible = False),),
                    width=700,
                    margin=dict(
                    r=10, l=10,
                    b=10, t=10)
                  )


upRand = 1.5*np.random.rand(1000,3)
centerRand = 1.5*np.random.rand(1000,3)

for i in range(1000):
    camera = dict(
        up=dict(x=upRand[i][0], y=upRand[i][1], z=upRand[i][2]),
        center=dict(x=centerRand[i][0], y=centerRand[i][1], z=centerRand[i][1]),
        eye=dict(x=3, y=3, z=3)
    )
    fig.update_layout(scene_camera=camera)
    place = ""
    if i<700:
        place = "dataset/train/"
    elif i>=700 and i <900:
        place = "dataset/test/"
    elif i>=900:
        place = "dataset/valid/"

    fig.write_image(place+"images/cone"+str(i)+".png")
    image = cv.imread(place+"images/cone"+str(i)+".png")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    th = cv.threshold(gray, 200, 255,
        cv.THRESH_BINARY_INV)[1]
    # find the largest contour in the threshold image
    cnts = cv.findContours(th.copy(), cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv.contourArea)
    (x, y, w, h) = cv.boundingRect(c)

    height, width = image.shape[:2]
    x=x/width
    y=y/height
    w=w/width
    h=h/height
    f = open(place+"labels/cone"+str(i)+".txt", "x")
    f.write("2 "+str(x+w/2)+" "+str(y+h/2)+" "+str(w)+" "+str(h))
    f.close()