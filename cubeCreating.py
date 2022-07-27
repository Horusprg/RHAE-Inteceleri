from turtle import bgcolor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import cv2 as cv
import imutils
import random
colors = ['red', 'green', 'blue', 'gold', 'mediumturquoise', 'magenta', 'black', 'yellow', 'orange', 'brown', "grey", "magenta"]
# Equation of ring cyclide
# see https://en.wikipedia.org/wiki/Dupin_cyclide
for j in range(20):
    color1, color2, color3, backc = random.choices(colors, k=4)
    fig = go.Figure(data=[
        go.Mesh3d(
            # 8 vertices of a cube
            x=[0, 0, 1, 1, 0, 0, 1, 1],
            y=[0, 1, 1, 0, 0, 1, 1, 0],
            z=[0, 0, 0, 0, 1, 1, 1, 1],
            colorbar_title='z',
            colorscale=[[0, color1],
                        [0.5, color2],
                        [1, color3]],
            # Intensity of each vertex, which will be interpolated and color-coded
            intensity = np.linspace(0, 1, 12, endpoint=True),
            intensitymode='cell',
            # i, j and k give the vertices of triangles
            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            name='y',
            showscale=False
        )
    ])

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

    upRand = np.random.rand(1000,3)
    centerRand = np.random.rand(1000,3)
    eyeRand = np.random.rand(1000,3)

    for i in range(1000):
        camera = dict(
            up=dict(x=upRand[i][0], y=upRand[i][1], z=upRand[i][2]),
            center=dict(x=centerRand[i][0], y=centerRand[i][1], z=centerRand[i][1]),
            eye=dict(x=1.5+eyeRand[i][0], y=1.5+eyeRand[i][1], z=1.5+eyeRand[i][2])
        )
        fig.update_layout(scene_camera=camera, paper_bgcolor = backc)
        place = ""
        if i<700:
            place = "dataset/train/"
        elif i>=700 and i <900:
            place = "dataset/valid/"
        elif i>=900:
            place = "dataset/test/"

        fig.write_image(place+"images/cube"+str(i+j*1000)+".png")
        image = cv.imread(place+"images/cube"+str(i+j*1000)+".png")
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
        f = open(place+"labels/cube"+str(i+j*1000)+".txt", "x")
        f.write("0 "+str(x+w/2)+" "+str(y+h/2)+" "+str(w)+" "+str(h))
        f.close()


"""
for i in range():
    fig.update_layout(scene_camera=camera)
    fig.write_image("images/fig1.png")



    image = cv.imread("images/fig"+i+".png")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    th = cv.threshold(gray, 200, 255,
        cv.THRESH_BINARY_INV)[1]

    # find the largest contour in the threshold image
    cnts = cv.findContours(th.copy(), cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv.contourArea)
    (x, y, w, h) = cv.boundingRect(c)
    # show the original contour image
    print("[INFO] {}, {}, {}, {}".format(x, y, x+w, y+h))
"""