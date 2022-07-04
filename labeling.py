import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
fig.write_image("images/fig1.png")
import cv2 as cv
import imutils

image = cv.imread("images/fig1.png")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
th = cv.threshold(gray, 200, 255,
	cv.THRESH_BINARY_INV)[1]

# find the largest contour in the threshold image
cnts = cv.findContours(th.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv.contourArea)
# draw the shape of the contour on the output image, compute the
# bounding box, and display the number of points in the contour
output = image.copy()
cv.drawContours(output, [c], -1, (0, 255, 0), 3)
(x, y, w, h) = cv.boundingRect(c)
text = "original, num_pts={}".format(len(c))
cv.putText(output, text, (x, y - 15), cv.FONT_HERSHEY_SIMPLEX,
	0.9, (0, 255, 0), 2)
# show the original contour image
print("[INFO] {}".format(c))
cv.imshow("result", output)
cv.waitKey(0)