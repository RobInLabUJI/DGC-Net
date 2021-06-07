from model.vision import DGCVision

import cv2
import numpy as np
import sys

COLORS = [(255,0,255),(255,255,0),(255,0,0),(0,255,0),(0,0,255)]
color = 0

# python point.py imgs/crossdoor/0560.png imgs/crossdoor/0053.png 
# python point.py imgs/crossdoor/0815.png imgs/crossdoor/0560.png 

img1 = cv2.cvtColor(cv2.imread(sys.argv[1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread(sys.argv[2], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
img1 = cv2.resize(img1, (640,360,))
img2 = cv2.resize(img2, (640,360,))
img = cv2.cvtColor(np.hstack((img1,img2,)), cv2.COLOR_RGB2BGR)

dgc = DGCVision()
dgc.predict(img1, img2)
warp_img = dgc.warp()
match_mask = dgc.getMatchabilityMask()
flow = dgc.getFlow()

cv2.namedWindow('Warped image')
cv2.imshow('Warped image', cv2.cvtColor(warp_img, cv2.COLOR_RGB2BGR))

cv2.namedWindow('Matchability mask')
cv2.imshow('Matchability mask', cv2.normalize(cv2.resize(match_mask, (640,360,)),None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX))

cv2.namedWindow('DGC-Net')

def onMouse(event, x, y, flags, param):
    global color
    if event == cv2.EVENT_LBUTTONDOWN:
       color = (color+1) % len(COLORS)
       cv2.circle(img,(x,y),5,COLORS[color],-1)
       if x < 640:
           print(x,y)
           xt, yt = dgc.target(x,y,0,0)
           cv2.circle(img,(round(xt+640),round(yt)),5,COLORS[color],-1)
       else:
           xs, ys = dgc.source(x-640,y)
           cv2.circle(img,(round(xs),round(ys)),5,COLORS[color],-1)

cv2.setMouseCallback('DGC-Net', onMouse)

while True:
    cv2.imshow('DGC-Net', img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == 32:
        img = cv2.cvtColor(np.hstack((img1,img2,)),cv2.COLOR_RGB2BGR)

