from model.vision import DGCVision

import cv2
import numpy as np
import sys

dgc = DGCVision()

COLORS = [(255,0,255),(255,255,0),(255,0,0),(0,255,0),(0,0,255)]

points = [(228, 189), (415, 191), (235, 251), (405, 251)]
ref_frame = '../gripper_camera/red_light/13/0068.png'

img1 = cv2.cvtColor(cv2.imread(ref_frame, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
img1 = cv2.resize(img1, (640,360,))

source = '../gripper_camera/red_light/08/'
ini_frame = 110
end_frame = 445

cv2.namedWindow('DGC-Net')

out = cv2.VideoWriter('output.mp4',0x7634706d , 15.0, (1280,360))

for frame in range(ini_frame, end_frame):
    filename = source + str(frame).zfill(4) + '.png'
    img2 = cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, (640,360,))

    img = cv2.cvtColor(np.hstack((img1,img2,)), cv2.COLOR_RGB2BGR)

    dgc.predict(img1, img2)
    warp_img = dgc.warp()
    match_mask = dgc.getMatchabilityMask()
    flow = dgc.getFlow()
    color = 0
    for p in points:
        (x, y) = p
        color = (color+1) % len(COLORS)
        cv2.circle(img,(x,y),5,COLORS[color],-1)
        xt, yt = dgc.target(x,y,0,0)
        cv2.circle(img,(round(xt+640),round(yt)),5,COLORS[color],-1)

    cv2.imshow('DGC-Net', img)
    cv2.waitKey(1)
    out.write(img)

out.release()

