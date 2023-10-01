
import cv2 as cv
import numpy as np
from stitching import Stitcher
from matplotlib import pyplot as plt
import sys
import time

start = time.time()  # 시작 시간 저장
#setting = {"warper_type": "stereographic", "detector": "orb", "nfeatures" : 200, "try_use_gpu": True}
#stitcher = Stitcher(**setting)

stitcher = Stitcher(detector="orb", confidence_threshold=0.5);
cap = cv.VideoCapture(0)
cap1 = cv.VideoCapture(1)
cap2 = cv.VideoCapture(2)
print('width :%d, height : %d' % (cap.get(3), cap.get(4)))
print('width :%d, height : %d' % (cap1.get(3), cap1.get(4)))
print('width :%d, height : %d' % (cap2.get(3), cap2.get(4)))

fourcc = cv.VideoWriter_fourcc(*'DIVX')
cam2 = cv.VideoWriter('cam2.avi', fourcc, 10.0, (640,480))
cam1 = cv.VideoWriter('cam1.avi', fourcc, 10.0,  (640,480))
out = cv.VideoWriter('result.avi', fourcc, 10.0, (1280,720))

def plot_image(img, figsize_in_inches=(5,5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

cnt = 1
while(True):
    cnt+=1
    ret0, frame0 = cap.read()    # Read 결과와 frame
    ret1, frame1 = cap1.read()    # Read 결과와 frame
    ret2, frame2 = cap2.read()    # Read 결과와 frame
    print(cnt)
    if(cnt==10):
        break
    
    if(ret1 & ret2) :
        frame_list = [frame1, frame2]
        pano = stitcher.stitch(frame_list)
        pano = np.array(pano) 
        pano = cv.resize(pano, (1280,720))
    
        #cv.imshow('frame_color',pano)
        cam1.write(frame1)
        cam2.write(frame2)
        out.write(pano)

        
    
        #if cv.waitKey(1) == ord('q'):
        #    break
        
out.release()
cam1.release()
cam2.release()
cap.release()
#cv.destroyAllWindows()