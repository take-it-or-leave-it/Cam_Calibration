import numpy as np
import cv2 as cv
import glob

def Cam_index_test():
    cap1 = cv.VideoCapture(3)
    
    while(cap1.isOpened()):
        ret1, frame1 = cap1.read()    # Read 결과와 frame
        
        if(ret1) :   
            cv.imshow('frame_color',frame1)
            
        if cv.waitKey(1) == ord('q'):
            break
     
    cap1.release()       


def Capture_and_Save():
    cap1 = cv.VideoCapture(1)
    
    cnt =1
    while(cap1.isOpened()):
        ret1, frame1 = cap1.read()    # Read 결과와 frame
        
        if(ret1) :   
            cv.imshow('frame_color',frame1)
            
            if cv.waitKey(1) == ord('s'):
                
                cv.imwrite('C:/Users/yongs/Desktop/RB4_Car_detection/data/chess'+str(cnt) +'.jpg',frame1)
                cnt+=1
            
        if cv.waitKey(1) == ord('q'):
            break
     
    cap1.release()       


def Calibration():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('./data/*.jpg')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,6),None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv.drawChessboardCorners(img, (7,6), corners2,ret)
            #cv.imshow('img',img)
            #cv.waitKey(1000)
    cv.destroyAllWindows()  
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    cap1 = cv.VideoCapture(1)
    ret1, img = cap1.read()    # Read 결과와 frame
    cv.imwrite('before_calib.png',img)
    #img = cv.imread('./data/chess1.jpg')
    h,  w = img.shape[:2]
    newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    print(mtx)
    print(dist)
    print(newcameramtx)
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('calibresult.png',dst)
    np.savez('calibration_parameters.npz',ret=ret, mtx=mtx, dist=dist, newcameramtx = newcameramtx)
#Capture_and_Save()
#Calibration()
Cam_index_test()
