import numpy as np
import cv2 as cv


FLANN_INDEX_LSH = 6
calib_parms = np.load('calibration_parameters.npz')
mtx = calib_parms['mtx']
dist = calib_parms['dist']
newcameramtx = calib_parms['newcameramtx']


def anorm2(a):
    return (a*a).sum(-1)
def anorm(a):
    return np.sqrt(anorm2(a))

def matchKeypoints(keyPoints1, keyPoints2, descriptors1, descriptors2 ):

    bf = cv.BFMatcher()

    
    raw_matches = bf.knnMatch(descriptors1,descriptors2,k=2)
    
    
    matches = []
    for m in raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.9:
            matches.append((m[0].trainIdx,m[0].queryIdx))
    
    if len(matches) >=3:
        
        keyPoints1 = np.float32([keyPoints1[i] for(_,i) in matches])
        keyPoints2 = np.float32([keyPoints2[i] for(i,_) in matches])
        
        H, status = cv.findHomography(keyPoints1,keyPoints2,cv.RANSAC, 2.0)
        print("%d / %d inliers/matches " % (np.sum(status),len(status)))
    else :
        H,status = None, None
    return matches, H, status

def main():

    cap1 = cv.VideoCapture(1)
    cap2 = cv.VideoCapture(2)
    cap3 = cv.VideoCapture(3)
    ret1, img1 = cap1.read()    # Read 결과와 frame
    ret2, img2 = cap2.read()    # Read 결과와 frame
    ret3, img3 = cap3.read()
    
    img1 = cv.undistort(img1, mtx, dist, None, newcameramtx)
    img2 = cv.undistort(img2, mtx, dist, None, newcameramtx)
    img3 = cv.undistort(img3, mtx, dist, None, newcameramtx)
    gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    gray3 = cv.cvtColor(img3,cv.COLOR_BGR2GRAY)
    
    detector = cv.SIFT_create()
    keyPoints1, descriptors1 = detector.detectAndCompute(gray1,None)
    keyPoints2, descriptors2 = detector.detectAndCompute(gray2,None)


    
    keyPoints1 = np.float32([keypoint.pt for keypoint in keyPoints1])
    keyPoints2 = np.float32([keypoint.pt for keypoint in keyPoints2])
    print("asdfasdf")
    print('img1 - %d features, img2 - %d features' %(len(keyPoints1), len(keyPoints2)))
    matches, H , status = matchKeypoints(keyPoints2, keyPoints1, descriptors2, descriptors1)
    print(H)
    #img_matching_result = drawMatches(img1, img2, keyPoints1, keyPoints2, matches, status)
    result = cv.warpPerspective(img2,H,(img2.shape[1]+img1.shape[1],img2.shape[0]))
    result[0:img1.shape[0],0:img1.shape[1]]=img1
    cv.imshow('img1', img1)
    cv.imshow('img2',img2)
    cv.imshow('img3',img3)
    cv.imshow('result', result)
    
    gray2 = cv.cvtColor(result,cv.COLOR_BGR2GRAY)
    keyPoints2, descriptors2 = detector.detectAndCompute(gray2,None)
    keyPoints3, descriptors3 = detector.detectAndCompute(gray3,None)
    keyPoints2 = np.float32([keypoint.pt for keypoint in keyPoints2])
    keyPoints3 = np.float32([keypoint.pt for keypoint in keyPoints3])
    print('img2 - %d features, img3 - %d features' %(len(keyPoints2), len(keyPoints3)))
    matches, H2 , status = matchKeypoints(keyPoints3, keyPoints2, descriptors3, descriptors2)
    result2 = cv.warpPerspective(img3,H2,(img3.shape[1]+result.shape[1],img3.shape[0]))
    result2[0:result.shape[0],0:result.shape[1]]=result
    
    #result2 = cv.undistort(result2,mtx,dist,None,newcameramtx)
    cv.imshow('result2', result2)
    
    
    
    #cv.imshow('matching result ',img_matching_result)
    return H



main()