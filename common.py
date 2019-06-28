import numpy as np
import cv2
from osgeo import gdal
from osgeo import osr

def georef(inputFile, referenceFile, outputFile, ratio = 0.75):
    """Georeference the input using the training image and save the result in outputFile. 
    A ratio can be given to select more or less matches (defaults to 0.75)."""
    img1 = cv2.imread(referenceFile,0) # queryImage (IMREAD_COLOR flag=0 to force grayscale)
    img2 = cv2.imread(inputFile,0) # trainImage (IMREAD_COLOR flag=0 to force grayscale)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.xfeatures2d.SURF_create()
    # sift = cv2.KAZE_create()
    # sift = cv2.AKAZE_create()# NOPE
    # sift = cv2.ORB_create()#NOPE
    # sift = cv2.BRISK_create()#NOPE

    # find the keypoints and descriptors with SIFT
    print ("sift on image 1")
    kp1, des1 = sift.detectAndCompute(img1,None)
    # # find the keypoints with ORB
    # kp1 = sift.detect(img1,None)
    # # compute the descriptors with ORB
    # kp1, des1 = sift.compute(img1, kp1)
    print ("sift on image 2")
    kp2, des2 = sift.detectAndCompute(img2,None)
    # # find the keypoints with ORB
    # kp2 = sift.detect(img2,None)
    # # compute the descriptors with ORB
    # kp2, des2 = sift.compute(img2, kp2)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=100)   # or pass empty dictionary

    print ("FlannBasedMatcher start")
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    print ("FlannBasedMatcher Knn Match")
    matches = flann.knnMatch(des1,des2,k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good.append(m)

    sortedMatches = sorted(good, key=lambda x:x.distance)
    for m in sortedMatches:
        print(str(m.distance) + ' => ' + str(m.queryIdx) + ' ' + str(m.trainIdx))

    MIN_MATCH_COUNT = 3

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        ds = gdal.Open(referenceFile) 
        xoffset, px_w, rot1, yoffset, px_h, rot2 = ds.GetGeoTransform()
        def georef( x, y ):
            "Get the geographic coordinates from the pixel coordinates x,y using the transform from the reference file (return a tuple)."
            posX = px_w * x + rot1 * y + xoffset
            posY = px_h * x + rot2 * y + yoffset
            # shift to the center of the pixel
            posX += px_w / 2.0
            posY += px_h / 2.0
            return (posX, posY);

        gcp_list = []
        # gcp_string = ''
        for i,goodmatch in enumerate(matchesMask):
            if goodmatch == 1:
                p1 = kp1[good[i].queryIdx].pt
                p2 = kp2[good[i].trainIdx].pt
                pp = georef(p1[0],p1[1])
                z = 0
                gcp = gdal.GCP(pp[0], pp[1], z, p2[0], p2[1])
                print ("GCP = " + str(pp[0]) +","+ str(pp[1]) + " " + str(p2[0]) +","+ str(p2[1]))
                # gcp_string += ' -gcp '+" ".join([str(p2[0]),str(p2[1]),str(pp[0]), str(pp[1])])
                gcp_list.append(gcp)

        # print (gcp_string)
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        src_ds = gdal.Open(inputFile)
        dst_ds = gdal.Translate('', src_ds, outputSRS = ds.GetProjection(), GCPs = gcp_list, format='VRT')
        dst_ds = gdal.Warp(outputFile, dst_ds, tps = False, polynomialOrder = 1, errorThreshold = 1)
        dst_ds = None

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        return img3;

    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
