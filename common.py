import numpy as np
import cv2
import math
import pandas as pd
from osgeo import gdal
from osgeo import osr
from osgeo import ogr

def getKeyPointsAndDescriptors(img, tile, offset, nFeatures=0):
    """Detect keypoints and compute descriptors. Later, we might something else than SIFT."""
    if (tile is not None and offset is not None):
        img_shape = img.shape
        nbTilesY = int(math.ceil(img_shape[0]/(offset[1] * 1.0)))
        nbTilesX = int(math.ceil(img_shape[1]/(offset[0] * 1.0)))
        #frames = []
        array = []
        for i in range(nbTilesY):
            print(i,"/",nbTilesY)
            for j in range(nbTilesX):
                xmin = offset[0]*j
                ymin = offset[1]*i
                cropped_img = img[ymin:min(offset[1]*i+tile[1], img_shape[0]), xmin:min(offset[0]*j+tile[0], img_shape[1])]
                sift = cv2.xfeatures2d.SIFT_create()
                kp, des = sift.detectAndCompute(cropped_img,None)
                if len(kp) > 0:
                    #print('kp',kp)
                    #print('des',des)
                    #for z in zip(kp, des):
                    #    print(z)
                    #arrays.append(zip(kp, des))
                    for z in zip(kp, des):
                        pt = z[0].pt
                        z[0].pt = (pt[0]+xmin,pt[1]+ymin)
                        array.append(z)
                    #                    df = pd.DataFrame({'keypoints':kp,'descriptors':des})
                    #                    frames.append(df)
        print(len(array))
        #array = np.concatenate(arrays)
        array.sort(key=lambda t: t[0].response, reverse=True)
        #df = pd.concat(frames)
        #df.assign(f = df['keypoints'].response).sort_values('f', ascending=False).drop('f', axis=1)
        sortedarray = array
        if nFeatures>0:
            sortedarray = array[:nFeatures]
            #df.truncate(after=nFeatures)
        #return df['keypoints'], df['descriptors']
        print(sortedarray[0][0])
        print(sortedarray[0][0].pt)
        print(sortedarray[0][1])
        return [ e[0] for e in sortedarray ], [ e[1] for e in sortedarray ]
    else:
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create(nFeatures)
        # find the keypoints and descriptors with SIFT
        print ("detect keypoints on image")
        #cv2.imwrite('tmp.png',img) 
        return sift.detectAndCompute(img,None)

def getMatches(des1, des2, ratio):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 100)   # or pass empty dictionary
    print ("FlannBasedMatcher start")
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    print ("FlannBasedMatcher Knn Match")
    matches = flann.knnMatch(np.asarray(des1,np.float32), np.asarray(des2,np.float32), k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good.append(m)
    return good

def getBinImage(image):
    # Otsu's thresholding with a gaussian blur
    blur = cv2.GaussianBlur(image,(5,5),0)
    th,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img

def saveGCPs(gcps, srs, outputFile):
    out_ds = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(outputFile)
    out_lyr = out_ds.CreateLayer('gcps', geom_type = ogr.wkbPoint, srs = srs)
    out_lyr.CreateField(ogr.FieldDefn('Id', ogr.OFTString))
    out_lyr.CreateField(ogr.FieldDefn('Info', ogr.OFTString))
    out_lyr.CreateField(ogr.FieldDefn('X', ogr.OFTReal))
    out_lyr.CreateField(ogr.FieldDefn('Y', ogr.OFTReal))
    for i in range(len(gcps)):
        f = ogr.Feature(out_lyr.GetLayerDefn())
        f.SetField('Id', gcps[i].Id)
        f.SetField('Info', gcps[i].Info)
        f.SetField('X', gcps[i].GCPPixel)
        f.SetField('Y', gcps[i].GCPLine)
        f.SetGeometry(ogr.CreateGeometryFromWkt('POINT(%f %f)' % (gcps[i].GCPX, gcps[i].GCPY)))
        out_lyr.CreateFeature(f)

    
def georef(inputFile, referenceFile, outputFile, gcpOutputFile, tile, offset, ratio = 0.75, convertToBinary = False):
    """Georeference the input using the training image and save the result in outputFile. 
    A ratio can be given to select more or less matches (defaults to 0.75)."""
    im1 = cv2.imread(referenceFile, cv2.IMREAD_GRAYSCALE) # queryImage (IMREAD_COLOR flag=cv.IMREAD_GRAYSCALE to force grayscale)
    im2 = cv2.imread(inputFile, cv2.IMREAD_GRAYSCALE) # trainImage (IMREAD_COLOR flag=cv.IMREAD_GRAYSCALE to force grayscale)

    img1 = im1
    if convertToBinary:
        img1 = getBinImage(im1)
    img2 = im2
    if convertToBinary:
        img2 = getBinImage(im2)
    
    kp1, des1 = getKeyPointsAndDescriptors(img1, tile, offset, 100000)
    print("points in img1",len(des1))
    kp2, des2 = getKeyPointsAndDescriptors(img2, tile, offset, 100000)
    print("points in img2",len(des2))

    # Match both ways
    good1 = getMatches(des1, des2, ratio)
    good2 = getMatches(des2, des1, ratio)
    # Only keep matches that are present in both ways
    good = [m for m in good1 if any(mm.queryIdx == m.trainIdx and mm.trainIdx == m.queryIdx for mm in good2)]

    # sortedMatches = sorted(good, key=lambda x:x.distance)
    # for m in sortedMatches:
    #     print(str(m.distance) + ' => ' + str(m.queryIdx) + ' ' + str(m.trainIdx))

    print("matches",len(good))
    MIN_MATCH_COUNT = 3

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        matchesMask = mask.ravel().tolist()

        ds = gdal.Open(referenceFile) 
        xoffset, px_w, rot1, yoffset, px_h, rot2 = ds.GetGeoTransform()

        gcp_list = []
        geo_t = ds.GetGeoTransform ()
        # gcp_string = ''
        print(str(len(matchesMask)) + " matches")
        for i, goodmatch in enumerate(matchesMask):
            if goodmatch == 1:
                p1 = kp1[good[i].queryIdx].pt
                p2 = kp2[good[i].trainIdx].pt
                pp = gdal.ApplyGeoTransform(geo_t,p1[0],p1[1])
                print ("GCP geot = (" + str(p2[0]) +","+ str(p2[1]) + ") -> (" + str(pp[0]) +","+ str(pp[1]) + ")")
                z = 0
                gcp = gdal.GCP(pp[0], pp[1], z, p2[0], p2[1])
                #print ("GCP     = (" + str(p2[0]) +","+ str(p2[1]) + ") -> (" + str(pp[0]) +","+ str(pp[1]) + ")")
                # gcp_string += ' -gcp '+" ".join([str(p2[0]),str(p2[1]),str(pp[0]), str(pp[1])])
                gcp_list.append(gcp)
        print(str(len(gcp_list)) + " GCPs")

        translate_t = gdal.GCPsToGeoTransform(gcp_list)
        translate_inv_t = gdal.InvGeoTransform(translate_t)
        print(str(len(translate_t)))
        print("geotransform = %s" % (translate_t,))
        print(str(len(translate_inv_t)))
        print("invgeotransform = %s " % (translate_inv_t,))
        #trans_gcp_list = []
        dst_gcp_list = []
        mapResiduals = 0.0
        geoResiduals = 0.0
        for gcp in gcp_list:
            # Inverse geotransform to get the corresponding pixel
            pix = gdal.ApplyGeoTransform(translate_inv_t,gcp.GCPX,gcp.GCPY)
            print ("GCP = (" + str(gcp.GCPPixel) +","+ str(gcp.GCPLine) + ") -> (" + str(gcp.GCPX) +","+ str(gcp.GCPY) + ")")
            print (" => (" + str(pix[0]) + ","+str(pix[1]))
            map_dX = gcp.GCPPixel - pix[0]
            map_dY = gcp.GCPLine - pix[1]
            map_residual = map_dX * map_dX + map_dY * map_dY
            mapResiduals = mapResiduals + map_residual
            #trans_gcp_list.append(pix)
            # Apply the transform to get the GCP location in the output SRS
            pp = gdal.ApplyGeoTransform(translate_t,gcp.GCPPixel,gcp.GCPLine)
            z = 0
            out_gcp = gdal.GCP(pp[0], pp[1], z, gcp.GCPPixel, gcp.GCPLine)
            print ("GCP = (" + str(out_gcp.GCPPixel) +","+ str(out_gcp.GCPLine) + ") -> (" + str(pp[0]) +","+ str(pp[1]) + ")")
            # gcp_string += ' -gcp '+" ".join([str(p2[0]),str(p2[1]),str(pp[0]), str(pp[1])])
            dX = gcp.GCPX - pp[0]
            dY = gcp.GCPY - pp[1]
            residual = dX * dX + dY * dY
            geoResiduals = geoResiduals + residual
            print("map residual = " + str(map_dX) + ","+str(map_dY) + " = "+str(map_residual))
            print("residual = " + str(dX) + ","+str(dY) + " = "+str(residual))
            dst_gcp_list.append(out_gcp)

        print("map residuals = "+str(mapResiduals))
        print("geo residuals = "+str(geoResiduals))
        # print (gcp_string)
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        src_ds = gdal.Open(inputFile)
        # translate and warp the inputFile using GCPs and polynomial of order 1
        dst_ds = gdal.Translate('', src_ds, outputSRS = ds.GetProjection(), GCPs = gcp_list, format='MEM')        
        dst_ds = gdal.Warp(outputFile, dst_ds, tps = False, polynomialOrder = 1, dstNodata = 1)

        if gcpOutputFile is not None:
            saveGCPs(dst_gcp_list, osr.SpatialReference(ds.GetProjection()), gcpOutputFile)
        
        dst_ds = None

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        return img3;

    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
