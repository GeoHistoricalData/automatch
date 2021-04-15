import cv2 as cv
import numpy as np
import math
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import logging
import datetime
import pickle

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH = 6


def init_feature(name):
    if name == 'sift':
        detector = cv.SIFT_create()
        norm = cv.NORM_L2
    elif name == 'surf':
        detector = cv.xfeatures2d_SURF.create(800)
        norm = cv.NORM_L2
    elif name == 'orb':
        detector = cv.ORB_create(400)
        norm = cv.NORM_HAMMING
    elif name == 'akaze':
        detector = cv.AKAZE_create()
        norm = cv.NORM_HAMMING
    elif name == 'brisk':
        detector = cv.BRISK_create()
        norm = cv.NORM_HAMMING
    else:
        return None, None
    return detector, norm


def init_matcher(flann, norm, checks=100):
    search_params = {'checks': checks}  # or pass empty dictionary
    if flann:
        logging.debug("FLANN")
        if norm == cv.NORM_L2:
            flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        else:
            flann_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
        matcher = cv.FlannBasedMatcher(flann_params, search_params)  # bug : need to pass empty dict (#1329)
    else:
        logging.debug("BF Matcher")
        matcher = cv.BFMatcher(norm, crossCheck=False)
    return matcher


def getKeyPointsAndDescriptors(detector, img, tile, offset, n_features=0):
    """Detect keypoints and compute descriptors. Later, we might something else than SIFT."""
    if tile and offset:
        img_shape = img.shape
        nbTilesY = int(math.ceil(img_shape[0] / (offset[1] * 1.0)))
        nbTilesX = int(math.ceil(img_shape[1] / (offset[0] * 1.0)))
        array = []
        point_set = set()
        # only used to have a fast 'contains' test before adding keypoints in order to avoid duplicates
        for i in range(nbTilesY):
            logging.debug("%s : %d/%d", datetime.datetime.now(), i, nbTilesY)
            for j in range(nbTilesX):
                x_min = offset[0] * j
                y_min = offset[1] * i
                cropped_img = img[
                              y_min:min(offset[1] * i + tile[1], img_shape[0]),
                              x_min:min(offset[0] * j + tile[0], img_shape[1])]
                kp, des = detector.detectAndCompute(cropped_img, None)
                if len(kp) > 0:
                    for z in zip(kp, des):
                        pt = z[0].pt
                        z[0].pt = (pt[0] + x_min, pt[1] + y_min)
                        if z[0].pt not in point_set:
                            point_set.add(z[0].pt)
                            array.append(z)
                        # array.append(z)
        logging.debug("%s : Found %d points", datetime.datetime.now(), len(array))
        array.sort(key=lambda t: t[0].response, reverse=True)
        sorted_array = array
        if n_features > 0:
            sorted_array = array[:n_features]
        return [e[0] for e in sorted_array], [e[1] for e in sorted_array]
    else:
        logging.debug("%s : detect keypoints on image", datetime.datetime.now())
        array = detector.detectAndCompute(img, None)
        logging.debug("%s : Found %d points", datetime.datetime.now(), len(array))
        array.sort(key=lambda t: t[0].response, reverse=True)
        sorted_array = array
        if n_features > 0:
            sorted_array = array[:n_features]
        return [e[0] for e in sorted_array], [e[1] for e in sorted_array]


def getMatches(matcher, des1, des2, ratio):
    # FLANN parameters
    #    FLANN_INDEX_KDTREE = 0
    #    index_params = {'algorithm': FLANN_INDEX_KDTREE,
    #                    'trees': 5}
    #    search_params = {'checks': 100}   # or pass empty dictionary
    #
    #    logging.info("FlannBasedMatcher start")
    #    flann = cv.FlannBasedMatcher(index_params, search_params)
    logging.info("%s : FlannBasedMatcher Knn Match", datetime.datetime.now())
    # matches = matcher.knnMatch(np.asarray(des1,np.float32), np.asarray(des2,np.float32), k=2)
    # matches = matcher.knnMatch(queryDescriptors = cv.UMat(des1), trainDescriptors = cv.UMat(des2), k = 2)
    matches = matcher.knnMatch(queryDescriptors=np.asarray(des1), trainDescriptors=np.asarray(des2), k=2)
    # Apply ratio test
    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
    return good_matches


def getBinImage(image):
    # Otsu's thresholding with a gaussian blur
    blur = cv.GaussianBlur(image, (5, 5), 0)
    th, img = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return img


def saveGCPsAsShapefile(gcps, srs, output_file):
    out_ds = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(output_file)
    out_lyr = out_ds.CreateLayer('gcps', geom_type=ogr.wkbPoint, srs=srs)
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


def saveGCPsAsText(gcps, output_file):
    file = open(output_file, "w+")
    file.write("mapX,mapY,pixelX,pixelY,enable,dX,dY,residual\n")
    for i in range(len(gcps)):
        file.write("%f,%f,%f,%f,1,0,0,0\n" % (gcps[i].GCPX, gcps[i].GCPY, gcps[i].GCPPixel, -gcps[i].GCPLine))
    file.close()


def getImageKeyPointsAndDescriptors(image_file, detector, tile, offset, convert_to_binary=False, n_features=100000):
    img = cv.imread(image_file,
                    cv.IMREAD_GRAYSCALE)  # queryImage (IMREAD_COLOR flag=cv.IMREAD_GRAYSCALE to force grayscale)
    if convert_to_binary:
        img = getBinImage(img)
    kp, des = getKeyPointsAndDescriptors(detector, img, tile, offset, n_features)
    logging.info(f'{len(des)} keypoints in the image')
    return img, kp, des


def match(norm, flann, des_train, des_query, ratio):
    # Match both ways
    matcher = init_matcher(flann, norm)
    if ratio is None:
        ratio = 0.75
    matches_train = getMatches(matcher, des_train, des_query, ratio)
    matches_query = getMatches(matcher, des_query, des_train, ratio)
    # Only keep matches that are present in both ways
    two_sides_matches = [m for m in matches_train if
                         any(mm.queryIdx == m.trainIdx and mm.trainIdx == m.queryIdx for mm in matches_query)]
    logging.debug("%s : Matches %d", datetime.datetime.now(), len(two_sides_matches))
    return two_sides_matches


def georef(input_file, reference_file, output_file,
           feature_name, flann, transform_type,
           gcp_output_shp, gcp_output_txt, tile, offset,
           ratio=0.75, convert=False):
    """Georeference the input using the training image and save the result in outputFile. 
    A ratio can be given to select more or less matches (defaults to 0.75)."""
    detector, norm = init_feature(feature_name)
    train, kp_train, des_train = getImageKeyPointsAndDescriptors(reference_file, detector, tile, offset, convert)
    query, kp_query, des_query = getImageKeyPointsAndDescriptors(input_file, detector, tile, offset, convert)
    two_sides_matches = match(norm, flann, des_train, des_query, ratio)

    MIN_MATCH_COUNT = 3

    affine = True
    if len(two_sides_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_train[m.queryIdx].pt for m in two_sides_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_query[m.trainIdx].pt for m in two_sides_matches]).reshape(-1, 1, 2)

        if affine:
            M, mask = cv.estimateAffine2D(src_pts, dst_pts)
        else:
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
        matchesMask = mask.ravel().tolist()

        ds = gdal.Open(reference_file)
        x_offset, px_w, rot1, y_offset, px_h, rot2 = ds.GetGeoTransform()
        logging.debug("%s : Transform: %f, %f, %f, %f, %f, %f", datetime.datetime.now(), x_offset, px_w, rot1, y_offset,
                      px_h, rot2)

        gcp_list = []
        geo_t = ds.GetGeoTransform()
        # gcp_string = ''
        logging.debug("%s : %d matches", datetime.datetime.now(), len(matchesMask))
        for i, good_match in enumerate(matchesMask):
            if good_match == 1:
                p1 = kp_train[two_sides_matches[i].queryIdx].pt
                p2 = kp_query[two_sides_matches[i].trainIdx].pt
                pp = gdal.ApplyGeoTransform(geo_t, p1[0], p1[1])
                logging.debug(f"GCP geot = (%f,%f) -> (%f,%f)", p1[0], p1[1], pp[0], pp[1])
                logging.debug(f"Matched with (%f,%f)", p2[0], p2[1])
                z = 0
                # info = "GCP from pixel %f, %f" % (p1[0], p1[1])
                gcp = gdal.GCP(pp[0], pp[1], z, p2[0], p2[1])  # , info, i)
                # print ("GCP     = (" + str(p2[0]) +","+ str(p2[1]) + ") -> (" + str(pp[0]) +","+ str(pp[1]) + ")")
                # gcp_string += ' -gcp '+" ".join([str(p2[0]),str(p2[1]),str(pp[0]), str(pp[1])])
                gcp_list.append(gcp)
        logging.debug("%s : %d GCPs", datetime.datetime.now(), len(gcp_list))

        translate_t = gdal.GCPsToGeoTransform(gcp_list)
        translate_inv_t = gdal.InvGeoTransform(translate_t)
        logging.debug(len(translate_t))
        logging.debug("geotransform = %s", translate_t)
        logging.debug(len(translate_inv_t))
        logging.debug("invgeotransform = %s", translate_inv_t)
        # trans_gcp_list = []
        dst_gcp_list = []
        mapResiduals = 0.0
        geoResiduals = 0.0
        for gcp in gcp_list:
            # Inverse geotransform to get the corresponding pixel
            pix = gdal.ApplyGeoTransform(translate_inv_t, gcp.GCPX, gcp.GCPY)
            logging.debug("GCP = (%d,%d) -> (%d,%d)", gcp.GCPPixel, gcp.GCPLine, gcp.GCPX, gcp.GCPY)
            logging.debug(" => (%d,%d)", pix[0], pix[1])
            map_dX = gcp.GCPPixel - pix[0]
            map_dY = gcp.GCPLine - pix[1]
            map_residual = map_dX * map_dX + map_dY * map_dY
            mapResiduals = mapResiduals + map_residual
            # trans_gcp_list.append(pix)
            # Apply the transform to get the GCP location in the output SRS
            pp = gdal.ApplyGeoTransform(translate_t, gcp.GCPPixel, gcp.GCPLine)
            z = 0
            out_gcp = gdal.GCP(pp[0], pp[1], z, gcp.GCPPixel, gcp.GCPLine)
            logging.debug("GCP = (%d,%d) -> (%d,%d)", out_gcp.GCPPixel, out_gcp.GCPLine, pp[0], pp[1])
            # gcp_string += ' -gcp '+" ".join([str(p2[0]),str(p2[1]),str(pp[0]), str(pp[1])])
            dX = gcp.GCPX - pp[0]
            dY = gcp.GCPY - pp[1]
            residual = dX * dX + dY * dY
            geoResiduals = geoResiduals + residual
            logging.debug("map residual = %f, %f = %f", map_dX, map_dY, map_residual)
            logging.debug("residual = %f, %f = %f", dX, dY, residual)
            dst_gcp_list.append(out_gcp)

        logging.debug(f"map residuals %s", mapResiduals)
        logging.debug(f"geo residuals %s", geoResiduals)
        # print (gcp_string)
        # h,w = train.shape
        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # dst = cv.perspectiveTransform(pts,M)

        # img2 = cv.polylines(query,[np.int32(dst)],True,255,3, cv.LINE_AA)

        src_ds = gdal.Open(input_file)
        # translate and warp the inputFile using GCPs and polynomial of order 1
        dst_ds = gdal.Translate('', src_ds, outputSRS=ds.GetProjection(), GCPs=gcp_list, format='MEM')
        tps = False
        if transform_type == 'poly1':
            polynomialOrder = 1
            gdal.Warp(output_file, dst_ds, tps=tps, polynomialOrder=polynomialOrder, dstNodata=1)
        elif transform_type == 'poly2':
            polynomialOrder = 2
            gdal.Warp(output_file, dst_ds, tps=tps, polynomialOrder=polynomialOrder, dstNodata=1)
        elif transform_type == 'poly3':
            polynomialOrder = 3
            gdal.Warp(output_file, dst_ds, tps=tps, polynomialOrder=polynomialOrder, dstNodata=1)
        elif transform_type == 'tps':
            tps = True
            gdal.Warp(output_file, dst_ds, tps=tps, dstNodata=1)
        else:
            polynomialOrder = 1
            gdal.Warp(output_file, dst_ds, tps=tps, polynomialOrder=polynomialOrder, dstNodata=1)

        if gcp_output_shp is not None:
            saveGCPsAsShapefile(dst_gcp_list, osr.SpatialReference(ds.GetProjection()), gcp_output_shp)
            saveGCPsAsText(dst_gcp_list, gcp_output_txt)

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv.drawMatches(train, kp_train, query, kp_query, two_sides_matches, None, **draw_params)
        return img3

    else:
        logging.error("Not enough matches are found - %d/%d", len(two_sides_matches), MIN_MATCH_COUNT)


def loadKeyPoints(points_file, n_features=0):
    with open(points_file, 'rb') as kpf:
        data = pickle.load(kpf)
        input_file = data['inputfile']
        feature_name = data['feature_name']
        logging.debug("%s : Loading keypoints extracted from %s using %s", datetime.datetime.now(), input_file,
                      feature_name)
        key_points = data['keypoints']
        logging.debug("%s : Found %d keypoints", datetime.datetime.now(), len(key_points))
        descriptors = data['descriptors']
        logging.debug("%s : Found %d descriptors", datetime.datetime.now(), len(descriptors))

        def make_cv_keypoint(kp):
            cvkp = cv.KeyPoint(x=kp[0], y=kp[1], _size=kp[2], _angle=kp[3], _response=kp[4], _octave=kp[5],
                               _class_id=kp[6])
            return cvkp

        cv_kp = [make_cv_keypoint(kp) for kp in key_points]
        # Sort the pairs of (keypoint, descriptor) based on keypoints responses, ascending
        sorted_array = sorted(zip(cv_kp, descriptors), key=lambda t: t[0].response, reverse=True)
        # Keep only nFeatures pairs is nFeatures is not 0
        sorted_array = sorted_array[:n_features] if n_features else sorted_array
        # Unzip so we get a list of sorted keypoints and a list of sorted descriptors
        unzipped = list(zip(*sorted_array))
        return input_file, feature_name, unzipped[0], unzipped[1]


def loadOrCompute(ki, input_file, feature_name, tile, offset):
    if ki is None:
        if feature_name is None:
            feature_name = 'brisk'
        detector, norm = init_feature(feature_name)
        img, kp, des = getImageKeyPointsAndDescriptors(input_file, detector, tile, offset)
        return img, feature_name, kp, des
    else:
        return loadKeyPoints(ki)


def getGCP(reference_file, kp_query, kp_train, two_sides_matches, min_matches=3):
    if len(two_sides_matches) > min_matches:
        src_pts = np.float32([kp_train[m.queryIdx].pt for m in two_sides_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_query[m.trainIdx].pt for m in two_sides_matches]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
        matchesMask = mask.ravel().tolist()

        ds = gdal.Open(reference_file)
        x_offset, px_w, rot1, y_offset, px_h, rot2 = ds.GetGeoTransform()
        logging.debug("%s : Transform: %f, %f, %f, %f, %f, %f", datetime.datetime.now(), x_offset, px_w, rot1, y_offset,
                      px_h, rot2)

        gcp_list = []
        geo_t = ds.GetGeoTransform()
        # gcp_string = ''
        logging.debug("%s : %d matches", datetime.datetime.now(), len(matchesMask))
        for i, good_match in enumerate(matchesMask):
            if good_match == 1:
                p1 = kp_train[two_sides_matches[i].queryIdx].pt
                p2 = kp_query[two_sides_matches[i].trainIdx].pt
                pp = gdal.ApplyGeoTransform(geo_t, p1[0], p1[1])
                logging.debug(f"GCP geot = (%f,%f) -> (%f,%f)", p1[0], p1[1], pp[0], pp[1])
                logging.debug(f"Matched with (%f,%f)", p2[0], p2[1])
                z = 0
                # info = "GCP from pixel %f, %f" % (p1[0], p1[1])
                gcp = gdal.GCP(pp[0], pp[1], z, p2[0], p2[1])  # , info, i)
                # print ("GCP     = (" + str(p2[0]) +","+ str(p2[1]) + ") -> (" + str(pp[0]) +","+ str(pp[1]) + ")")
                # gcp_string += ' -gcp '+" ".join([str(p2[0]),str(p2[1]),str(pp[0]), str(pp[1])])
                gcp_list.append(gcp)
        logging.debug("%s : %d GCPs", datetime.datetime.now(), len(gcp_list))

        translate_t = gdal.GCPsToGeoTransform(gcp_list)
        translate_inv_t = gdal.InvGeoTransform(translate_t)
        logging.debug(len(translate_t))
        logging.debug("geotransform = %s", translate_t)
        logging.debug(len(translate_inv_t))
        logging.debug("invgeotransform = %s", translate_inv_t)
        # trans_gcp_list = []
        dst_gcp_list = []
        mapResiduals = 0.0
        geo_residuals = 0.0
        for gcp in gcp_list:
            # Inverse geotransform to get the corresponding pixel
            pix = gdal.ApplyGeoTransform(translate_inv_t, gcp.GCPX, gcp.GCPY)
            logging.debug("GCP = (%d,%d) -> (%d,%d)", gcp.GCPPixel, gcp.GCPLine, gcp.GCPX, gcp.GCPY)
            logging.debug(" => (%d,%d)", pix[0], pix[1])
            map_dX = gcp.GCPPixel - pix[0]
            map_dY = gcp.GCPLine - pix[1]
            map_residual = map_dX * map_dX + map_dY * map_dY
            mapResiduals = mapResiduals + map_residual
            # Apply the transform to get the GCP location in the output SRS
            pp = gdal.ApplyGeoTransform(translate_t, gcp.GCPPixel, gcp.GCPLine)
            z = 0
            out_gcp = gdal.GCP(pp[0], pp[1], z, gcp.GCPPixel, gcp.GCPLine)
            logging.debug("GCP = (%d,%d) -> (%d,%d)", out_gcp.GCPPixel, out_gcp.GCPLine, pp[0], pp[1])
            dX = gcp.GCPX - pp[0]
            dY = gcp.GCPY - pp[1]
            residual = dX * dX + dY * dY
            geo_residuals = geo_residuals + residual
            logging.debug("map residual = %f, %f = %f", map_dX, map_dY, map_residual)
            logging.debug("residual = %f, %f = %f", dX, dY, residual)
            dst_gcp_list.append(out_gcp)

        logging.debug(f"map residuals %s", mapResiduals)
        logging.debug(f"geo residuals %s", geo_residuals)
        return ds.GetProjection(), gcp_list, dst_gcp_list
    else:
        logging.error("Not enough matches are found - %d/%d", len(two_sides_matches), min_matches)
        return None, None, None


def saveGeoref(input_file, output_file, projection, transform_type, gcp_list, dst_gcp_list, points_shp, points_txt):
    src_ds = gdal.Open(input_file)
    # translate and warp the inputFile using GCPs and polynomial of order 1
    dst_ds = gdal.Translate('', src_ds, outputSRS=projection, GCPs=gcp_list, format='MEM')
    tps = False
    if transform_type == 'poly1':
        polynomialOrder = 1
        gdal.Warp(output_file, dst_ds, tps=tps, polynomialOrder=polynomialOrder, dstNodata=1)
    elif transform_type == 'poly2':
        polynomialOrder = 2
        gdal.Warp(output_file, dst_ds, tps=tps, polynomialOrder=polynomialOrder, dstNodata=1)
    elif transform_type == 'poly3':
        polynomialOrder = 3
        gdal.Warp(output_file, dst_ds, tps=tps, polynomialOrder=polynomialOrder, dstNodata=1)
    elif transform_type == 'tps':
        tps = True
        gdal.Warp(output_file, dst_ds, tps=tps, dstNodata=1)
    else:
        polynomialOrder = 1
        gdal.Warp(output_file, dst_ds, tps=tps, polynomialOrder=polynomialOrder, dstNodata=1)
    # save the points to file
    if points_shp is not None:
        saveGCPsAsShapefile(dst_gcp_list, osr.SpatialReference(projection), points_shp)
    if points_txt is not None:
        saveGCPsAsText(dst_gcp_list, points_txt)
