#!/usr/bin/python3
import argparse
import logging
from matplotlib import pyplot as plt
import common


# Configure the global logger
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

parser = argparse.ArgumentParser(prog='automatch.py',description='Parse the automatch arguments.')
parser.add_argument('-i', type=str, help='inputfile')
parser.add_argument('-r', type=str, help='referencefile')
parser.add_argument('-o', type=str, help='outputfile')
parser.add_argument('--feature', type=str, help='<sift|surf|orb|akaze|brisk>')
parser.add_argument('--flann', action='store_true', help='use flann matcher')
parser.add_argument('--gcps', action='store_true', help='export gcps')
parser.add_argument('--ratio', type=float, help='ratio test explained by D.Lowe')
parser.add_argument('--show', action='store_true', help='show the matches')
parser.add_argument('-tileX', type=int, help='')
parser.add_argument('-tileY', type=int, help='')
parser.add_argument('-offsetX', type=int, help='')
parser.add_argument('-offsetY', type=int, help='')
parser.add_argument('-ki', type=str, help='keypoint inputfile')
parser.add_argument('-kr', type=str, help='keypoint referencefile')

def automatch(inputfile, referencefile, outputfile, feature_name, flann, gcps, show, tileX=None, tileY=None, offsetX=None, offsetY=None, ratio = 0.75, binary = False):
   logging.debug('Input file is %s', inputfile)
   logging.debug('Reference file is %s', referencefile)
   logging.debug('Output file is %s', outputfile)
   logging.debug('Feature is %s', feature_name)
   logging.debug('Matcher is %d', flann)
   logging.debug('GCPs is %d', gcps)
   logging.debug('Ratio is %f', ratio)
   logging.debug('Show %d', show)
   logging.debug('tileX %d', tileX)
   logging.debug('tileY %d', tileY)
   logging.debug('offsetX %d', offsetX)
   logging.debug('offsetY %d', offsetY)
   tile = None
   offset = None
   if tileX is not None and tileY is not None and offsetX is not None and offsetY is not None:
      tile = (tileX, tileY)
      offset = (offsetX, offsetY)

   pointsShp = None
   pointsTxt = None
   if gcps:
      pointsShp = outputfile+'.shp'
      pointsTxt = outputfile+'.points'
   if not feature_name:
      feature_name = 'brisk'
   img = common.georef(inputfile, referencefile, outputfile, feature_name, flann, pointsShp, pointsTxt, tile, offset, ratio, binary)
   if show and img is not None:
       plt.imshow(img, 'gray'),plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    tile = None
    offset = None
    if args.tileX is not None and args.tileY is not None:
        tile = (args.tileX, args.tileY)
        if args.offsetX is not None and args.offsetY is not None:
            offset = (args.offsetX, args.offsetY)
        else:
            offset = tile # if no offset is set, use the tile sizes
    img_query, feature_name_query, kp_query, des_query = common.loadOrCompute(args.ki, args.i, args.feature, tile, offset)
    img_train, feature_name_train, kp_train, des_train = common.loadOrCompute(args.kr, args.r, args.feature, tile, offset)
    assert feature_name_train == feature_name_query
    _, norm = common.init_feature(feature_name_query)
    ratio = args.ratio
    if ratio is None:
        ratio = 0.75
    # Match both ways
    two_sides_matches = common.match(norm, args.flann, des_train, des_query, ratio)
    projection, gcps, dst_gcps = common.getGCP(img_train, kp_query, kp_train, two_sides_matches, min_matches = 3)
    pointsShp = None
    pointsTxt = None
    if gcps:
        pointsShp = args.o+'.shp'
        pointsTxt = args.o+'.points'
    common.saveGeoref(img_query, args.o, projection, gcps, dst_gcps, pointsShp, pointsTxt)

    #automatch(args.i, args.r, args.o, args.feature, args.flann, args.gcps, args.show, args.tileX, args.tileY, args.offsetX, args.offsetY, ratio)
