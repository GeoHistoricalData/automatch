#!/usr/bin/python3
import argparse
import logging
from matplotlib import pyplot as plt
from common import georef


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
#parser.add_argument('-ki', type=str, help='keypoint inputfile')
#parser.add_argument('-kr', type=str, help='keypoint referencefile')

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
   img = georef(inputfile, referencefile, outputfile, feature_name, flann, pointsShp, pointsTxt, tile, offset, ratio, binary)
   if show and img is not None:
       plt.imshow(img, 'gray'),plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    if args.ratio is not None:
       automatch(args.i, args.r, args.o, args.feature, args.flann, args.gcps, args.show, args.tileX, args.tileY, args.offsetX, args.offsetY, args.ratio)
    else:
       automatch(args.i, args.r, args.o, args.feature, args.flann, args.gcps, args.show, args.tileX, args.tileY, args.offsetX, args.offsetY)
