#!/usr/bin/python3

import sys
from matplotlib import pyplot as plt
from common import georef
import argparse

parser = argparse.ArgumentParser(prog='automatch.py',description='Parse the automatch arguments.')
parser.add_argument('-i', type=str, help='inputfile')
parser.add_argument('-r', type=str, help='referencefile')
parser.add_argument('-o', type=str, help='outputfile')
parser.add_argument('--gcps', action='store_true', help='export gcps')
parser.add_argument('--ratio', type=float, help='ratio test explained by D.Lowe')
parser.add_argument('--show', action='store_true', help='show the matches')
parser.add_argument('-tileX', type=int, help='')
parser.add_argument('-tileY', type=int, help='')
parser.add_argument('-offsetX', type=int, help='')
parser.add_argument('-offsetY', type=int, help='')

def automatch(inputfile, referencefile, outputfile, gcps, show, tileX=None, tileY=None, offsetX=None, offsetY=None, ratio = 0.75, binary = False):
   print ('Input file is', inputfile)
   print ('Reference file is', referencefile)
   print ('Output file is', outputfile)
   print ('GCPs is', gcps)
   print ('Ratio is', ratio)
   print ('Show', show)
   print ('tileX', tileX)
   print ('tileY', tileY)
   print ('offsetX', offsetX)
   print ('offsetY', offsetY)
   tile = None
   offset = None
   if tileX is not None and tileY is not None and offsetX is not None and offsetY is not None:
      tile = (tileX, tileY)
      offset = (offsetX, offsetY)

   points = None
   if gcps:
      points = outputfile+'.shp'
   img = georef(inputfile, referencefile, outputfile, points, tile, offset, ratio, binary)
   if show and img is not None:
       plt.imshow(img, 'gray'),plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    if args.ratio is not None:
       automatch(args.i, args.r, args.o, args.gcps, args.show, args.tileX, args.tileY, args.offsetX, args.offsetY, args.ratio)
    else:
       automatch(args.i, args.r, args.o, args.gcps, args.show, args.tileX, args.tileY, args.offsetX, args.offsetY)
