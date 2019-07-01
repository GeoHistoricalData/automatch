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

def automatch(inputfile, referencefile, outputfile, gcps, show, ratio = 0.75):
   print ('Input file is ', inputfile)
   print ('Reference file is ', referencefile)
   print ('Output file is ', outputfile)
   print ('GCPs is ', gcps)
   print ('Ratio is ', ratio)
   print ('Show ', show)
   points = None
   if gcps:
      points = outputfile+'.shp'
   img = georef(inputfile, referencefile, outputfile, points, ratio)
   if show and img is not None:
       plt.imshow(img, 'gray'),plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    if args.ratio is not None:
       automatch(args.i, args.r, args.o, args.gcps, args.show, args.ratio)
    else:
       automatch(args.i, args.r, args.o, args.gcps, args.show)
