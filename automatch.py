#!/usr/bin/python3

import sys
from matplotlib import pyplot as plt
from common import georef
import argparse

parser = argparse.ArgumentParser(prog='automatch.py',description='Parse the automatch arguments.')
parser.add_argument('-i', type=str, help='inputfile')
parser.add_argument('-r', type=str, help='referencefile')
parser.add_argument('-o', type=str, help='outputfile')
parser.add_argument('--ratio', type=float, help='ratio test explained by D.Lowe')
parser.add_argument('--show', action='store_true', help='show the matches')

def automatch(inputfile, referencefile, outputfile, show, ratio = 0.75):
   print ('Input file is ', inputfile)
   print ('Reference file is ', referencefile)
   print ('Output file is ', outputfile)
   print ('Ratio is ', ratio)
   print ('Show ', show)
   img = georef(inputfile, referencefile, outputfile, ratio)
   if show and img is not None:
       plt.imshow(img, 'gray'),plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    automatch(args.i, args.r, args.o, args.show, args.ratio)
