import argparse
import logging
from matplotlib import pyplot as plt
import cv2 as cv
import pickle

# Configure the global logger
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

parser = argparse.ArgumentParser(prog='load_keypoints.py',description='Parse the load keypoints arguments.')
parser.add_argument('-i', type=str, help='inputfile')
parser.add_argument('-k', type=str, help='keypoints')
parser.add_argument('-n', type=int, help='number of keypoints to load')

def loadKeyPoints(keypoints_file, nFeatures):
    f = open(keypoints_file,'rb')
    keypoints = pickle.load(f)
    logging.debug("Found %d keypoints", len(keypoints))
    descriptors = pickle.load(f)
    logging.debug("Found %d descriptors", len(descriptors))
    kp = []
    for point in keypoints:
        temp = cv.KeyPoint(x=point[0],y=point[1],_size=point[2], _angle=point[3], _response=point[4], _octave=point[5], _class_id=point[6]) 
        kp.append(temp)
    sortedarray = sorted(zip(kp, descriptors), key=lambda t: t[0].response, reverse=True)
    #array.sort(key=lambda t: t[0].response, reverse=True)
    if nFeatures>0:
        sortedarray = sortedarray[:nFeatures]
    return [ e[0] for e in sortedarray ], [ e[1] for e in sortedarray ]

if __name__ == "__main__":
    args = parser.parse_args()
    nFeatures = 0
    if args.n:
        nFeatures = args.n
    kp,_ = loadKeyPoints(args.k, nFeatures)
    img = cv.imread(args.i)
    imm = cv.drawKeypoints(img, kp, img)
    plt.imshow(cv.cvtColor(imm, cv.COLOR_BGR2RGB)),plt.show()
