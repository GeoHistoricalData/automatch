import argparse
import logging
from matplotlib import pyplot as plt
import cv2 as cv
import pickle
import datetime

# Configure the global logger
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

parser = argparse.ArgumentParser(prog='load_keypoints.py',description='Parse the load keypoints arguments.')
parser.add_argument('keypoints', type=str, help='keypoints')
parser.add_argument('-n', type=int, help='number of keypoints to load')

def loadKeyPoints(keypoints_file, nFeatures = 0):
    f = open(keypoints_file,'rb')
    inputFile = pickle.load(f)
    feature_name = pickle.load(f)
    logging.debug("%s : Loading keypoints extracted from %s using %s", datetime.datetime.now(), inputFile, feature_name)
    keypoints = pickle.load(f)
    logging.debug("%s : Found %d keypoints", datetime.datetime.now(), len(keypoints))
    descriptors = pickle.load(f)
    logging.debug("%s : Found %d descriptors", datetime.datetime.now(), len(descriptors))
    kp = []
    for point in keypoints:
        temp = cv.KeyPoint(x=point[0],y=point[1],_size=point[2], _angle=point[3], _response=point[4], _octave=point[5], _class_id=point[6]) 
        kp.append(temp)
    sortedarray = sorted(zip(kp, descriptors), key=lambda t: t[0].response, reverse=True)
    #array.sort(key=lambda t: t[0].response, reverse=True)
    if nFeatures>0:
        sortedarray = sortedarray[:nFeatures]
    return inputFile, feature_name, [ e[0] for e in sortedarray ], [ e[1] for e in sortedarray ]

if __name__ == "__main__":
    args = parser.parse_args()
    nFeatures = 0
    if args.n:
        nFeatures = args.n
    inputFile, _, kp, _ = loadKeyPoints(args.keypoints, nFeatures)
    img = cv.imread(inputFile)
    imm = cv.drawKeypoints(img, kp, img)
    plt.imshow(cv.cvtColor(imm, cv.COLOR_BGR2RGB)),plt.show()
