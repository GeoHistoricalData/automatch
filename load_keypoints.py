import argparse
import logging
from matplotlib import pyplot as plt
import cv2 as cv
from common import loadKeyPoints

# Configure the global logger
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

parser = argparse.ArgumentParser(prog='load_keypoints.py',description='Parse the load keypoints arguments.')
parser.add_argument('keypoints', type=str, help='keypoints')
parser.add_argument('-n', type=int, help='number of keypoints to load')

if __name__ == "__main__":
    args = parser.parse_args()
    nFeatures = 0
    if args.n:
        nFeatures = args.n
    inputFile, _, kp, _ = loadKeyPoints(args.keypoints, nFeatures)
    img = cv.imread(inputFile)
    imm = cv.drawKeypoints(img, kp, img)
    plt.imshow(cv.cvtColor(imm, cv.COLOR_BGR2RGB)),plt.show()
