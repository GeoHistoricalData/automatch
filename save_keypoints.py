import argparse
import logging
from common import getImageKeyPointsAndDescriptors, init_feature
import pickle

class My_KeyPoint:
    def __init__(self, point):
        self.pt = point.pt
        self.size = point.size
        self.angle = point.angle
        self.response = point.response
        self.octave = point.octave
        self.class_id = point.class_id

# Configure the global logger
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

parser = argparse.ArgumentParser(prog='save_keypoints.py',description='Parse the save keypoints arguments.')
parser.add_argument('-i', type=str, help='inputfile')
parser.add_argument('-o', type=str, help='outputfile')
parser.add_argument('--feature', type=str, help='<sift|surf|orb|akaze|brisk>[-flann]')
parser.add_argument('-n', type=int, help='number of features')
parser.add_argument('-tileX', type=int, help='')
parser.add_argument('-tileY', type=int, help='')
parser.add_argument('-offsetX', type=int, help='')
parser.add_argument('-offsetY', type=int, help='')

def saveKeyPoints(inputFile, outputFile, feature_name, nFeatures, tile, offset, convertToBinary = False):
    logging.debug('Input file is %s', inputFile)
    logging.debug('Output file is %s', outputFile)
    logging.debug('Feature is %s', feature_name)
    logging.debug('Number of Features is %d', nFeatures)
    logging.debug('tile %d, %d', tile[0], tile[1])
    logging.debug('offset %d, %d', offset[0], offset[1])
    detector, matcher = init_feature(feature_name)
    img, kp, des = getImageKeyPointsAndDescriptors(inputFile, detector, tile, offset, convertToBinary, nFeatures)
    out_s = open(outputFile, 'wb')
    index = []
    for point in kp:
        temp = (point.pt[0], point.pt[1], point.size, point.angle, point.response, point.octave, point.class_id)
        index.append(temp)
    pickle.dump(index, out_s)
    pickle.dump(des, out_s)
    out_s.close()

if __name__ == "__main__":
    args = parser.parse_args()
    tile = None
    offset = None
    if args.tileX is not None and args.tileY is not None and args.offsetX is not None and args.offsetY is not None:
        tile = (args.tileX, args.tileY)
        offset = (args.offsetX, args.offsetY)
    feature_name = args.feature
    if feature_name is None:
        feature_name = "brisk"
    nFeatures = args.n
    if nFeatures is None:
        nFeatures = 0
    saveKeyPoints(args.i, args.o, feature_name, nFeatures, tile, offset)
