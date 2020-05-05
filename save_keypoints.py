import logging
import pickle
import click
from common import getImageKeyPointsAndDescriptors, init_feature

# Configure the global logger
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


@click.command()
@click.argument('inputfile', type=click.File('rb', lazy=True))
@click.argument('outputfile', type=click.File('wb', lazy=True))
@click.option("--feature", "-f", 'feature_name', default='brisk', help="<sift|surf|orb|akaze|brisk>[-flann]")
@click.option("--n_features", "-n", 'n_features', default=0, help="Number of features")
@click.option("--tile", type=(int, int), default=(512,512), help="Tile dimensions (width, height)")
@click.option("--offset", type=(int, int), default=(1,1), help="Offset as a pair (offsetX, offsetY)")
def save_keypoints(inputfile, outputfile, feature_name, n_features, tile, offset, to_binary=False):
    """Compute the keypoints and desctiptors of an image and save them in a store file.
    All the arguments used for the point detection will be saved."""

    logging.debug('Input file is %s', inputfile)
    logging.debug('Output file is %s', outputfile)
    logging.debug('Feature is %s', feature_name)
    logging.debug('Number of Features is %i', n_features)
    logging.debug('tile is %s', tile)
    logging.debug('offset is %s', offset)

    detector, norm = init_feature(feature_name)
    _, kp, des = getImageKeyPointsAndDescriptors(inputfile.name, detector, tile, offset, to_binary, n_features)

    def get_point_data(p):
        (p.pt[0], p.pt[1], p.size, p.angle, p.response, p.octave, p.class_id)

    index = [get_point_data(point) for point in kp]
    dump_me = [inputfile.name, feature_name, index, des]
    pickle.dump(dump_me, outputfile)


if __name__ == "__main__":
    save_keypoints()
