from matplotlib import pyplot as plt
import cv2 as cv
import click
from common import loadKeyPoints


@click.command()
@click.argument('keypoints', type=click.File('rb', lazy=True))
@click.option('--n-keypoints', '-n', default=0, help='Number of keypoints to load')
def load(keypoints, n_keypoints):
    """Parse the load keypoints arguments"""
    inputfile, _, kp, _ = loadKeyPoints(keypoints.name, n_keypoints)
    img = cv.imread(inputfile)
    im_kp = cv.drawKeypoints(img, kp, img)
    plt.imshow(cv.cvtColor(im_kp, cv.COLOR_BGR2RGB)), plt.show()


if __name__ == "__main__":
    load()