from skimage import transform as tf
from skimage.feature import (match_descriptors, ORB, plot_matches)
import matplotlib.pyplot as plt
import cv2
from skimage.filters.rank import enhance_contrast
from skimage import morphology
from skimage.measure import ransac as skransac
from skimage.transform import EuclideanTransform
import numpy as np
from ex4_part1 import *
from scipy import misc


# img1 = misc.imread('BL05.bmp', flatten=0)
# img1 = cv2.medianBlur(img1[:,:,0, 15)

# img2 = misc.imread('FU05.bmp', flatten=0)
# img2 = cv2.medianBlur(img2[:,:,0, 15)


img1 = cv2.imread('BL01-no.tif', 0)
img1 = cv2.medianBlur(img1, 15)
img1 = enhance_contrast(img1, morphology.square(10))
img1 = img1 / img1.max(axis=0)

img2 = cv2.imread('FU01-no.tif', 0)
img2 = cv2.medianBlur(img2, 15)
img2 = enhance_contrast(img2, morphology.square(10))
img2 = img2 / img2.max(axis=0)


descriptor_extractor = ORB(n_keypoints=200, harris_k=0.00001)

descriptor_extractor.detect_and_extract(img1)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(img2)
keypoints2 = descriptor_extractor.keypoints
descriptors2 = descriptor_extractor.descriptors

matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

display_matches(img1, img2, keypoints1[matches12[:, 0]], keypoints2[matches12[:, 1]])

fig, ax = plt.subplots(nrows=2, ncols=1)

plt.gray()

plot_matches(ax[0], img1, img2, keypoints1, keypoints2, matches12)
# plt.scatter(keypoints1[matches12[:, 0]][:, 1], keypoints1[matches12[:, 0]][:, 0])
# plt.scatter(keypoints2[matches12[:, 1]][:, 1] + img1.shape[1], keypoints2[matches12[:, 1]][:, 0])
ax[0].axis('off')
ax[0].set_title("Original Image vs. Transformed Image")

f, inliers = ransac(keypoints1[matches12[:, 0]], keypoints2[matches12[:, 1]], calcPointBasedReg, calcDist,
                    minPtNum=3, iterNum=20000, thDist=5, thInlrRatio=0.2)

# g, skinliers = skransac((keypoints1[matches12[:, 0]], keypoints2[matches12[:, 1]]), EuclideanTransform, min_samples=2,
#                     residual_threshold=10, max_trials=20000)
# inliers_idx = np.nonzero(inliers)[0]
inliers_idx = matches12[inliers]
plot_matches(ax[1], img1, img2, keypoints1, keypoints2, inliers_idx)
# plt.scatter(keypoints1[inliers[:, 0]][:, 1], keypoints1[inliers[:, 0]][:, 0])
# plt.scatter(keypoints2[inliers[:, 1]][:, 1] + img1.shape[1], keypoints2[inliers[:, 1]][:, 0])
ax[1].axis('off')
ax[1].set_title("ransac")

plt.show()

display_matches(img1, img2, keypoints1[matches12[:, 0]], keypoints2[matches12[:, 1]], inliers)

