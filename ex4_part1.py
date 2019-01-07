from PIL import Image
import matplotlib.pyplot as plt
from utils import *
from skimage import transform as tf
import copy


########################################################################################################################
# USER GUIDE:
# The given images should be in the same directory as the .py file, or full path should be provided
# main function can be uncommented at the end of the file for running the different functions
########################################################################################################################


def display_matches(BL_im, FU_im, BL_points, FU_points, inliers=[]):
    """
    A function that displays the two given images and plots the matching points in each of the corresponding images.
    """
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(BL_im)
    plt.title('BL01')
    for point in range(len(BL_points)):
        if len(inliers) > 0:
            if point in inliers:
                plt.scatter(BL_points[point, 0], BL_points[point, 1], marker='o', c='r')
            else:
                plt.scatter(BL_points[point, 0], BL_points[point, 1], marker='o', c='b')
            plt.annotate(str(point + 1), (BL_points[point, 0], BL_points[point, 1]))
            continue
        plt.scatter(BL_points[point, 0], BL_points[point, 1], marker='o')
        plt.annotate(str(point + 1), (BL_points[point, 0], BL_points[point, 1]))

    fig.add_subplot(1, 2, 2)
    plt.imshow(FU_im)
    plt.title('FU01')
    for point in range(len(FU_points)):
        if len(inliers) > 0:
            if point in inliers:
                plt.scatter(FU_points[point, 0], FU_points[point, 1], marker='o', c='r')
            else:
                plt.scatter(FU_points[point, 0], FU_points[point, 1], marker='o', c='b')
            plt.annotate(str(point + 1), (FU_points[point, 0], FU_points[point, 1]))
            continue
        plt.scatter(FU_points[point, 0], FU_points[point, 1], marker='o')
        plt.annotate(str(point + 1), (FU_points[point, 0], FU_points[point, 1]))
    plt.show()


def calcPointBasedReg(BLPoints, FUPoints):
    """
    A funciton that calculates the rigid transformation between the given two sets of points using SVD calculation
    """
    n = len(BLPoints)
    # compute the centroid of each points set:
    BL_centroid = [np.mean(BLPoints[:, 0]), np.mean(BLPoints[:, 1])]
    FU_centroid = [np.mean(FUPoints[:, 0]), np.mean(FUPoints[:, 1])]

    # compute the 2x2 covariance matrix:
    X = FUPoints - FU_centroid
    W = np.eye(n) / n  # create a diagonal matrix with equal wights for all points
    Y = BLPoints - BL_centroid
    S = np.matmul(np.matmul(X.T, W), Y)

    # compute the SVD and find the rotation:
    u, sigma, vh = np.linalg.svd(S)
    s_mat = np.eye(2)
    s_mat[1, 1] = np.linalg.det(np.matmul(vh.T, u.T))
    R = np.matmul(np.matmul(vh.T, s_mat), u.T)

    # find the optimal translation:
    t = BL_centroid - np.dot(R, FU_centroid)

    rigidReg = np.zeros((3, 3))
    rigidReg[0:2, 0:2] = R
    rigidReg[0, 2] = t[0]
    rigidReg[1, 2] = t[1]
    rigidReg[2, 2] = 1
    return rigidReg


def apply_registration(points, rigidReg):
    """
    A function that receives an array with points and transforms the points using the given rigid registration
    :param points: The points that should be transformed, as a [N,2] array
    :param rigidReg: The registration that should be used for transforming the points
    :return: An array of shape [N, 2] with the coordinates of 'points' after the transformation
    """
    # transform the points into homogeneous points:
    n = len(points)
    homog_points = np.ones((3, n))  # create homogeneous representation to calculate the transformation
    homog_points[0:2, :] = points.T
    return np.matmul(rigidReg, homog_points)[0:n - 1, :].T  # todo: check about return value of homogeneous coordinates


def calcDist(BL_Points, FU_Points, rigidReg):
    """
    A function that calculates the distance of every matching point between FUPoints and BLPoints, by first applying
    the registration of FUPoints
    """
    est_BL = apply_registration(FU_Points, rigidReg)  # todo: check about the shape!
    return np.power((BL_Points[:, 0] - est_BL[:, 0]) ** 2 + (BL_Points[:, 1] - est_BL[:, 1]) ** 2, .5)


def display_registration(BL_im, FU_im, BLPoints, FUPoints):
    """
    A function that receives two images and the matching points between them, calculates the registration between them
    and then, moves one towards the other and displays them together.
    """
    BL_copy = copy.deepcopy(BL_im)
    rigidReg = calcPointBasedReg(BLPoints, FUPoints)
    warped_im = tf.warp(FU_im, np.linalg.inv(rigidReg))

    kernel = np.ones((5, 5), np.float32) / 25
    warped_im = cv2.filter2D(warped_im, -1, kernel)
    warped_im[warped_im < 0.3] = 0
    warped_im[warped_im > 0] = 1

    FU_im = cv2.filter2D(FU_im, -1, kernel)
    FU_im[FU_im < 0.25 * 255] = 0
    FU_im[FU_im > 0] = 1

    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.imshow(BL_copy)
    plt.title('original BL image')
    fig.add_subplot(1, 3, 2)
    BL_copy[FU_im == 0] = (255, 56, 56)
    plt.imshow(BL_copy)
    plt.title('before registration')
    fig.add_subplot(1, 3, 3)
    BL_im[warped_im == 0] = (255, 56, 56)  # mark the edges of the warped image in red
    plt.imshow(BL_im)
    plt.title('after registration')
    plt.show()


def calcRobustPointBasedReg(FUPoints, BLPoints):
    return ransac(BLPoints, FUPoints, calcPointBasedReg, calcDist, minPtNum=3, iterNum=100, thDist=15, thInlrRatio=0.1)


########################################################################################################################
# UNCOMMENT main and the functions to execute the different steps in the exercise
########################################################################################################################
# if __name__ == '__main__':
    # FU_im_orig = Image.open('FU01.tif')
    # FU_im = np.array(FU_im_orig)
    # BL_im_orig = Image.open('BL01.tif')
    # BL_im = np.array(BL_im_orig)
    # BL_points, FU_points = getPoints('with_outliers')  # change here between 'no/with _outliers'
    # rigidReg = calcPointBasedReg(BL_points, FU_points)

    # Question 1
    # UNCOMMENT: to display the graph of the matches between the two images
    # display_matches(BL_im, FU_im, BL_points, FU_points)

    # Question 3
    # UNCOMMENT: to calculate the mean distance in pixels
    # dist_vec = calcDist(BL_points, FU_points, rigidReg)
    # RMSE = np.mean(dist_vec)
    # print('RMSE: ', RMSE)

    # Question 4
    # UNCOMMENT: to display the images before and after registration
    # display_registration(BL_im, FU_im[:, :, 0], BL_points, FU_points)

    # Question 7
    # UNCOMMENT: to repeat steps 1-4 when using the robust function. Outliers status should be 'with_outliers'.
    # f, inliers = calcRobustPointBasedReg(BL_points, FU_points)
    # rigidReg = calcPointBasedReg(BL_points[inliers, :], FU_points[inliers, :])
    # display_matches(BL_im, FU_im, BL_points, FU_points, inliers)
    # dist_vec = calcDist(BL_points[inliers, :], FU_points[inliers, :], rigidReg)
    # RMSE = np.mean(dist_vec)
    # print('RMSE: ', RMSE)
    # display_registration(BL_im, FU_im[:, :, 0], BL_points[inliers, :], FU_points[inliers, :])

