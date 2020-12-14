""" The code heavily based on scivision/pyoptflow project (https://github.com/scivision/pyoptflow). """

import os
import cv2 as cv
import numpy as np
from OpticalFlow.utils import getfiles
from scipy.ndimage.filters import convolve as filter2


HSKERN = np.array([[1/12, 1/6, 1/12],
                   [1/6,    0, 1/6],
                   [1/12, 1/6, 1/12]], float)

kernelX = np.array([[-1, 1],
                    [-1, 1]]) * .25  # kernel for computing d/dx

kernelY = np.array([[-1, -1],
                    [1, 1]]) * .25  # kernel for computing d/dy

kernelT = np.ones((2, 2))*.25


def HornSchunck(im1, im2, alpha, Niter):
    """

    Parameters
    ----------

    im1: numpy.ndarray
        image at t=0
    im2: numpy.ndarray
        image at t=1
    alpha: float
        regularization constant
    Niter: int
        number of iteration
    """
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    # set up initial velocities
    uInitial = np.zeros([im1.shape[0], im1.shape[1]])
    vInitial = np.zeros([im1.shape[0], im1.shape[1]])

    # Set initial value for the flow vectors
    U = uInitial
    V = vInitial

    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    # Iteration to reduce error
    for _ in range(Niter):
        # Compute local averages of the flow vectors
        uAvg = filter2(U, HSKERN)
        vAvg = filter2(V, HSKERN)
        # common part of update step
        der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)
        # iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U, V


def computeDerivatives(im1, im2):
    fx = filter2(im1, kernelX) + filter2(im2, kernelX)
    fy = filter2(im1, kernelY) + filter2(im2, kernelY)

    # ft = im2 - im1
    ft = filter2(im1, kernelT) + filter2(im2, -kernelT)

    return fx, fy, ft


def hs_from_image(path):
    flist = getfiles(path)
    os.makedirs('./hs_results/')
    
    for k in range(len(flist) - 1):
        im1 = cv.imread(flist[k])
        im2 = cv.imread(flist[k+1])
        gray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

        hsv = np.zeros_like(im1)
        # Sets image saturation to maximum
        hsv[..., 1] = 255

        U, V = HornSchunck(gray1, gray2, alpha=1.0, Niter=8)

        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(U, V)
        # Sets image hue according to the optical flow direction
        hsv[..., 0] = angle * 180 / np.pi / 2
        # Sets image value according to the optical flow magnitude (normalized)
        hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        # Converts HSV to RGB (BGR) color representation
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        cv.imwrite(f'./hs_results/optical_hs_{k}.png',bgr)


def hs_from_video(file):
    # The video feed is read in as a VideoCapture object
    cap = cv.VideoCapture(file)
    # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
    ret, first_frame = cap.read()
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    # Creates an image filled with zero intensities with the same dimensions as the frame
    hsv = np.zeros_like(first_frame)
    # Sets image saturation to maximum
    hsv[..., 1] = 255
    # Index of current frame
    idx = 0

    while(cap.isOpened()):
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()
        # Opens a new window and displays the input frame
        cv.imshow("input", frame)
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Horn-Schunck method
        U, V = HornSchunck(prev_gray, gray, alpha=1.0, Niter=8)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(U, V)
        # Sets image hue according to the optical flow direction
        hsv[..., 0] = angle * 180 / np.pi / 2
        # Sets image value according to the optical flow magnitude (normalized)
        hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        # Converts HSV to RGB (BGR) color representation
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        # Opens a new window and displays the output frame
        cv.imshow("dense optical flow", bgr)

        # Updates previous frame
        prev_gray = gray
        idx += 1
        # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
        # Save current frame when the user presses the 's' key
        k = cv.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('s'):
            cv.imwrite(f'optical_hs_{idx}.png',bgr)
        
    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()
