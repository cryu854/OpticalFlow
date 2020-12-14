""" The code heavily based on opencv (https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html). """

import os
import cv2 as cv
import numpy as np
from utils import getfiles


def fb_from_image(path):
    flist = getfiles(path)
    os.makedirs('./fb_results/')

    for i in range(len(flist) - 1):
        im1 = cv.imread(flist[i])
        im2 = cv.imread(flist[i+1])
        gray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

        hsv = np.zeros_like(im1)
        # Sets image saturation to maximum
        hsv[..., 1] = 255
        # Calculates dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # Sets image hue according to the optical flow direction
        hsv[..., 0] = angle * 180 / np.pi / 2
        # Sets image value according to the optical flow magnitude (normalized)
        hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        # Converts HSV to RGB (BGR) color representation
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        cv.imwrite(f'./fb_results/optical_fb_{i}.png',bgr)


def fb_from_video(video_file):
    # The video feed is read in as a VideoCapture object
    cap = cv.VideoCapture(video_file)
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
        # Calculates dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
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
            cv.imwrite(f'optical_fb_{idx}.png',bgr)
    
    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()
