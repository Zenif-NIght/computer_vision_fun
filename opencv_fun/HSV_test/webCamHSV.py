# # https://pysource.com/2019/02/15/detecting-colors-hsv-color-space-opencv-with-python/

# import cv2
# import numpy as np

# cap = cv2.VideoCapture(0)

# while True:
#     _, frame = cap.read()
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Red color
#     low_red = np.array([161, 155, 84])
#     high_red = np.array([179, 255, 255])
#     red_mask = cv2.inRange(hsv_frame, low_red, high_red)
#     red = cv2.bitwise_and(frame, frame, mask=red_mask)

#     # Blue color
#     low_blue = np.array([94, 80, 2])
#     high_blue = np.array([126, 255, 255])
#     blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
#     blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

#     # Green color
#     low_green = np.array([25, 52, 72])
#     high_green = np.array([102, 255, 255])
#     green_mask = cv2.inRange(hsv_frame, low_green, high_green)
#     green = cv2.bitwise_and(frame, frame, mask=green_mask)

#     # Every color except white
#     low = np.array([0, 42, 0])
#     high = np.array([179, 255, 255])
#     mask = cv2.inRange(hsv_frame, low, high)
#     result = cv2.bitwise_and(frame, frame, mask=mask)

#     cv2.imshow("Frame", frame)
#     cv2.imshow("Red", red)
#     cv2.imshow("Blue", blue)
#     cv2.imshow("Green", green)
#     cv2.imshow("Result", result)

#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# https://docs.opencv.org/master/da/d97/tutorial_threshold_inRange.html
import cv2 as cv
import argparse
import matplotlib.pyplot as plt


max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

def main():
    parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
    parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
    args = parser.parse_args()
    cap = cv.VideoCapture(args.camera)
    cv.namedWindow(window_capture_name)
    cv.namedWindow(window_detection_name)
    cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
    cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
    cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
    cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
    cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
    cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
    while True:
        
        ret, frame = cap.read()

        if frame is None:
            break
        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        
        # fg = cv.bitwise_or(frame_HSV,frame_threshold, mask=frame_threshold)
        
        # cv.imshow("Masked",fg)
        cv.imshow(window_capture_name, frame)
        cv.imshow(window_detection_name, frame_threshold)
        
        key = cv.waitKey(30)
        if key == 27:
            break

if __name__ == "__main__":
    main()