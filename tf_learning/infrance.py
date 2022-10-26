# this proram will open the webcam and detect using the torch model 

import cv2
import numpy as np
import torch
from torchvision import datasets, models, transforms
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
import sys
import argparse

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils

def load_model(model_path):
    # Load the model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = models.resnet18(pretrained=True)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    # Load the saved model
    model.load_state_dict(torch.load(model_path))
    # Set the model to inference mode
    model.eval()
    return model


def open_webcam( model):
    # initialize the vide stream, allow the camera sensor to warm up and initialize FPS counter
    print('[INFO] starting video stream...')
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()

    classes = ['bees', 'ants']

    # loop over the frames from video stream
    while True:
        # grab the frame from threaded video stream and resize it to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        orig = frame.copy()

        # conver the frame from BGR to RGB channels ordering and change from from channels last to channels first ordering
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.transpose((2, 0, 1))

        # add a batch dimension, scale the raw pixel intensities to the range [0, 1] and convert frame to a
        # floating point tensor
        frame = np.expand_dims(frame, axis=0)
        frame = frame/255.0
        frame = torch.FloatTensor(frame)

        # send the input to device and pass it through the model to get detections and predictions
        frame = frame.to('cpu')
        detections = model(frame)[0]

        #  loop over the detections
        for i in range(0, detections.shape[0]):
            # extract the confidence associated with predictions
            confidence = detections[i]

            # filter weak detections by ensuring the confidence is greater then minimum confidence
            if confidence > 0.5:
                # extract the index of class label from detections then compute (x, y) coordinates of bounding box for object
                idx = int(detections['labels'][i])
                box = detections['boxes'][i].detach().cpu().numpy()
                (startX, startY, endX, endY) = box.astype('int')

                # display predictions to our terminal
                label = f'{classes[idx]}: {confidence * 100:.2f}%'

                # draw bounding box and label on image
                cv2.rectangle(orig, (startX, startY), (endX, endY), colors[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cv2.colors[idx], 2)

        cv2.imshow('Frame', orig)
        key = cv2.waitKey(1)

        # if q is pressed breat the loop
        if key == ord('q'):
            break

        # update FPS counter
        fps.update()

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="tf_learning/model2.pt", help="path to model")

    args = vars(ap.parse_args())

    # load the model
    model = load_model(args["model"])
    # open the webcam
    open_webcam(model)


if __name__ == "__main__":
    main()