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

def load_model(model_path):
    # Load the model
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = models.resnet18(pretrained=True)
    
    # Load the saved model
    model.load_state_dict(torch.load(model_path))
    model = torch.load(model_path)
    # Set the model to inference mode
    model.eval()
    return model


def detect_object(img, model):
    # draw the bounding box and label on the image
    def draw_bounding_box(img, boxes, class_ids, labels, obj_count):
        # Draw the bounding box
        for i in range(obj_count):
            (x, y, w, h) = boxes[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Draw the label
            text = "{}: {:.4f}".format(labels[class_ids[i]], scores[i])
            cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        return img
    
    # Convert the image from OpenCV BGR format to PyTorch RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert the image from NumPy / OpenCV to PIL format
    img = Image.fromarray(img)
    # Resize the image
    img = img.resize((300, 300))
    # Convert the image to a PyTorch Tensor
    img = transforms.ToTensor()(img)
    # Add an extra batch dimension since PyTorch treats all images as batches
    img = img.unsqueeze(0)
    # Get the detections
    detections = model(img)
    # Get the detections predicted scores and class
    scores = list(detections[0]['scores'].detach().numpy())
    class_ids = list(detections[0]['labels'].detach().numpy())
    # Get the bounding box coordinates
    boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(detections[0]['boxes'].detach().numpy())]
    obj_count = len(scores)
    labels = ['bees', 'ants']
    
    # draw the bounding box and label on the image
    img = draw_bounding_box(img, boxes, class_ids, labels, obj_count)

def open_webcam( model):
    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Display the resulting frame
        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # detect the object
        detect_object(frame, model)
        cv2.imshow('frame',frame)

    # When everything done, release the capture 
    cap.release()
    cv2.destroyAllWindows()

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