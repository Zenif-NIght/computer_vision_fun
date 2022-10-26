# Computer Vision is FUN! 

There are many good resources for learning computer vision. This repo is a collection of resources that I have found useful.
One  of the best is the [OpenCV-Python Tutorials](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html).

and This PyIageSearch blog is also very good: [PyImageSearch](www.pyimagesearch.com/)
https://github.com/BasitJaved/PyImageSearch_University


# Install miniconda 
## YOU NEED THIS IN YOUR LIFE 😱
Google it you will figure it out 

# Intall pyTorch

Check out the right command for you:
[https://pytorch.org/get-started/locally/]

to create in conda env from scratch you could do this:
```
conda create -n pytorch torchvision -c pytorch
```

## Notes on Conda 
### For M1 macs if you need 3.6
```
CONDA_SUBDIR=osx-64 conda create -n torch_36_x86 python=3.6
conda activate torch_36_x86

```
### For Intel macs and the rest of the world
```
conda create -n torch_36 python=3.6
conda activate torch_36
```
# OpenCV Fun
This folder holds some fun I had with openCV
the readme in the folder has more info

I found this simple example in OpenCV "MobileNetSSD_deploy" 
https://github.com/Beomus/Python-Realtime-Object-Detector/blob/master/main.py

# PyTorch 

## tf_learning folder
this holds an example of houw to us transfer learning with pytorch
the inference method is not working yet

## torch video folder
this holds an example of how to use pytorch on a camera feed 

# One Yolo to rule them all
well like 5 of them... there are more added all the time it seems.
Here is an example of pytorch Yolo v3.
```
git clone https://github.com/nrsyed/pytorch-yolov3.git

cd pytorch-yolov3
pip install .
./get_weights.sh

```

if you have a wget error on macos you can use brew to install wget and then run the script again

```
brew install wget
```

if you don't have brew installed you can go install it and try again

 

