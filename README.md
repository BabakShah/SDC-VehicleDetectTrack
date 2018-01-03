# Self-Driving Car Project: Vehicle Detection and Tracking

In this project, I wrote a software pipeline to detect and track vehicles in a video.

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### The Dataset

For this project, I used a labeled dataset of vehicles and non-vehicles to train my classifier. The data consists of a combination of images (8792 vehicles and 8968 non-vehicles) from GTI vehicle image database and KITTI vision benchmark suite. You can find the links to database here:

-[GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html)

-[KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/)

Also, you can see a random selection of the dataset below:

##### Cars:
<img src=output_images/Cars.png />

##### Not cars:
<img src=output_images/notCars.png />

