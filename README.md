# Self-Driving Car Project: Vehicle Detection and Tracking

In this project, I wrote a software pipeline to detect and track vehicles in a video.

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### The Dataset

For this project, I used a labeled dataset of vehicles and non-vehicles to train my classifier. The data consists of a combination of 64x64 pixels images (8792 vehicles and 8968 non-vehicles) from GTI vehicle image database and KITTI vision benchmark suite. You can find the links to database here:

-[GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html)

-[KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/)

Also, you can see a random selection of the dataset below:

##### Cars:
<img src=output_images/Cars.png />

##### Not cars:
<img src=output_images/notCars.png />

### HOG Visualization

<img src=output_images/HOGVis.png />

### SVM Classification

Using: 9 orientations 8 pixels per cell and 2 cells per block

Feature vector length: 5292

Test Accuracy of SVC =  0.9761

My SVC predicts:  [ 1.  1.  1.  0.  1.  0.  0.  1.  1.  0.]
For these 10 labels:  [ 1.  1.  1.  0.  1.  0.  0.  1.  1.  0.]

### Sliding Window Search

<img src=output_images/SlidingWindow1.png />
<img src=output_images/SlidingWindow2.png />
<img src=output_images/SlidingWindow3.png />
<img src=output_images/SlidingWindow4.png />