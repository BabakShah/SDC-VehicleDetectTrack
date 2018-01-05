# Self-Driving Car Project: Vehicle Detection and Tracking

<center>
<img src="output_images/Main.jpg" width="90%" alt="Vehicle Detection" />
</center>

Hello there! I'm Babak. Let me introduce you to my project. In this project, I wrote a software pipeline to detect and track vehicles in a video. Using different training dataset and features to train the classifier; this algorithm can be used to detect and teack other objects such as pedestrains or cyclists. This project was written using Python object oriented programming.


### Pipeline:

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images 
* Train a Linear Support Vector Machine (SVM) classifier on the HOG features. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### The Dataset

For this project, I used a labeled dataset of vehicles and non-vehicles to train my classifier. The data consists of a combination of 64x64 pixels images (8792 vehicles and 8968 non-vehicles) from GTI vehicle image database and KITTI vision benchmark suite. You can see a random selection of the dataset below:

Cars:
<img src=output_images/Cars.png style="width: 100%;"/>

Not cars:
<img src=output_images/notCars.png style="width: 100%;"/>

You can find the links to database here:

-[GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html)

-[KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/)

---
### Pipeline (image)

#### 1. HOG Visualization

<img src=output_images/HOGVis.png />

#### 2. SVM Classification

Using: 9 orientations 8 pixels per cell and 2 cells per block

Feature vector length: 5292

Test Accuracy of SVC =  0.9761

My SVC predicts:  [ 1.  1.  1.  0.  1.  0.  0.  1.  1.  0.]
For these 10 labels:  [ 1.  1.  1.  0.  1.  0.  0.  1.  1.  0.]

#### 3. Sliding Window Search

<img src=output_images/SlidingWindow1.png />
<img src=output_images/SlidingWindow3.png />
<img src=output_images/SlidingWindow4.png />

#### 4. Heatmap

<img src=output_images/heat.png />

---

### Pipeline (video)

#### 1. Final video output 

Here's a [link to my video result](./project_video_output.mp4) 

---

### Discussion on making the pipeline more robust?