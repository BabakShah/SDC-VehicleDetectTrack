import numpy as np
import cv2
import pickle
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
imageio.plugins.ffmpeg.download()
from ipywidgets import interact, interactive, fixed
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from skimage.feature import hog

images = glob.glob('vehicles/*/*.png')
cars = []
for fname in images:
    cars.append(fname)
    
images = glob.glob('non-vehicles/*/*.png')
notcars = []
for fname in images:
    notcars.append(fname)
    
print("No. of car images = ")
print(len(cars))

print("No. of not car images = ")
print(len(notcars))


# Define feature parameters
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

orient = 9  # HOG orientation angles
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"  # aka 1st feature channel. Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

n_samples = 1000
random_idxs = np.random.randint(0, len(cars), n_samples)
# test_cars = np.array(cars)[random_idxs]  # Train on only random 1000 images. Takes ~=10 seconds.
test_cars = cars  # Trains on all available data. Takes along time!
# test_notcars = np.array(notcars)[random_idxs]  # Train on only random 1000 images. Takes ~=10 seconds.
test_notcars = notcars  # Trains on all available data. Takes along time!

car_features = extract_features(test_cars, color_space=color_space, spatial_size=spatial_size,
                                hist_bins=hist_bins, orient=orient,
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                hog_channel=hog_channel,
                                spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(test_notcars, color_space=color_space, spatial_size=spatial_size,
                                   hist_bins=hist_bins, orient=orient,
                                   pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                   hog_channel=hog_channel,
                                   spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomised training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=rand_state)

print('Using :', orient, 'orientations,', pix_per_cell, 'pixels per cell,', cell_per_block, 'cells per block,',
      hist_bins, 'histogram bins, and', spatial_size, 'spatial sampling')

print('Feature vector length :', len(X_train[0]))

# Use a linear SVC
svc = LinearSVC()

# Check the training time for the SVC
svc.fit(X_train, y_train)

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# Save Linear SVC Classifier
dist_pickle = {}
dist_pickle['svc'] = svc
dist_pickle['X_scaler'] = X_scaler
dist_pickle['orient'] = orient
dist_pickle['pix_per_cell'] = pix_per_cell
dist_pickle['cell_per_block'] = cell_per_block
dist_pickle['spatial_size'] = spatial_size
dist_pickle['hist_bins'] = hist_bins
pickle.dump(dist_pickle, open("svc_pickle.p", "wb"))