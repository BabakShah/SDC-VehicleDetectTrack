#=============================================================================
#=== Importing libraries =====================================================
#=============================================================================

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
from random import *

car_images = glob.glob('vehicles/*/*.png')
car = []
for fname in car_images:
    car.append(fname)
    
notcar_images = glob.glob('non-vehicles/*/*.png')
notcar = []
for fname in notcar_images:
    notcar.append(fname)
    
print("No. of car images = ")
print(len(car))

print("No. of not car images = ")
print(len(notcar))

fig, axs = plt.subplots(8,8, figsize=(10,20))
axs = axs.ravel()

for i in range(0, 64):
    car_img = cv2.imread(car_images[randint(0,len(car_images))])
    car_img = cv2.cvtColor(car_img,cv2.COLOR_BGR2RGB)
    axs[i].axis('off')
    axs[i].imshow(car_img)
plt.show()

for i in range(0, 64):
    notcar_img = cv2.imread(notcar_images[randint(0,len(notcar_images))])
    notcar_img = cv2.cvtColor(notcar_img,cv2.COLOR_BGR2RGB)
    axs[i].axis('off')
    axs[i].imshow(notcar_img)
plt.show()

# # Define feature parameters
# color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

# orient = 9  # HOG orientation angles
# pix_per_cell = 8
# cell_per_block = 2
# hog_channel = "ALL"  # aka 1st feature channel. Can be 0, 1, 2, or "ALL"
# spatial_size = (32, 32)  # Spatial binning dimensions
# hist_bins = 32  # Number of histogram bins
# spatial_feat = True  # Spatial features on or off
# hist_feat = True  # Histogram features on or off
# hog_feat = True  # HOG features on or off

# n_samples = 1000
# random_idxs = np.random.randint(0, len(cars), n_samples)
# # test_cars = np.array(cars)[random_idxs]  # Train on only random 1000 images. Takes ~=10 seconds.
# test_cars = cars  # Trains on all available data. Takes along time!
# # test_notcars = np.array(notcars)[random_idxs]  # Train on only random 1000 images. Takes ~=10 seconds.
# test_notcars = notcars  # Trains on all available data. Takes along time!

# car_features = extract_features(test_cars, color_space=color_space, spatial_size=spatial_size,
#                                 hist_bins=hist_bins, orient=orient,
#                                 pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
#                                 hog_channel=hog_channel,
#                                 spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
# notcar_features = extract_features(test_notcars, color_space=color_space, spatial_size=spatial_size,
#                                    hist_bins=hist_bins, orient=orient,
#                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
#                                    hog_channel=hog_channel,
#                                    spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

# X = np.vstack((car_features, notcar_features)).astype(np.float64)
# # Fit a per-column scaler
# X_scaler = StandardScaler().fit(X)
# # Apply the scaler to X
# scaled_X = X_scaler.transform(X)

# # Define the labels vector
# y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# # Split up data into randomised training and test sets
# rand_state = np.random.randint(0, 100)
# X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=rand_state)

# print('Using :', orient, 'orientations,', pix_per_cell, 'pixels per cell,', cell_per_block, 'cells per block,',
#       hist_bins, 'histogram bins, and', spatial_size, 'spatial sampling')

# print('Feature vector length :', len(X_train[0]))

# # Use a linear SVC
# svc = LinearSVC()

# # Check the training time for the SVC
# svc.fit(X_train, y_train)

# # Check the score of the SVC
# print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# # Save Linear SVC Classifier
# dist_pickle = {}
# dist_pickle['svc'] = svc
# dist_pickle['X_scaler'] = X_scaler
# dist_pickle['orient'] = orient
# dist_pickle['pix_per_cell'] = pix_per_cell
# dist_pickle['cell_per_block'] = cell_per_block
# dist_pickle['spatial_size'] = spatial_size
# dist_pickle['hist_bins'] = hist_bins
# pickle.dump(dist_pickle, open("svc_pickle.p", "wb"))

# dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
# svc = dist_pickle["svc"]
# X_scaler = dist_pickle["scaler"]
# orient = dist_pickle["orient"]
# pix_per_cell = dist_pickle["pix_per_cell"]
# cell_per_block = dist_pickle["cell_per_block"]
# spatial_size = dist_pickle["spatial_size"]
# hist_bins = dist_pickle["hist_bins"]

# img = mpimg.imread('./test_images/test1.jpg')



# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return draw_img
    
# ystart = 400
# ystop = 656
# scale = 1.5
    
# out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

# plt.imshow(out_img)