#=============================================================================
#=== Importing libraries =====================================================
#=============================================================================

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from Functions import *
from scipy.ndimage.measurements import label

svc = pickle.load( open("svc_pickle.p", "rb" ) )
X_scaler = pickle.load( open("X_scaler_pickle.p", "rb" ) )
orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = (32, 32)
hist_bins = 32

# img = mpimg.imread('test_images/test6.jpg')

#=============================================================================
#=== Find Cars Function ======================================================
#=============================================================================

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
  boxes = []
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
      # test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
      # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
      test_features = X_scaler.transform((hog_features).reshape(1, -1))
      test_prediction = svc.predict(test_features)
      # test_prediction = 1

      if test_prediction == 1:
        xbox_left = np.int(xleft*scale)
        ytop_draw = np.int(ytop*scale)
        win_draw = np.int(window*scale)
        cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
        boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

  # print(spatial_features.shape)
  # print(hist_features.shape)  
  # print(hog_features.shape)   
  # print(test_features.shape) 
  return draw_img, boxes

#=============================================================================
#=== Heat, Threshold and Label Functions =====================================
#=============================================================================

def add_heat(heatmap, bbox_list):
  # Iterate through list of bboxes
  for box in bbox_list:
    # Add += 1 for all pixels inside each bbox
    # Assuming each "box" takes the form ((x1, y1), (x2, y2))
    heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

  # Return updated heatmap
  return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
  # Zero out pixels below the threshold
  heatmap[heatmap <= threshold] = 0
  # Return thresholded map
  return heatmap

def draw_labeled_bboxes(img, labels):
  # Iterate through all detected cars
  for car_number in range(1, labels[1]+1):
    # Find pixels with each car_number label value
    nonzero = (labels[0] == car_number).nonzero()
    # Identify x and y values of those pixels
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Define a bounding box based on min/max x and y
    bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
    # Draw the box on the image
    cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
  # Return the image
  return img

ystart = 400
ystop = 656
scale = 1.5

#=============================================================================
#=== Process Image, Video Pipeline ===========================================
#=============================================================================

def process_image(img):
  out_img, boxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
  # plt.imshow(out_img)
  # plt.show()

  box_list = boxes
  # print(box_list)
  # Read in image similar to one shown above 
  # image = mpimg.imread('test_image.jpg')
  heat = np.zeros_like(img[:,:,0]).astype(np.float)

  # Add heat to each box in box list
  heat = add_heat(heat,box_list)

  # Apply threshold to help remove false positives
  heat = apply_threshold(heat,1)

  # Visualize the heatmap when displaying    
  heatmap = np.clip(heat, 0, 255)

  # Find final boxes from heatmap using label function
  labels = label(heatmap)
  draw_img = draw_labeled_bboxes(np.copy(img), labels)

  # fig = plt.figure()
  # plt.subplot(121)
  # plt.imshow(draw_img)
  # plt.title('Car Positions')
  # plt.subplot(122)
  # plt.imshow(heatmap, cmap='hot')
  # plt.title('Heat Map')
  # fig.tight_layout()
  # plt.show()

  return draw_img

from moviepy.editor import VideoFileClip
from IPython.display import HTML
import moviepy as mve

#clip1 = VideoFileClip("project_video.mp4")
input_video = VideoFileClip("project_video.mp4")#.subclip(40,44)
output_video = input_video.fl_image(process_image) #NOTE: this function expects color images!!
output_video.write_videofile('project_video_output.mp4', audio=False)