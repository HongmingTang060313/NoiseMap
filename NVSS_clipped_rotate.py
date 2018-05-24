#!/usr/bin/env python
# The script is to rotate images upon clipped NVSS postage stamp images.
# Also, it will fill the blank space with randomized noise map
# Author: Hongming Tang

#from __future__ import division
from PIL import Image # yes
from astropy.stats import sigma_clipped_stats # yes
import numpy as np
import os
import random
from scipy.misc import imrotate
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve, Gaussian2DKernel
import math
# Set the root directory
rootdir = 'Directory_you_import_the_images'

# A function online to convert JPEG into 2D numpy array
def jpg_image_to_array(img_size, image_path, im_arr = None):
  """
  Load JPEG images into 3D Numpy array of shape
  (width, height, channels)
  """
  with Image.open(image_path) as image:
    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((img_size, img_size))
  return im_arr

# A function to store JPEG filenames in an folder.
def filename(rootdir):
    '''
    Save JPEG image filenames into 1D Numpy array
    (image1.jpeg, image2.jpeg, image3.jpeg, ...)
    '''
    filenames = []
    # Iterate files over directory
    for filename in os.listdir(rootdir):
        if filename.endswith(".jpeg"):
            name = os.path.join(filename)
            filenames = np.append(name,filenames)
        else:
            continue
    # Sort filenames
    filenames = filenames[::-1]
    return filenames

# Create Noise Map using np.random.multivariate_normal()
def Ran_Norm_2(mu,sigma,cov_x_y, size):
    """
    Create nosie map before and after convolvement
    mean pixel value = mu
    pixel std = sigma
    pixel width / height = size
    """
    a = math.sqrt(pow(mu,2)/2.)
    mean = [a, a]
    print(pow(sigma,2.))
    cov = [[pow(sigma,2.), cov_x_y], [cov_x_y, pow(sigma,2.)]]  # diagonal covariance, cov[0,0] and cov[1,1] represents the variance in x and y directions, respectively.
    x, y = np.random.multivariate_normal(mean, cov, size * size).T
    value = []
    for i in range(0,len(x)):
        tmp = math.sqrt(pow(x[i],2.) + pow(y[i],2.))
        value = np.append(tmp,value)
    # noise map before convolving with the VLA clean beam (FWHM = 45'' == 3 pixels)
    value = np.reshape(value,(size,size))
    # noise map after convolving with the VLA clean beam
    kernel = Gaussian2DKernel(0.785) # VLA FWHM: 45'' == 3 pixel
    value1 = convolve(value,kernel)
    return value, value1

# A function to calculate covariance
def transpose(im_arr):
    im_arr.tolist()
    im_arr = zip(*im_arr)
    im_arr = np.asarray(im_arr)
    return im_arr

def covariance_x_y (im_arr):
    x_val = im_arr.flatten()
    x_mean = np.mean(x_val)
    y_val = transpose(im_arr).flatten()
    y_mean = np.mean(y_val)
    covari_x_y = 0
    for ii in range(len(x_val)):
        covari_x_y += (x_val[ii]- x_mean) * (y_val[ii] - y_mean)
    covari_x_y /= len(x_val)
    #print('covarix_y',covari_x_y)
    return(covari_x_y)

# A function to rotate given images.
def image_rotate(rootdir,angle):
    # Import filenames in a folder
    filenames = filename(rootdir)
    for i in range(0,len(filenames)):
        image_path = rootdir + '/' + filenames[i]
        # Open image and convert it into pixels
        im_arr = jpg_image_to_array(60,image_path)
        cov_x_y = covariance_x_y(im_arr)
        # Rotate given JPEG images
        im_arr = imrotate(im_arr,angle,interp='bilinear')
        sigma = sigma_clipped_stats(im_arr,sigma=3.0,iters=5) # produce [mean, median, std] of an image
        print(sigma) # print the mean, median, and std of the array
        Ran_val, Ran_val_2 = Ran_Norm_2(sigma[0],sigma[2],cov_x_y,len(im_arr[0]))
        Ran_val = np.reshape(Ran_val_2,(len(im_arr[0]),len(im_arr[0])))

        for k in range(0,len(im_arr[0])):
            for j in range(0,len(im_arr[0])):
                if(im_arr[j][k] <= 0):
                    im_arr[j][k] = Ran_val[j][k] # Q: Is it reasonable to take median value of an image? Or perhaps should we use 0 as usual?
        im = Image.fromarray(im_arr)
        im.save("Directory_you_save_the_images" + str(i+1) + "_rotate_" + str(angle) +"_deg.jpeg")

# Run the script
image_rotate(rootdir,120)
