"""This code imports necessary libraries"""

import os
# from pathlib import Path
# import logging
# import datetime
# import random

# import json
# import cv2
import glob
# from osgeo import gdal, gdalnumeric as gdn

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from functools import partial
# import tensorflow as tf
# import albumentations as A
# import segmentation_models as sm


# Get Number of Files
"""This code gets the number of files in a directory"""

def get_files(dirname, pattern = "*.tif"):
    """
    Get TIFF files in a directory.
    """
    # Get the list of files in the directory
    file_list = glob.glob(os.path.join(dirname, pattern))
    # Return the list
    return file_list


def get_num_files(dirname, pattern="*.tif"):
    """
    Get the number of TIFF files in a directory.
    """
    # Get the list of files in the directory
    return len(get_files(dirname, pattern="*.tif")

def tf_gdal_get_image_tensor(image_path):
    """
    Reads a satellite image geotiff using gdal, and converts to a numpy tensor.
    Image geotiffs have 3 channels (RGB) and are of type float32.
  
    Channels go last in the index order for the output tensor, as is standard for tensorflow.
    
    :param tf.StringTensor image_path: path to the image file, as a tensorflow String Tensor.
    """

    # Open the raster image file using GDAL
    ds = gdal.Open(mask_path.numpy())

    # Extract the individual raster bands as a list
    bands = [ds.GetRasterBand(i) for i in range(1, ds.RasterCount + 1)]

    # Iterate over each band and read it as a NumPy array
    band_arrays = []
    for band_num, current_band in enumerate(bands):
      band_arrays.append(gdn.BandReadAsArray(current_band))

    # Stack the band arrays together along the last axis to create a multi-band array
    image = np.stack(band_arrays, axis=-1)

    # normalize the float image to [0,1]
    return image/np.max(image)


def tf_gdal_get_mask_tensor(mask_path):
    """
    Reads a mask geotiff image using gdal, and converts to a numpy tensor.
    Masks have datatype uint8, and a single channel.
    mask_path: path to a gdal-readable image.

    Channels go last in the index order for the output tensor, as is standard for tensorflow.

    :param tf.StringTensor mask_path: path to the image file, as a tensorflow String Tensor.
    """

    # Open the raster image file using GDAL
    ds = gdal.Open(mask_path.numpy())

    # Extract the individual raster bands as a list
    bands = [ds.GetRasterBand(i) for i in range(1, ds.RasterCount + 1)]

    # Iterate over each band and read it as a NumPy array
    band_arrays = []
    for band_num, current_band in enumerate(bands):
      band_arrays.append(gdn.BandReadAsArray(current_band))

    # Stack the band arrays together along the last axis to create a multi-band array
    mask = np.stack(band_arrays, axis=-1)

    return mask


