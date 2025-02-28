
import os
os.chdir(r"D:\THESIS\Thesis_data")
import rasterio
from rasterio.plot import show
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import view_as_windows
# Load stacked backscatter images (assuming 3 layers: VV, VH, HV)
with rasterio.open('Lope/D/P_band_rotated_clipped.tif') as src:
    stacked_polarizations = src.read()  # Read all bands/layers
# Extract individual polarization layers
HH_image = stacked_polarizations[0, :, :]  # HH polarization
HV_image = stacked_polarizations[1, :, :]  # HV polarization
VV_image = stacked_polarizations[2, :, :]  # VV polarization


# Function to normalize floating-point image to uint8
def normalize_to_uint8(image):
    min_val = np.nanmin(image)
    max_val = np.nanmax(image)
    normalized_image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return normalized_image

# Normalize the images to uint8
hh_image_uint8 = normalize_to_uint8(HH_image)
hv_image_uint8 = normalize_to_uint8(HV_image)
vv_image_uint8 = normalize_to_uint8(VV_image)

# Function to compute GLCM features in a sliding window manner
def compute_glcm_features_local(image, distances, angles, levels=256, window_size=3):
    # Create an empty array to hold the GLCM feature maps (same size as the image)
    feature_maps = np.zeros((image.shape[0], image.shape[1], 3))  # 3 features: correlation, energy, homogeneity
    
    # Create a sliding window view of the image
    windows = view_as_windows(image, (window_size, window_size))
    
    # Iterate over each window and compute GLCM features
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            window = windows[i, j]
            glcm = graycomatrix(window, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
            correlation = graycoprops(glcm, 'correlation').mean()
            energy = graycoprops(glcm, 'energy').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            
            # Assign the computed GLCM features to the feature_maps array
            feature_maps[i + window_size // 2, j + window_size // 2, :] = [correlation, energy, homogeneity]
    
    return feature_maps

# Define parameters for GLCM
distances = [1]  # Pixel distance
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Directions (0°, 45°, 90°, 135°)

# Compute GLCM features for each normalized polarization (uint8) using a sliding window approach
vv_glcm_features = compute_glcm_features_local(vv_image_uint8, distances, angles)
vh_glcm_features = compute_glcm_features_local(vh_image_uint8, distances, angles)
hv_glcm_features = compute_glcm_features_local(hv_image_uint8, distances, angles)

# Stack the original polarization layers with GLCM feature layers
# This will result in a new set of layers
stacked_features = np.stack([
    vv_image, vh_image, hv_image,              # Original polarization layers
    vv_glcm_features[:, :, 0], vh_glcm_features[:, :, 0], hv_glcm_features[:, :, 0],  # Correlation
    vv_glcm_features[:, :, 1], vh_glcm_features[:, :, 1], hv_glcm_features[:, :, 1],  # Energy
    vv_glcm_features[:, :, 2], vh_glcm_features[:, :, 2], hv_glcm_features[:, :, 2]   # Homogeneity
], axis=-1)

# Save the new stacked features as a multi-layer GeoTIFF
with rasterio.open('P_stacked_glcm_features.tif', 'w', driver='GTiff',
                   height=vv_image.shape[0], width=vv_image.shape[1], count=stacked_features.shape[2],
                   dtype='float32') as dst:
    for i in range(stacked_features.shape[2]):
        dst.write(stacked_features[:, :, i], i+1)