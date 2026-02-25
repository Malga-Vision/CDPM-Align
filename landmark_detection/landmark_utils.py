import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import torch


def get_scale_factor(original_image_shape, downsampled_image_shape):
    width_aspect_ratio = original_image_shape[1] / downsampled_image_shape[1]
    height_aspect_ratio = original_image_shape[0] / downsampled_image_shape[0]
    
    return width_aspect_ratio, height_aspect_ratio
    
def fuse_heatmaps(heatmaps):
    return np.maximum.reduce(heatmaps)

def keypoints2heatmaps(landmarks, img_size, fuse=False, sigma=0):
    heatmaps = []
    for i in range(len(landmarks)):
        heatmap = np.zeros(img_size, dtype=np.float32)
        x, y = landmarks[i].astype(int)
        
        heatmap[y, x] = 1.0
        
        # Apply Gaussian filter and threshold to get a binary image
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        heatmap = np.where(heatmap < 0.5 * heatmap.max(), 0, 1)
        assert is_binary_image(heatmap), f"Heatmap Image is not binary: {np.unique(heatmap)}"
        heatmaps.append(heatmap)
    
    return fuse_heatmaps(heatmaps) if fuse else np.array(heatmaps)
                     
def is_binary_image(image):
    return len(np.unique(image)) == 2


def get_mode_probability(heatmap):
    return np.max(heatmap)

def get_hottest_point(heatmap):
    # Find the index of the maximum value
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    # Get the maximum value and its neighbors
    max_val = heatmap[y, x]
    
    # Check if the maximum is at the edge of the heatmap
    if x > 0 and x < heatmap.shape[1] - 1 and y > 0 and y < heatmap.shape[0] - 1:
        # Quadratic interpolation in both x and y directions
        dx = 0.5 * (heatmap[y, x+1] - heatmap[y, x-1]) / (2 * max_val - heatmap[y, x+1] - heatmap[y, x-1])
        dy = 0.5 * (heatmap[y+1, x] - heatmap[y-1, x]) / (2 * max_val - heatmap[y+1, x] - heatmap[y-1, x])
    else:
        dx, dy = 0, 0
        
    return x + dx, y + dy


def extract_landmark(heatmap, use_argmax=True):
    if use_argmax:
        # Argmax method
        x, y = get_hottest_point(heatmap)

    else:
        # binary heatmap from negative and positive max values
        binary_heatmap = np.where(heatmap >= 0.5 * np.max(heatmap), 1, 0)
        assert is_binary_image(binary_heatmap), f"Image is not binary: {np.unique(binary_heatmap)}\nMax: {np.max(heatmap)}, Min: {np.min(heatmap)}"
        
        # Convert to uint8
        binary_heatmap = (binary_heatmap).astype(np.uint8)

        # FindContours method        
        contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(max_contour)

            if M['m00'] != 0:
                x = M['m10'] / (M['m00'] + 1e-8)
                y = M['m01'] / (M['m00'] + 1e-8)
            else:
                # Save heatmap
                x, y = get_hottest_point(heatmap)
                print(f"Zero division error in moment calculation, falling back to argmax at ({x}, {y})")
        else:
            raise ValueError(f"No contours found to extract landmark")
        
    return np.array([x, y], dtype=np.float64)
    
def calculate_ere(heatmap, predicted_point_scaled, pixel_size, significant_pixel_cutoff=0.05):
    norm = heatmap / np.max(heatmap)
    norm = np.where(norm > significant_pixel_cutoff, norm, 0.0)
    norm /= np.sum(norm)
    ys, xs = np.mgrid[0:norm.shape[0], 0:norm.shape[1]]
    coords = np.stack([xs, ys], axis=-1).reshape(-1, 2) * pixel_size  # (H*W, 2)
    weights = norm.ravel()
    dists   = np.linalg.norm(coords - predicted_point_scaled, axis=1)
    return float(np.dot(dists, weights))

def compute_euclidean_distance(predicted_point: np.ndarray, target_point: np.ndarray) -> float:
    euclidean_dist = np.linalg.norm(predicted_point - target_point)
    return euclidean_dist

def compute_mean_std(distance_list):
    means = np.mean(distance_list)
    std_devs = np.std(distance_list)
    return means, std_devs

def compute_sdr(distance_list, thresholds=[2, 2.5, 3, 4, 6, 9, 10]):
    successful_detection_rates = []
    for threshold in thresholds:
        sdr = 100 * (np.sum(distance_list <= threshold) / len(distance_list))
        successful_detection_rates.append(sdr)
    return successful_detection_rates, thresholds


def compute_p95(radial_errors) -> float:
    return float(np.percentile(np.asarray(radial_errors, dtype=np.float64), 95))