import cv2
import numpy as np

from skimage.feature import local_binary_pattern

def chi_square_distance(x, y):
    """Compute chi-square distance between two vectors"""
    eps = 1e-10  # small constant to avoid division by zero
    return 0.5 * np.sum((x - y)**2 / (x + y + eps))


def extract_lbp_features(image, radius=1, n_points=8, method='uniform', grid_size=(8,8)):
    """Extract LBP features from aligned face"""
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        img = image

    
    # Rest of the feature extraction remains the same
    h, w = img.shape
    bh, bw = h//grid_size[0], w//grid_size[1]
    
    y, x = np.mgrid[0:grid_size[0], 0:grid_size[1]]
    center_y, center_x = grid_size[0]//2, grid_size[1]//2
    weights = 1 / (1 + np.sqrt((y-center_y)**2 + (x-center_x)**2))
    
    features = []
    radii = [1, 2, 3]
    
    for radius in radii:
        n_points = radius * 8
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                block = img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                lbp = local_binary_pattern(block, n_points, radius, method=method)
                n_bins = n_points + 2
                hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
                hist = hist / np.linalg.norm(hist) if np.linalg.norm(hist) > 0 else hist
                hist = hist * weights[i,j]
                features.extend(hist)
    
    return np.array(features)

from skimage.feature import hog

def extract_hog_features(image, params):
    """Extract HOG features with given parameters"""
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        img = image
    
    features = hog(img, 
                  orientations=params['orientations'],
                  pixels_per_cell=params['pixels_per_cell'],
                  cells_per_block=params['cells_per_block'],
                  block_norm='L2-Hys')
    
    return features

def extract_dense_sift(image, params):
    """Extract Dense SIFT features with given parameters"""
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        img = image
    
    # Resize image first for consistency and speed
    img = cv2.resize(img, (128, 128))
    
    # Use OpenCV's built-in dense SIFT
    sift = cv2.SIFT_create()
    
    # More efficient way to create keypoints
    step_size = params['step_size']
    patch_size = params['patch_size']
    
    # Create grid of keypoints more efficiently using numpy
    y, x = np.mgrid[0:img.shape[0]:step_size, 
                    0:img.shape[1]:step_size]
    kp = [cv2.KeyPoint(float(x_), float(y_), patch_size) 
          for x_, y_ in zip(x.flatten(), y.flatten())]
    
    # Compute descriptors
    _, descriptors = sift.compute(img, kp)
    
    return descriptors

def calculate_cmc(gallery_features, gallery_ids, probe_features, probe_ids, max_rank=10):
    """Calculate CMC curve using chi-square distance"""
    gallery_ids = np.array(gallery_ids)
    probe_ids = np.array(probe_ids)

    # Calculate pairwise chi-square distances
    distances = np.zeros((len(probe_features), len(gallery_features)))
    for i, probe in enumerate(probe_features):
        for j, gallery in enumerate(gallery_features):
            distances[i,j] = chi_square_distance(probe, gallery)
    
    sorted_indices = np.argsort(distances, axis=1)
    
    n_probes = len(probe_ids)
    cmc = np.zeros(max_rank)
    
    for i, probe_id in enumerate(probe_ids):
        gallery_rankings = gallery_ids[sorted_indices[i]]
        correct_matches = np.where(gallery_rankings == probe_id)[0]
        if len(correct_matches) > 0:
            first_match = correct_matches[0]
            cmc[first_match:] += 1
    
    cmc = cmc / n_probes
    return cmc