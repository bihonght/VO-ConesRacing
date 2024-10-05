import cv2
import numpy as np
from common import params  # Assuming `basics.Config` is a config management system like a JSON or YAML file

def calcKeyPoints(image):
    # Retrieve configuration parameters
    num_keypoints = params.max_number_keypoints
    scale_factor = params.scale_factor
    level_pyramid = params.level_pyramid
    score_threshold = params.score_threshold
    
    # Create ORB object
    orb = cv2.ORB_create(nfeatures=num_keypoints, scaleFactor=scale_factor, nlevels=level_pyramid, 
                         edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, 
                         patchSize=31, fastThreshold=score_threshold)
    # Detect keypoints
    keypoints = orb.detect(image, None)
    # Apply uniform selection of keypoints
    keypoints = selectUniformKptsByGrid(keypoints, image.shape[0], image.shape[1])
    return keypoints

def calcDescriptors(image, keypoints):
    # Retrieve configuration parameters
    num_keypoints = params.max_number_keypoints
    scale_factor = params.scale_factor
    level_pyramid = params.level_pyramid
    # Create ORB object
    orb = cv2.ORB_create(nfeatures=num_keypoints, scaleFactor=scale_factor, nlevels=level_pyramid)
    # Compute descriptors
    keypoints, descriptors = orb.compute(image, keypoints)
    return keypoints, descriptors

def selectUniformKptsByGrid(keypoints, image_rows, image_cols):
    """
    Select keypoints uniformly across the image using a grid-based approach.
    """
    max_num_keypoints = params.max_number_keypoints
    grid_size = params.kpts_uniform_selection_grid_size
    max_pts_per_grid = params.kpts_uniform_selection_max_pts_per_grid
    
    rows = image_rows // grid_size
    cols = image_cols // grid_size
    
    # Initialize an empty grid
    grid = np.zeros((rows, cols), dtype=int)
    
    # List for storing selected keypoints
    tmp_keypoints = []
    cnt = 0
    
    # Insert keypoints into the grid
    for kpt in keypoints:
        row = int(kpt.pt[1] / grid_size)
        col = int(kpt.pt[0] / grid_size)
        
        if grid[row, col] < max_pts_per_grid:
            tmp_keypoints.append(kpt)
            grid[row, col] += 1
            cnt += 1
            
            # Stop if we've reached the maximum number of keypoints
            if cnt >= max_num_keypoints:
                break
    
    return tmp_keypoints

def matchByRadiusAndBruteForce(keypoints_1, keypoints_2, descriptors_1, descriptors_2, max_matching_pixel_dist):
    """
    Perform brute-force matching with a radius constraint on the pixel distance between keypoints.
    Args:
        keypoints_1 (list of cv2.KeyPoint): Keypoints from the first image.
        keypoints_2 (list of cv2.KeyPoint): Keypoints from the second image.
        descriptors_1 (np.ndarray): Descriptors corresponding to keypoints_1.
        descriptors_2 (np.ndarray): Descriptors corresponding to keypoints_2.
        max_matching_pixel_dist (float): Maximum pixel distance allowed between keypoints.
    Returns:
        matches (list of cv2.DMatch): List of matches between keypoints from the two images.
    """
    N1 = len(keypoints_1)
    N2 = len(keypoints_2)
    
    # Ensure that the number of keypoints and descriptors match
    assert N1 == descriptors_1.shape[0]
    assert N2 == descriptors_2.shape[0]
    
    matches = []
    r2 = max_matching_pixel_dist * max_matching_pixel_dist  # Squared radius for comparison

    for i in range(N1):
        kpt1 = keypoints_1[i]
        x1, y1 = kpt1.pt
        is_matched = False
        min_feature_dist = float('inf')
        target_idx = -1
        
        for j in range(N2):
            x2, y2 = keypoints_2[j].pt

            # Check if the distance between keypoints is within the maximum allowed pixel distance
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 <= r2:
                # Compute the feature distance (L1 norm between descriptors)
                feature_dist = np.sum(np.abs(descriptors_1[i] - descriptors_2[j])) / descriptors_1.shape[1]
                
                if feature_dist < min_feature_dist:
                    min_feature_dist = feature_dist
                    target_idx = j
                    is_matched = True
        
        # If a match is found, append it to the matches list
        if is_matched:
            matches.append(cv2.DMatch(i, target_idx, float(min_feature_dist)))

    return matches


def matchFeatures(descriptors_1, descriptors_2, method_index, is_print_res, keypoints_1=None, keypoints_2=None, max_matching_pixel_dist=None):
    """
    Match features between two sets of descriptors based on the specified method.
    Args:
        descriptors_1 (np.ndarray): Descriptors from the first image (query descriptors).
        descriptors_2 (np.ndarray): Descriptors from the second image (train descriptors).
        method_index (int): The matching method to use (1: Xiang Gao's method, 2: Lowe's method, 3: Brute-force with radius).
        is_print_res (bool): Whether to print matching results.
        keypoints_1 (list of cv2.KeyPoint, optional): Keypoints from the first image (required for method 3).
        keypoints_2 (list of cv2.KeyPoint, optional): Keypoints from the second image (required for method 3).
        max_matching_pixel_dist (float, optional): Maximum allowed pixel distance for matches (required for method 3).
    Returns:
        matches (list of cv2.DMatch): List of good matches between descriptors.
    """
    # Configuration constants (replace with your actual config fetching mechanism)
    xiang_gao_method_match_ratio = params.xiang_gao_method_match_ratio
    lowe_method_dist_ratio = params.lowe_method_dist_ratio
    method_3_feature_dist_threshold = params.method_3_feature_dist_threshold

    # Matcher setup
    # matcher_flann = cv2.FlannBasedMatcher(cv2.flann.LSHIndexParams(5, 10, 2))
    index_params = params.Matching_Params.index_params
    search_params = params.Matching_Params.search_params
    matcher_flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = []
    min_dis = float('inf')
    max_dis = 0

    if method_index == 1 or method_index == 3:  # Xiang Gao's method or Brute-force with radius
        all_matches = []

        if method_index == 3:
            all_matches = matchByRadiusAndBruteForce(keypoints_1, keypoints_2, descriptors_1, descriptors_2, max_matching_pixel_dist)
        else:
            all_matches = matcher_flann.match(descriptors_1, descriptors_2)

        # Compute distance threshold
        for match in all_matches:
            dist = match.distance
            if dist < min_dis:
                min_dis = dist
            if dist > max_dis:
                max_dis = dist

        distance_threshold = max(min_dis * xiang_gao_method_match_ratio, 30.0)

        # Select good matches based on distance threshold
        for match in all_matches:
            if match.distance < distance_threshold:
                matches.append(match)

    else:
        raise ValueError("Invalid method_index specified in matchFeatures")

    # Remove duplicated matches based on trainIdx
    matches = removeDuplicatedMatches(matches)

    # Print the results if requested
    if is_print_res:
        print(f"Matching features using method {method_index}, threshold = {distance_threshold}")
        print(f"Number of matches: {len(matches)}")
        print(f"-- Max dist: {max_dis}")
        print(f"-- Min dist: {min_dis}")
    
    return matches

def removeDuplicatedMatches(matches):
    """
    Remove duplicated matches based on trainIdx to ensure uniqueness.
    Args:
        matches (list of cv2.DMatch): List of matches.
    Returns:
        matches (list of cv2.DMatch): List of unique matches.
    """
    # Sort matches by trainIdx
    matches = sorted(matches, key=lambda m: m.trainIdx)

    # Remove duplicates based on trainIdx
    unique_matches = []
    if matches:
        unique_matches.append(matches[0])
    
    for i in range(1, len(matches)):
        if matches[i].trainIdx != matches[i - 1].trainIdx:
            unique_matches.append(matches[i])
    
    return unique_matches

def compute_mean_dist_between_keypoints(keypts1, keypts2, matches):
    """
    Computes the mean distance between matched keypoints.
    Args:
        kpts1 (list of cv2.KeyPoint): The keypoints from the first image.
        kpts2 (list of cv2.KeyPoint): The keypoints from the second image.
        matches (list of cv2.DMatch): The list of matches between the keypoints.
    Returns:
        float: The mean distance between matched keypoints.
    """
    dists_between_kpts = []

    for match in matches:
        p1 = keypts1[match.queryIdx].pt  # Get the point in the first image
        p2 = keypts2[match.trainIdx].pt  # Get the point in the second image
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dist = np.sqrt(dx * dx + dy * dy)     # Calculate the distance between the two points
        dists_between_kpts.append(dist)

    mean_dist = np.mean(dists_between_kpts)  # Calculate the mean of the distances
    return mean_dist