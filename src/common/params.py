import numpy as np

FEATURE_FAST = 0
FEATURE_ORB = 1
FEATURE_ORB_NEW = 1.5
FEATURE_SHI_TOMASI = 2

# K = np.loadtxt("K_matrix.txt")
K = np.array([[2174.3363,  0.,  800],
              [0.,         2178.2863,   200],
              [0.,         0.,          1.]])

TOTAL_FRAME_COUNT = 159

# ORB feature extraction
max_number_keypoints = 20000 # 8000
min_number_keypoints = 2000 # 1200
scale_factor = 1.2 # 1.2
level_pyramid = 4 # 4 or 6
score_threshold = 20

class Matching_Params:
    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1, 
                        trees = 5
                        )
    search_params = dict(checks = 50)

# Matching parameters
feature_match_method_index_initialization = 1
feature_match_method_index_triangulation = 1
feature_match_method_index_pnp = 1
# Matching descriptos
feature_match_method_index = 3
  # method 1: the method in Dr. Xiang Gao's slambook:
  #           distance_threshold = max<float>(min_dis * match_ratio, 30.0);
  # method 2: the method in Lowe's 2004 paper:
  #           min dis / second min dis < 0.8
  # method 3: match a point with its neighboring points and then apply Dr. Xiang Gao's criteria.
xiang_gao_method_match_ratio = 2 # This is for method 1 used in Dr. Xiang Gao's slambook:
lowe_method_dist_ratio = 0.8 # This is for method 2 of Lowe's 2004 SIFT paper
method_3_feature_dist_threshold = 50.0
# Method 3 parameters:
max_matching_pixel_dist_in_initialization = 100
max_matching_pixel_dist_in_triangulation = 100
max_matching_pixel_dist_in_pnp = 50

# remove wrong matches
kpts_uniform_selection_grid_size = 16
kpts_uniform_selection_max_pts_per_grid = 8

# ------------------- RANSAC Essential matrix -------------------
findEssentialMat_prob = 0.9999 # 0.999
findEssentialMat_threshold = 0.9 #0.9  1.0

# ------------------- Triangulation -------------------
min_triang_angle = 1.0
max_ratio_between_max_angle_and_median_angle = 20

# ------------------- Initialization -------------------

min_inlier_matches = 15
min_pixel_dist = 50
min_median_triangulation_angle = 2.0
assumed_mean_pts_depth_during_vo_init = 0.8

# ------------------- Tracking -------------------
min_dist_between_two_keyframes = 0.03
max_possible_dist_to_prev_keyframe = 0.3

# ------------------- Optimization -------------------
is_enable_ba = True                # Use bundle adjustment for camera and points in single frame. 1 for true, 0 for false
num_prev_frames_to_opti_by_ba = 5      # <= 20. I set the "kBuffSize_" in "vo.h" as 20, so only previous 20 frames are stored.
information_matrix = [1.0, 0.0, 0.0, 1.0] # "1.0 0.0 0.0 1.0"
is_ba_fix_map_points = True # TO DEBUG: If I set it to true and optimize both camera pose and map points, there is huge error.
# UPDATE_MAP_PTS: "" # This equals (!is_ba_fix_map_points) by default