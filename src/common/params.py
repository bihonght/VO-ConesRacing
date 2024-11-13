import numpy as np

FEATURE_FAST = 0
FEATURE_ORB = 1
FEATURE_ORB_NEW = 1.5
FEATURE_SHI_TOMASI = 2

K = np.array([[2174.3363,  0.,  800],
              [0.,         2178.2863,   200],
              [0.,         0.,          1.]])

K2 = np.array([[760.75395685,   0.        , 480.        ],
       [  0.        , 566.06677175, 270.        ],
       [  0.        ,   0.        ,   1.        ]])

# K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
#               [0, 7.188560000000e+02, 1.852157000000e+02],
#               [0, 0, 1]])

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
max_matching_pixel_dist_in_pnp = 100 #c [50]

# remove wrong matches
kpts_uniform_selection_grid_size = 16
kpts_uniform_selection_max_pts_per_grid = 8

# ------------------- RANSAC Essential matrix -------------------
findEssentialMat_prob = 0.999 #c            [0.999] 
findEssentialMat_threshold = 0.9 #c [0.9  1.0]

# ------------------- Triangulation -------------------
min_triang_angle = 0.1 # 0.5 or [1.0] 
max_ratio_between_max_angle_and_median_angle = 70

# ------------------- Initialization -------------------

min_inlier_matches = 15
min_pixel_dist = 45 # 45 or [50]
min_median_triangulation_angle = 0.15 # 1.5 or [2.0]
assumed_mean_pts_depth_during_vo_init = 0.8
display_init_matches = 0
display_init_matches_inliers = 1
# ------------------- Tracking -------------------
min_dist_between_two_keyframes = 0.03 #c min dist to be consider motion as large move # 0.02 or [0.03]
max_possible_dist_to_prev_keyframe = 3.99 #c 0.45 or [0.3]
display_pnp = 1 # debugging 
display_tracking_triangular = 0 #
# ------------------- Optimization -------------------
scale_factor_max = 2 # 1.8 - 2.5
scale_factor_min = 0.1

# ------------------- Data Association NN -------------------
ConfidenceNN = 0.996 #5
ConfidenceNewBeacon = 0.97 # 0.97

dist_threshold = 3.68
dist_threshold_for_match = 2.34 #2.45 #1 
dist_threshold_for_new_beacon = 1.56 # 3.04 

