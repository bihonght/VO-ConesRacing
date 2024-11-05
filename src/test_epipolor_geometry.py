import os 
import time
import cv2
from cv2.xfeatures2d import matchGMS
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
# from utils import * 
from vo.State import State
from common import params, common
from geometry import motion_estimate, epipolar, feature_matching
from detection import detection
import display.display as display

from scipy.optimize import minimize, least_squares

def is_rotation_matrix(R):
    Rt = np.transpose(R)
    should_be_identity = np.dot(Rt, R)
    I = np.eye(3)
    
    return np.linalg.norm(I - should_be_identity) < 1e-6

show_match = 0
draw_bbox_pair = 0
bbox1 = []
def main():
    frame_count = 1
    K = params.K
    camera_pose = np.eye(4)
    start = time.process_time()

    prev_state = State()
    # curr_state = State() 
    H_count = 0
    TOTAL_FRAME_COUNT = 159

    fig_cones, ax_cones, sc_cones = display.init_cone_plot()
    fig_traj, ax_traj, sc_traj = display.init_trajectory_plot()
    x_coord = []
    z_coord = []
    for frame_count in range(1, TOTAL_FRAME_COUNT):
        print("---- FRAME_COUNT ---- : " + str(frame_count))
        if frame_count == 159: 
            break
        curr_state = State() 
        curr_state.read_image(frame_count)
        curr_state.read_txt()
        curr_state.estimate_cones_location()
        if curr_state.image is None:
            raise Exception("An error occurred.")
        if prev_state.image is None:
            print("Previous state Image is not available")
            prev_state = curr_state
            continue
        if prev_state.kpoints is None or len(prev_state.kpoints) < 2000: 
            print("number keypoints of tracked features: " + str(len(prev_state.keypoints)))
            print(f"detecting features in previous state {frame_count-1}..." )
            prev_state.detect_keypoints_and_descriptors(1.5)

        curr_state.detect_keypoints_and_descriptors(1.5)
        # Feature matching
        print(f"matching features in previous state {frame_count-1} and current state {frame_count}...")
        matches = feature_matching.matchFeatures(prev_state.descriptors, curr_state.descriptors, method_index=1, is_print_res=1, 
                      keypoints_1=prev_state.keypoints, keypoints_2=curr_state.keypoints, max_matching_pixel_dist=params.max_matching_pixel_dist_in_initialization)
        
        matches_gms = matchGMS(prev_state.image.shape[:2], curr_state.image.shape[:2], prev_state.keypoints, curr_state.keypoints, matches, 
                                                withScale=True, withRotation=False, thresholdFactor=4)
        if show_match: 
            display.draw_matches(prev_state.image, curr_state.image, matches, prev_state.keypoints, curr_state.keypoints, "Inliers Matching [E]")

        prev_state.kpoints, curr_state.kpoints = epipolar.extractPtsFromMatches(prev_state.keypoints, curr_state.keypoints, matches)
        is_print_res, is_calc_homo, is_frame_change = False, True, True
        best_sol, list_R, list_t, list_matches, list_normal, sols_pts3d_in_cam1_by_triang = motion_estimate.helper_estimate_possible_relative_poses_by_epipolar_geometry(prev_state.keypoints, 
                                                                                                                                                                         curr_state.keypoints, 
                                                                                                                                                                         matches, K,  is_print_res, 
                                                                                                                                                                         is_calc_homo, is_frame_change)

        print("++++++++++++++++ BEST +++++++++++++, ", best_sol)
        if best_sol != 0: 
            H_count += 1
        
        R_curr_to_prev, t_curr_to_prev = list_R[best_sol], list_t[best_sol]
        matches = list_matches[best_sol]
        print("+++++++ Number of R solution +++++++++", len(list_matches))
        if show_match:
            display.draw_matches(prev_state.image, curr_state.image, list_matches[0], prev_state.keypoints, curr_state.keypoints, "Inliers Matching [E]")
            display.draw_matches(prev_state.image, curr_state.image, matches_gms, prev_state.keypoints, curr_state.keypoints, "Inliers Matching [GMS]")
        if not is_rotation_matrix(R_curr_to_prev):
            print("Rotation matrix is not valid.")
            continue
        if np.linalg.det(R_curr_to_prev) < 0:
                R_curr_to_prev = -R_curr_to_prev
                t_curr_to_prev = -t_curr_to_prev
        
        # prev_state.kpoints, curr_state.kpoints = good_matches_features(prev_state.kpoints, curr_state.kpoints, mask)
        boxes_pair1 = detection.matching_bouding_boxes(matches_gms, curr_state.cones, prev_state.cones, curr_state.keypoints, prev_state.keypoints)
        boxes_pair2 = detection.matching_bouding_boxes(matches, curr_state.cones, prev_state.cones, curr_state.keypoints, prev_state.keypoints)
        boxes_pair = boxes_pair1
        for pair in boxes_pair2:
            if pair not in boxes_pair: 
                boxes_pair.append(pair)
        if boxes_pair is not None: 
            print(f"Number of bbox pairs GMS: {len(boxes_pair1)}\t, FLANN methods: {len(boxes_pair2)}")
            print("Number of total bbox pair: ", len(boxes_pair))
            bbox1.append(frame_count) if len(boxes_pair) == 1 else None
            if draw_bbox_pair: 
                display.draw_match_boxes(boxes_pair, prev_state, curr_state)
        else:
            raise Exception("No matching bounding boxes found.")

        if (1):
            display.update_cone_location(fig_cones, sc_cones, curr_state.cones3D, index_list=boxes_pair)

        pair_prev_cones, pair_curr_cones = pairing_cones(prev_state.cones3D, curr_state.cones3D, boxes_pair)
        # alpha = least_squares_alpha(pair_prev_cones, pair_curr_cones, R_curr_to_prev, t_curr_to_prev)
        scipy_alpha = find_optimal_alpha(pair_prev_cones, pair_curr_cones, R_curr_to_prev, t_curr_to_prev)
        print_error(R_curr_to_prev, t_curr_to_prev, pair_prev_cones, pair_curr_cones, alpha=1, df=True)

        # est_t = estimate_translation_xz(pair_prev_cones, pair_curr_cones, R_curr_to_prev, t_curr_to_prev)
        print_error(R_curr_to_prev, t_curr_to_prev, pair_prev_cones, pair_curr_cones, alpha=scipy_alpha, df=False)
        Rt = common.convert_rt_to_T(R_curr_to_prev, t_curr_to_prev*scipy_alpha)
        camera_pose = camera_pose @ Rt
        # x_coord = camera_pose[0, -1]
        # z_coord = camera_pose[2, -1]
        x_coord.append(camera_pose[0, -1])  # Example trajectory X
        z_coord.append(-camera_pose[2, -1])
        display.update_trajectory_plot(fig_traj, ax_traj, sc_traj, x_coord, z_coord)
        # plt.scatter(x_coord, -z_coord, color='b') 
        plt.pause(0.00001)
        frm = cv2.resize(curr_state.image, (0,0), fx=0.5, fy=0.5)
        cv2.imshow('Frame', frm)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        # Update the previous state
        prev_state = curr_state
    
    print(bbox1)
    cv2.destroyAllWindows()
    plt.show()
    end = time.process_time() - start
    print("Time taken: ", end)

def print_error(R, t, prev_cones_pos, curr_cones_pos, alpha=1, df=False):
    print("alpha: ", alpha)
    prev_cone_pos_transformed = []
    curr_cone_pos = []
    total_err = 0
    for i in range(len(prev_cones_pos)):
        prev_cone_pos = prev_cones_pos[i][:3]
        transformed_p = common.trans_coord(prev_cone_pos, R, alpha*t)

        prev_cone_pos_transformed.append(transformed_p)
        curr_cone_pos.append(curr_cones_pos[i][:3])
        total_err += np.linalg.norm(transformed_p[:3:2] - curr_cones_pos[i][:3:2])
        # prev_cone_pos_transformed = np.asarray(prev_cone_pos_transformed)
    if not df: 
        print("ERROR %.4f" % total_err)
        print(np.asarray(prev_cone_pos_transformed))
        print(np.asarray(curr_cone_pos))
    else: 
        print("DEFAULT ERROR %.4f" % total_err)
        print(np.asarray(prev_cone_pos_transformed))
        print(np.asarray(curr_cone_pos))

def pairing_cones(previous_cones, current_cones, pairs):
    prev_cones_pos_XZ = np.array([
        previous_cones[pair[0]][:3] 
        for pair in pairs 
        if previous_cones[pair[0]][2] <= 35 and current_cones[pair[1]][2] <= 30
    ])
    curr_cones_pos_XZ = np.array([
        current_cones[pair[1]][:3] 
        for pair in pairs 
        if previous_cones[pair[0]][2] <= 35 and current_cones[pair[1]][2] <= 30
    ])
    return prev_cones_pos_XZ, curr_cones_pos_XZ

def least_squares_alpha(previous_cones, current_cones, R, t):
    numerator = 0.0
    denominator = 0.0
    for prev_p, curr_p in zip(previous_cones, current_cones):
        transformed_p = R @ prev_p
        diff = curr_p - transformed_p
        numerator += t.T @ diff
        denominator += np.linalg.norm(t)**2
    # Compute the optimal alpha
    alpha = numerator / denominator
    return alpha

def least_squares_alpha_xz(previous_cones, current_cones, R, t):
    # Only take X and Z components of t
    t_xz = t[[0, 2]].flatten()  # Flatten to get a (2,) vector for dot product
    numerator = 0.0
    denominator = 0.0
    for prev_p, curr_p in zip(previous_cones, current_cones):
        transformed_p_xz = (R @ prev_p)[[0, 2]] 
        curr_p_xz = curr_p[[0, 2]]   
        diff_xz = curr_p_xz - transformed_p_xz  
        numerator += np.dot(t_xz, diff_xz)
        denominator += np.dot(t_xz, t_xz)
    alpha = numerator / denominator
    return alpha

def objective_function(alpha, previous_cones, current_cones, R, t):
    scaled_t = t * alpha
    total_error = 0.0
    for prev_p, curr_p in zip(previous_cones, current_cones):
        transformed_p = (R @ prev_p) + scaled_t.flatten()
        transformed_p_xz = transformed_p[[0, 2]]
        curr_p_xz = curr_p[[0, 2]]
        error = np.linalg.norm(transformed_p_xz - curr_p_xz)**2
        total_error += error
    return total_error

def find_optimal_alpha(previous_cones, current_cones, R, t):
    initial_alpha = 1.0
    result = minimize(objective_function, initial_alpha, args=(previous_cones, current_cones, R, t))
    return result.x[0]  # Extract the scalar value for alpha

def estimate_translation_xz(previous_cones, current_cones, R, init_t):
    prev_xz = (R @ previous_cones.T).T[:, [0, 2]]  # Shape (N, 2)
    curr_xz = current_cones[:, [0, 2]]  # Shape (N, 2)
    def objective(t_xz):
        transformed_xz = prev_xz + t_xz
        return (transformed_xz - curr_xz).flatten()

    initial_guess = init_t[[0, 2]].flatten()
    # Use least squares optimization to find the best t_xz
    result = least_squares(objective, initial_guess)
    # Extract the optimized translation vector for X and Z
    t_xz = result.x
    init_t[[0, 2]] = t_xz.reshape(2,1)
    return init_t

def save_current_data(new_R, new_t, cones3D, frame_count):
    tx = new_t[0,0]
    ty = -new_t[2,0]  # Due to the coordinate system mapping
    # Extract theta from the rotation matrix new_R
    theta = np.arctan2(new_R[1, 0], new_R[0, 0])
    # Create the odom matrix
    odom = np.array([
        [tx, 1, frame_count + 1, frame_count],
        [ty, 1, frame_count + 1, frame_count],
        [theta, 1, frame_count + 1, frame_count]
    ])

    len_cones = cones3D.shape[0]
    observations = np.ones((len_cones * 2, 4))

    frame_val = frame_count + 1
    cones_idx = np.arange(1, len_cones + 1)
    
    for i in range(len_cones):
        if (cones3D[i, 2] < 25):
            observations[2 * i, 0] = cones3D[i, 0]
            observations[2 * i + 1, 0] = cones3D[i, 2]
            observations[2 * i:2 * i + 2, 1] = 2
            observations[2 * i:2 * i + 2, 2] = cones_idx[i]
            observations[2 * i:2 * i + 2, 3] = frame_val
        else: 
            observations[2*i : 2*i+2, 0] = -100
    
    observations = observations[observations[:,0]>-100, :]
    perception_data = np.vstack((odom, observations))
    return perception_data

if __name__ == "__main__":
    main()