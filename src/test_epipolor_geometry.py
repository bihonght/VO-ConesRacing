import os 
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
# from utils import * 

from vo.State import State

from common import params
from geometry import motion_estimate, epipolar, feature_matching
import display.display as display

def is_rotation_matrix(R):
    Rt = np.transpose(R)
    should_be_identity = np.dot(Rt, R)
    I = np.eye(3)
    
    return np.linalg.norm(I - should_be_identity) < 1e-6

def main():
    frame_count = 1
    K = params.K
    camera_pose = np.eye(4)
    start = time.process_time()

    prev_state = State()
    # curr_state = State() 
    H_count = 0
    TOTAL_FRAME_COUNT = 159
    for frame_count in range(1, TOTAL_FRAME_COUNT):
        print("FRAME_COUNT : " + str(frame_count))
        if frame_count == 159: 
            break
        curr_state = State() 
        curr_state.read_image(frame_count)
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
            # prev_state.optimized_grid_based_orb_extraction()
            # prev_state.keypoints = calcKeyPoints(prev_state.image)
            # prev_state.keypoints, prev_state.descriptors = calcDescriptors(prev_state.image, prev_state.keypoints)

        curr_state.detect_keypoints_and_descriptors(1.5)
        # curr_state.keypoints = calcKeyPoints(curr_state.image)
        # curr_state.keypoints, curr_state.descriptors = calcDescriptors(prev_state.image, prev_state.keypoints)
        # draw_keypoints(prev_state.image, prev_state.keypoints)
        # draw_keypoints(curr_state.image, curr_state.keypoints)
        # Feature matching
        print(f"matching features in previous state {frame_count-1} and current state {frame_count}...")

        matches = feature_matching.matchFeatures(prev_state.descriptors, curr_state.descriptors, method_index=1, is_print_res=1, 
                      keypoints_1=prev_state.keypoints, keypoints_2=curr_state.keypoints, max_matching_pixel_dist=params.max_matching_pixel_dist_in_initialization)
        
        ()

        # display.draw_matches(prev_state.image, curr_state.image, matches, prev_state.keypoints, curr_state.keypoints)
        
        prev_state.kpoints, curr_state.kpoints = epipolar.extractPtsFromMatches(prev_state.keypoints, curr_state.keypoints, matches)
    
        is_print_res, is_calc_homo, is_frame_change = False, True, True
        best_sol, list_R, list_t, list_matches, list_normal, sols_pts3d_in_cam1_by_triang = motion_estimate.helper_estimate_possible_relative_poses_by_epipolar_geometry(prev_state.keypoints, 
                                                                                                                                                                         curr_state.keypoints, 
                                                                                                                                                                         matches, K,  is_print_res, 
                                                                                                                                                                         is_calc_homo, is_frame_change)

        print("++++++++++++++++ BEST +++++++++++++, ", best_sol)
        if best_sol != 0: 
            H_count += 1
        
        new_R, new_t = list_R[best_sol], list_t[best_sol]

        # essential_mat, _ = cv2.findEssentialMat(prev_state.kpoints, curr_state.kpoints, K, method=cv2.RANSAC, prob=0.9999, threshold=1)
        # _, new_R, new_t, mask = cv2.recoverPose(essential_mat, prev_state.kpoints, curr_state.kpoints, K)
        if not is_rotation_matrix(new_R):
            print("Rotation matrix is not valid.")
            continue
        if np.linalg.det(new_R) < 0:
                new_R = -new_R
                new_t = -new_t
        
        # prev_state.kpoints, curr_state.kpoints = good_matches_features(prev_state.kpoints, curr_state.kpoints, mask)

        new_pose = np.column_stack((new_R, new_t))
        new_pose = np.vstack((new_pose, np.array([0,0,0,1])))
        camera_pose = camera_pose @ new_pose
        x_coord = camera_pose[0, -1]
        z_coord = camera_pose[2, -1]
        plt.scatter(x_coord, -z_coord, color='b') 
        plt.pause(0.00001)
        frm = cv2.resize(prev_state.image, (0,0), fx=0.5, fy=0.5)
        cv2.imshow('Frame', frm)

        # Update the previous state
        prev_state = curr_state
    
    print(H_count)
    cv2.destroyAllWindows()
    plt.show()
    end = time.process_time() - start
    print("Time taken: ", end)


if __name__ == "__main__":
    main()