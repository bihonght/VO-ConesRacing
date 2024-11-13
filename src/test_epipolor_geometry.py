import os 
import time
import cv2
from cv2.xfeatures2d import matchGMS
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO

from utils import * 
from vo.State import State
from common import params, common
from geometry import motion_estimate, epipolar, feature_matching
from detection import detection
import display.display as display
from optimization import optimization
from scipy.optimize import minimize, least_squares
from scipy.io import savemat
from odometry.ekf import EKF

show_match = 0
draw_bbox_pair = 0
yolo_model = YOLO('model/weights/best.pt')
bbox1 = []
def main():
    frame_count = 1
    camera_pose = np.eye(4)
    start = time.process_time()

    prev_state = State(yolo_model)
    # curr_state = State() 
    H_count = 0
    TOTAL_FRAME_COUNT = 159 # 540 159

    fig_cones, ax_cones, sc_cones = display.init_cone_plot()
    fig_traj, ax_traj, sc_traj, cones_sc = display.init_trajectory_plot()
    x_coord = []
    z_coord = []
    slam_data = []

    ekf = EKF()
    for frame_count in range(1, TOTAL_FRAME_COUNT):
        print("---- FRAME_COUNT ---- : " + str(frame_count))
        if frame_count == TOTAL_FRAME_COUNT: 
            break
        curr_state = State(yolo_model) 
        curr_state.read_image(frame_count)
        # curr_state.read_txt()
        curr_state.cone_detection()
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
        print(f"matching features in previous state {prev_state.frame_id} and current state {curr_state.frame_id}...")
        matches = feature_matching.matchFeatures(prev_state.descriptors, curr_state.descriptors, method_index=1, is_print_res=1, 
                      keypoints_1=prev_state.keypoints, keypoints_2=curr_state.keypoints, max_matching_pixel_dist=params.max_matching_pixel_dist_in_initialization)
        
        matches_gms = matchGMS(prev_state.image.shape[:2], curr_state.image.shape[:2], prev_state.keypoints, curr_state.keypoints, matches, 
                                                withScale=True, withRotation=False, thresholdFactor=4)

        new_matches = feature_matching.removeDuplicatedMatches(matches + list(matches_gms))            

        prev_state.kpoints, curr_state.kpoints = epipolar.extractPtsFromMatches(prev_state.keypoints, curr_state.keypoints, matches)
        is_print_res, is_calc_homo, is_frame_change = False, True, True
        best_sol, list_R, list_t, list_matches, list_normal, sols_pts3d_in_cam1_by_triang = motion_estimate.helper_estimate_possible_relative_poses_by_epipolar_geometry(prev_state.keypoints, 
                                                                                                                                                                         curr_state.keypoints, 
                                                                                                                                                                         new_matches, params.K,  is_print_res, 
                                                                                                                                                                         is_calc_homo, is_frame_change)

        print("++++++++++++++++ BEST +++++++++++++, ", best_sol)
        if best_sol != 0: 
            H_count += 1

        R_curr_to_prev, t_curr_to_prev = list_R[best_sol], list_t[best_sol]
        matches = list_matches[best_sol]
        print("+++++++ Number of R solution +++++++++", len(list_matches))

        boxes_pair12 = detection.matching_bouding_boxes(new_matches, prev_state.cones, curr_state.cones,  prev_state.keypoints, curr_state.keypoints)
        curr_state.cones_pairings_with_ref_ = boxes_pair12

        if len(boxes_pair12) > 0: 
            bbox1.append(frame_count) if len(boxes_pair12) == 1 else None
            print("Number of NEW bbox pair: ", len(boxes_pair12))
        else:
            print("No matching bounding boxes found.")
        
        if np.linalg.det(R_curr_to_prev) < 0:
                R_curr_to_prev = -R_curr_to_prev
                t_curr_to_prev = -t_curr_to_prev
        
        if len(boxes_pair12) > 0:
            pair_prev_cones, pair_curr_cones = detection.pairing_cones(prev_state.cones3D, curr_state.cones3D, boxes_pair12)
            scipy_alpha = optimization.find_optimal_alpha(pair_prev_cones, pair_curr_cones, R_curr_to_prev, t_curr_to_prev, x_weight=3, z_weight=2)
            min_scale = params.scale_factor_min
            max_scale = params.scale_factor_max
            scale = max(min_scale, min(max_scale, scipy_alpha))
            ################ Scale translation ##################
            default_error = common.calc_cones_error(R_curr_to_prev, t_curr_to_prev, pair_prev_cones, pair_curr_cones, alpha=1,  print_error=False, print_cones_updates=False)
            error = common.calc_cones_error(R_curr_to_prev, t_curr_to_prev, pair_prev_cones, pair_curr_cones, alpha=scale, print_error=True, print_cones_updates=False)
            Rt = common.convert_rt_to_T(R_curr_to_prev, t_curr_to_prev*scale)
            if pair_curr_cones.shape[0] > 1: 
                newR, newT = optimization.estimate_and_modify_rotation_translation(R_curr_to_prev, t_curr_to_prev, pair_prev_cones, pair_curr_cones, x_weight=2, z_weight=1)
                new_error = common.calc_cones_error(newR, newT, pair_prev_cones, pair_curr_cones, alpha=1, print_error=True, print_cones_updates=False)
                if new_error < error:
                    Rt = common.convert_rt_to_T(newR, newT)
                    R_curr_to_prev = newR
                    t_curr_to_prev = newT
            elif error < default_error: 
                Rt = common.convert_rt_to_T(R_curr_to_prev, t_curr_to_prev*scale)
                t_curr_to_prev *= scale
            else: 
                Rt = common.convert_rt_to_T(R_curr_to_prev, t_curr_to_prev)
        else: 
            Rt = common.convert_rt_to_T(R_curr_to_prev, t_curr_to_prev)
        curr_state.Rt = Rt
        camera_pose = camera_pose @ (Rt)
        # x_coord.append(camera_pose[0, -1]) 
        # z_coord.append(-camera_pose[2, -1])

        if (1):
            display.update_cone_location(fig_cones, sc_cones, curr_state.cones3D, index_list=boxes_pair12)
            display.update_trajectory_plot(fig_traj, ax_traj, sc_traj, x_coord, z_coord) 
            if show_match:
                # display.draw_matches(prev_state.image, curr_state.image, list_matches[0], prev_state.keypoints, curr_state.keypoints, "Inliers Matching [E]")
                display.draw_matches(prev_state.image, curr_state.image, matches, prev_state.keypoints, curr_state.keypoints, "Inliers Matching [E]")
                display.draw_matches(prev_state.image, curr_state.image, matches_gms, prev_state.keypoints, curr_state.keypoints, "Inliers Matching [GMS]")
            if draw_bbox_pair: 
                display.draw_match_boxes(boxes_pair12, prev_state, curr_state)
        # plt.scatter(x_coord, -z_coord, color='b') 
        plt.pause(0.00001)
        frm = cv2.resize(curr_state.image, (0,0), fx=0.5, fy=0.5)
        cv2.imshow('Frame', frm)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Update the previous state
        prev_state = curr_state

        movement_data = common.current_data(R_curr_to_prev, t_curr_to_prev, curr_state.cones3D, frame_count)
        curr_state.movement_and_observation_ = movement_data
        slam_data.append(movement_data)
        ekf.add_frame(curr_state)

        x_robot, z_robot = ekf.st_global[:2, 0]
        x_coord.append(x_robot)  # Example trajectory X
        z_coord.append(z_robot)
        landmarks = ekf.st_global[3:, 0].reshape(-1, 2)
        display.update_trajectory_plot(fig_traj, ax_traj, sc_traj, x_coord, z_coord, cones_sc, cones_x=landmarks[:, 0], cones_z=landmarks[:, 1]) 


    print(bbox1)
    # common.save_slam_data(slam_data, 'FAE_Slam.mat')
    cv2.destroyAllWindows()
    plt.show()
    end = time.process_time() - start
    print("Time taken: ", end)

if __name__ == "__main__":
    main()