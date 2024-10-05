import numpy as np 
import cv2 

from geometry import feature_matching
from common import common, params
from vo.vo import VisualOdometry 
from vo.State import State 
from vo.map import Map

class VisualOdometry(VisualOdometry):
    def add_frame(self, State: State):
        """
        Adds a new frame to the visual odometry system and processes it.
        Args:
            frame (Frame): The frame to process.
        """
        # Settings
        self.push_frame_to_buff(State)
        # Renamed vars
        self.curr_ = State
        img_id = self.curr_.frame_id
        K = self.curr_.K

        # Start
        print("\n\n=============================================")
        print(f"Start processing the {img_id}th image.")

        self.curr_.detect_keypoints_and_descriptors(1.5)
        print(f"Number of keypoints: {len(self.curr_.keypoints)}")
        self.prev_ref_ = self.ref_

        # Visual Odometry state: BLANK -> DOING_INITIALIZATION
        if self.vo_state_ == self.VOState.BLANK:
            self.curr_.T_w_c_ = np.eye(4, dtype=np.float64)
            self.vo_state_ = self.VOState.DOING_INITIALIZATION
            self.add_key_frame(self.curr_)  # curr_ becomes the reference
        elif self.vo_state_ == self.VOState.DOING_INITIALIZATION:
            # Match features
            max_matching_pixel_dist = params.max_matching_pixel_dist_in_initialization
            method_index = params.feature_match_method_index_initialization
            self.curr_.matches_with_ref_ = feature_matching.matchFeatures(
                self.ref_.descriptors, self.curr_.descriptors, method_index, 
                False, self.ref_.keypoints, self.curr_.keypoints,
                max_matching_pixel_dist)

            print(f"Number of matches with the 1st frame: {len(self.curr_.matches_with_ref_)}")

            # Estimate motion and triangulate points
            self.estimate_motion_and_3d_points()
            print(f"Number of inlier matches: {len(self.curr_.inliers_matches_for_3d_)}")

            # Check initialization condition
            print("\nCheck VO init conditions: ")
            if self.is_vo_good_to_init():
                print(f"Large movement detected at frame {img_id}. Start initialization")
                self.push_curr_points_to_map()
                self.add_key_frame(self.curr_)
                self.vo_state_ = self.VOState.DOING_TRACKING
                print("Initialization success!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            else:
                self.curr_.T_w_c_ = self.ref_.T_w_c_
                print("Not initializing VO...")
        elif self.vo_state_ == self.VOState.DOING_TRACKING:
            print("\nDoing tracking")
            self.curr_.T_w_c_ = self.ref_.T_w_c_.copy()  # Initial estimation of the current pose
            is_pnp_good = self.pose_estimation_pnp()
            if not is_pnp_good:
                num_matches = len(self.curr_.matches_with_map_)
                print("PnP failed.")
                print(f"    Num inlier matches: {num_matches}.")
                if num_matches >= 5:
                    print("    Computed world to camera transformation:")
                    print(self.curr_.T_w_c_)
                print("PnP result has been reset as R=identity, t=zero.")
            '''
            else:
                self.call_bundle_adjustment()
                # Insert a keyframe if motion is large
                if self.check_large_move_for_add_key_frame(self.curr_, self.ref_):
                    max_matching_pixel_dist_triang = params.max_matching_pixel_dist_in_triangulation
                    method_index = params.feature_match_method_index_pnp
                    geometry.match_features(
                        self.ref_.descriptors_, self.curr_.descriptors_, self.curr_.matches_with_ref_,
                        method_index, False, self.ref_.keypoints_, self.curr_.keypoints_,
                        max_matching_pixel_dist_triang)

                    # Find inliers by epipolar constraint
                    self.curr_.inliers_matches_with_ref_ = geometry.helper_find_inlier_matches_by_epipolar_cons(
                        self.ref_.keypoints_, self.curr_.keypoints_, self.curr_.matches_with_ref_, K)

                    print(f"For triangulation: Matches with prev keyframe: {len(self.curr_.matches_with_ref_)}; Num inliers: {len(self.curr_.inliers_matches_with_ref_)}")

                    # Triangulate points
                    self.curr_.inliers_pts3d_ = geometry.helper_triangulate_points(
                        self.ref_.keypoints_, self.curr_.keypoints_,
                        self.curr_.inliers_matches_with_ref_, self.get_motion_from_frame1to2(self.curr_, self.ref_), K)

                    self.retain_good_triangulation_result_()

                    # Update state
                    self.push_curr_points_to_map_()
                    self.optimize_map_()
                    self.add_key_frame_(self.curr_)
                '''
        # Print relative motion
        if self.vo_state_ == self.VOState.DOING_TRACKING:
            if not hasattr(self, "T_w_to_prev"):
                self.T_w_to_prev = np.eye(4, dtype=np.float64)
            T_w_to_curr = self.curr_.T_w_c_
            T_prev_to_curr = np.linalg.inv(self.T_w_to_prev) @ T_w_to_curr
            R, t = common.get_rt_from_T(T_prev_to_curr)
            print("\nCamera motion:")
            print(f"R_prev_to_curr: {R}")
            print(f"t_prev_to_curr: {t.T}")

        self.prev_ = self.curr_
        print("\nEnd of a frame")

        