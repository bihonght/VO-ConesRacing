import cv2
import numpy as np
from collections import deque
from enum import Enum
from typing import List, Optional, Dict

# Assuming the existence of these classes based on the C++ code
# You need to implement or import these appropriately
from vo.map import Map
from vo.map_point import MapPoint
from vo.State import State, PtConn

# import basics
from common import params, common
from geometry import motion_estimate, epipolar, feature_matching, camera
from display import display
from optimization import optimization
from detection import detection

# import optimization



class Odometry():
    class VOState(Enum):
        BLANK = 0
        DOING_INITIALIZATION = 1
        DOING_TRACKING = 2
        LOST = 3

    def __init__(self):
        # Map
        self.map_ = Map()
        
        # VO State
        self.vo_state_ = self.VOState.BLANK
        
        # Frame members
        self.curr_: Optional[State] = None
        self.prev_: Optional[State] = None
        self.ref_: Optional[State] = None
        self.prev_ref_: Optional[State] = None
        self.newest_frame_: Optional[State] = None
        self.prev_T_w_c_: Optional[np.ndarray] = None  # Assuming 4x4 transformation matrix
        
        # Buffer to store previous frames
        self.kBuffSize_ = 20
        self.frames_buff_ = deque(maxlen=self.kBuffSize_)
        
        # Map features
        self.keypoints_curr_: List[cv2.KeyPoint] = []
        self.descriptors_curr_: Optional[np.ndarray] = None
        
        self.matched_pts_3d_in_map_: List[np.ndarray] = []
        self.matched_pts_2d_idx_: List[int] = []
        
    # Public Methods
    def is_initialized(self) -> bool:
        return self.vo_state_ == self.VOState.DOING_TRACKING
    
    def get_prev_ref(self) -> Optional[State]:
        return self.prev_ref_
    
    def get_map(self) -> 'Map':
        return self.map_
    
    # Private Methods
    def push_frame_to_buff(self, State: State):
        """
        Push a frame to the buffer.
        """
        self.frames_buff_.append(State)
        if len(self.frames_buff_) > self.kBuffSize_:
            self.frames_buff_.popleft()
#  --------------------------------- Initialization ---------------------------------

    def estimate_motion_and_3d_points(self):
        """
        Estimate the motion between the current frame and the reference frame,
        and triangulate 3D points based on the inlier matches.
        """
        if not self.curr_ or not self.ref_:
            print("Current or reference frame is not set.")
            return
        # -- Rename output
        inlier_matches: List[cv2.DMatch] = self.curr_.inliers_matches_with_ref_
        pts3d_in_curr: List[np.ndarray] = self.curr_.inliers_pts3d_
        inliers_matches_for_cones: List[cv2.DMatch] = self.curr_.inliers_matches_for_cones_
        T = self.curr_.T_w_c_
        # -- Start: call this big function to compute everything
        # (1) motion from Essential & Homography,
        # (2) inliers indices,
        # (3) triangulated points
        is_print_res = False
        is_frame_cam2_to_cam1 = True
        is_calc_homo = True
        K = params.K2
        # Call the helper function to estimate possible relative poses
        best_sol, list_R, list_t, list_matches, list_normal, sols_pts3d_in_cam1_by_triang = motion_estimate.helper_estimate_possible_relative_poses_by_epipolar_geometry(self.ref_.keypoints, 
                                                                                                                                                                         self.curr_.keypoints, 
                                                                                                                                                                         self.curr_.matches_with_ref_, K,  
                                                                                                                                                                         is_print_res, is_calc_homo, 
                                                                                                                                                                         is_frame_cam2_to_cam1)
        if best_sol < 0 or best_sol >= len(list_R):
            print("No valid solution found for relative pose estimation.")
            return
        # -- Only retain the data of the best solution
        R_curr_to_prev = list_R[best_sol]
        t_curr_to_prev = list_t[best_sol]
        inlier_matches = list_matches[best_sol]

        inlier_matches_gms = feature_matching.removeDuplicatedMatches(inlier_matches + list(self.curr_.matches_gms_with_ref_))
        self.curr_.cones_pairings_with_ref_ = detection.matching_bouding_boxes(inlier_matches_gms, self.ref_.cones, self.curr_.cones, self.ref_.keypoints, self.curr_.keypoints)
        pair_prev_cones, pair_curr_cones = detection.pairing_cones(self.ref_.cones3D, self.curr_.cones3D, self.curr_.cones_pairings_with_ref_)

        # -- Compute camera pose
        Rt = common.convert_rt_to_T(R_curr_to_prev, t_curr_to_prev)
        T = self.ref_.T_w_c_ @ (Rt)
        self.curr_.T_w_c_ = T.copy()
        # Get points that are used for triangulating new map points
        self.curr_.inliers_matches_with_ref_ = inlier_matches
        self.curr_.inliers_pts3d_ = pts3d_in_curr
        # Normalize Points Depth to 1
        
        scale, success = self.check_motion_with_scale(pair_prev_cones, pair_curr_cones, R_curr_to_prev, t_curr_to_prev)
        t_curr_to_prev *= scale
        self.curr_.T_w_c_[0:3, 3] = self.curr_.T_w_c_[0:3, 3] * scale

        # Update camera pose after scaling
        Rt_scaled = common.convert_rt_to_T(R_curr_to_prev, t_curr_to_prev)
        T_scaled = self.ref_.T_w_c_ @ (Rt_scaled)
        self.curr_.T_w_c_ = T_scaled.copy()
        
        print("Motion and 3D points estimation completed successfully.")

    def check_motion_with_scale(self, pair_prev_cones, pair_curr_cones, R_curr_to_prev, t_curr_to_prev):
        # Rename input
        prev_cones = pair_prev_cones
        curr_cones = pair_curr_cones
        R = R_curr_to_prev
        t = t_curr_to_prev
        
        # Compute the scale
        scale = optimization.least_squares_alpha(prev_cones, curr_cones, R, t)
        default_error = common.calc_cones_error(R, t, prev_cones, curr_cones, alpha=1,  print_error=False, print_cones_updates=True)
        error = common.calc_cones_error(R, t, prev_cones, curr_cones, alpha=scale, print_error=False, print_cones_updates=True)
        # normalize error 
        num_matching_cones = prev_cones.shape[0]
        default_error /= num_matching_cones
        error /= num_matching_cones
        # Print error
        print(f"Scaling : {scale}")
        print(f"Scale error: {error:.2f} (default: {default_error:.2f})")
        # Check if the scale is large
        if error < default_error: 
            success = True
            scale = min(max(scale, 0.2), 1.8)
            error = common.calc_cones_error(R, t, prev_cones, curr_cones, alpha=scale)
            print(f"Scaling at {float(scale):.2f} with error: {float(error/num_matching_cones):.2f}")
        elif error > 4:
            print(f"Scaling : {scale}")
            # print(f"Scaling > 4: {scale:.2f}. It's not efficient: > default.")
            success = False
            scale = min(max(scale, 0.2), 1.8)
            error = common.calc_cones_error(R, t, prev_cones, curr_cones, alpha=scale)
            print(f"Scaling at {scale:.2f} with error: {error}")
        elif default_error < 4: # error > 0.6*default_error
            scale = 1.0
            print(f"Scaling success: {scale:.2f}. It's too small.")
            success = True
        return scale, success

    
# ------------------------------- Triangulation -------------------------------
    # --------------------------------------------------
    # -------------- Tracking Methods ------------------
    def check_large_move_for_add_key_frame(self, curr: State, ref: State) -> bool:
        """
        Check if the movement between current frame and reference frame is large enough to add a keyframe.
        """
        T_key_to_curr = np.linalg.inv(ref.T_w_c_) @ curr.T_w_c_
        R, t = common.get_rt_from_T(T_key_to_curr)
        R_vec, _ = cv2.Rodrigues(R)
        _, _, _, euler_angles = cv2.RQDecomp3x3(R)      
        yaw = euler_angles[1]  # Rotation around Y-axis
        min_dist = params.min_dist_between_two_keyframes
        moved_dist = np.linalg.norm(t[:3:2])
        rotated_angle = np.linalg.norm(R_vec)
        
        print(f"Wrt prev keyframe, relative dist = {moved_dist:.5f}, angle = {yaw:.5f}")
        
        return moved_dist > min_dist
    
    def is_vo_good_to_init(self) -> bool:
        """
        Check if visual odometry is good to be initialized.
        """
        if not self.ref_ or not self.curr_:
            return False
        
        # Rename input
        init_kpts = self.ref_.keypoints
        curr_kpts = self.curr_.keypoints
        matches = self.curr_.inliers_matches_with_ref_
        
        # Parameters (these should be configurable)
        min_inlier_matches = params.min_inlier_matches
        min_pixel_dist = params.min_pixel_dist
        min_median_triangulation_angle = params.min_median_triangulation_angle
        
        # Check Criteria 0: Number of inliers should be large
        criteria_0 = len(matches) >= min_inlier_matches
        if not criteria_0:
            print(f"{len(matches)} inlier points are too few... threshold is {min_inlier_matches}.")
        
        # Check Criteria 1: Mean distance between keypoints
        mean_dist = feature_matching.compute_mean_dist_between_keypoints(init_kpts, curr_kpts, matches)
        print(f"Pixel movement of matched keypoints: {mean_dist:.1f}. Threshold is {min_pixel_dist:.1f}")
        criteria_1 = mean_dist > 15 #min_pixel_dist
        
        # Check Criteria 2: Median triangulation angle
        
        return criteria_0 and criteria_1 # and criteria_2

    def pose_estimation(self) -> bool:
        """
        Estimate the camera pose using PnP.
        """
        if not self.curr_ or not self.map_:
            return False
        
        # Get 3D-2D correspondences
        candidate_mappoints_in_map, candidate_2d_pts_in_image, corresponding_descriptors = self.get_mappoints_in_current_view()
        candidate_kpts = common.pts2keypts(candidate_2d_pts_in_image)
        if params.display_pnp:  
            display.draw_keypoints(self.ref_.image, candidate_kpts)
        # Feature matching
        max_dist = params.max_matching_pixel_dist_in_pnp
        method_index = params.feature_match_method_index_pnp
        self.curr_.matches_with_map_ = feature_matching.matchFeatures(
            corresponding_descriptors, self.curr_.descriptors,
            method_index,
            is_print_res=False,
            keypoints_1=candidate_kpts,
            keypoints_2=self.curr_.keypoints,
            max_matching_pixel_dist=max_dist
        )
        if params.display_pnp:
            title = f"PnP Matches ref frame {self.ref_.frame_id} with {self.curr_.frame_id}"
            display.draw_matches(self.ref_.image, self.curr_.image, self.curr_.matches_with_map_, candidate_kpts, self.curr_.keypoints, title)
        
        num_matches = len(self.curr_.matches_with_map_)
        print(f"Number of 3D-2D pairs: {num_matches}")
        
        # -- Solve PnP, get T_world_to_camera
        pts_3d = []
        pts_2d = []
        for i in range(num_matches):
            match = self.curr_.matches_with_map_[i]
            mappoint = candidate_mappoints_in_map[match.queryIdx]
            pts_3d.append(mappoint.pos_)
            pts_2d.append(self.curr_.keypoints[match.trainIdx].pt)

        kMinPtsForPnP = 5
        max_possible_dist_to_prev_keyframe = params.max_possible_dist_to_prev_keyframe
        is_pnp_good = num_matches >= kMinPtsForPnP
        if is_pnp_good:
            is_pnp_good, R_vec, t, pnp_inliers_mask = cv2.solvePnPRansac(
                np.array(pts_3d), np.array(pts_2d), self.curr_.K, None,
                useExtrinsicGuess=False, iterationsCount=100, reprojectionError=2.0, confidence=0.999
            ) # reprojectionError=2.0, confidence=0.999

            R, _ = cv2.Rodrigues(R_vec)  # Convert rotation vector to matrix
            
            # -- Get inlier matches used in PnP
            tmp_pts_2d = []
            tmp_matches_with_map_ = []
            num_inliers = pnp_inliers_mask.shape[0]
            
            for i in range(num_inliers):
                good_idx = pnp_inliers_mask[i, 0]
                # good match
                match = self.curr_.matches_with_map_[good_idx]
                tmp_matches_with_map_.append(match)
                # good pts 2d
                tmp_pts_2d.append(pts_2d[good_idx])
                # good pts 3d
                inlier_mappoint = candidate_mappoints_in_map[match.queryIdx]
                inlier_mappoint.matched_times_ += 1
                # Update graph info
                self.curr_.inliers_to_mappt_connections_[match.trainIdx] = PtConn(-1, inlier_mappoint.id_)
            
            pts_2d = tmp_pts_2d
            self.curr_.matches_with_map_ = tmp_matches_with_map_
            # -- Update current camera pose
            self.curr_.T_w_c_ = np.linalg.inv(common.convert_rt_to_T(R, t))
            # -- Check relative motion with previous frame
            R_prev, t_prev = common.get_rt_from_T(self.prev_.T_w_c_)
            R_curr, t_curr = common.get_rt_from_T(self.curr_.T_w_c_)
            dist_to_prev_keyframe = np.linalg.norm(t_prev - t_curr)
            
            if params.display_pnp: 
                title = f"PnP Inlier Matches ref frame {self.ref_.frame_id} with {self.curr_.frame_id}"
                display.draw_matches(self.ref_.image, self.curr_.image, self.curr_.matches_with_map_, candidate_kpts, self.curr_.keypoints, title)
            if dist_to_prev_keyframe >= max_possible_dist_to_prev_keyframe:
                print(f"PnP: distance with prev keyframe is {dist_to_prev_keyframe:.3f}. Threshold is {max_possible_dist_to_prev_keyframe:.3f}.")
                is_pnp_good = False
        
        else:
            print(f"PnP num inlier matches: {num_matches}")
        
        if not is_pnp_good:
            self.curr_.T_w_c_ = self.prev_.T_w_c_.copy()
        
        return is_pnp_good

    def get_mappoints_in_current_view(self):
        """
        Retrieve map points that are visible in the current frame.
        """
        if not self.curr_:
            return [], [], None
        
        candidate_mappoints_in_map = []
        candidate_2d_pts_in_image = []
        corresponding_descriptors = []
        
        for mappt_id, mappoint in self.map_.map_points_.items():
            # Transform point to camera frame
            p_cam = common.pre_translate_point3f(mappoint.pos_, np.linalg.inv(self.curr_.T_w_c_))  # T_c_w * p_w = p_c
            if p_cam[2] < 0:
                continue
            
            pixel = camera.cam_to_pixel(p_cam, self.curr_.K)
            if not (0 <= pixel[0] < self.curr_.image.shape[1] and 0 <= pixel[1] < self.curr_.image.shape[0]):
                continue
            
            candidate_mappoints_in_map.append(mappoint)
            candidate_2d_pts_in_image.append(pixel)
            corresponding_descriptors.append(mappoint.descriptor_)
            mappoint.visible_times_ += 1
        
        corresponding_descriptors = np.array(corresponding_descriptors)
        return candidate_mappoints_in_map, candidate_2d_pts_in_image, corresponding_descriptors
    
    # Bundle Adjustment Methods
    def call_bundle_adjustment(self):
        """
        Perform bundle adjustment to optimize camera poses and map points.
        """
        # Read settings from config
        is_enable_ba = params.is_enable_ba
        num_prev_frames = params.num_prev_frames_to_opti_by_ba
        im = params.information_matrix  # Should return a list of 4 doubles
        information_matrix = np.array(im).reshape(2, 2)
        is_ba_fix_map_points = params.is_ba_fix_map_points
        is_ba_update_map_points = not is_ba_fix_map_points
        
        if not is_enable_ba:
            print("\nNot using bundle adjustment ...")
            return
        
        print(f"\nCalling bundle adjustment on {min(num_prev_frames, len(self.frames_buff_) - 1)} frames ...")
        
        # Prepare data for bundle adjustment
        v_pts_2d = []
        v_pts_2d_to_3d_idx = []
        um_pts_3d_in_prev_frames = {}
        v_pts_3d_only_in_curr = []
        v_camera_poses = []
        
        kTotalFrames = len(self.frames_buff_)
        kNumFramesForBA = min(num_prev_frames, kTotalFrames - 1)
        
        for ith_frame_in_buff in range(kTotalFrames - 1, kTotalFrames - kNumFramesForBA - 1, -1):
            frame = self.frames_buff_[ith_frame_in_buff]
            num_mappt = len(frame.inliers_to_mappt_connections_)
            if num_mappt < 3:
                continue
            print(f"Frame id: {frame.frame_id}, num map points = {num_mappt}")
            v_pts_2d.append([])
            v_pts_2d_to_3d_idx.append([])
            v_camera_poses.append(frame.T_w_c_)
            
            for kpt_idx, conn in frame.inliers_to_mappt_connections_.items():
                mappt_idx = conn.pt_map_idx #conn['pt_map_idx']
                if mappt_idx not in self.map_.map_points_:
                    continue
                v_pts_2d[-1].append(frame.keypoints[kpt_idx].pt)
                v_pts_2d_to_3d_idx[-1].append(mappt_idx)
                um_pts_3d_in_prev_frames[mappt_idx] = self.map_.map_points_[mappt_idx].pos_
                if ith_frame_in_buff == kTotalFrames - 1:
                    v_pts_3d_only_in_curr.append(self.map_.map_points_[mappt_idx].pos_)
        
        if not v_camera_poses:
            print("No frames available for bundle adjustment.")
            return
        
        # Perform bundle adjustment
        pose_src = common.get_pos_from_T(self.curr_.T_w_c_)
        optimization.bundle_adjustment(
            v_pts_2d, v_pts_2d_to_3d_idx, self.curr_.K,
            um_pts_3d_in_prev_frames, v_camera_poses,
            information_matrix,
            is_ba_fix_map_points, is_ba_update_map_points
        )
        
        # Print result
        pose_new = common.get_pos_from_T(self.curr_.T_w_c_)
        print(f"Cam pos: Before: {pose_src.flatten()}, After: {pose_new.flatten()}")
        print("Bundle adjustment finishes...\n")
    
    # -----------------------------------------------
    # ------------------- Mapping -------------------
    def add_key_frame(self, State: State):
        """
        Add a keyframe to the map.
        """
        self.map_.insert_key_frame(State)
        self.ref_ = State

    def optimize_map(self):
        """
        Optimize the map by removing unreliable map points.
        """
        default_erase = 0.1
        map_point_erase_ratio = default_erase
        curr_frame = self.curr_.frame_id
        to_erase = []
        not_in_map = []
        for map_id, mappoint in self.map_.map_points_.items():
            # if not self.curr_.is_in_frame(mappoint.pos_):
            #     to_erase.append(map_id)
            #     not_in_map.append(map_id)
            #     continue
            
            if mappoint.created_frame_ < curr_frame-20:
                to_erase.append(map_id)
                continue

            match_ratio = mappoint.matched_times_ / mappoint.visible_times_
            if match_ratio < map_point_erase_ratio:
                to_erase.append(map_id)
                continue
            
            # angle = self.get_view_angle(self.curr_, mappoint)
            # if angle > (np.pi / 4):
            #     to_erase.append(map_id)
            #     continue

        for map_id in to_erase:
            del self.map_.map_points_[map_id]
        
        if len(self.map_.map_points_) > 1000:
            map_point_erase_ratio += 0.05
        else:
            map_point_erase_ratio = default_erase
        
        print(f"Map points: {len(self.map_.map_points_)}")
    
    def push_curr_points_to_map(self):
        """
        Push current inlier points to the map.
        """
        if not self.curr_:
            return
        
        inliers_pts3d = self.curr_.inliers_pts3d_
        T_w_curr = self.curr_.T_w_c_
        descriptors = self.curr_.descriptors
        # kpts_colors = self.curr_.kpts_colors_
        matches_for_3d = self.curr_.inliers_matches_for_3d_
        connections = self.curr_.inliers_to_mappt_connections_
        frame_time = self.curr_.frame_id
        for i, match in enumerate(matches_for_3d):
            pt_idx = match.trainIdx
            mappt_id = -1
            
            if self.ref_ and self.ref_.is_mappoint(match.queryIdx):
                mappt_id = self.ref_.inliers_to_mappt_connections_[match.queryIdx].pt_map_idx
                if mappt_id in self.map_.map_points_:
                    mappoint = self.map_.map_points_[mappt_id]
                    # mappoint.descriptor_ = descriptors[pt_idx].copy()
                    mappoint.created_frame_ = frame_time
            else:
                world_pos = common.pre_translate_point3f(inliers_pts3d[i], T_w_curr)
                descriptor = descriptors[pt_idx].copy()
                # color = kpts_colors[pt_idx]
                mappoint = MapPoint(
                    pos=world_pos,
                    descriptor=descriptor,
                    norm=common.get_normalized_mat(world_pos - self.curr_.get_cam_center()),
                    frame_time=frame_time
                    # r=color[0],
                    # g=color[1],
                    # b=color[2]
                )
                self.map_.insert_map_point(mappoint)
                mappt_id = mappoint.id_
            
            # connections[pt_idx] = {'queryIdx': match.queryIdx, 'pt_map_idx': mappt_id} 
            connections[pt_idx] = PtConn(match.queryIdx, mappt_id) # queryIdx is the id of keypoint in the first picture

    def get_view_angle(self, state: State, point: MapPoint) -> float:
        """
        Calculate the view angle between the camera and the map point.
        """
        n = point.pos_ - state.get_cam_center()
        n = common.get_normalized_mat(n)
        dot_product = np.dot(n.flatten(), point.norm_.flatten())
        return np.arccos(dot_product)
    
    # ------------------- Utility Methods -------------------------- 
    def get_motion_from_frame1to2(self, frame1: State, frame2: State): 
        T_w_to_f1 = frame1.T_w_c_.copy()
        T_w_to_f2 = frame2.T_w_c_.copy()
        T_f1_to_f2 = np.linalg.inv(T_w_to_f1)@T_w_to_f2
        return T_f1_to_f2
