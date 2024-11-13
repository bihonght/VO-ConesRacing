import cv2, os
import numpy as np
from collections import deque
from enum import Enum
from typing import List, Optional

# Assuming the existence of these classes based on the C++ code
# You need to implement or import these appropriately
from vo.State import State

# import basics
from common import params, common
from geometry import motion_estimate, epipolar, feature_matching, camera
from optimization import optimization
from scipy.sparse import triu
from scipy.stats import chi2
from scipy.linalg import block_diag
from scipy.io import savemat

class EKF():

    def __init__(self):
        # Frame members
        self.curr_: Optional[State] = None
        self.prev_: Optional[State] = None

        # EKF SLAM initialization
        self.st_global_previous = np.zeros((3, 2))
        self.st_global = np.zeros((3, 2), dtype="float64")   # global landmark id start from 1 
        # st_global format [values, global_id] ---- first three elements are translation and yaw
        self.covariance = np.eye(3)*0.01
        self.covariance[2,2] /= 100  

        self.total_step = 0
        self.localmap_id = 1

        self.temporary_match = None

    def add_frame(self, frame):
        self.curr_ = frame
        # if len(self.curr_.cones_pairings_with_ref_) < 1 or self.total_step > 5: 
        #     localmap_P = self.covariance
        #     localmap_st = self.st_global[:, [1, 0]].copy()
        #     localmap_st[0:3, 0] = 0
        #     output_dir = "./localmap"
        #     os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        #     file_path = os.path.join(output_dir, f"localmap_{self.localmap_id}.mat")
        #     savemat(file_path, {'localmap_P': localmap_P,
        #                     'localmap_st': localmap_st
        #                     })
        #     self.st_global = np.zeros((3, 2), dtype="float64")
        #     self.covariance = np.eye(3)*0.01
        #     self.covariance[2,2] /= 100
        #     self.total_step = 0
        #     self.localmap_id += 1

        self.predict_step()
        match_local_map = self.data_association()
        self.curr_.matches_with_ekf_global_ = match_local_map.copy()

        self.update_step(match_local_map, self.curr_.matches_with_ekf_global_)
        
        self.curr_.global_pose_ = self.st_global.copy()
        self.prev_ = self.curr_
        self.total_step += 1

    def predict_step(self):
        # Update the state using odometry [tx, tz, yaw]
        odom = self.curr_.movement_and_observation_[:, :3]
        # Motion noise
        motion_noise = 0.01
        Q3 = np.array([
            [motion_noise, 0, 0],
            [0, motion_noise, 0],
            [0, 0, motion_noise / 100]
        ])
        # Odometry readings
        dx, dy, dtheta = odom[0, 0], odom[1, 0], odom[2, 0]
        theta = self.st_global[2, 0]
        # Sparsity matrix F
        F = np.zeros((3, self.st_global.shape[0]))
        F[:3, :3] = np.eye(3)
        # Update st_global position with odometry
        rotation_matrix = self.rotation(-self.st_global[2, 0])
        self.st_global[:2, 0] += rotation_matrix.T @ (np.array([dx, dy]))
        self.st_global[2, 0] = self.wrap_to_pi(self.st_global[2, 0] + dtheta)
        # Calculate Gx matrix
        Gx = np.array([
            [1, 0, -dx * np.sin(theta) - dy * np.cos(theta)],
            [0, 1,  dx * np.cos(theta) - dy * np.sin(theta)],
            [0, 0, 1]
        ])
        # Gx = csr_matrix(Gx)
        # Update covariance
        P_xx = self.covariance[:3, :3]
        P_xm = self.covariance[:3, 3:]
        self.covariance[:3, :3] = Gx @ P_xx @ Gx.T + Q3
        self.covariance[:3, 3:] = Gx @ P_xm
        self.covariance = (triu(self.covariance) + triu(self.covariance, 1).T).toarray()
        # For odometry
        curr_frame_id = self.curr_.frame_id
        self.st_global[:3, 1] = -curr_frame_id * np.ones((3))
        # return self.st_global[:2, 0]
    
    def update_step(self, match_local_map, match):
        localmap_st = self.curr_.movement_and_observation_[:, [0, 2]]
        n2 = localmap_st.shape[0]
        obs = localmap_st[3:n2,:]
        num_obs_beacon = int(obs.shape[0]//2)
        # match = match_local_map.copy()     # format (n, 2) where [local id, global beacon_id]
        last_landmark_id = int(len(self.st_global)-3)//2

        for i in range(num_obs_beacon):
            # Define the measurement noise covariance matrix W
            W = np.array([
                        [0.1**2, 0],
                        [0, 0.5**2]])

            # Check if the landmark has been observed before
            if match[i, 1] > 0:
                # Landmark has been observed before
                # Extract the beacon ID from Match.local
                beacon_id = match[i, 1]
                # Find the index in Est.st_global where the beacon ID matches
                # Assumption: Est.st_global has a specific structure where beacon IDs are stored
                # For example, if Est.st_global is a (n, 2) array where the second column contains IDs
                # Adjust this indexing based on your actual data structure
                indices = np.where(self.st_global[:, 1] == beacon_id)[0]
                if len(indices) == 0:
                    print(f"Warning: Beacon ID {beacon_id} not found in Est.st_global.")
                    continue  # Skip to the next iteration
                index = indices[0]  # Get the first occurrence
                # Extract the measurement Z (assumed to be a 2D vector)
                Z = obs[2*i:2*i+2, 0]  # Shape (2,)
                # Compute the difference between beacon position and robot position
                x_beac = self.st_global[index, 0]
                y_beac = self.st_global[index + 1, 0]
                x_robot = self.st_global[0, 0]
                y_robot = self.st_global[1, 0]
                dx = x_beac - x_robot
                dy = y_beac - y_robot
                # Extract the robot's orientation
                theta = self.st_global[2, 0]
                # Compute the expected measurement based on the current state
                expectedZ = self.rotation(theta).T @ np.array([dx, dy])  # Shape (2,)
                # Compute the Jacobian Hi of the measurement function h for this observation
                n_state = self.st_global.shape[0]
                Hi = np.zeros((2, n_state))

                Hi[0, 0] = -np.cos(theta)
                Hi[0, 1] = -np.sin(theta)
                Hi[0, 2] = -dx * np.sin(theta) + dy * np.cos(theta)
                Hi[0, index] = np.cos(theta)
                Hi[0, index + 1] = np.sin(theta)

                Hi[1, 0] = np.sin(theta)
                Hi[1, 1] = -np.cos(theta)
                Hi[1, 2] = -dx * np.cos(theta) - dy * np.sin(theta)
                Hi[1, index] = -np.sin(theta)
                Hi[1, index + 1] = np.cos(theta)

                # Compute the innovation covariance S
                S = Hi @ self.covariance @ Hi.T + W  # Shape (2, 2)
                # Compute the Kalman Gain Ki
                Ki = self.covariance @ Hi.T @ np.linalg.inv(S)  # Shape (n_state, 2)
                # Compute the difference between the actual and expected measurements
                z_diff = Z - expectedZ  # Shape (2,)
                # Update the state estimate
                self.st_global[:, 0] += Ki @ z_diff  # Shape (n_state,)
                # Update the covariance matrix
                self.covariance = (np.eye(n_state) - Ki @ Hi) @ self.covariance
                # Normalize the robot's orientation angle to [-pi, pi]
                self.st_global[2, 0] = self.wrap_to_pi(self.st_global[2, 0])

            elif match[i, 1] == -100:
                # Number of states before adding a new landmark
                nState = len(self.st_global)
                # Landmark is observed for the first time
                match[i, 1] = last_landmark_id + 1
                last_landmark_id += 1
                landmark_location = (self.st_global[:2, 0] + 
                    self.rotation(-self.st_global[2, 0]).T @ obs[2*i:2*i+2, 0])
                
                # Concatenate landmark_location and Match.local[i:i+2, :3] to Est.st_global
                landmark_location = landmark_location.reshape(-1, 1)  # Reshape for concatenation
                self.st_global = np.vstack([self.st_global, np.hstack([landmark_location, match[i, 1]*np.ones((2,1))])])
                indices = np.where(self.st_global[:, 1] == match[i, 1])[0]
                index = indices[0]
                dx = self.st_global[index, 0] - self.st_global[0, 0]
                dy = self.st_global[index + 1, 0] - self.st_global[1, 0]
                
                # Initialize Jacobians with zeros
                jGx = np.zeros((2, nState))
                jGz = np.zeros((2, 2))
                # Fill in values for jGx
                jGx[0, 0] = -np.cos(self.st_global[2, 0])
                jGx[0, 1] = -np.sin(self.st_global[2, 0])
                jGx[0, 2] = -dx * np.sin(self.st_global[2, 0]) + dy * np.cos(self.st_global[2, 0])
                jGx[1, 0] = np.sin(self.st_global[2, 0])
                jGx[1, 1] = -np.cos(self.st_global[2, 0])
                jGx[1, 2] = -dx * np.cos(self.st_global[2, 0]) - dy * np.sin(self.st_global[2, 0])
                # Fill in values for jGz
                jGz[0, 0] = np.cos(self.st_global[2, 0])
                jGz[0, 1] = np.sin(self.st_global[2, 0])
                jGz[1, 0] = -np.sin(self.st_global[2, 0])
                jGz[1, 1] = np.cos(self.st_global[2, 0])
                # Update covariance
                P = self.covariance
                self.covariance = np.block([
                    [P, P @ jGx.T],
                    [jGx @ P, jGx @ P @ jGx.T + jGz @ W @ jGz.T]
                ])

    def data_association(self):
        localmap_st = self.curr_.movement_and_observation_[:, [0, 2]]
        # localmap_P = Est.P
        n2 = localmap_st.shape[0]
        obs = localmap_st[3:n2,0]
        num_obs_beacon = int(obs.shape[0]//2)
        obscov = generate_covariance_matrix(num_obs_beacon, sigma_x=0.3, sigma_y=2) #0.1*np.eye(obs.shape[0])

        global_map_st = self.st_global
        global_map_st_cov = self.covariance
        num = global_map_st.shape[0]

        match_local_map = np.zeros((num_obs_beacon, 3), dtype="int32") # [index, beacon_id, local_id]
        if num == 3:
            # match_local_map = np.zeros((num_obs_beacon,2))
            for i in range(num_obs_beacon):
                local_id = int(localmap_st[2*i+3, 1])
                match_local_map[i, :] = np.array([i, -100, local_id])
        else: 
            beac,beaccov = self.trans_localmap_robot(global_map_st, global_map_st_cov)
            correspondance = self.joint_match_beacons_NN(beac, beaccov, obs, obscov)
            newcorrespondance = correspondance.copy()

            # Update the second column of correspondance based on Est_SelectedBeaconForDataAssociation
            for j in range(correspondance.shape[0]):
                beac_idx = correspondance[j, 1]
                newcorrespondance[j, 1] = beac_idx
                local_id = int(localmap_st[2*j+3, 1])
                match_local_map[j, 2] = local_id
            match_local_map[:, :2] = newcorrespondance[newcorrespondance[:, 0].argsort()]
        return match_local_map
    
    def trans_localmap_robot(self, slam_state, state_cov):
        if (slam_state.shape[0] - 3) % 2 != 0:
            raise ValueError("Invalid slam_state size. Must be 3 + 2*nmb_beac.")
        # Number of beacons
        nmb_beac = (slam_state.shape[0] - 3) // 2
        # Initialize beacon position vector and Jacobian matrix
        beac = np.zeros(2 * nmb_beac)  # 1D array for beacon positions
        JH = np.zeros((2 * nmb_beac, 2 * nmb_beac + 3))  # Jacobian matrix
        # Extract robot's orientation
        phi = slam_state[2, 0]
        # Precompute trigonometric functions for efficiency
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        # Transformation for each beacon
        for i in range(nmb_beac):
            # Calculate beacon indices
            beacon_x_idx = 2 * i + 3  # Python index for beacon_x
            beacon_y_idx = 2 * i + 4  # Python index for beacon_y
            # Ensure indices are within bounds
            if beacon_y_idx >= slam_state.shape[0]:
                raise IndexError(f"Beacon index out of bounds for beacon {i+1}.")
            # Compute differences
            dx = slam_state[beacon_x_idx, 0] - slam_state[0, 0]  # x difference
            dy = slam_state[beacon_y_idx, 0] - slam_state[1, 0]  # y difference
            # Transform beacon position
            transformed_x = cos_phi * dx + sin_phi * dy
            transformed_y = -sin_phi * dx + cos_phi * dy
            beac[2 * i : 2 * i + 2] = [transformed_x, transformed_y]
            # Populate Jacobian matrix JH for this beacon
            # Derivatives w.r.t robot's state [x, y, phi]
            if state_cov is not None:
                JH[2 * i : 2 * i + 2, 0:3] = [
                    [-cos_phi, -sin_phi, -sin_phi * dx + cos_phi * dy],
                    [ sin_phi, -cos_phi, -cos_phi * dx - sin_phi * dy]
                ]
                # Derivatives w.r.t beacon's state [x, y]
                JH[2 * i : 2 * i + 2, 3 + 2 * i : 5 + 2 * i] = [
                    [cos_phi, sin_phi],
                    [-sin_phi, cos_phi]
                ]
        # Calculate the transformed beacon covariance
        beaccov = JH @ state_cov @ JH.T 
        return beac, beaccov
        
    def joint_match_beacons_NN(self, beac, beac_cov, obs, obs_cov):
        # Start the main match code
        # Determine the number of observations and beacons
        nmb_obs = len(obs) // 2
        nmb_beac = len(beac) // 2
        # Initialize arrays to record matches
        poor_obs = []
        new_beacon = []
        nearest_match = []
        # Compute thresholds using problem-dependent parameters
        ConfidenceNN = params.ConfidenceNN #5
        ConfidenceNewBeacon = params.ConfidenceNewBeacon # 0.97
        maha_dist_threshold = np.sqrt(chi2.ppf(ConfidenceNN, df=2))
        maha_dist_threshold_for_new_beacon = np.sqrt(chi2.ppf(ConfidenceNewBeacon, df=2))

        dist_threshold = params.dist_threshold
        dist_threshold_for_match = params.dist_threshold_for_match #2.45 #1 
        dist_threshold_for_new_beacon = params.dist_threshold_for_new_beacon

        # Initialize distance matrices
        dist = np.zeros((nmb_obs, nmb_beac))
        mahadist = np.zeros((nmb_obs, nmb_beac))
        # Reshape observations and beacons for easier indexing
        obs_reshaped = obs.reshape(nmb_obs, 2)
        beac_reshaped = beac.reshape(nmb_beac, 2)
        # Reshape covariance matrices
        # Assuming covariance matrices are block diagonal with 2x2 blocks
        # Compute Euclidean and Mahalanobis distances
        for j in range(nmb_obs):
            x_obs, y_obs = obs_reshaped[j]
            cov_obs = obs_cov[j*2:(j+1)*2, j*2:(j+1)*2]
            for i in range(nmb_beac):
                x_beac, y_beac = beac_reshaped[i]
                cov_beac = beac_cov[i*2:(i+1)*2, i*2:(i+1)*2]
                # Innovation vector
                innov = np.array([x_beac - x_obs, y_beac - y_obs]).reshape(2,1)
                # Euclidean distance
                dist[j, i] = np.linalg.norm(innov)
                # Total covariance
                total_cov = cov_beac + cov_obs
                # Mahalanobis distance
                try:
                    mahalanobis_sq = innov.T @ np.linalg.inv(total_cov) @ innov
                    mahadist[j, i] = np.sqrt(mahalanobis_sq)
                except np.linalg.LinAlgError:
                    # Handle singular matrix
                    mahadist[j, i] = np.inf

        # Find minimal distances from beacons to observations
        min_maha_beac_to_obs = np.min(mahadist, axis=0)
        index_min_maha_beac_to_obs = np.argmin(mahadist, axis=0)
        min_dist_beac_to_obs = np.min(dist, axis=0)
        index_min_dist_beac_to_obs = np.argmin(dist, axis=0)
        # Find minimal distances from observations to beacons
        min_maha_obs_to_beac = np.min(mahadist, axis=1)
        index_min_maha_obs_to_beac = np.argmin(mahadist, axis=1)
        min_dist_obs_to_beac = np.min(dist, axis=1)
        index_min_dist_obs_to_beac = np.argmin(dist, axis=1)
        # Iterate over each observation to determine matches
        for j in range(nmb_obs):
            # Conditions for Case 1: Nearest Neighbor based on Mahalanobis and Euclidean distances
            cond1 = (
                min_maha_obs_to_beac[j] < maha_dist_threshold and
                min_dist_obs_to_beac[j] < dist_threshold and
                index_min_maha_obs_to_beac[j] == index_min_dist_obs_to_beac[j] and
                index_min_maha_beac_to_obs[index_min_maha_obs_to_beac[j]] == j and
                index_min_dist_beac_to_obs[index_min_maha_obs_to_beac[j]] == j
            )
            # Conditions for Case 2: Nearest Neighbor based on Euclidean distance
            cond2 = (
                not cond1 and
                min_dist_obs_to_beac[j] < dist_threshold_for_match and
                index_min_maha_obs_to_beac[j] == index_min_dist_obs_to_beac[j] and
                index_min_maha_beac_to_obs[index_min_maha_obs_to_beac[j]] == j and
                index_min_dist_beac_to_obs[index_min_maha_obs_to_beac[j]] == j
            )
            # Conditions for Case 3: New Beacon
            cond3 = (
                not cond1 and
                not cond2 and
                min_maha_obs_to_beac[j] > maha_dist_threshold_for_new_beacon and
                min_dist_obs_to_beac[j] > dist_threshold_for_new_beacon
            )

            if cond1 or cond2:
                # Valid nearest neighbor match
                beac_index = index_min_maha_obs_to_beac[j]
                # nearest_match.append([j + 1, beac_index + 1])  # +1 for 1-based indexing
                nearest_match.append([j, beac_index+1])  # +1 for 1-based indexing 
            elif cond3:
                # New beacon
                new_beacon.append([j, -100])  # -100 indicates a new beacon
            else:
                # Poor observation
                poor_obs.append([j, -1])  # -1 indicates a poor observation

        # Combine all matches into the match matrix
        match_matrix = []

        if nearest_match:
            match_matrix.extend(nearest_match)
        if new_beacon:
            match_matrix.extend(new_beacon)
        if poor_obs:
            match_matrix.extend(poor_obs)

        match_matrix = np.array(match_matrix)

        return match_matrix
    
    def rotation(self, theta):
        # Helper function to create a rotation matrix for a given angle theta
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
    
    def wrap_to_pi(self, angle):
        # Helper function to wrap an angle to the range [-pi, pi]
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def data_association_visual_matching(self): # data association using image processing (bbox matching)
        if len(self.st_global) > 3 and len(self.curr_.cones_pairings_with_ref_) > 0: 
            num_observations = int((len(self.curr_.movement_and_observation_)-3)//2)
            temporary_match = np.zeros((num_observations, 3), dtype="int32")
            cones_pairings_with_ref_ = np.array(self.curr_.cones_pairings_with_ref_)
            prev_matches_with_ekf_global_ = self.prev_.matches_with_ekf_global_
            for i in range(num_observations):
                curr_local_id = self.curr_.movement_and_observation_[2*i+3, 2]
                temporary_match[i, :] = np.array([i, -100, curr_local_id])
                # using data association from image matching
                prev_match_local_id = np.where(cones_pairings_with_ref_[:, 1] == curr_local_id)[0]
                if prev_match_local_id.size > 0:
                    prev_match_local_id = cones_pairings_with_ref_[prev_match_local_id[0], 0]
                    temp_global_id = np.where(prev_matches_with_ekf_global_[:, 2] == prev_match_local_id)[0]
                    if temp_global_id.size > 0:
                        temp_global_id = prev_matches_with_ekf_global_[temp_global_id[0], 1]
                        temporary_match[i, :] = np.array([i, temp_global_id, curr_local_id])
            self.temporary_match = temporary_match

    def extract_cones_pairing(self):
        # calculate cones' position in the previous camera frame and in the current camera frame
        temporary_match = self.temporary_match
        temporary_match = temporary_match[temporary_match[:, 1] >= 0] # only consider valid matches
        
        nmb_beac = (temporary_match.shape[0])
        # Initialize beacon position vector and Jacobian matrix
        beac = np.zeros(2 * nmb_beac)  # 1D array for beacon positions
        curr_obs = np.zeros(2 * nmb_beac)
        # Extract robot's orientation
        prev_slam_state = self.prev_.global_pose_
        phi = prev_slam_state[2, 0]
        # Precompute trigonometric functions for efficiency
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        # Transformation for each beacon
        for i in range(nmb_beac):
            # Extract beacon's position from the temporary match
            idx = np.where(prev_slam_state[:, 1] == temporary_match[i, 1])[0][0]
            # Calculate beacon indices
            beacon_x_idx = idx  # Python index for beacon_x
            beacon_y_idx = idx + 1  # Python index for beacon_y
            # Compute differences
            dx = prev_slam_state[beacon_x_idx, 0] - prev_slam_state[0, 0]  # x difference
            dy = prev_slam_state[beacon_y_idx, 0] - prev_slam_state[1, 0]  # y difference
            # Transform beacon position
            transformed_x = cos_phi * dx + sin_phi * dy
            transformed_y = -sin_phi * dx + cos_phi * dy
            beac[2 * i : 2 * i + 2] = [transformed_x, transformed_y]
            # 
            obs_idx = int(temporary_match[i, 0])
            curr_obs[2 * i : 2 * i + 2] = self.curr_.movement_and_observation_[2 * obs_idx + 3: 2 * obs_idx + 5, 0]
        return beac, curr_obs
    
def generate_covariance_matrix(orthogonal_length, sigma_x=0.3, sigma_y=2.0):
    single_cov = np.array([
        [sigma_x**2, 0],
        [0, sigma_y**2]
    ])
    
    # Create a list of identical covariance blocks
    cov_blocks = [single_cov for _ in range(orthogonal_length)]
    
    # Generate the block diagonal covariance matrix
    cov_matrix = block_diag(*cov_blocks)
    
    return cov_matrix

def iterative_svd_align_2d(P, Q, T_initial=np.zeros((3, 1)), max_iterations=50, tolerance=1):
    # Ensure input arrays are numpy arrays
    P = np.asarray(P)
    Q = np.asarray(Q)
    T_initial = np.asarray(T_initial)
    # Validate shapes of P and Q
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("Source points P must be a 2D array with shape (n, 2).")
    if Q.ndim != 2 or Q.shape[1] != 2:
        raise ValueError("Target points Q must be a 2D array with shape (n, 2).")
    if P.shape[0] != Q.shape[0]:
        raise ValueError("Source and target point sets must have the same number of points.")
    # Validate T_initial shape
    if T_initial.shape == (3, 1):
        T_initial = T_initial.flatten()  # Convert to 1D array
    elif T_initial.shape == (3,):
        pass  # Already 1D array
    else:
        raise ValueError("Initial transformation T_initial must be a 3-element array with shape (3,1) or (3,).")
    # Initialize transformation parameters
    tx, ty, theta = T_initial
    theta *= -1
    history = []
    for iteration in range(max_iterations):
        # Construct the rotation matrix from current theta
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])  # Shape: (2, 2)
        # Apply current transformation to P
        P_transformed = (R @ P.T).T + np.array([tx, ty])  # Shape: (n, 2)
        # Compute current alignment error (e.g., Root Mean Squared Error)
        error = np.sqrt(np.mean(np.sum((P_transformed - Q)**2, axis=1)))
        history.append(error)
        # Check for convergence
        if iteration > 0 and abs(history[-2] - history[-1]) < tolerance:
            print(f"Converged after {iteration} iterations.")
            break
        # Compute centroids of both point sets
        centroid_P = np.mean(P_transformed, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        # Center the points
        P_centered = P_transformed - centroid_P  # Shape: (n, 2)
        Q_centered = Q - centroid_Q              # Shape: (n, 2)
        # Compute the covariance matrix
        H = P_centered.T @ Q_centered  # Shape: (2, 2)
        # Perform SVD on the covariance matrix
        U, S, Vt = np.linalg.svd(H)
        # Compute rotation matrix
        R_new = Vt.T @ U.T
        # Handle reflection: ensure a proper rotation (determinant = 1)
        if np.linalg.det(R_new) < 0:
            Vt[1, :] *= -1
            R_new = Vt.T @ U.T
        # Compute the optimal translation
        t_new = centroid_Q - R_new @ centroid_P
        # Extract rotation angle from the new rotation matrix
        theta_new = np.arctan2(R_new[1, 0], R_new[0, 0])
        # Update transformation parameters
        tx, ty, theta = t_new[0], t_new[1], theta_new
    else:
        print(f"Reached maximum iterations ({max_iterations}) without full convergence.")

    return tx, ty, theta, history