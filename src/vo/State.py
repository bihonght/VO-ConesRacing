import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

from utils import * 
from common import params, common 
from geometry import feature_matching, camera
from detection import detection
from scipy.optimize import least_squares
from collections import namedtuple

PtConn = namedtuple('PtConn', ['pt_ref_idx', 'pt_map_idx']) # Define the PtConn namedtuple

class State(): 
    def __init__(self, model):
        self.dir = "dataset/"
        # self.dir = "data_stereo_amz/"
        # self.dir = "image_0/"
        self.model = model
        self.frame_id : int
        self.image: np.ndarray = None  # grayscale image of the current frame

        self.K = params.K # intrinsic matrix
        self.cones: np.ndarray = None
        self.cones3D: np.ndarray = None

        if self.image is not None: 
            self.estimate_cones_location()
        
        self.kpoints: np.ndarray = None
        self.keypoints: List[cv2.KeyPoint] = []  # 2D keypoints within camera image
        self.descriptors: np.ndarray = None  # keypoint descriptors
        self.matches_with_ref_: List[cv2.DMatch] = []  # keypoint matches between previous and current frame
        self.inliers_matches_with_ref_ = []  # Inliers after E or H constraints
        self.matches_gms_with_ref_: List[cv2.DMatch] = [] 
        self.cones_pairings_with_ref_ = [] # The 1st column is the index of the reference cone, the second column is the index of the current cone
        # -- Vectors for triangulation
        self.triangulation_angles_of_inliers_ = []  # Triangulation angles
        self.inliers_matches_for_3d_ = []  # Good triangulation results after retain funtion
        self.inliers_pts3d_ = []  # 3D points from triangulation
        self.inliers_to_mappt_connections_ = {}  # Unordered map for point connections // matches between inliers points and 3D map points in the world map
        self.inliers_matches_for_cones_ = [] 
        # -- Camera
        self.camera_ = None
        # -- Current pose and odometry 
        self.movement_and_observation_ = []  # Movement and observation vectors
        self.T_w_c_ = None  # Transform from world to camera (from )
        self.R_curr_to_prev_: np.ndarray = None
        self.t_curr_to_prev_: np.ndarray = None
        self.Rt: np.ndarray = None
        # -- EKF state updates
        self.global_pose_ = None  # Current pose (in camera coordinates   (x, y, theta)
        self.matches_with_ekf_global_: np.ndarray = None  # Matches for global map points from EKF

    def read_image(self, frame_id):
        dir = self.dir  # Assuming the dataset is in the same directory as the code. Adjust as needed.  
        self.frame_id = frame_id
        filename = os.path.join(dir, f"amz_{self.frame_id:03d}.jpg")
        # filename = os.path.join(dir, f"{self.frame_id:06d}.png")
        if os.path.exists(filename):
            # image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(filename)
            self.image = image
        else:
            print("Could not find image file ", filename)
            self.image = None
    
    def read_txt(self):
        dir = self.dir  # Assuming the dataset is in the same directory as the code. Adjust as needed.
        filetxt = os.path.join(dir, f"amz_{self.frame_id:03d}.txt")
        if os.path.exists(filetxt):
            # Read the file and process the lines
            with open(filetxt, 'r') as file:
                blue_pixels = np.array(file.readline().strip().split(), dtype=int).reshape(-1, 4)
                yellow_pixels = np.array(file.readline().strip().split(), dtype=int).reshape(-1, 4)
            blue, yellow = detection.bbox_detection(self.image, self.model)
            blue_pixels = common.filter_out(blue_pixels, -1)     # -1 denotes blue cones 
            yellow_pixels = common.filter_out(yellow_pixels, -2)   # -2 denotes yellow cones
            # Create the interleaved array directly without initializing an empty array
            combined = np.vstack((blue_pixels, yellow_pixels))
            self.blue_count = len(blue_pixels)
            self.yellow_count = len(yellow_pixels)
            self.num_cones = len(combined)
            indices = np.arange(combined.shape[0]).reshape(-1, 1)
            self.cones = np.hstack([combined, indices])
        else:
            print("Could not find txt file ", filetxt)
            return None
    
    def cone_detection(self):
        blue, yellow = detection.bbox_detection(self.image, self.model)
        blue_pixels = common.filter_out(blue, -1)     # -1 denotes blue cones 
        yellow_pixels = common.filter_out(yellow, -2)   # -2 denotes yellow cones
        combined = np.vstack((blue_pixels, yellow_pixels))
        self.blue_count = len(blue_pixels)
        self.yellow_count = len(yellow_pixels)
        self.num_cones = len(combined)
        indices = np.arange(combined.shape[0]).reshape(-1, 1)
        self.cones = np.hstack([combined, indices])

    def estimate_cones_location(self):
        self.cones3D = np.zeros([self.num_cones, 5])
        # [-0.1129, 0, 0.1129], [0.1129, 0, 0.1129], 
        object_points = np.array([[-0.1129, 0, 0],
                                [0.1129, 0, 0], 
                                [0, 0.325, 0]])
        dist_coeffs = np.zeros((4, 1))
        img_points_proj = np.zeros([3*self.num_cones, 2])

        for i in range(0, self.num_cones):
            u1, v1, u2, v2 = self.cones[i,:4]
            point = np.array([[u1, v2], 
                            [u2, v2], 
                            [(u1+u2)//2, v1]], dtype=np.float64)
            success, rvec, tvec = cv2.solvePnP(object_points, point, self.K, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
            if success:
                self.cones3D[i,0:3] = tvec.T 
                self.cones3D[i,3] = self.cones[i,4] # color and id 
            else:
                print("SolvePnP failed to find a solution.")
        self.cones3D[:,4] = self.cones[:,5] # local id 
    def detect_keypoints_and_descriptors(self, extractor_type):
        mask = np.zeros_like(self.image)
        mask[20:600, :] = 255     #75
        mask[465:800, 550:1250] = 0
        # mask[10:500, :] = 255     #75
        # mask[313:, 20:840] = 0
        # mask[265:, 320:460] = 0
        masked_gray = cv2.bitwise_and(self.image, mask)
        if extractor_type == params.FEATURE_FAST:
            # FAST algorithm
            fast_threshold = 20
            nonmax_suppression = True
            fast = cv2.FastFeatureDetector_create(threshold=fast_threshold, nonmaxSuppression=nonmax_suppression)
            keypoints = fast.detect(masked_gray, None)
        
        elif extractor_type == params.FEATURE_ORB:
            orb = cv2.ORB_create(10000)
            keypoints, descriptors = orb.detectAndCompute(masked_gray, None)

        elif extractor_type == params.FEATURE_ORB_NEW:
            print("doing new feature detection", params.FEATURE_ORB_NEW)
            keypoints = feature_matching.calcKeyPoints(masked_gray)
            keypoints, descriptors = feature_matching.calcDescriptors(masked_gray, keypoints)

        self.kpoints = cv2.KeyPoint_convert(keypoints)
        self.keypoints = keypoints
        self.descriptors = descriptors

    def project_world_point_to_image(self, p_world: np.ndarray) -> np.ndarray:
        """Projects a 3D world point to 2D image coordinates."""
        p_cam = common.pre_translate_point3f(p_world, np.linalg.inv(self.T_w_c_))
        pixel = camera.cam_to_pixel(p_cam, self.K)
        return pixel

    def is_in_frame(self, p_world: np.ndarray) -> bool:
        """Checks if a 3D point is in the frame."""
        p_cam = common.pre_translate_point3f(p_world, np.linalg.inv(self.T_w_c_))
        if p_cam[2] < 0:
            return False
        pixel = camera.cam_to_pixel(p_cam, self.K)
        return (0 < pixel[0] < self.image.shape[1]) and (0 < pixel[1] < self.image.shape[0])

    def is_mappoint(self, idx: int) -> bool:
        """Checks if the index corresponds to a map point."""
        return idx in self.inliers_to_mappt_connections_

    def get_cam_center(self) -> np.ndarray:
        """Gets the camera center from the current pose."""
        return common.get_pos_from_T(self.T_w_c_)

    def insert_inliers_to_mappt_connections(self, pt_idx, pt_ref_idx, pt_map_idx):
        self.inliers_to_mappt_connections_[pt_idx] = PtConn(pt_ref_idx, pt_map_idx)

    def update_cones_3D(self):
        """Updates the 3D coordinates of the map points."""
        for idx, conn in self.inliers_to_mappt_connections_.items():
            mappt_idx = conn.pt_map_idx
            mappt = self.map_.map_points_[mappt_idx]
            mappt.pos_ = self.cones3D[idx]
