import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

from utils import * 
from common import params, common 
from geometry import feature_matching, camera
from scipy.optimize import least_squares
from collections import namedtuple

PtConn = namedtuple('PtConn', ['pt_ref_idx', 'pt_map_idx']) # Define the PtConn namedtuple

class State(): 
    def __init__(self):
        self.dir = "dataset/"
        # self.dir = "image_0/"
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

        # -- Vectors for triangulation
        self.triangulation_angles_of_inliers_ = []  # Triangulation angles
        self.inliers_matches_for_3d_ = []  # Good triangulation results after retain funtion
        self.inliers_pts3d_ = []  # 3D points from triangulation
        self.inliers_to_mappt_connections_ = {}  # Unordered map for point connections // matches between inliers points and 3D map points in the world map

        # -- Matches with map points (for PnP)
        self.matches_with_map_ = []  # Inliers for map points
        # -- Camera
        self.camera_ = None
        # -- Current pose
        self.T_w_c_ = None  # Transform from world to camera (from )

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
    
    def estimate_cones_location(self):
        self.cones3D = np.zeros([self.num_cones, 5])
        self.cones3D_op = np.zeros([self.num_cones, 5])

        # self.cone_3Dpoints = np.empty([self.num_cones, 11]) 
        # self.cones_3pixels = np.empty([self.num_cones, 8])
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
            # self.cones_3pixels[i, :6] = point.reshape(1, -1)
            # self.cones_3pixels[i, 6:] = self.cones[i,4:5]
            if success:
                self.cones3D[i,0:3] = tvec.T 
                self.cones3D[i,3] = self.cones[i,4]
                # R, _ = cv2.Rodrigues(rvec)
                # M_i = np.hstack([R, tvec])
                # img_points_proj[3*i:3*i+3, :] = reprojectPoints(object_points, M_i, self.K) 
                # self.cone_3Dpoints[i,:9] = cone3D_transform(object_points, M_i).reshape(1, -1)
                # self.cone_3Dpoints[i,9:10] = self.cones[i,4:5]

                # M = least_squares(pixel_reproject_err, M_i.ravel(), args = (object_points, self.K, point), max_nfev = 100) 
                # self.cones3D_op[i,0:3] = M.x[3::4]

            else:
                print("SolvePnP failed to find a solution.")

    def detect_keypoints_and_descriptors(self, extractor_type):
        mask = np.zeros_like(self.image)
        mask[20:600, :] = 255     #75
        mask[465:800, 550:1250] = 0
        masked_gray = cv2.bitwise_and(self.image, mask)
        # masked_gray = self.image

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

    def optimized_grid_based_orb_extraction(self, grid_size=(4, 4), max_features_per_cell=20):
        mask = np.zeros_like(self.image)
        mask[25:600, :] = 255 # 75:600
        mask[465:800, 550:1250] = 0
        masked_gray = cv2.bitwise_and(self.image, mask)
        height, width = mask.shape[:2]
        h_step = height // grid_size[0]
        w_step = width // grid_size[1]
        orb = cv2.ORB_create(nfeatures=max_features_per_cell * grid_size[0] * grid_size[1])
        # num_keypoints = params.max_number_keypoints
        # scale_factor = params.scale_factor
        # level_pyramid = params.level_pyramid
        # score_threshold = params.score_threshold
        # orb = cv2.ORB_create(nfeatures=max_features_per_cell, scaleFactor=scale_factor, nlevels=level_pyramid, 
        #                  edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, 
        #                  patchSize=31, fastThreshold=score_threshold)
        keypoints_all = []
        descriptors_all = []

        # Loop over each cell in the grid
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                # Define the cell boundaries
                x_start, x_end = j * w_step, (j + 1) * w_step
                y_start, y_end = i * h_step, (i + 1) * h_step
                # Extract the region of interest (ROI) or cell from the image
                cell = masked_gray[y_start:y_end, x_start:x_end]
                # Detect ORB keypoints and descriptors in this cell
                keypoints, descriptors = orb.detectAndCompute(cell, None)
                
                if keypoints:
                    # Adjust the keypoints to the original image coordinates using NumPy
                    for kp in keypoints:
                        kp.pt = (kp.pt[0] + x_start, kp.pt[1] + y_start)
                    keypoints_all.extend(keypoints)

                    # Efficiently append descriptors
                    if descriptors is not None:
                        descriptors_all.append(descriptors)

        # Convert list of descriptors to a NumPy array if non-empty
        if descriptors_all:
            descriptors_all = np.vstack(descriptors_all)
        else:
            descriptors_all = None
        self.keypoints = keypoints_all
        self.descriptors = descriptors_all
        # return keypoints_all, descriptors_all

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


''' testing draft
def pixel_reproject_err(M, points_3D, K, points_2D): 
    M = M.reshape(3, 4)
    projected_points = reprojectPoints(points_3D, M, K) 
    err = projected_points - points_2D
    return err.ravel()

    def __call__(self):
        print(self.cones)
        print(self.num_cones)
        plt.imshow(self.image, cmap="gray")
        M1 = np.hstack((np.eye(3), np.zeros((3,1))))
        reprojected = reprojectPoints(self.cone_3Dpoints[:,:9].reshape(-1,3), M1, self.K)
        plt.scatter(reprojected[:,0], reprojected[:,1], marker="x", color="blue", s=8)
        # plt.scatter(self.cones[:self.blue_count, [0, 2]].flatten(), self.cones[:self.blue_count, [1, 3]].flatten(), marker="x", color="blue", s=8)
        # plt.scatter(self.cones[self.blue_count:, 2], self.cones[self.blue_count:, 3], marker="+", color="yellow", s=8)
        plt.show()      
        plt.scatter(self.cones3D_op[:, 0], self.cones3D_op[:, 2], marker="o", color='r')  # Plotting in 2D (X vs Z)
        plt.scatter(self.cones3D[:, 0], self.cones3D[:, 2], marker="o", color='g', s=16)  # Plotting in 2D (X vs Z)
        # Draw X and Z axis arrows
        plt.xlim(-20, 20)
        plt.ylim(0, 20)
        plt.arrow(0, 0, 0.5, 0, head_width=0.3, head_length=1, fc='blue', ec='blue')  # X-axis
        plt.arrow(0, 0, 0, 0.5, head_width=0.3, head_length=1, fc='green', ec='green')  # Z-axis
        plt.grid(True)
        plt.pause(0.1)  # Pause to create a sequence effect
        plt.show()
'''