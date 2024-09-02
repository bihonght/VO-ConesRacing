import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

from utils import * 
from scipy.optimize import least_squares

class State(): 
    def __init__(self, frame_id, K):
        dir = "../dataset/"
        self.frame_id =  frame_id
        self.K = K # intrinsic matrix
        self.cones = None
        self.cones3D = None

        self.read_image(dir)
        self.read_txt(dir)
        if self.image is not None: 
            self.estimate_cones_location()
        
        self.keypoints: List[cv2.KeyPoint] = []  # 2D keypoints within camera image
        self.descriptors: np.ndarray = None  # keypoint descriptors
        self.kptMatches: List[cv2.DMatch] = []  # keypoint matches between previous and current frame

        self.kptsAbove300: List[cv2.KeyPoint] = []
        self.desAbove300: np.ndarray = None

        self.matched_pixels: List[np.array] = []
        self.matched_3Dpoints: List[np.array] = []
    def read_image(self, dir):
        filename = os.path.join(dir, f"amz_{self.frame_id:03d}.jpg")
        if os.path.exists(filename):
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            self.image = image
        else:
            print("Could not find image file ", filename)
            self.image = None
    
    def read_txt(self, dir):
        filetxt = os.path.join(dir, f"amz_{self.frame_id:03d}.txt")
        if os.path.exists(filetxt):
            # Read the file and process the lines
            with open(filetxt, 'r') as file:
                blue_pixels = np.array(file.readline().strip().split(), dtype=int).reshape(-1, 4)
                yellow_pixels = np.array(file.readline().strip().split(), dtype=int).reshape(-1, 4)
            blue_pixels = filter_out(blue_pixels, -1)     # -1 denotes blue cones 
            yellow_pixels = filter_out(yellow_pixels, -2)   # -2 denotes yellow cones
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

        self.cone_3Dpoints = np.empty([self.num_cones, 11]) 
        self.cones_3pixels = np.empty([self.num_cones, 8])
        object_points = np.array([[-0.1129, 0, 0.1129],
                                [0.1129, 0, 0.1129], 
                                [0, 0.325, 0]])
        dist_coeffs = np.zeros((4, 1))
        img_points_proj = np.zeros([3*self.num_cones, 2])

        for i in range(0, self.num_cones):
            u1, v1, u2, v2 = self.cones[i,:4]
            point = np.array([[u1, v2], 
                            [u2, v2], 
                            [(u1+u2)//2, v1]], dtype=np.float64)
            success, rvec, tvec = cv2.solvePnP(object_points, point, self.K, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
            self.cones_3pixels[i, :6] = point.reshape(1, -1)
            self.cones_3pixels[i, 6:] = self.cones[i,4:5]
            if success:
                self.cones3D[i,0:3] = tvec.T 
                self.cones3D[i,3] = self.cones[i,4]
                R, _ = cv2.Rodrigues(rvec)
                M_i = np.hstack([R, tvec])
                img_points_proj[3*i:3*i+3, :] = reprojectPoints(object_points, M_i, self.K) 
                self.cone_3Dpoints[i,:9] = cone3D_transform(object_points, M_i).reshape(1, -1)
                self.cone_3Dpoints[i,9:10] = self.cones[i,4:5]
                # print(f"Error {i}: \n", reprojectPoints(object_points, M_i, K) - point)
                # print(self.cone_3Dpoints[i,:9].reshape(-1, 3))
                # M_est = [elem for elem in M_i.reshape(-1)]

                M = least_squares(pixel_reproject_err, M_i.ravel(), args = (object_points, self.K, point), max_nfev = 100) 
                self.cones3D_op[i,0:3] = M.x[3::4]
                # print(f"Error {i}: \n", reprojectPoints(object_points, M.x.reshape(3,4), K) - point)
                # print(self.cones3D[i,:])
                # print("\n")
            else:
                print("SolvePnP failed to find a solution.")
            
    # def __call__(self):
    #     print(self.cones)
    #     print(self.num_cones)
    #     plt.imshow(self.image, cmap="gray")
    #     M1 = np.hstack((np.eye(3), np.zeros((3,1))))
    #     reprojected = reprojectPoints(self.cone_3Dpoints[:,:9].reshape(-1,3), M1, self.K)
    #     plt.scatter(reprojected[:,0], reprojected[:,1], marker="x", color="blue", s=8)
    #     # plt.scatter(self.cones[:self.blue_count, [0, 2]].flatten(), self.cones[:self.blue_count, [1, 3]].flatten(), marker="x", color="blue", s=8)
    #     # plt.scatter(self.cones[self.blue_count:, 2], self.cones[self.blue_count:, 3], marker="+", color="yellow", s=8)
    #     plt.show()      
    #     plt.scatter(self.cones3D_op[:, 0], self.cones3D_op[:, 2], marker="o", color='r')  # Plotting in 2D (X vs Z)
        
    #     plt.scatter(self.cones3D[:, 0], self.cones3D[:, 2], marker="o", color='g', s=16)  # Plotting in 2D (X vs Z)
    #     # Draw X and Z axis arrows
    #     plt.xlim(-20, 20)
    #     plt.ylim(0, 20)
    #     plt.arrow(0, 0, 0.5, 0, head_width=0.3, head_length=1, fc='blue', ec='blue')  # X-axis
    #     plt.arrow(0, 0, 0, 0.5, head_width=0.3, head_length=1, fc='green', ec='green')  # Z-axis
    #     plt.grid(True)
    #     plt.pause(0.1)  # Pause to create a sequence effect
    #     plt.show()

def pixel_reproject_err(M, points_3D, K, points_2D): 
    M = M.reshape(3, 4)
    projected_points = reprojectPoints(points_3D, M, K) 
    err = projected_points - points_2D
    return err.ravel()

# K = np.loadtxt("K_matrix.txt")
# robot_state = State(24, K)
# robot_state()