import os 
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
# from utils import * 

from vo.State import State
from vo.vo_addframe import VisualOdometry
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
    Vo = VisualOdometry()
    print(os.getcwd())

    for frame_count in range(1, TOTAL_FRAME_COUNT):
        print("FRAME_COUNT : " + str(frame_count))
        if frame_count == 10: 
            break
        curr_state = State() 
        curr_state.read_image(frame_count)
        if curr_state.image is None:
            raise Exception("An error occurred.")
        
        
        Vo.add_frame(curr_state)
        if Vo.vo_state_ == 2: 
            break
        # prev_state.kpoints, curr_state.kpoints = good_matches_features(prev_state.kpoints, curr_state.kpoints, mask)

        new_pose = np.column_stack((curr_state.T_w_c_))
        # new_pose = np.vstack((new_pose, np.array([0,0,0,1])))
        camera_pose = camera_pose @ (new_pose)
        x_coord = camera_pose[0, -1]
        z_coord = camera_pose[2, -1]
        plt.scatter(x_coord, -z_coord, color='b') 
        plt.pause(0.00001)
        # frm = cv2.resize(prev_state.image, (0,0), fx=0.5, fy=0.5)
        # cv2.imshow('Frame', frm)

        # Update the previous state
        prev_state = curr_state
    
    print(H_count)
    cv2.destroyAllWindows()
    plt.show()
    end = time.process_time() - start
    print("Time taken: ", end)


if __name__ == "__main__":
    main()