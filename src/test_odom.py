import os 
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
# from utils import * 

from vo.State import State
from odometry.odom_addframe import Odometry
from common import params
from geometry import motion_estimate, epipolar, feature_matching
import display.display as display
from ultralytics import YOLO

def is_rotation_matrix(R):
    Rt = np.transpose(R)
    should_be_identity = np.dot(Rt, R)
    I = np.eye(3)
    return np.linalg.norm(I - should_be_identity) < 1e-6

def main():
    frame_count = 1
    K = params.K2
    camera_pose = np.eye(4)
    start = time.process_time()

    yolo_model = YOLO('model/weights/best.pt')
    prev_state = State(yolo_model)
    H_count = 0
    TOTAL_FRAME_COUNT = 160
    Vo = Odometry()
    fig_cones, ax_cones, sc_cones = display.init_cone_plot()
    fig_traj, ax_traj, sc_traj = display.init_trajectory_plot()
    x_coord = []
    z_coord = []
    # Vo.vo_state_ = Vo.VOState.DOING_INITIALIZATION

    for frame_count in range(1, TOTAL_FRAME_COUNT):
        print("FRAME_COUNT : " + str(frame_count))
        if frame_count == 159: 
            break
        curr_state = State(yolo_model) 
        curr_state.read_image(frame_count)
        curr_state.cone_detection()
        curr_state.estimate_cones_location()
        if curr_state.image is None:
            raise Exception("An error occurred.")
        
        
        Vo.add_frame(curr_state)
        # if Vo.vo_state_ == 2: 
        #     break

        new_pose = np.column_stack((curr_state.T_w_c_))
        # camera_pose = camera_pose @ (Vo.curr_.T_w_c_)
        camera_pose = Vo.curr_.T_w_c_
        x_coord.append(camera_pose[0, -1])
        z_coord.append(-camera_pose[2, -1])
        display.update_trajectory_plot(fig_traj, ax_traj, sc_traj, x_coord, z_coord)
        # plt.scatter(x_coord, -z_coord, color='b') 
        plt.pause(0.00001)
        frm = cv2.resize(curr_state.image, (0,0), fx=0.5, fy=0.5)
        cv2.imshow('Frame', frm)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

        # Update the previous state
        # prev_state = curr_state
    
    cv2.destroyAllWindows()
    plt.show()
    end = time.process_time() - start
    print("Time taken: ", end)


if __name__ == "__main__":
    main()