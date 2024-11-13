import numpy as np
import cv2
def reprojectPoints(P, M_tilde, K):
    # Reproject 3D points given a projection matrix
    #
    # P         [n x 3] coordinates of the 3d points in the world frame
    # M_tilde   [3 x 4] projection matrix
    # K         [3 x 3] camera matrix
    #
    # Returns [n x 2] coordinates of the reprojected 2d points
    P_homo = np.concatenate((P, np.ones((P.shape[0], 1))), axis=-1) 
    p_homo = (K@M_tilde@P_homo.T).T
    p = p_homo[:, :2] / p_homo[:, -1][:, np.newaxis]
    return p

def cone3D_transform(P, M_tilde): 
    """
    Transforms 3D points given a transformation matrix [R|T]
    P         [n x 3] coordinates of the 3d points in the world frame
    M_tilde   [3 x 4] matrix between the cone frame and the camera frame
    Returns [n x 3] coordinates of cones in the camera frame
    """
    P_homo = np.concatenate((P, np.ones((P.shape[0], 1))), axis=-1) 
    return (M_tilde@P_homo.T).T

def filter_out(cones_pixels, color_value): 
    cone_boxes = np.array([[u1, v1, u2, v2, color_value] for u1, v1, u2, v2 in cones_pixels
                        if np.abs(v1 - v2) <= 1.55 * np.abs(u1 - u2)], dtype=int)
    if len(cone_boxes) == 0: 
        return np.empty((0,5), dtype=int)
    return cone_boxes

def point_in_box(x, y, box):
    return box[0]-10 <= x <= box[2]+10 and box[1]-10 <= y <= box[3]+10

def get_linear_triangulated_points(pose, point_list1, point_list2):
    P = np.eye(3,4)
    P_dash = pose
    points_3D = []
    num_points = len(point_list1)
    for i in range(num_points):
        point1 = point_list1[i]
        point2 = point_list2[i]
        '''
        A = np.array([
            (point1[Y] * P[ROW3]) - P[ROW2],
            P[ROW1] - (point1[X]*P[ROW3]),
            (point2[Y] * P_dash[ROW3]) - P_dash[ROW2],
            P_dash[ROW1] - (point2[X] * P_dash[ROW3])
        ])
        '''
        point1_cross = np.array([
            [0, -point1[2], point1[1]],
            [point1[2], 0, -point1[0]],
            [-point1[1], point1[0], 0]
        ])

        point2_cross = np.array([
            [0, -point2[2], point2[1]],
            [point2[2], 0, -point2[0]],
            [-point2[1], point2[0], 0]
        ])

        point1_cross_P = point1_cross @ P
        point2_cross_P_dash = point2_cross @ P_dash

        A = np.vstack((point1_cross_P, point2_cross_P_dash))

        _, _, VT = np.linalg.svd(A)
        solution = VT.T[:, -1]
        solution /= solution[-1]

        points_3D.append([solution[0], solution[1], solution[2]])
        #yield [solution[X], solution[Y], solution[Z]]        
    return points_3D
