import numpy as np
import cv2
from typing import Union
from scipy.io import savemat

def inv_rt(R, t):
    T = convert_rt_to_T(R, t)
    R_inv, t_inv = get_rt_from_T(np.linalg.inv(T))
    return R_inv, t_inv

def convert_rt_to_T(R, t):
    T = np.array([[R[0, 0], R[0, 1], R[0, 2], t[0, 0]],
                  [R[1, 0], R[1, 1], R[1, 2], t[1, 0]],
                  [R[2, 0], R[2, 1], R[2, 2], t[2, 0]],
                  [0, 0, 0, 1]], dtype=np.float64)
    return T

def get_rt_from_T(T):
    R = T[:3, :3].copy()  # Extract the 3x3 rotation matrix
    t = T[:3, 3].reshape(3, 1).copy()  # Extract the 3x1 translation vector
    return R, t


def get_pos_from_T(T: np.ndarray) -> np.ndarray:
    """
    Extracts the position vector from a 4x4 transformation matrix.
    """
    if T.shape != (4, 4):
        raise ValueError("Input matrix T must be a 4x4 matrix.")
    # Extract the last column (indices 0:3, 3)
    pos = T[0:3, 3].copy()
    # Ensure it's a column vector with shape (3, 1)
    pos = pos.reshape(3, 1)
    return pos

def calc_dist(p1, p2):
    """
    Calculate the Euclidean distance between two 2D points.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return np.sqrt(dx * dx + dy * dy)

def calc_mean_depth(pts_3d):
    """
    Calculate the mean depth (z-value) of a list of 3D points.
    """
    if not pts_3d:
        return 0
    mean_depth = sum(p[2] for p in pts_3d) / len(pts_3d)
    return mean_depth

# def scale_point_pos(p, scale):
#     p[0] *= scale
#     p[1] *= scale
#     p[2] *= scale
#     return p

def scale_point_pos(p, scale):
    p = p.flatten()  # Ensure p is in shape (3,)
    p[0] *= scale
    p[1] *= scale
    p[2] *= scale
    return p.reshape(-1, 1) if p.ndim == 1 else p  # Return in (3,1) if it was originally

def point3f_to_mat3x1(p):
    if p.shape == (3,1):
        return p
    return np.array([[p[0]], [p[1]], [p[2]]], dtype=float)

def pre_translate_point3f(p3x1, T4x4):
    """
    Transforms a 3D point using a 4x4 transformation matrix (translation and rotation).
    """
    # Convert the point to homogeneous coordinates
    p_homogeneous = np.vstack((p3x1, [[1]]))
    # Apply the transformation matrix (T4x4) to the homogeneous point
    res_homogeneous = np.dot(T4x4, p_homogeneous)
    # Return the first three components (x, y, z) as a 3x1 NumPy array
    return res_homogeneous[:3]

def trans_coord(p: Union[np.ndarray, list, tuple], R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Transforms a 3D point from one coordinate system to another using rotation and translation.
    Parameters:
        p (np.ndarray or list or tuple): The 3D point to transform. Shape should be (3,) or (3, 1).
        R (np.ndarray): The 3x3 rotation matrix.
        t (np.ndarray): The 3x1 translation vector.
    Returns:
        np.ndarray: The transformed 3D point. Shape is (3,).
    """
    # Convert input point to a NumPy array if it's not already
    p = np.asarray(p, dtype=np.float64).reshape(3, 1)  # Ensure shape is (3, 1)

    # Validate input shapes
    if R.shape != (3, 3):
        raise ValueError(f"Rotation matrix R must be of shape (3, 3), but got {R.shape}")
    if t.shape not in [(3, 1), (3,)]:
        raise ValueError(f"Translation vector t must be of shape (3, 1) or (3,), but got {t.shape}")
    # Ensure t is a column vector
    t = t.reshape(3, 1)
    # Apply the transformation: p2 = R * p + t
    p2 = R @ p + t  # Matrix multiplication
    # Flatten the result to shape (3,)
    p2_flat = p2.flatten()
    # p2_flat = p2.reshape(3, 1)
    return p2_flat

def pts2keypts(pts):
    """
    Converts a list of 2D points into a list of OpenCV KeyPoint objects.
    Args:
        pts (list of tuple or np.ndarray): A list of 2D points (x, y).
    Returns:
        list of cv2.KeyPoint: A list of corresponding KeyPoint objects.
    """
    keypts = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 10) for pt in pts]
    return keypts

def get_normalized_mat(mat):
    """
    Normalizes the input matrix by dividing it by its norm.
    """
    length = np.linalg.norm(mat)
    if length == 0:
        raise ValueError("Matrix norm is zero, cannot normalize.")
    return mat / length

def calc_angle_between_two_vectors(vec1, vec2):
    """
    Calculate the angle between two vectors using the dot product formula.
    cos(angle) = vec1.dot(vec2) / (||vec1|| * ||vec2||)
    Args:
        vec1 (np.ndarray): First vector (Nx1).
        vec2 (np.ndarray): Second vector (Nx1).
    Returns:
        float: The angle in radians between the two vectors.
    """
    assert vec1.shape == vec2.shape and vec1.shape[1] == 1, "Vectors must have the same shape and be column vectors"
    
    # Dot product
    dot_product = np.dot(vec1.T, vec2)[0, 0]
    # Norm of each vector
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    # Ensure the norms are not zero to avoid division by zero
    assert norm_vec1 != 0 and norm_vec2 != 0, "Vector norms must be non-zero"
    # Calculate the angle (in radians)
    cos_angle = dot_product / (norm_vec1 * norm_vec2)
    # Clip cos_angle to the valid range [-1, 1] to avoid numerical issues with acos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    # Return the angle in radians
    angle = np.arccos(cos_angle)
    
    return angle

def filter_out(cones_pixels, color_value): 
    cone_boxes = np.array([[u1, v1, u2, v2, color_value] for u1, v1, u2, v2 in cones_pixels
                        if np.abs(v1 - v2) <= 1.55 * np.abs(u1 - u2)], dtype=int)
    if len(cone_boxes) == 0: 
        return np.empty((0,5), dtype=int)
    return cone_boxes

def point_in_box(x, y, box):
    return box[0]-10 <= x <= box[2]+10 and box[1]-10 <= y <= box[3]+10

def calc_cones_error(R, t, prev_cones_pos, curr_cones_pos, alpha=1, print_error=False, print_cones_updates=False):
    prev_cone_pos_transformed = []
    curr_cone_pos = []
    total_err = 0
    num_matching_cones = prev_cones_pos.shape[0]
    for i in range(len(prev_cones_pos)):
        prev_cone_pos = prev_cones_pos[i][:3]
        transformed_p = trans_coord(prev_cone_pos, R, alpha*t)
        prev_cone_pos_transformed.append(transformed_p)
        curr_cone_pos.append(curr_cones_pos[i][:3])
        total_err += np.linalg.norm(transformed_p[:3:2] - curr_cones_pos[i][:3:2])
    if print_error: 
        print(f"Scaling at {float(alpha):.2f} with error: {float(total_err/num_matching_cones):.2f}")
    if print_cones_updates:  # print the transformed cones
        print(np.asarray(prev_cone_pos_transformed))
        print(np.asarray(curr_cone_pos))
    return total_err

def calc_weighted_error(transformed_cones, current_cones, x_weight=2.0, z_weight=1.0):
    # Extract only X and Z components if points are in 3D
    if transformed_cones.shape[1] == 3:
        transformed_cones = transformed_cones[:, [0, 2]]
    if current_cones.shape[1] == 3:
        current_cones = current_cones[:, [0, 2]]
    # Calculate weighted squared differences
    x_diff_squared = (transformed_cones[:, 0] - current_cones[:, 0]) ** 2 * x_weight
    z_diff_squared = (transformed_cones[:, 1] - current_cones[:, 1]) ** 2 * z_weight
    # Sum the errors and normalize by the sum of weights
    total_error = np.sum(x_diff_squared + z_diff_squared)
    normalized_error = total_error / (x_weight + z_weight)
    return normalized_error

def current_data(R, new_t, cones3D, frame_count):
    tx = new_t[0,0]
    ty = -new_t[2,0]  # Due to the coordinate system mapping
    # Extract theta from the rotation matrix new_R
    theta_y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    # Create the odom matrix
    odom = np.array([
        [tx, 1, frame_count, frame_count-1],
        [ty, 1, frame_count, frame_count-1],
        [theta_y, 1, frame_count, frame_count-1]
    ])

    len_cones = cones3D.shape[0]
    observations = np.ones((len_cones * 2, 4))

    frame_val = frame_count
    # cones_idx = np.arange(1, len_cones + 1)
    
    for i in range(len_cones):
        if (cones3D[i, 2] < 25):
            observations[2 * i, 0] = cones3D[i, 0]
            observations[2 * i + 1, 0] = cones3D[i, 2]
            observations[2 * i:2 * i + 2, 1] = 2
            observations[2 * i:2 * i + 2, 2] = cones3D[i, 4]
            observations[2 * i:2 * i + 2, 3] = frame_val
        else: 
            observations[2*i : 2*i+2, 0] = -100
    
    observations = observations[observations[:,0]>-100, :]
    perception_data = np.vstack((odom, observations))
    return perception_data

def save_slam_data(data, filename): 
    data = np.concatenate(data, axis=0)
    zeros_array = np.zeros((3, data.shape[1]))  # Create a row of zeros with the same shape as data
    zeros_array[:, 1:3] = np.ones((3, 2)) #
    data = np.vstack((zeros_array, data))
    data = {
    'FAE_motorsport': data
    }
    savemat(filename, data)
    print('successfully saved data to %s' % filename)

def calc_theta(R): 
    theta = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    theta = (theta + np.pi / 2) % np.pi - np.pi / 2
    return theta