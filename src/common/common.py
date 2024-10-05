import numpy as np
import cv2
from typing import Union
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
    # p2_flat = p2.flatten()
    p2_flat = p2.reshape(3, 1)
    return p2_flat

def pts2keypts(pts):
    """
    Converts a list of 2D points into a list of OpenCV KeyPoint objects.
    Args:
        pts (list of tuple or np.ndarray): A list of 2D points (x, y).
    Returns:
        list of cv2.KeyPoint: A list of corresponding KeyPoint objects.
    """
    keypts = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=10) for pt in pts]
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