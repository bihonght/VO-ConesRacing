import numpy as np

def pixel_to_cam_norm_plane(p, K):
    return np.array([
        (p[0] - K[0, 2]) / K[0, 0],  # x-coordinate normalized by focal length and center offset
        (p[1] - K[1, 2]) / K[1, 1]   # y-coordinate normalized by focal length and center offset
    ])

def pixel_to_cam(p, K, depth):
    return np.array([
        depth * (p[0] - K[0, 2]) / K[0, 0],  # x in camera coordinates
        depth * (p[1] - K[1, 2]) / K[1, 1],  # y in camera coordinates
        depth  # z is the depth provided
    ])

def cam_to_pixel(p, K):
    return np.array([
        K[0, 0] * p[0] / p[2] + K[0, 2],  # project x in pixel coordinates
        K[1, 1] * p[1] / p[2] + K[1, 2]   # project y in pixel coordinates
    ])

def cam_to_pixel_mat(p, K): 
    # Convert 3D homogeneous coordinates to pixel coordinates
    p0 = np.dot(K, p)  # project onto image
    pp = p0 / p0[2, 0]  # normalize by z-coordinate (third row of p0)
    return np.array([pp[0, 0], pp[1, 0]])  # return normalized 2D point
