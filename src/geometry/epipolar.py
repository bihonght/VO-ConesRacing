import cv2
import numpy as np

from common import params
from math import sqrt

def estiMotionByEssential(pts_in_img1, pts_in_img2, camera_intrinsics):
    """
    Estimate motion by essential matrix, and recover rotation and translation.

    Args:
        pts_in_img1 (list of cv2.Point2f): Points from the first image.
        pts_in_img2 (list of cv2.Point2f): Points from the second image.
        camera_intrinsics (np.ndarray): Camera intrinsic matrix.
    
    Returns:
        essential_matrix (np.ndarray): Estimated essential matrix.
        R (np.ndarray): Rotation matrix.
        t (np.ndarray): Translation vector.
        inliers_index (list of int): Indices of inliers.
        inliers_pts_in_img1 (list of cv2.Point2f): Inlier points from the first image.
        inliers_pts_in_img2 (list of cv2.Point2f): Inlier points from the second image.
    """
    # Extract focal length and principal point from camera intrinsics
    K = camera_intrinsics
    principal_point = (K[0, 2], K[1, 2])  # Optical center
    focal_length = (K[0, 0] + K[1, 1]) / 2  # Average focal length
    
    # Define RANSAC parameters
    method = cv2.RANSAC
    findEssentialMat_prob = params.findEssentialMat_prob
    findEssentialMat_threshold = params.findEssentialMat_threshold
    
    # Calculate essential matrix
    essential_matrix, inliers_mask = cv2.findEssentialMat(
        pts_in_img1, pts_in_img2, focal_length, principal_point,
        method, prob=findEssentialMat_prob, threshold=findEssentialMat_threshold)
    
    essential_matrix /= essential_matrix[2, 2]  # Normalize essential matrix

    # Get inliers' indices
    inliers_index = [i for i in range(inliers_mask.shape[0]) if inliers_mask[i, 0] == 1]

    # Recover rotation (R) and translation (t) from essential matrix
    _, R, t, _ = cv2.recoverPose(essential_matrix, pts_in_img1, pts_in_img2, cameraMatrix=K, mask=inliers_mask)

    # Normalize translation vector
    norm_t = sqrt(t[0, 0]**2 + t[1, 0]**2 + t[2, 0]**2)
    t = t / norm_t

    return essential_matrix, R, t, inliers_index

def removeWrongRtOfHomography(pts_on_np1, pts_on_np2, inliers, Rs, ts, normals):
    """
    Removes incorrect R and t solutions based on visibility of points in front of the camera.

    Args:
        pts_on_np1 (list of cv2.Point2f): Points on the normalized plane from image 1.
        pts_on_np2 (list of cv2.Point2f): Points on the normalized plane from image 2.
        inliers (list of int): List of inlier indices.
        Rs (list of np.ndarray): List of rotation matrices.
        ts (list of np.ndarray): List of translation vectors.
        normals (list of np.ndarray): List of normal vectors.
    
    Returns:
        Updated Rs, ts, and normals by filtering out incorrect solutions.
    """
    # Collect inlier points based on inliers index
    inl_pts_on_np1 = pts_on_np1[inliers][:, np.newaxis, :] # new axis for CV232F in fileterHomographyDecomByVisibleRefpoints
    inl_pts_on_np2 = pts_on_np2[inliers][:, np.newaxis, :]
    
    inl_pts_on_np1 = inl_pts_on_np1.astype(np.float32)
    inl_pts_on_np2 = inl_pts_on_np2.astype(np.float32)

    # Use OpenCV's filterHomographyDecompByVisibleRefpoints
    possible_solutions = cv2.filterHomographyDecompByVisibleRefpoints(Rs, normals, inl_pts_on_np1, inl_pts_on_np2) 

    print("Valid decomposition indices:", possible_solutions)

    # Filter Rs, ts, and normals based on valid solutions
    res_Rs = []
    res_ts = []
    res_normals = []
    
    for idx in possible_solutions:
        res_Rs.append(Rs[idx[0]])    # Extract the valid R
        res_ts.append(ts[idx[0]])    # Extract the valid t
        res_normals.append(normals[idx[0]])  # Extract the valid normal
    
    # Return the filtered Rs, ts, and normals
    return res_Rs, res_ts, res_normals

def estiMotionByHomography(pts_in_img1, pts_in_img2, camera_intrinsics):
    """
    Estimate motion by homography matrix, and recover rotation and translation.

    Args:
        pts_in_img1 (list of cv2.Point2f): Points from the first image.
        pts_in_img2 (list of cv2.Point2f): Points from the second image.
        camera_intrinsics (np.ndarray): Camera intrinsic matrix.

    Returns:
        homography_matrix (np.ndarray): The computed homography matrix.
        Rs (list of np.ndarray): List of rotation matrices.
        ts (list of np.ndarray): List of translation vectors.
        normals (list of np.ndarray): List of normal vectors.
        inliers_index (list of int): List of inlier indices.
    """
    # Homography computation using RANSAC
    ransac_reproj_threshold = 3.0  # Set the RANSAC reprojection threshold
    method = cv2.RANSAC
    
    # Find homography matrix with RANSAC
    homography_matrix, inliers_mask = cv2.findHomography(
        pts_in_img1, pts_in_img2, method, ransac_reproj_threshold)
    
    # Normalize the homography matrix
    homography_matrix /= homography_matrix[2, 2]
    
    # Extract inliers' indices based on inliers_mask
    inliers_index = [i for i in range(inliers_mask.shape[0]) if inliers_mask[i, 0] == 1]
    
    # Decompose homography matrix into possible R, t, and normal vectors
    num_sol, rotations, translations, normals = cv2.decomposeHomographyMat(homography_matrix, camera_intrinsics) 
    # num_sol possible solutions
    ts = list(translations)
    Rs = list(rotations)
    # Normalize each translation vector
    for i in range(len(ts)):
        norm_t = sqrt(ts[i][0, 0]**2 + ts[i][1, 0]**2 + ts[i][2, 0]**2)
        ts[i] = ts[i] / norm_t
    
    return homography_matrix, Rs, ts, normals, inliers_index

def doTriangulation(pts_on_np1, pts_on_np2, R_cam2_to_cam1, t_cam2_to_cam1, inliers):

    # Extract inlier points
    inlier_pts_on_np1 = [pts_on_np1[i, :] for i in inliers]
    inlier_pts_on_np2 = [pts_on_np2[i, :] for i in inliers]

    # Setup projection matrices for triangulation
    T_cam1_to_world = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0]], dtype=np.float32)

    # Concatenate R and t to create the 3x4 projection matrix for cam2
    T_cam2_to_world = np.hstack((R_cam2_to_cam1, t_cam2_to_cam1))

    # Triangulate points
    pts4d_in_world = cv2.triangulatePoints(T_cam1_to_world, T_cam2_to_world, 
                                           np.array(inlier_pts_on_np1).T, 
                                           np.array(inlier_pts_on_np2).T)

    # Convert from homogeneous coordinates to 3D coordinates
    pts3d_in_cam1 = []
    for i in range(pts4d_in_world.shape[1]):
        x = pts4d_in_world[:, i]
        x /= x[3]  # Normalize by the fourth coordinate (homogeneous to Euclidean conversion)
        pt3d_in_world = np.array([x[0], x[1], x[2]])
        pts3d_in_cam1.append(pt3d_in_world)

    return pts3d_in_cam1


def extractPtsFromMatches(points_1, points_2, matches):
    pts1 = []
    pts2 = []

    for match in matches:
        pts1.append(points_1[match.queryIdx].pt)
        pts2.append(points_2[match.trainIdx].pt)
    
    pts1 = np.ascontiguousarray(pts1)
    pts2 = np.ascontiguousarray(pts2)

    return pts1, pts2
