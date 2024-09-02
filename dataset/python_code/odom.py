import os 
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import * 
from scipy.optimize import least_squares
from State import State

def draw_match_boxes(matched_boxes, img1, img2): 
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # Draw matched boxes and annotate with numbers
    for idx, (box1, box2) in enumerate(matched_boxes):
        # Draw box on img1
        x1_1, y1_1, x2_1, y2_1 = map(int, box1[:4])
        cv2.rectangle(img1, (x1_1, y1_1), (x2_1, y2_1), color=(0, 255, 0), thickness=2)
        # Annotate with a number
        cv2.putText(img1, str(idx + 1), (x1_1, y1_1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Draw box on img2
        x1_2, y1_2, x2_2, y2_2 = map(int, box2[:4])
        cv2.rectangle(img2, (x1_2, y1_2), (x2_2, y2_2), color=(0, 255, 0), thickness=2)
        # Annotate with a number
        cv2.putText(img2, str(idx + 1), (x1_2, y1_2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the images with the matched boxes and annotations
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))  # Convert to RGB for display in matplotlib
    axes[0].set_title("Image 1 with Matched Boxes")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))  # Convert to RGB for display in matplotlib
    axes[1].set_title("Image 2 with Matched Boxes")
    axes[1].axis('off')

    # plt.show()

def boxes_matching(pre_state: State, curr_state: State): 
    image1 = pre_state.image
    image2 = curr_state.image
    boxes1 = pre_state.cones
    boxes2 = curr_state.cones
    
    mask = np.zeros_like(image2)
    mask[300:600, :] = 255
    mask[465:800, 550:1250] = 0
    masked_gray1 = cv2.bitwise_and(image1, mask)
    masked_gray2 = cv2.bitwise_and(image2, mask) 

    orb = cv2.ORB_create(1000) 
    kp1, des1 = orb.detectAndCompute(masked_gray1,None)
    kp2, des2 = orb.detectAndCompute(masked_gray2,None)
    # index_params = dict(algorithm=6,
    #                     table_number=6,
    #                     key_size=12,
    #                     multi_probe_level=2, 
    #                     )
    # search_params = dict(checks = 50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1,des2,k=2)
    # good_matches = [] # Apply Lowe's ratio test
    # for m, n in matches:
    #     if m.distance < 0.95 * n.distance:
    #         good_matches.append(m)  
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x: x.distance)
    good_matches = matches[:90]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # Use RANSAC to find the homography matrix and remove outliers
    if len(pts1) >= 4:  # At least 4 points are required to compute the homography
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 6.0)
        # H, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=0.5)
        matches_mask = mask.ravel().tolist()
    else:
        print("Not enough points to compute homography.")
        matches_mask = None

    boxes1T = {tuple(box) for box in boxes1[:,:]}
    boxes2T = {tuple(box) for box in boxes2[:,:]}

    features1 = []
    features2 = []
    inlier_matches = []
    matched_boxes = []
    matched1_boxes2D = np.empty((0, 6), dtype=np.int64)
    matched2_boxes2D = np.empty((0, 6), dtype=np.int64)
    matched1_boxes3D = np.empty((0, 9), dtype=np.float64)
    for i, m in enumerate(good_matches): 
        if matches_mask[i]:  # Inlier
            inlier_matches.append(m)
            # Extract the matching points
            x1, y1 = kp1[m.queryIdx].pt
            x2, y2 = kp2[m.trainIdx].pt 
            features1.append([x1, y1, 1])
            features2.append([x2, y2, 1])
            # Find the matching box in 
            matched1_box = None
            matched1_color = None
            for box1 in boxes1T:
                if point_in_box(x1, y1, box1):
                    matched1_box = box1
                    matched1_color = box1[4]
                    matched1_id = box1[5]
                    break
            # Find the matching box in 
            matched2_box = None
            for box2 in boxes2T:
                if point_in_box(x2, y2, box2) and matched1_color==box2[4]:
                    matched2_box = box2
                    matched2_color = box2[4]
                    matched2_id = box2[5]
                    break      
            # If both boxes are found, append them to matched_boxes and remove from the set
            if matched1_box is not None and matched2_box is not None:
                matched2_boxes2D = np.vstack((matched2_boxes2D, curr_state.cones_3pixels[matched2_id, :6]))
                matched1_boxes2D = np.vstack((matched1_boxes2D, pre_state.cones_3pixels[matched1_id, :6]))

                matched1_boxes3D = np.vstack((matched1_boxes3D, pre_state.cone_3Dpoints[matched1_id, :9]))
                
                matched_boxes.append((matched1_box, matched2_box))
                boxes1T.remove(matched1_box)
                boxes2T.remove(matched2_box) 
    # draw_match_boxes(matched_boxes, image1, image2)
    features1 = np.ascontiguousarray(features1)
    features2 = np.ascontiguousarray(features2)
    return matched1_boxes2D, matched2_boxes2D, matched1_boxes3D
    # return features1, features2, matched_boxes, matched_boxes3D

def feature_extraction(img1, img2):
    mask = np.zeros_like(img1)
    mask[80:600, :] = 255
    # mask[530:800, 600:1250] = 0
    # mask[470:800, 300:1320] = 0
    mask[470:800, 600:1250] = 0
    # Apply the mask
    masked_gray1 = cv2.bitwise_and(img1, img1, mask=mask)
    masked_gray2 = cv2.bitwise_and(img2, img2, mask=mask)
    orb = cv2.ORB_create(1500) 
    kp1, des1 = orb.detectAndCompute(masked_gray1, None)
    kp2, des2 = orb.detectAndCompute(masked_gray2, None)

    # index_params = dict(algorithm=6,
    #                     table_number=6,
    #                     key_size=12,
    #                     multi_probe_level=1, 
    #                     trees = 5)
    # search_params = dict(checks = 50)

    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1,des2,k=2)
    features1 = []
    features2 = []
    # # store all the good matches as per Lowe's ratio test.
    # good_matches = []
    # for i, (m, n) in enumerate(matches):
    #     if m.distance < 0.8 * n.distance:
    #         # good_matches.append(m)
    #         x1, y1 = kp1[m.queryIdx].pt
    #         x2, y2 = kp2[m.trainIdx].pt
    #         features1.append([x1, y1, 1])
    #         features2.append([x2, y2, 1])

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x: x.distance)

    for i, m in enumerate(matches[:500]):
        # good_matches.append(m)
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt
        features1.append([x1, y1, 1])
        features2.append([x2, y2, 1])
    features1 = np.ascontiguousarray(features1)
    features2 = np.ascontiguousarray(features2)
    return features1, features2

MIN_MATCH_COUNT = 50
# INIT_IMAGE = 1
def camera_motion():
    frame_count = 1
    camera_pose = np.eye(4)
    K = np.loadtxt("K_matrix.txt")
    start = time.process_time()
    while True:
        print("FRAME_COUNT: " + str(frame_count))
        if frame_count == 159: 
            break
        # print(frame_count)
        prev_state = State(frame_count, K)
        curr_state = State(frame_count+1, K) 
        features1, features2 = feature_extraction(prev_state.image, curr_state.image)
        # print(matched2_boxes.shape,'\n', matched1_boxes.shape)
        essential_mat, _ = cv2.findEssentialMat(features1[:, :2], features2[:, :2], K, method=cv2.RANSAC, prob=0.999, threshold=0.5)
        _, new_R, new_t, mask = cv2.recoverPose(essential_mat, features1[:, :2], features2[:, :2], K)
        # new_R, new_t = compute_camera_motion_pnp(matched1_boxes3D, matched2_boxes, K)
        if np.linalg.det(new_R) < 0:
            new_R = -new_R
            new_t = -new_t
        scale = 1
        matched1_boxes, matched2_boxes, matched1_boxes3D = boxes_matching(prev_state, curr_state)
        scale = compute_weighted_scale_factor(matched1_boxes3D, matched1_boxes, matched2_boxes, new_R, new_t, K)
        new_pose = np.column_stack((new_R, scale*new_t))
        # print(pixel_reproject_err(new_pose.ravel(), matched1_boxes3D.reshape(-1, 3), K, matched2_boxes.reshape(-1, 2)) )
        # get_approximate_odometry(matched2_boxes, matched1_boxes3D, new_pose, K, )
        new_pose = np.vstack((new_pose, np.array([0,0,0,1])))
        camera_pose = camera_pose @ new_pose
        x_coord = camera_pose[0, -1]
        z_coord = camera_pose[2, -1]
        print(x_coord, -z_coord)
        plt.scatter(x_coord, -z_coord, color='b') 
        plt.pause(0.00001)

        frm = cv2.resize(prev_state.image, (0,0), fx=0.5, fy=0.5)
        cv2.imshow('Frame', frm)
        frame_count += 1
    print('\n\nTime taken: ', (time.process_time() - start))
    cv2.destroyAllWindows()
    plt.show()

def compute_weighted_scale_factor(matched1_box3D, matched1_box, matched2_box, new_R, new_t, K):
    """
    Compute the scale factor using weighted depths of the triangulated 3D points.
    """
    matched2_box = matched2_box.reshape(-1, 2)
    matched1_box = matched1_box.reshape(-1, 2)
    matched1_box3D = matched1_box3D.reshape(-1, 3)
    # Construct projection matrices for the two cameras
    proj_matrix1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for the first camera
    proj_matrix2 = K @ np.hstack((new_R, new_t.reshape(3, 1)))   # Projection matrix for the second camera
    # Triangulate the 3D points in the second camera frame
    points4D_hom = cv2.triangulatePoints(proj_matrix1, proj_matrix2, matched1_box.T, matched2_box.T)
    # Convert homogeneous coordinates to 3D points
    points3D_triangulated = points4D_hom[:3] / points4D_hom[3]
    points3D_triangulated = points3D_triangulated.T
        # Ensure all points have the same sign for their depths
    depths_triangulated = points3D_triangulated[:, 2]
    if np.any(depths_triangulated < 0) and np.any(depths_triangulated > 0):
        # If there are mixed signs, flip the sign of all points to make depths consistent
        if np.sum(depths_triangulated > 0) < np.sum(depths_triangulated < 0):
            points3D_triangulated *= -1
    # distances_original = np.linalg.norm(matched1_box3D, axis=1)
    # distances_triangulated = np.linalg.norm(points3D_triangulated, axis=1)

    # # Compute the scale factor as the ratio of the distances
    # scale_factors = distances_original / distances_triangulated
    # scale_factor = np.mean(scale_factors)
    # return scale_factor

    # Extract the Z-coordinates (depths)
    depths_original = matched1_box3D[:, 2]
    depths_triangulated = points3D_triangulated[:, 2]
    print(matched1_box3D)
    print(points3D_triangulated)
    # Define the weight function based on depth reliability
    def weight(depth):
        if 6 <= depth <= 15:
            return 0.6  # High weight for reliable depths
        else:
            return 1  # Lower weight for less reliable depths

    # Compute the weights for each point
    weights = np.array([weight(d) for d in depths_original])

    # Calculate the weighted scale factor
    scale_factors = (depths_original / depths_triangulated) * weights
    scale_factor = np.sum(scale_factors) / np.sum(weights)
    print("SCALE FACTOR: ", scale_factor)
    print("================================")
    return scale_factor

def get_approximate_odometry(matched2_boxes, matched1_boxes3D, pose, K):
    matched2_boxes = matched2_boxes.reshape(-1, 2)
    matched1_boxes3D = matched1_boxes3D.reshape(-1, 3)
    # Calculate the translation and rotation from the matched boxes and 3D points

    res = least_squares(pixel_reproject_err, pose.ravel(), args = (matched1_boxes3D, K, matched2_boxes), max_nfev = 100) 
    # print('after: ', pixel_reproject_err(res.x, matched1_boxes3D, K, matched2_boxes))
    return res.x.reshape(3, 4)
def pixel_reproject_err(M, points_3D, K, points_2D): 
    M = M.reshape(3, 4)
    projected_points = reprojectPoints(points_3D, M, K) 
    err = projected_points - points_2D
    return err.ravel()

def compute_camera_motion_pnp(points1_3D, points2_2D, K):
    """Compute camera motion using the PnP algorithm."""
    # Convert the input points to the required format (float32)
    points2_2D = points2_2D.reshape(-1, 2)
    points1_3D = points1_3D.reshape(-1, 3)
    points1_3D = np.array(points1_3D, dtype=np.float32)
    points2_2D = np.array(points2_2D, dtype=np.float32)
    
    # Initial guess for rotation and translation (can be zeros or based on prior knowledge)
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    
    # Solve PnP to estimate the camera pose (rotation and translation)
    success, rotation_vector, translation_vector = cv2.solvePnP(
        points1_3D, points2_2D, K, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
    )
    
    if not success:
        raise ValueError("PnP algorithm failed to find a solution.")
    # Convert the rotation vector to a rotation matrix
    R, _ = cv2.Rodrigues(rotation_vector)
    t = translation_vector.reshape(-1)
    out = get_approximate_odometry(points2_2D, points1_3D, np.column_stack([R, t]), K)
    return out[:, :3], out[:, 3]
    return R, t

if __name__ == "__main__":
    camera_motion()


