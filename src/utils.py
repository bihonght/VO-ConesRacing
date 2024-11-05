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

def boxes_matching(image1, image2, box_pixels1, box_pixels2, color_boundary): 
    mask = np.zeros_like(image2)
    mask[300:600, :] = 255
    mask[465:800, 550:1250] = 0
    masked_gray1 = cv2.bitwise_and(image1, mask)
    masked_gray2 = cv2.bitwise_and(image2, mask) 

    orb = cv2.ORB_create(1200) 
    kp1, des1 = orb.detectAndCompute(masked_gray1,None)
    kp2, des2 = orb.detectAndCompute(masked_gray2,None)
    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2, 
                        )
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good_matches = [] # Apply Lowe's ratio test
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good_matches.append(m)     
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # Use RANSAC to find the homography matrix and remove outliers
    if len(pts1) >= 4:  # At least 4 points are required to compute the homography
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        # H, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=0.5)
        matches_mask = mask.ravel().tolist()
    else:
        print("Not enough points to compute homography.")
        matches_mask = None
    
    blue_1 = {tuple(box) for box in box_pixels1[:color_boundary,:]}
    blue_2 = {tuple(box) for box in box_pixels2[:color_boundary,:]}
    yellow_1 = {tuple(box) for box in box_pixels1[color_boundary:,:]}
    yellow_2 = {tuple(box) for box in box_pixels1[color_boundary:,:]}
    inlier_matches = []
    matched_boxes = []

    for i, m in enumerate(good_matches): 
        if matches_mask[i]:  # Inlier
            inlier_matches.append(m)

            # Extract the matching points
            x1, y1 = kp1[m.queryIdx].pt
            x2, y2 = kp2[m.trainIdx].pt 
            # Find the matching box in 
            matched_box1 = None
            for box1 in yellow_1:
                if point_in_box(x1, y1, box1):
                    matched_box1 = box1
                    matched_index1 = box1[5]
                    break
            # Find the matching box in 
            matched_box2 = None
            for box2 in yellow_2:
                if point_in_box(x2, y2, box2):
                    matched_box2 = box2
                    matched_index2 = box2[5]
                    break      
            # If both boxes are found, append them to matched_boxes and remove from the set
            if matched_box1 is not None and matched_box2 is not None:
                matched_boxes.append((matched_box1, matched_box2))
                yellow_1.remove(matched_box1)
                yellow_2.remove(matched_box2)
            
            
            for box1 in blue_1:
                if point_in_box(x1, y1, box1):
                    matched_box1 = box1
                    break
            # Find the matching box in filtered_boxes2_set
            matched_box2 = None
            for box2 in blue_2:
                if point_in_box(x2, y2, box2):
                    matched_box2 = box2
                    break
            # If both boxes are found, append them to matched_boxes and remove from the set
            if matched_box1 is not None and matched_box2 is not None:
                matched_boxes.append((matched_box1, matched_box2))
                
                blue_1.remove(matched_box1)
                blue_2.remove(matched_box2)
    
def find_matched_boxes(x1, y1, x2, y2, boxes1, boxes2): 
    matched_boxes = []
    for i in range(boxes1.shape[0]):
        matched1 = None
        box1 = boxes1[i, :]
        if point_in_box(x1, y1, box1): 
            matched1 = box1[i, 4]
            boxes1.remove(box1)
            break
    for box2 in boxes2:
        matched2 = None
        if point_in_box(x2, y2, box2):
            matched_boxes.append((box1, box2))
            matched2 = box2
            boxes2.remove(box2)
            if matched1 is not None and matched2 is not None: 
                matched_boxes.append((matched1, matched2))
            break
    return matched_boxes

def save_current_data(new_R, new_t, cones3D, frame_count):
    tx = new_t[0,0]
    ty = -new_t[2,0]  # Due to the coordinate system mapping
    # Extract theta from the rotation matrix new_R
    theta = np.arctan2(new_R[1, 0], new_R[0, 0])
    # Create the odom matrix
    odom = np.array([
        [tx, 1, frame_count + 1, frame_count],
        [ty, 1, frame_count + 1, frame_count],
        [theta, 1, frame_count + 1, frame_count]
    ])

    len_cones = cones3D.shape[0]
    observations = np.ones((len_cones * 2, 4))

    frame_val = frame_count + 1
    cones_idx = np.arange(1, len_cones + 1)
    
    for i in range(len_cones):
        if (cones3D[i, 2] < 25):
            observations[2 * i, 0] = cones3D[i, 0]
            observations[2 * i + 1, 0] = cones3D[i, 2]
            observations[2 * i:2 * i + 2, 1] = 2
            observations[2 * i:2 * i + 2, 2] = cones_idx[i]
            observations[2 * i:2 * i + 2, 3] = frame_val
        else: 
            observations[2*i : 2*i+2, 0] = -100
    
    observations = observations[observations[:,0]>-100, :]
    perception_data = np.vstack((odom, observations))
    return perception_data