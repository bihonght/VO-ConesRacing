import numpy as np
import cv2

from geometry import epipolar
from geometry import camera
from common import common
def helper_estimate_possible_relative_poses_by_epipolar_geometry(
    keypoints_1, keypoints_2, matches, K,             # Input
    is_print_res, is_calc_homo, is_frame_cam2_to_cam1):   # Settings

    list_R, list_t, list_matches, list_normal, sols_pts3d_in_cam1_by_triang = [], [], [], [], []
    # Convert keypoints to points
    # pts_img1_all = convert_keypoints_to_point2f(keypoints_1)
    # pts_img2_all = convert_keypoints_to_point2f(keypoints_2)
    # pts_img1, pts_img2 = [], []

    kpts_img1, kpts_img2 = epipolar.extractPtsFromMatches(keypoints_1, keypoints_2, matches)
    # pts_on_np1, pts_on_np2 = [], []
    num_matched = len(matches)
    kpts_on_np1 = np.empty((num_matched, 2))
    kpts_on_np2 = np.empty((num_matched, 2))

    for i in range(num_matched):
        kpts_on_np1[i] = camera.pixel_to_cam_norm_plane(kpts_img1[i], K)
        kpts_on_np2[i] = (camera.pixel_to_cam_norm_plane(kpts_img2[i], K))
    
    # Estimate motion by Essential Matrix
    R_e, t_e, essential_matrix = None, None, None
    inliers_index_e = []

    essential_matrix, R_e, t_e, inliers_index_e = epipolar.estiMotionByEssential(
        kpts_img1, kpts_img2, K)

    if is_print_res:
        epipolar.print_result_estimate_motion_by_essential(essential_matrix, inliers_index_e, R_e, t_e)

    # Estimate motion by Homography Matrix (if enabled)
    R_h_list, t_h_list, normal_list = [], [], []
    inliers_index_h = []
    homography_matrix = None

    if is_calc_homo:
        homography_matrix, R_h_list, t_h_list, normal_list, inliers_index_h = epipolar.estiMotionByHomography(
            kpts_img1, kpts_img2, K)
        R_h_list, t_h_list, normal_list = epipolar.removeWrongRtOfHomography(kpts_on_np1, kpts_on_np2, inliers_index_h, R_h_list, t_h_list, normal_list)

    num_h_solutions = len(R_h_list)

    if is_print_res and is_calc_homo:
        epipolar.print_result_estimate_motion_by_homography(homography_matrix, inliers_index_h, R_h_list, t_h_list, normal_list)

    # Combine the motions from Essential/Homography
    list_inliers = []
    # list_matches, list_R, list_t, list_normal = [], [], [], []

    list_R.append(R_e)
    list_t.append(t_e)
    list_normal.append(None)
    list_inliers.append(inliers_index_e)

    for i in range(num_h_solutions):
        list_R.append(R_h_list[i])
        list_t.append(t_h_list[i])
        list_normal.append(normal_list[i])
        list_inliers.append(inliers_index_h)

    num_solutions = len(list_R)

    # Convert [inliers of matches] to the [cv2.DMatch of all keypoints]
    for i in range(num_solutions):
        list_matches.append([])
        inliers = list_inliers[i]
        for idx in inliers:
            list_matches[i].append(cv2.DMatch(matches[idx].queryIdx, matches[idx].trainIdx, matches[idx].distance))

    # Triangulation for all solutions
    for i in range(num_solutions):
        pts3d_in_cam1 = epipolar.doTriangulation(kpts_on_np1, kpts_on_np2, list_R[i], list_t[i], list_inliers[i])
        sols_pts3d_in_cam1_by_triang.append(pts3d_in_cam1)

    # Change frame if needed
    if not is_frame_cam2_to_cam1:
        for i in range(num_solutions):
            list_R[i], list_t[i] = common.inv_rt(list_R[i], list_t[i])

    # Debugging
    if is_print_res and not is_calc_homo:
        print_epipolar_error_and_triangulation_result_by_solution(
            pts_img1, pts_img2, pts_on_np1, pts_on_np2, sols_pts3d_in_cam1_by_triang, list_inliers, list_R, list_t, K)
    elif is_print_res and is_calc_homo:
        print_epipolar_error_and_triangulation_result_by_common_inlier(
            pts_img1, pts_img2, pts_on_np1, pts_on_np2, sols_pts3d_in_cam1_by_triang, list_inliers, list_R, list_t, K)

    # Evaluate and choose the best solution
    score_E = check_essential_score(essential_matrix, K, kpts_img1, kpts_img2, inliers_index_e)
    score_H = check_homography_score(homography_matrix, kpts_img1, kpts_img2, inliers_index_h)

    ratio = score_H / (score_E + score_H)
    print(f"Evaluate E/H score: E = {score_E:.1f}, H = {score_H:.1f}, H/(E+H)={ratio:.3f}")

    best_sol = 0
    if ratio > 0.5:
        best_sol = 1
        largest_norm_z = abs(list_normal[1][2, 0])
        for i in range(2, num_solutions):
            norm_z = abs(list_normal[i][2, 0])
            if norm_z > largest_norm_z:
                largest_norm_z = norm_z
                best_sol = i

    print(f"Best index = {best_sol}, which is [{'E' if best_sol == 0 else 'H'}].\n")

    return best_sol, list_R, list_t, list_matches, list_normal, sols_pts3d_in_cam1_by_triang

def helper_esti_motion_by_essential(keypoints_1, keypoints_2, matches, K, is_print_res=False):
    # Extract points from matches
    pts_in_img1, pts_in_img2 = epipolar.extractPtsFromMatches(keypoints_1, keypoints_2, matches)
    # Estimate motion by essential matrix
    _, R, t, inliers_index = epipolar.estiMotionByEssential(pts_in_img1, pts_in_img2, K)
    # Collect inlier matches
    inlier_matches = [matches[idx] for idx in inliers_index]
    return R, t, inlier_matches

def helper_find_inlier_matches_by_epipolar_cons(keypoints_1, keypoints_2, matches, K):
    # Estimate inlier matches using the essential matrix
    _, _, inlier_matches = helper_esti_motion_by_essential(keypoints_1, keypoints_2, matches, K)
    return inlier_matches

def helper_triangulate_points(prev_kpts, curr_kpts, curr_inlier_matches, T_curr_to_prev, K):
    """Wrapper to call triangulation with a 4x4 transformation matrix."""
    R_curr_to_prev, t_curr_to_prev = common.get_rt_from_T(T_curr_to_prev)
    return helper_triangulate_points_Rt(prev_kpts, curr_kpts, curr_inlier_matches, R_curr_to_prev, t_curr_to_prev, K) 

def helper_triangulate_points_Rt(prev_kpts, curr_kpts, curr_inlier_matches, R_curr_to_prev, t_curr_to_prev, K):
    """Triangulate points using rotation and translation matrices."""
    # Extract matched keypoints and convert to camera normalized plane
    pts_img1, pts_img2 = epipolar.extractPtsFromMatches(prev_kpts, curr_kpts, curr_inlier_matches)
    pts_on_np1 = np.array([camera.pixel_to_cam_norm_plane(pt, K) for pt in pts_img1])
    pts_on_np2 = np.array([camera.pixel_to_cam_norm_plane(pt, K) for pt in pts_img2])
    # Set inlier 
    inliers = []
    for i in range(pts_img1.shape[0]):
        inliers.append(i)
    # Triangulate points
    pts_3d_in_prev = epipolar.doTriangulation(pts_on_np1, pts_on_np2, R_curr_to_prev, t_curr_to_prev, inliers)
    # Change 3D point positions to the current frame
    pts_3d_in_curr = [common.trans_coord(pt3d, R_curr_to_prev, t_curr_to_prev) for pt3d in pts_3d_in_prev]
    return pts_3d_in_curr

def check_essential_score(E21, K, pts_img1, pts_img2, inliers_index, sigma=1):
    inliers_index_new = []

    # Convert Essential matrix to Fundamental matrix
    Kinv = np.linalg.inv(K)
    F21 = Kinv.T @ E21 @ Kinv

    f11, f12, f13 = F21[0, 0], F21[0, 1], F21[0, 2]
    f21, f22, f23 = F21[1, 0], F21[1, 1], F21[1, 2]
    f31, f32, f33 = F21[2, 0], F21[2, 1], F21[2, 2]

    score = 0
    th = 3.841
    thScore = 5.991
    invSigmaSquare = 1.0 / (sigma ** 2)

    N = len(inliers_index)
    for i in range(N):
        good_point = True

        p1 = pts_img1[inliers_index[i]]
        p2 = pts_img2[inliers_index[i]]

        u1, v1 = p1[0], p1[1]
        u2, v2 = p2[0], p2[1]

        # Reprojection error in second image (epipolar constraint error)
        a2 = f11 * u1 + f12 * v1 + f13
        b2 = f21 * u1 + f22 * v1 + f23
        c2 = f31 * u1 + f32 * v1 + f33

        num2 = a2 * u2 + b2 * v2 + c2
        squareDist1 = num2**2 / (a2**2 + b2**2)
        chiSquare1 = squareDist1 * invSigmaSquare

        if chiSquare1 > th:
            good_point = False
        else:
            score += thScore - chiSquare1

        # Reprojection error in first image
        a1 = f11 * u2 + f21 * v2 + f31
        b1 = f12 * u2 + f22 * v2 + f32
        c1 = f13 * u2 + f23 * v2 + f33

        num1 = a1 * u1 + b1 * v1 + c1
        squareDist2 = num1**2 / (a1**2 + b1**2)
        chiSquare2 = squareDist2 * invSigmaSquare

        if chiSquare2 > th:
            good_point = False
        else:
            score += thScore - chiSquare2

        if good_point:
            inliers_index_new.append(inliers_index[i])

    print(f"E score: sum = {score:.1f}, mean = {score / len(inliers_index):.2f}")
    inliers_index[:] = inliers_index_new
    return score

def check_homography_score(H21, pts_img1, pts_img2, inliers_index, sigma=1):
    score = 0
    inliers_index_new = []
    H12 = np.linalg.inv(H21)

    h11, h12, h13 = H21[0, 0], H21[0, 1], H21[0, 2]
    h21, h22, h23 = H21[1, 0], H21[1, 1], H21[1, 2]
    h31, h32, h33 = H21[2, 0], H21[2, 1], H21[2, 2]

    h11inv, h12inv, h13inv = H12[0, 0], H12[0, 1], H12[0, 2]
    h21inv, h22inv, h23inv = H12[1, 0], H12[1, 1], H12[1, 2]
    h31inv, h32inv, h33inv = H12[2, 0], H12[2, 1], H12[2, 2]

    th = 5.991
    invSigmaSquare = 1.0 / (sigma ** 2)

    N = len(inliers_index)
    for i in range(N):
        good_point = True

        p1 = pts_img1[inliers_index[i]]
        p2 = pts_img2[inliers_index[i]]

        u1, v1 = p1[0], p1[1]
        u2, v2 = p2[0], p2[1]

        # Reprojection error in first image
        w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv)
        u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv
        v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv

        squareDist1 = (u1 - u2in1)**2 + (v1 - v2in1)**2
        chiSquare1 = squareDist1 * invSigmaSquare

        if chiSquare1 > th:
            good_point = False
        else:
            score += th - chiSquare1

        # Reprojection error in second image
        w1in2inv = 1.0 / (h31 * u1 + h32 * v1 + h33)
        u1in2 = (h11 * u1 + h12 * v1 + h13) * w1in2inv
        v1in2 = (h21 * u1 + h22 * v1 + h23) * w1in2inv

        squareDist2 = (u2 - u1in2)**2 + (v2 - v1in2)**2
        chiSquare2 = squareDist2 * invSigmaSquare

        if chiSquare2 > th:
            good_point = False
        else:
            score += th - chiSquare2

        if good_point:
            inliers_index_new.append(inliers_index[i])

    print(f"H score: sum = {score:.1f}, mean = {score / len(inliers_index):.2f}")
    inliers_index[:] = inliers_index_new
    return score

def print_epipolar_error_and_triangulation_result_by_common_inlier(
    pts_img1, pts_img2, pts_on_np1, pts_on_np2,
    sols_pts3d_in_cam1, list_inliers,
    list_R, list_t, K):
    
    kMaxNumPtsToCheckAndPrint = 1000
    num_solutions = len(list_R)
    inliers_index_e = list_inliers[0]
    inliers_index_h = list_inliers[1]

    print("\n---------------------------------------")
    print("Check [Epipolar error] and [Triangulation result]")
    print(f"for the first {kMaxNumPtsToCheckAndPrint} points:")

    # Iterate through points
    cnt = 0
    num_points = len(pts_img1)
    for i in range(num_points):
        if cnt >= kMaxNumPtsToCheckAndPrint:
            break
        
        if i not in inliers_index_e or i not in inliers_index_h:
            continue

        ith_in_e_inliers = inliers_index_e.index(i)
        ith_in_h_inliers = inliers_index_h.index(i)
        
        print("\n--------------")
        print(f"Printing the {cnt}th (in common) and {i}th (in matched) point's real position in image:")
        cnt += 1

        # Print point pos in image frame
        p1 = pts_img1[i]
        p2 = pts_img2[i]
        print(f"cam1, pixel pos (u,v): {p1}")
        print(f"cam2, pixel pos (u,v): {p2}")

        # Print result of each method
        p_cam1 = pts_on_np1[i]  # point pos on the normalized plane
        p_cam2 = pts_on_np2[i]
        
        for j in range(num_solutions):
            R = list_R[j]
            t = list_t[j]

            # Print epipolar error
            err_epipolar = compute_epipolar_cons_error(p1, p2, R, t, K)
            print(f"===solu {j}: epipolar_error*1e6 is {err_epipolar * 1e6}")

            # Print triangulation result
            if j == 0:
                ith_in_curr_sol = ith_in_e_inliers
            else:
                ith_in_curr_sol = ith_in_h_inliers

            pts3dc1 = point3f_to_mat3x1(sols_pts3d_in_cam1[j][ith_in_curr_sol])  # 3D pos in camera 1
            pts3dc2 = R @ pts3dc1 + t
            pts2dc1 = cam_to_pixel(pts3dc1, K)
            pts2dc2 = cam_to_pixel(pts3dc2, K)

            print(f"-- In img1, pos: {pts2dc1}")
            print(f"-- In img2, pos: {pts2dc2}")
            print(f"-- On cam1, pos: {pts3dc1.T}")
            print(f"-- On cam2, pos: {pts3dc2.T}")

        print()

def ransac_inliers_well_distributed_y(keypoints1, keypoints2, matches, 
                                      num_bins=10, inliers_per_bin=8, 
                                      reproj_threshold=20.0):
    """
    Perform RANSAC to find inliers matches and ensure they are well-distributed along the y-axis.
    Parameters:
    - keypoints1: List of cv2.KeyPoint objects from image1.
    - keypoints2: List of cv2.KeyPoint objects from image2.
    - matches: List of cv2.DMatch objects representing matches.
    - num_bins: Number of bins to divide the y-axis into.
    - inliers_per_bin: Number of inliers to select per bin.
    - reproj_threshold: RANSAC reprojection threshold.
    Returns:
    - selected_inliers: List of cv2.DMatch objects representing the selected inliers.
    """
    if len(matches) < 4:
        # RANSAC requires at least 4 matches to compute homography
        print("Not enough matches to compute homography.")
        return []
    
    kpts_img1, kpts_img2 = epipolar.extractPtsFromMatches(keypoints1, keypoints2, matches)
    # Compute homography using RANSAC
    H, mask = cv2.findHomography(kpts_img1, kpts_img2, cv2.RANSAC, reproj_threshold)
    if mask is None:
        print("Homography could not be computed.")
        return []
    mask = mask.ravel().astype(bool)
    # Extract inlier matches
    inlier_matches = [m for m, inlier in zip(matches, mask) if inlier]
    if len(inlier_matches) == 0:
        print("No inliers found.")
        return []
    # Get y-coordinates of inliers from image1
    y_coords = [ keypoints1[m.queryIdx].pt[1] for m in inlier_matches ]
    y_coords = np.array(y_coords)
    # Define the bins for y-axis distribution
    y_min, y_max = y_coords.min(), y_coords.max()
    bins = np.linspace(y_min, y_max, num_bins + 1)
    # Assign each inlier to a bin
    bin_indices = np.digitize(y_coords, bins) - 1  # bin indices start at 0
    # Initialize selected inliers list
    selected_inliers = []
    # For each bin, select up to inliers_per_bin matches
    for i in range(num_bins):
        # Find matches in the current bin
        indices_in_bin = np.where(bin_indices == i)[0]
        if len(indices_in_bin) == 0:
            continue  # No matches in this bin
        # Shuffle the indices to select random matches
        np.random.shuffle(indices_in_bin)
        # Select up to inliers_per_bin matches
        selected_indices = indices_in_bin[:inliers_per_bin]
        selected_inliers.extend([ inlier_matches[idx] for idx in selected_indices ])
    return inlier_matches

# Helper Functions (should be implemented):
# - convert_keypoints_to_point2f
# - extract_pts_from_matches
# - pixel_to_cam_norm_plane
# - estimate_motion_by_essential
# - estimate_motion_by_homography
# - remove_wrong_rt_of_homography
# - triangulate
# - invert_rt
# - print_result_estimate_motion_by_essential
# - print_result_estimate_motion_by_homography
# - print_epipolar_error_and_triangulation_result_by_solution
# - print_epipolar_error_and_triangulation_result_by_common_inlier
# - check_essential_score
# - check_homography_score
