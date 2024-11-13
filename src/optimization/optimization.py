import numpy as np
import g2o
import time
import sophuspy as sophus
from collections import defaultdict
from scipy.optimize import minimize, least_squares
from common import common

def find_optimal_alpha(previous_cones, current_cones, R, t, x_weight=2, z_weight=1):
    def objective_function(alpha, previous_cones, current_cones, R, t):
        scaled_t = t * alpha
        total_error = 0.0
        transformed_cones = (R @ previous_cones.T).T + scaled_t.flatten()
        total_error = common.calc_weighted_error(transformed_cones, current_cones, x_weight=x_weight, z_weight=z_weight)
        return total_error
    initial_alpha = 1.0
    result = minimize(objective_function, initial_alpha, args=(previous_cones, current_cones, R, t))
    return result.x[0]  # Extract the scalar value for alpha

def objective(params, previous_cones, current_cones,  x_weight, z_weight):
    theta, t_x, t_z = params
    # Construct 2D rotation matrix for angle theta
    R_xz = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    # Apply rotation and translation to previous_cones
    transformed_cones = (R_xz @ previous_cones.T).T + np.array([t_x, t_z])
    # Calculate the sum of squared errors
    error = common.calc_weighted_error(transformed_cones, current_cones, x_weight=x_weight, z_weight=z_weight)
    return error

def estimate_and_modify_rotation_translation(R, t, previous_cones_3d, current_cones_3d,  x_weight=1, z_weight=1, output_type='Rt'):
    # Extract X and Z components of 3D points for 2D minimization
    if previous_cones_3d.shape[1] > 2:
        previous_cones_xz = previous_cones_3d[:, [0, 2]]
        current_cones_xz = current_cones_3d[:, [0, 2]]
    else:
        previous_cones_xz = previous_cones_3d
        current_cones_xz = current_cones_3d
    # Initial guesses for theta, t_x, and t_z
    initial_theta = common.calc_theta(R)  # Initial guess for rotation angle (in radians)
    initial_t_x = t[0, 0]
    initial_t_z = t[2, 0]
    initial_params = [initial_theta, initial_t_x, initial_t_z]
    # Minimize the objective function
    result = minimize(objective, initial_params, args=(previous_cones_xz, current_cones_xz, x_weight, z_weight))
    # Extract the optimized parameters
    theta_opt, t_x_opt, t_z_opt = result.x
    if output_type == 'Rt':
        # Update the original 3D rotation matrix with the optimized theta
        R_y = np.array([
            [np.cos(theta_opt), 0, np.sin(theta_opt)],
            [0, 1, 0],
            [-np.sin(theta_opt), 0, np.cos(theta_opt)]
        ])
        R_modified = R_y
        # Update the translation vector in the X and Z directions
        t_modified = t.copy()
        t_modified[0, 0] = t_x_opt
        t_modified[2, 0] = t_z_opt
        return R_modified, t_modified
    else:
        return t_x_opt, t_z_opt, theta_opt

def least_squares_alpha(previous_cones, current_cones, R, t):
    numerator = 0.0
    denominator = 0.0
    for prev_p, curr_p in zip(previous_cones, current_cones):
        transformed_p = R @ prev_p
        diff = curr_p - transformed_p
        numerator += t.T @ diff
        denominator += np.linalg.norm(t)**2
    # Compute the optimal alpha
    alpha = numerator / denominator
    return alpha

















def bundleAdjustment(v_pts_2d, v_pts_2d_to_3d_idx, K, pts_3d, v_camera_g2o_poses, information_matrix, is_fix_map_pts=False, is_update_map_pts=True):

    # Change pose format from OpenCV to Sophus::SE3 equivalent in Python
    num_frames = len(v_camera_g2o_poses)
    v_T_cam_to_world = []
    for i in range(num_frames):
        T_cam_to_world = np.linalg.inv(v_camera_g2o_poses[i])
        v_T_cam_to_world.append(T_cam_to_world)

    # Init g2o optimizer
    solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
    optimizer = g2o.SparseOptimizer()
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)

    # Add camera pose vertices
    vertex_id = 0
    g2o_poses = []
    for i in range(num_frames):
        pose = g2o.VertexSE3Expmap()
        pose.set_id(vertex_id)
        pose.set_estimate(g2o.SE3Quat(v_T_cam_to_world[i][:3, :3], v_T_cam_to_world[i][:3, 3]))
        optimizer.add_vertex(pose)
        g2o_poses.append(pose)
        vertex_id += 1

    # Add camera intrinsics as parameter
    fx, _, cx, _, fy, cy, _, _, _ = K.ravel()
    camera = g2o.CameraParameters(fx, np.array([cx, cy]), 0)
    camera.set_id(0)
    optimizer.add_parameter(camera)

    # Add point vertices
    pts3dID_to_vertexID = {}
    g2o_points_3d = {}
    for pt3d_id, p in pts_3d.items():
        point = g2o.VertexPointXYZ()
        point.set_id(vertex_id)
        point.set_estimate(np.array([p[0], p[1], p[2]]))
        if is_fix_map_pts:
            point.set_fixed(True)
        optimizer.add_vertex(point)
        pts3dID_to_vertexID[pt3d_id] = vertex_id
        g2o_points_3d[pt3d_id] = point
        vertex_id += 1

    # Set information matrix
    information_matrix_eigen = np.array(information_matrix)

    # Add edges (error terms)
    edge_id = 0
    for i in range(num_frames):
        num_pts_2d = len(v_pts_2d[i])
        for j in range(num_pts_2d):
            p = v_pts_2d[i][j]
            pt3d_id = v_pts_2d_to_3d_idx[i][j]

            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_id(edge_id)
            edge.set_vertex(0, g2o_points_3d[pt3d_id])
            edge.set_vertex(1, g2o_poses[i])
            edge.set_measurement(np.array([p[0], p[1]]))
            edge.set_information(information_matrix_eigen)
            edge.set_parameter_id(0, 0)
            edge.set_robust_kernel(g2o.RobustKernelHuber())
            optimizer.add_edge(edge)
            edge_id += 1

    # Optimize
    optimizer.initialize_optimization()
    optimizer.optimize(50)

    # Retrieve results
    print(f"BA: Number of frames = {num_frames}, 3d points = {vertex_id - num_frames}")

    # Update camera poses
    for i in range(num_frames):
        pose_se3 = g2o_poses[i].estimate()
        R = pose_se3.rotation()
        t = pose_se3.translation()
        v_camera_g2o_poses[i][:3, :3] = np.array(R)
        v_camera_g2o_poses[i][:3, 3] = t

    # Update 3D points
    if is_update_map_pts:
        for pt3d_id, point in g2o_points_3d.items():
            p_res = point.estimate()
            pts_3d[pt3d_id] = [p_res[0], p_res[1], p_res[2]]

def mat2eigen(mat):
    return np.array(mat)
