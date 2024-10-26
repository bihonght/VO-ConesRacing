import numpy as np
import g2o
import time
import sophuspy as sophus
from collections import defaultdict

def transT_cv2sophus(T_cv):
    """
    Convert a transformation matrix from OpenCV format to Sophus SE3.

    Args:
        T_cv (np.ndarray): 4x4 transformation matrix in OpenCV format.

    Returns:
        sophus.SE3: Transformation in Sophus SE3 format.
    """
    R = T_cv[:3, :3]
    t = T_cv[:3, 3]
    return sophus.SE3(R, t)

def transT_sophus2cv(T_sophus):
    """
    Convert a transformation from Sophus SE3 format to OpenCV 4x4 matrix.

    Args:
        T_sophus (sophus.SE3): Transformation in Sophus SE3 format.

    Returns:
        np.ndarray: 4x4 transformation matrix in OpenCV format.
    """
    R = T_sophus.rotation().matrix()  
    t = T_sophus.translation()
    T_cv = np.eye(4)
    T_cv[:3, :3] = R
    T_cv[:3, 3] = t
    return T_cv


def bundle_adjustment(v_pts_2d, v_pts_2d_to_3d_idx, K, pts_3d, v_camera_poses, information_matrix, is_fix_map_points, is_update_map_points):
    """
    Perform bundle adjustment to optimize camera poses and 3D points.

    Args:
        v_pts_2d (list of list of np.ndarray): 2D points observed in each frame.
        v_pts_2d_to_3d_idx (list of list of int): Indices mapping 2D points to 3D points.
        K (np.ndarray): Camera intrinsic matrix (3x3).
        pts_3d (dict): Dictionary of 3D points.
        v_camera_poses (list of np.ndarray): Initial camera poses.
        information_matrix (np.ndarray): Information matrix for edge weights.
        is_fix_map_points (bool): Whether to fix the 3D points during optimization.
        is_update_map_points (bool): Whether to update 3D points after optimization.
    """
    num_frames = len(v_camera_poses)

    # Convert camera poses to Sophus SE3 format
    v_T_cam_to_world = [transT_cv2sophus(np.linalg.inv(v_camera_pose)) for v_camera_pose in v_camera_poses]

    # Initialize g2o optimizer
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)

    # Add camera poses as vertices to the optimizer
    g2o_poses = []
    for i, T_cam_to_world in enumerate(v_T_cam_to_world):
        pose_vertex = g2o.VertexSE3Expmap()
        pose_vertex.set_id(i)
        pose_vertex.set_estimate(g2o.SE3Quat(T_cam_to_world.rotationMatrix(), T_cam_to_world.translation()))
        # if num_frames > 1 and i == num_frames - 1:
        #     pose_vertex.set_fixed(True)  # Optionally fix the earliest frame
        optimizer.add_vertex(pose_vertex)
        g2o_poses.append(pose_vertex)

    # Add camera intrinsics as parameter
    camera = g2o.CameraParameters(K[0, 0], np.array([K[0, 2], K[1, 2]]), 0)
    camera.set_id(0)
    optimizer.add_parameter(camera)

    # Add 3D points as vertices to the optimizer
    g2o_points_3d = {}  # Dictionary to hold g2o vertices for 3D points
    vertex_id = num_frames
    pts3dID_to_vertexID = {}
    for pt3d_id, pt in pts_3d.items():
        point_vertex = g2o.VertexPointXYZ()
        point_vertex.set_id(vertex_id)
        point_vertex.set_estimate(pt)
        if is_fix_map_points:
            point_vertex.set_fixed(True)
        point_vertex.set_marginalized(True)
        optimizer.add_vertex(point_vertex)
        g2o_points_3d[pt3d_id] = point_vertex
        pts3dID_to_vertexID[pt3d_id] = vertex_id
        vertex_id += 1

    # Set information matrix
    information_matrix_eigen = information_matrix

    # Add edges representing the observations between camera poses and 3D points
    edge_id = 0
    for ith_frame in range(num_frames):
        for j, p_2d in enumerate(v_pts_2d[ith_frame]):
            pt3d_id = v_pts_2d_to_3d_idx[ith_frame][j]
            point_vertex = optimizer.vertex(pts3dID_to_vertexID[pt3d_id])
            pose_vertex = optimizer.vertex(ith_frame)

            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_id(edge_id)
            edge.set_vertex(0, point_vertex)  # XYZ point
            edge.set_vertex(1, pose_vertex)   # Camera pose
            edge.set_measurement(p_2d)
            edge.set_parameter_id(0, 0)
            edge.set_information(information_matrix_eigen)
            edge.set_robust_kernel(g2o.RobustKernelHuber())
            optimizer.add_edge(edge)
            edge_id += 1

    # Optimize the graph
    optimizer.initialize_optimization()
    optimizer.optimize(50)

    print(f"BA: Number of frames = {num_frames}, 3d points = {vertex_id - num_frames}")

    # Update camera poses and 3D points
    for i, pose_vertex in enumerate(g2o_poses):
        updated_pose = pose_vertex.estimate()
        v_camera_poses[i][:] = np.linalg.inv(transT_sophus2cv(updated_pose))  # Update original camera pose

    if is_update_map_points:
        for pt3d_id, point_vertex in g2o_points_3d.items():
            pts_3d[pt3d_id][:] = point_vertex.estimate()  # Update 3D points

    print("Bundle adjustment finished.")



























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
