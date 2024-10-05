
import os 
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import * 
from vo.State import State
from odom import compute_weighted_scale_factor, pixel_reproject_err
def draw_match_boxes(matched_boxes, prev_frame, curr_frame): 
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # Draw matched boxes and annotate with numbers
    for idx, idbox1, in enumerate(matched_boxes):
        idbox2 = matched_boxes[idbox1]
        box1 = prev_frame.cones[idbox1]
        box2 = curr_frame.cones[idbox2]
        # Draw box on img1
        x1_1, y1_1, x2_1, y2_1 = map(int, box1[:4])
        cv2.rectangle(prev_frame.image, (x1_1, y1_1), (x2_1, y2_1), color=(0, 255, 0), thickness=2)
        # Annotate with a number
        cv2.putText(prev_frame.image, str(idx + 1), (x1_1, y1_1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        # Draw box on img2
        x1_2, y1_2, x2_2, y2_2 = map(int, box2[:4])
        cv2.rectangle(curr_frame.image, (x1_2, y1_2), (x2_2, y2_2), color=(0, 255, 0), thickness=2)
        # Annotate with a number
        cv2.putText(curr_frame.image, str(idx + 1), (x1_2, y1_2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the images with the matched boxes and annotations
    axes[0].imshow(cv2.cvtColor(prev_frame.image, cv2.COLOR_BGR2RGB))  # Convert to RGB for display in matplotlib
    axes[0].set_title("Image 1 with Matched Boxes")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(curr_frame.image, cv2.COLOR_BGR2RGB))  # Convert to RGB for display in matplotlib
    axes[1].set_title("Image 2 with Matched Boxes")
    axes[1].axis('off')

def match_bounding_boxes(matches, prev_frame, curr_frame):
    bb_temp_matches = []
    c = 0
    for match_pair in matches:
        if True: 
            best_match = match_pair #[0]
            prev_img_box_id = -1
            curr_img_box_id = -1
            # Loop through all the BBoxes in previous image and find the box ID of the 'query' keypoint (best match keypoint in previous image)
            for bbox in (prev_frame.cones):
                # key_pt = prev_frame.keypoints[best_match.queryIdx]
                key_pt = prev_frame.kptsAbove300[best_match.queryIdx]
                if point_in_box(key_pt.pt[0], key_pt.pt[1], bbox[:4]):
                    # bbox.keypoints.append(key_pt)
                    prev_box_color = bbox[4]
                    prev_img_box_id = bbox[5]
                    break

            # Loop through all the BBoxes in current image and find the box ID of the 'train' keypoint (best match keypoint in current image)
            for i, bbox in enumerate(curr_frame.cones):
                # key_pt = curr_frame.keypoints[best_match.trainIdx]
                key_pt = curr_frame.kptsAbove300[best_match.trainIdx]
                if point_in_box(key_pt.pt[0], key_pt.pt[1], bbox[:4]):
                    # bbox.keypoints.append(key_pt)
                    curr_box_color = bbox[4]
                    curr_img_box_id = bbox[5]
                    break

            # Store the box ID pairs in a temporary list
            if prev_img_box_id != -1 and curr_img_box_id != -1 and (prev_box_color == curr_box_color):  # Exclude pairs which are not part of either BBoxes
                bb_temp_matches.append((prev_img_box_id, curr_img_box_id))
        # STEP 2: For each BBox pair count the number of keypoint matches
    count_map = defaultdict(int)

    # Loop through each element in the temporary matches list
    for prev_box_id, curr_box_id in bb_temp_matches:
        count_map[(prev_box_id, curr_box_id)] += 1

    # STEP 3: The BBox pair with highest number of keypoint match occurrences is the best matched BBox pair
    bb_best_matches = {}
    unique_keys = set(prev_box_id for prev_box_id, _ in bb_temp_matches)

    for prev_box_id in unique_keys:
        max_keypoint_count = -1
        best_match = (-1, -1)

        # Loop through all the BBox matched pairs and find the ones with highest keypoint occurrences
        for (prev_box, curr_box), count in count_map.items():
            if prev_box == prev_box_id and count > max_keypoint_count:
                max_keypoint_count = count
                best_match = (prev_box, curr_box)
        

        if best_match != (-1, -1):  # Exclude pairs which are not part of either BBoxes
            bb_best_matches[best_match[0]] = best_match[1]
            # Store the matched pixels in the respective frames
            prev_frame.matched_pixels.append(prev_frame.cones_3pixels[best_match[0], 4:6])
            curr_frame.matched_pixels.append(curr_frame.cones_3pixels[best_match[1], 4:6])
            # Store the matched cones in 3D world in the respective frames
            prev_frame.matched_3Dpoints.append(prev_frame.cone_3Dpoints[best_match[0], 6:9])
            curr_frame.matched_3Dpoints.append(curr_frame.cone_3Dpoints[best_match[1], 6:9])
    return bb_best_matches

def main():
    frame_count = 1
    camera_pose = np.eye(4)
    start = time.process_time()
    while True:
        print("FRAME_COUNT : " + str(frame_count))
        if frame_count == 159: 
            break
        # print(frame_count)
        prev_state = State(frame_count, K)
        curr_state = State(frame_count+1, K) 
        prev_state.detect_keypoints_and_descriptors()
        curr_state.detect_keypoints_and_descriptors()

        index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2, 
                        trees = 5
                        )
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        TEST = True
        if not TEST: 
            matches = flann.knnMatch(prev_state.descriptors, curr_state.descriptors, k=2)
            good_matches = [] # Apply Lowe's ratio test
            for m, n in matches:
                if m.distance < 0.95 * n.distance:
                    good_matches.append(m)  
            pts1 = np.float32([prev_state.keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            pts2 = np.float32([curr_state.keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            # Use RANSAC to find the homography matrix and remove outliers
            if len(pts1) >= 4:  # At least 4 points are required to compute the homography
                H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 8.0)
                # H, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=0.5)
                matches_mask = mask.ravel().tolist()
            else:
                print("Not enough points to compute homography.")
                matches_mask = None
            inlier_matches = []
            for i, m in enumerate(good_matches): 
                if matches_mask[i]:  # Inlier
                    inlier_matches.append(m)
        
        if TEST: 
            matches = flann.knnMatch(prev_state.desAbove300, curr_state.desAbove300, k=2)
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2: 
                    if m_n[0].distance < 0.9 * m_n[1].distance:
                        good_matches.append(m_n[0])  
            # good_matches = [m for m, n in matches if m.distance < 0.95 * n.distance]

            # Ensure there are enough matches to proceed
            if len(good_matches) < 4:
                print("Not enough points to compute homography.")
                inlier_matches = []
            else:
                # Convert keypoints to numpy arrays directly in one step
                pts1 = np.float32([prev_state.kptsAbove300[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([curr_state.kptsAbove300[m.trainIdx].pt for m in good_matches])
                # Use RANSAC to find the homography matrix and remove outliers
                H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 6.0)
                # Extract inlier matches based on the mask
                inlier_matches = [m for m, msk in zip(good_matches, mask.ravel()) if msk]
        
        bb_best_matches = match_bounding_boxes(inlier_matches, prev_state, curr_state)
        print("Best matches: ", bb_best_matches)
        # print("Matched ")
        # draw_match_boxes(bb_best_matches, prev_state, curr_state)
        # plt.show()
        
        # print("Pixels Matching: ", np.array(prev_state.matched_pixels))
        # print("Matched pixels", np.array(curr_state.matched_pixels))
        # M = np.hstack((np.eye(3), np.zeros((3,1))))
        # error1 = pixel_reproject_err(M, np.array(prev_state.matched_3Dpoints).reshape(-1, 3), K, np.array(prev_state.matched_pixels).reshape(-1, 2) )
        # print("Error 1: ", sum(error1)) if sum(error1) > 0.00000001 else None
        # error2 = pixel_reproject_err(M, np.array(curr_state.matched_3Dpoints).reshape(-1, 3), K, np.array(curr_state.matched_pixels).reshape(-1, 2) )
        # print("Error 2: ", sum(error2)) if sum(error2) > 0.00000001 else None
        if True: 
            matching = 'FLANN'
            features1, features2 = [], []
            if matching == 'FLANN':
                index_params['multi_probe_level'] = 1
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(prev_state.descriptors, curr_state.descriptors, k=2)
                good_matches = []
                for i, (m, n) in enumerate(matches):
                    if m.distance < 0.8 * n.distance:
                        # good_matches.append(m)
                        x1, y1 = prev_state.keypoints[m.queryIdx].pt
                        x2, y2 = curr_state.keypoints[m.trainIdx].pt
                        features1.append([x1, y1, 1])
                        features2.append([x2, y2, 1])

            if matching == 'BF':
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
                matches = bf.match(prev_state.descriptors, curr_state.descriptors)
                matches = sorted(matches, key = lambda x: x.distance)
                for i, m in enumerate(matches[:200]):
                    # good_matches.append(m)
                    x1, y1 = prev_state.keypoints[m.queryIdx].pt
                    x2, y2 = curr_state.keypoints[m.trainIdx].pt
                    features1.append([x1, y1, 1])
                    features2.append([x2, y2, 1])
            features1 = np.ascontiguousarray(features1)
            features2 = np.ascontiguousarray(features2)

            essential_mat, _ = cv2.findEssentialMat(features1[:, :2], features2[:, :2], K, method=cv2.RANSAC, prob=0.9999, threshold=0.5)
            _, new_R, new_t, mask = cv2.recoverPose(essential_mat, features1[:, :2], features2[:, :2], K)

            if np.linalg.det(new_R) < 0:
                new_R = -new_R
                new_t = -new_t
            scale = 1
            # matched1_boxes3D = np.array(prev_state.matched_3Dpoints)
            # matched1_boxes = np.array(prev_state.matched_pixels)
            # matched2_boxes = np.array(curr_state.matched_pixels)
            # scale = compute_weighted_scale_factor(matched1_boxes3D, matched1_boxes, matched2_boxes, new_R, new_t, K)
            new_pose = np.column_stack((new_R, scale*new_t))
            # print(pixel_reproject_err(new_pose.ravel(), matched1_boxes3D.reshape(-1, 3), K, matched2_boxes.reshape(-1, 2)) )
            # get_approximate_odometry(matched2_boxes, matched1_boxes3D, new_pose, K, )
            new_pose = np.vstack((new_pose, np.array([0,0,0,1])))
            camera_pose = camera_pose @ new_pose
            x_coord = camera_pose[0, -1]
            z_coord = camera_pose[2, -1]

            plt.scatter(x_coord, -z_coord, color='b') 
            plt.pause(0.00001)
            frm = cv2.resize(prev_state.image, (0,0), fx=0.5, fy=0.5)
            cv2.imshow('Frame', frm)
            ########################################################
        frame_count += 1
    
    print('\n\nTime taken: ', (time.process_time() - start))
    cv2.destroyAllWindows()
    plt.show()

if __name__ == "__main__":
    main()
