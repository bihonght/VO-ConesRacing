import torch
import cv2
import numpy as np
from common import common

def keypoint_detect(model,img,img_size=80):
    image = img
    image_size = (img_size, img_size)
    h, w, _ = image.shape
    image = cv2.resize(image, image_size)
    image = (image.transpose((2, 0, 1)) / 255.0)[np.newaxis, :]
    image = torch.from_numpy(image).type('torch.FloatTensor')
    output = model(image)
    tensor_output=output[1][0].cpu().data
    return tensor_output

def bbox_detection(image, model, model_kpt):
    cone_types = [0, 1]   #   0: blue_cone 1: yellow_cone 2: orange_cone 3: large_orange_cone 4: unknown_cone
    conf_threshold = 0.75   # Set confidence threshold

    results = model.predict(image)
    # Extract bounding boxes, classes, names, and confidences
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()
    # Iterate through the results
    blue_boxes = []
    yellow_boxes = []
    for box, cls, conf in zip(boxes, classes, confidences):
        if int(cls) in cone_types and conf > conf_threshold:
            x1, y1, x2, y2 = map(int, box)
            blue_boxes.extend([x1, y1, x2, y2]) if int(cls) == 0 else None
            yellow_boxes.extend([x1, y1, x2, y2]) if int(cls) == 1 else None
            # Detect keypoints using the keypoint model
            cone_img = image[y1:y2, x1:x2]            
            h, w, _ = cone_img.shape
            i = 0
            points = keypoint_detect(model_kpt, cone_img)
            for pt in np.array(points):
                x_original = int(pt[0] * w + x1)
                y_original = int(pt[1] * h + y1)
                i += 1

def matching_bouding_boxes(matches, curr_boxes, prev_boxes, curr_keypoints, prev_keypoints):
    N = prev_boxes.shape[0]
    M = curr_boxes.shape[0]
    counts = np.zeros((N, M))

    for match_pair in matches:
        # Assuming you've already applied Lowe's ratio test
        # if match_pair[0].distance < 0.95 * match_pair[1].distance:
        prev_key_pt = prev_keypoints[match_pair.queryIdx]
        curr_key_pt = curr_keypoints[match_pair.trainIdx]
        prev_box_id = find_box_containing(prev_boxes, prev_key_pt.pt[0], prev_key_pt.pt[1])
        curr_box_id = find_box_containing(curr_boxes, curr_key_pt.pt[0], curr_key_pt.pt[1])
        if prev_box_id!= -1 and curr_box_id!= -1:
            counts[prev_box_id, curr_box_id] += 1
        
    # Determine the top B for each A / Determine the curr box id with highest matching for each prev box  
    max_prev_boxes = np.zeros((N))
    top_pick_for_prev = [[] for _ in range(N)]
    for j in range(M): 
        for i in range(N): 
            if counts[i, j] > max_prev_boxes[i]:
                max_prev_boxes[i] = counts[i, j] # highest matching for each box in the previous
                top_pick_for_prev[i] = [j]
            elif counts[i, j] == max_prev_boxes[i] and counts[i, j] > 0:
                top_pick_for_prev[i].append(j)

    # Determine the top A for each B/ Determine the prev box id with highest matching for each curr box
    max_curr_boxes = np.zeros((M))
    top_pick_for_curr = [[] for _ in range(M)]
    for i in range(N): 
        for j in range(M): 
            if counts[i, j] > max_curr_boxes[j]:
                max_curr_boxes[j] = counts[i, j]
                top_pick_for_curr[j] = [i]
            elif counts[i, j] == max_curr_boxes[j] and counts[i, j] > 0:
                top_pick_for_curr[j].append(i)

    selected_pairs = []
    for i in range(N):
        if max_prev_boxes[i] == 0:
            continue
        if len(top_pick_for_prev[i]) != 1: 
            continue # multiple boxes matching for same index
        
        j = top_pick_for_prev[i][0]
        if counts[i, j] == max_curr_boxes[j] and len(top_pick_for_curr[j]) == 1:
            selected_pairs.append((i, j))
    return selected_pairs

def find_box_containing(boxes, x, y): 
    for i, box in enumerate(boxes):
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            return i
    return -1