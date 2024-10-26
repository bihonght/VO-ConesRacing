import cv2
import matplotlib.pyplot as plt
import numpy as np
def draw_matches(img1, img2, matches, kp1s, kp2s, title="Matches between Images"):
    img_matches = cv2.drawMatches(img1, kp1s, img2, kp2s, matches[:], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
    # Display the matches using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(img_matches)
    plt.title(title)
    plt.axis('off')  # Hide axes
    # plt.show()

def draw_matches_inliners_index(img1, img2, matches, kp1s, kp2s, inliers_index):
    matches = [matches[i] for i in inliers_index]
    img_matches = cv2.drawMatches(img1, kp1s, img2, kp2s, matches[:], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
    # Display the matches using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(img_matches)
    plt.axis('off')  # Hide axes
    plt.show()

def draw_keypoints(image, keypoints):
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    # Convert the image from BGR to RGB (if necessary, here it's grayscale so not required)
    # img_with_keypoints = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)
    # Display the image with keypoints using matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(img_with_keypoints, cmap='gray')
    plt.axis('off')  # Hide axes
    # plt.show()

def update_cone_location(fig, sc, cone_locations): 
    colors = np.where(cone_locations[:, 3] == -2, 'yellow', 'blue')  # Yellow for -2, blue for -1
    sc.set_offsets(np.c_[cone_locations[:, 0], cone_locations[:, 2]])  # Update X and Z coordinates
    sc.set_color(colors)
    fig.canvas.draw_idle()

def update_trajectory_plot(fig, ax, sc, x, z):
    # Get the current data from the line object
    sc.set_offsets(np.c_[x, z])
    ax.relim()  # Recompute limits based on data
    ax.autoscale_view(True, True, True)
    fig.canvas.draw_idle()

def init_cone_plot():
    # Create a figure and 2D axis
    fig, ax = plt.subplots()
    # Initialize the scatter plot on X (cones3D[:, 0]) and Z (cones3D[:, 2]) axes
    init_location = np.zeros((3, 3))
    sc = ax.scatter(init_location[:, 0], init_location[:, 2])
    # Set axis limits (adjust according to your scene)
    ax.set_xlim([-15, 15])
    ax.set_ylim([-5, 20])
    # Labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Z axis')
    ax.set_title('Cone Locations on X-Z Plane')
    ax.arrow(0, 0, 1, 0, head_width=0.1, head_length=0.2, fc='blue', ec='blue')  # X-axis arrow
    ax.arrow(0, 0, 0, 1, head_width=0.1, head_length=0.2, fc='red', ec='red')   # Z-axis arrow
    return fig, ax, sc

def init_trajectory_plot():
    fig, ax = plt.subplots()
    # Initialize an empty line for the trajectory
    line, = ax.plot([], [], 'b-', marker='o', label='Camera Trajectory')
    sc = ax.scatter([], [], color='b', marker='o', label='Camera Trajectory')
    # Set axis limits (adjust according to your scene)
    ax.set_xlim([-50, 40])
    ax.set_ylim([-30, 40])
    # Labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Z axis')
    ax.set_title('Camera Trajectory on X-Z Plane')
    ax.legend()
    return fig, ax, sc

def print_R_t(R, t):
    # Convert rotation matrix to rotation vector
    rvec, _ = cv2.Rodrigues(R)
    
    # Print the rotation vector and translation vector (transposed for compact printing)
    print("R_vec is:", rvec.T)
    print("t is:", t.T)

def draw_match_boxes(matched_boxes, prev_frame, curr_frame): 
    fig, axes = plt.subplots(1, 2)
    prev_img = prev_frame.image.copy()
    curr_img = curr_frame.image.copy()
    # Draw matched boxes and annotate with numbers
    for idx, pair in enumerate(matched_boxes):
        idbox1 = pair[0]
        idbox2 = pair[1] # matched_boxes[idbox1]
        box1 = prev_frame.cones[idbox1]
        box2 = curr_frame.cones[idbox2]
        # Draw box on img1
        x1_1, y1_1, x2_1, y2_1 = map(int, box1[:4])
        cv2.rectangle(prev_img, (x1_1, y1_1), (x2_1, y2_1), color=(0, 255, 0), thickness=2)
        # Annotate with a number
        cv2.putText(prev_img, str(idbox1), (x1_1, y1_1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        # Draw box on img2
        x1_2, y1_2, x2_2, y2_2 = map(int, box2[:4])
        cv2.rectangle(curr_img, (x1_2, y1_2), (x2_2, y2_2), color=(0, 255, 0), thickness=2)
        # Annotate with a number
        cv2.putText(curr_img, str(idbox1), (x1_2, y1_2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the images with the matched boxes and annotations
    axes[0].imshow(cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB))  # Convert to RGB for display in matplotlib
    axes[0].set_title("Image 1 with Matched Boxes")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB))  # Convert to RGB for display in matplotlib
    axes[1].set_title("Image 2 with Matched Boxes")
    axes[1].axis('off')