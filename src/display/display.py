import cv2
import matplotlib.pyplot as plt

def draw_matches(img1, img2, matches, kp1s, kp2s):
    img_matches = cv2.drawMatches(img1, kp1s, img2, kp2s, matches[:], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
    # Display the matches using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(img_matches)
    plt.axis('off')  # Hide axes
    plt.show()

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
    plt.show()


def print_R_t(R, t):
    # Convert rotation matrix to rotation vector
    rvec, _ = cv2.Rodrigues(R)
    
    # Print the rotation vector and translation vector (transposed for compact printing)
    print("R_vec is:", rvec.T)
    print("t is:", t.T)