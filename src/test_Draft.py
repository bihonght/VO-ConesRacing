import numpy as np
import cv2
import os

# Example points (ensure you replace this with your actual points)
refpoints1 = np.array([[100, 150], [200, 250], [300, 350], [400, 450]], dtype=np.float32)
refpoints2 = np.array([[105, 155], [205, 255], [305, 355], [405, 455]], dtype=np.float32)

# Ensure the points are in the correct shape and format (CV_32FC2)
# This means the shape should be (N, 2) and type should be np.float32
print(refpoints1.shape)  # Should print (N, 2)
print(refpoints2.shape)  # Should print (N, 2)
print(refpoints1.dtype)  # Should print float32
print(refpoints2.dtype)  # Should print float32

# Check if they are already in the correct format, otherwise, correct the shape
if refpoints1.shape[1] != 2:
    raise ValueError("refpoints1 should have shape (N, 2)")

if refpoints2.shape[1] != 2:
    raise ValueError("refpoints2 should have shape (N, 2)")

# Ensure dtype is correct, if not, convert them
if refpoints1.dtype != np.float32:
    refpoints1 = refpoints1.astype(np.float32)

if refpoints2.dtype != np.float32:
    refpoints2 = refpoints2.astype(np.float32)

# Sample homography decompositions and normals (ensure these are already computed)
# Example rotations and normals (dummy values for illustration)
rotations = [np.eye(3), np.eye(3)]  # Two dummy identity rotation matrices
normals = [np.array([[0], [0], [1]]), np.array([[0], [0], [1]])]  # Dummy normals

refpoints1 = refpoints1[:, np.newaxis, :]
refpoints2 = refpoints2[:, np.newaxis, :]
# Filter the homography decompositions by visible reference points
valid_indices = cv2.filterHomographyDecompByVisibleRefpoints(rotations, normals, refpoints1, refpoints2)

# Print the valid indices
print("Valid decomposition indices:", valid_indices)

def convert_image(frame_id):
    dir = "dataset/"  # Assuming the dataset is in the same directory as the code. Adjust as needed.  
    filename = os.path.join(dir, f"amz_{frame_id:03d}.jpg")
    if os.path.exists(filename):
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        mask = np.zeros_like(image)
        mask[25:600, :] = 255 # 75:600
        mask[465:800, 550:1250] = 0
        image = cv2.bitwise_and(image, mask)
        return image
    else:
        print("Could not find image file ", filename)
        return None

def read_image(frame_id):
    dir = "dataset/"  # Assuming the dataset is in the same directory as the code. Adjust as needed.  
    filename = os.path.join(dir, f"amz_{frame_id:03d}.jpg")
    if os.path.exists(filename):
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        print("Could not find image file ", filename)
        image = None

def write_image(image, index): 
    dir = "new_data/"
    filename = os.path.join(dir, f"{index:06d}.png")
    cv2.imwrite(filename, image)
    print("Image saved at ", filename)

id_out = 0
for frame_id in range(1, 159):  # Replace 101 with the total number of frames in the dataset
    img = convert_image(frame_id)
    if img is None:
        continue
    write_image(img, id_out)
    id_out += 1