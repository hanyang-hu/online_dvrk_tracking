import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch as th

from diffcali.utils.detection_utils import detect_lines


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the clicked coordinates
        params["keypoints"].append((x, y))
        # Display the point on the image
        cv2.circle(params["image"], (x, y), 3, (0, 1, 0), -1)
        cv2.imshow("Reference Image", params["image"])


def get_reference_keypoints(image, num_keypoints=2):
    # Load the image
    # image = cv2.imread(ref_img_path)
    # clone = image.copy()
    image = image.cpu().numpy()

    params = {"image": image, "keypoints": []}

    # Set up the mouse callback
    cv2.namedWindow("Reference Image")
    cv2.setMouseCallback("Reference Image", click_event, params)

    print(f"Please click {num_keypoints} points on the reference image.")

    while True:
        cv2.imshow("Reference Image", params["image"])
        key = cv2.waitKey(1) & 0xFF

        # Break when the required number of keypoints are collected
        if len(params["keypoints"]) >= num_keypoints:
            break

        # Exit on 'q' key press
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    return params["keypoints"]


def get_reference_keypoints_auto(ref_img_path, num_keypoints=2, ref_img=None):
    # Read data
    if ref_img_path is None and ref_img is None:
        raise ValueError("Either ref_img_path or ref_mask must be provided.")
    cv_img = ref_img if ref_img_path is None else cv2.imread(ref_img_path)

    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    ref_img = cv2.inRange(cv_img, np.ones(3) * 128, np.ones(3) * 255) 
    binary_mask = ref_img.astype(np.uint8)
    region_mask = (binary_mask > 0).astype(np.uint8)
    max_corners = num_keypoints        # Maximum number of corners to find
    quality_level = 0.1     # Minimum quality of corners (lower means more corners)
    min_distance = 5         # Minimum distance between detected corners
    block_size = 15          # Size of the neighborhood considered for corner detection
    
    corners = cv2.goodFeaturesToTrack(
        binary_mask,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
        mask=region_mask
    )

    output_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Draw the corners
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(output_image, (int(x), int(y)), radius=4, color=(0, 0, 255), thickness=-1)  # Red circles for corners

    # # Visualize the result
    # plt.figure(figsize=(8, 8))      
    # plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    # plt.title("Corners of the Mask")
    # plt.axis('off')
    # plt.pause(2)  # Pause for 4 seconds
    # plt.close('all')

    ref_keypoints = corners
    # print(f"detected keypoint shape: {ref_keypoints.shape}")  # [2, 1, 2] will squeeze 
     
    return ref_keypoints.squeeze(1).tolist()  # Convert to list of tuples


def get_det_line_params(mask):
    ref_mask_np = mask.detach().cpu().numpy()
    longest_lines = detect_lines(ref_mask_np, output=True)
    longest_lines = np.array(longest_lines, dtype=np.float64)

    if longest_lines.shape[0] < 2:
        # print(
        #     "WARNING: Not enough lines found by Hough transform. Skipping cylinder loss."
        # )
        ret = None

    else:
        # print(f"debugging the longest lines {longest_lines}")
        x1 = longest_lines[:, 0]
        y1 = longest_lines[:, 1]
        x2 = longest_lines[:, 2]
        y2 = longest_lines[:, 3]
        # print(f"debugging the end points x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
        # Calculate line parameters (a, b, c) for detected lines
        a = y2 - y1
        b = x1 - x2
        c = x1 * y2 - x2 * y1  # Determinant for the line equation

        # Normalize to match the form au + bv = 1
        # norm = c + 1e-6  # Ensure no division by zero
        norm = np.abs(c)  # Compute the absolute value
        norm = np.maximum(norm, 1e-6)  # Clamp to a minimum value of 1e-6
        a /= norm
        b /= norm

        # Stack line parameters into a tensor and normalize to match au + bv = 1 form
        detected_lines = th.from_numpy(np.stack((a, b), axis=-1)).cuda()
        ret = detected_lines

    return ret



if __name__ == "__main__":
    # Example usage
    ref_img_path = "data/consistency_evaluation/easy/0/00026.jpg"
    keypoints = get_reference_keypoints_auto(ref_img_path)

