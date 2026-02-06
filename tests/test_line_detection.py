import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch as th

if __name__ == "__main__":

    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, main_dir)

    # Load the binary mask (assuming it's already binary and in grayscale)
    def readRefImage(ref_img_file):
        cv_img = cv2.imread(ref_img_file)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img = cv2.inRange(cv_img, np.ones(3) * 128, np.ones(3) * 255)
        return img

    img_path = "data/extractions/bag_1/masks/00012.jpg"
    binary_mask = readRefImage(img_path)

    filtered_mask = cv2.medianBlur(binary_mask, 13)
    blurred = cv2.GaussianBlur(filtered_mask, (13, 13), 0)

    # kernel = np.ones((3, 3), np.uint8)

    # # For smoothing small holes, try closing: \
    # smoothed_mask = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel, iterations=2)

    edges = cv2.Canny(blurred, 50, 150)

    plt.figure(figsize=(10, 10))
    plt.title("Canny Edge Detection")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")  # Hide axis ticks
    plt.show()
    plt.close()

    # Perform Hough Line Transform to detect lines
    # Use cv2.HoughLinesP for probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=60, minLineLength=50, maxLineGap=40
    )

    # Create a copy of the binary mask to draw lines on it
    line_image = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    # Draw the lines
    # Check if any lines are detected
    assert lines is not None
    # Calculate lengths of all lines
    line_lengths = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        line_lengths.append((length, line[0]))

    # Sort lines by length in descending order
    line_lengths.sort(key=lambda x: x[0], reverse=True)

    # Limit the number of lines detected (e.g., top 2 longest lines)
    N = 2
    longest_lines = [line_lengths[i][1] for i in range(min(N, len(line_lengths)))]

    print(f"lines number: {len(longest_lines)}")
    print(f"debugging the longest lines {longest_lines}")
    # Show the result
    for x1, y1, x2, y2 in longest_lines:
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title("Original Binary Mask")
    plt.imshow(binary_mask, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Detected Lines")
    plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
    plt.show()

    longest_lines = np.array(longest_lines)
    print(longest_lines.shape)

    if longest_lines.shape[0] < 2:
        # Force skip cylinder or fallback
        print(
            "WARNING: Not enough lines found by Hough transform. Skipping cylinder loss."
        )

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

        detected_lines = th.from_numpy(np.stack((a, b), axis=-1))

    def lines_to_standard_form(lines):
        """
        Convert multiple lines (in x1, y1, x2, y2 format) to au + bv = 1.
        :param lines: NumPy array of shape (N, 4), where each row is [x1, y1, x2, y2]
        :return: NumPy array of shape (N, 2), where each row is [a, b]
        """
        lines = np.array(lines, dtype=np.float64)
        x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]

        # Handle vertical lines separately
        vertical_mask = (x2 - x1) == 0
        a = np.zeros_like(x1, dtype=np.float32)
        b = np.zeros_like(x1, dtype=np.float32)
        norm_factors = np.zeros_like(x1, dtype=np.float32)

        # For non-vertical lines
        slopes = np.divide(
            (y2 - y1), (x2 - x1), out=np.zeros_like(y1), where=~vertical_mask
        )
        intercepts = y1 - slopes * x1
        a[~vertical_mask] = -slopes
        b[~vertical_mask] = 1
        norm_factors[~vertical_mask] = intercepts

        # For vertical lines
        a[vertical_mask] = 1
        b[vertical_mask] = 0
        norm_factors[vertical_mask] = x1[vertical_mask]

        # Normalize so that c = 1
        a /= norm_factors
        b /= norm_factors

        return np.stack([a, b], axis=1)

    # Example usage
    longest_lines = np.array(longest_lines)
    print(f"debugging the shape: {longest_lines.shape}")
    line_params = lines_to_standard_form(longest_lines)
    print("Line parameters in the form au + bv = 1:")
    print(line_params)
