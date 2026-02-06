import cv2
import matplotlib.pyplot as plt
import numpy as np

def detect_lines(mask, output=False):
    """Remember to tune the parameters of single optimizer as well"""
    ref_mask_np = mask
    if ref_mask_np.dtype != np.uint8:
        ref_mask_np = (ref_mask_np * 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(ref_mask_np, (13, 13), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, rho=0.5, theta=np.pi / 360, threshold=50, minLineLength=70, maxLineGap=10
    )

    if lines is not None:
        # Extract all line endpoints (shape: [num_lines, 1, 4])
        lines = lines[:, 0, :]  # Shape now: [num_lines, 4]

        # Vectorized calculation of line lengths
        x1 = lines[:, 0]
        y1 = lines[:, 1]
        x2 = lines[:, 2]
        y2 = lines[:, 3]

        # Compute lengths for all lines (no loop)
        lengths = np.hypot(x2 - x1, y2 - y1)

        # Get the indices of the lines sorted by length (descending order)
        sorted_indices = np.argsort(-lengths)

        # Select the top N longest lines
        N = 2
        top_indices = sorted_indices[:N]

        # Extract the top N longest lines
        longest_lines = lines[top_indices]
    else:
        longest_lines = []

    if output == True:
        return longest_lines
    # Instant checking detected lines:
    line_image = cv2.cvtColor(ref_mask_np, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in longest_lines:
        cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # plt.title("Detected Lines")
    # plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
    # plt.show()
    # plt.close("all")


