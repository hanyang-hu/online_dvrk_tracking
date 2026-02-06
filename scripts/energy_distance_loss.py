import torch
import torch.nn.functional as F
import numpy as np
import FastGeodis
import cv2
import time
import matplotlib.pyplot as plt
import skfmm
import kornia

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    # Read binary mask (as grayscale image)
    img_dir_lst = [
        "data/consecutive_prediction/0617/10/00010.png",
        "data/consecutive_prediction/0617/30/00030.png",
        "data/consecutive_prediction/0617/50/00050.png",
        "data/consecutive_prediction/0617/100/00100.png",
        "data/consecutive_prediction/0617/0/00000.png",
    ]
    img_lst = [
        cv2.resize(cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE), (64, 48)) for img_dir in img_dir_lst
    ]

    img_batch = np.stack(img_lst, axis=0)
    mask = torch.from_numpy(img_batch).cuda().float().unsqueeze(1) / 255 # [B, 1, H, W]

    B = 70
    mask = mask.repeat(B // mask.shape[0], 1, 1, 1)  # [B, 1, H, W]
    
    # Use kornia distance transform for comparison
    ret = kornia.contrib.distance_transform(mask, kernel_size=33)

    start_time = time.time()
    ret = kornia.contrib.distance_transform(mask, kernel_size=33)
    end_time = time.time()

    # Resize to (640, 480) for fair comparison
    mask = F.interpolate(mask, size=(480, 640), mode='bilinear', align_corners=False)
    ret = F.interpolate(ret, size=(480, 640), mode='bilinear', align_corners=False)
    ret *= 10

    print("Kornia DT Time for batch size {}: {:.4f} seconds".format(mask.shape[0], end_time - start_time))

    # Plot results
    for i in range(min(2, B)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(mask[i, 0].cpu().numpy(), cmap='gray')
        plt.title('Input Mask')
        plt.axis('off')

        plt.subplot(2, 2, i + 3)
        plt.imshow(ret[i, 0].cpu().numpy())
        plt.title('Kornia DT')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
