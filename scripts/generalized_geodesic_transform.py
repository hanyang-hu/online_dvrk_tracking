import torch
import torch.nn.functional as F
import numpy as np
import FastGeodis
import cv2
import time
import matplotlib.pyplot as plt
import skfmm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def compute_weighting_mask(shape, center_weight=0.5, edge_weight=1.0):
    """
    Copied from your single-sample code: creates a weighting mask for the MSE.
    shape: (H,W)
    """
    h, w = shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
    max_distance = np.sqrt(center_y**2 + center_x**2)
    normalized_distance = distance / max_distance
    weights = edge_weight + (center_weight - edge_weight) * (
        1 - normalized_distance
    )
    weighting_mask = torch.from_numpy(weights).float().cuda()

    return weighting_mask


def compute_edge_aware_cost(image_gray, lambda_val=1.0):
    image_gray = image_gray.astype(np.float32) / 255.0
    grad_x = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    cost_map = 1 + lambda_val * gradient_mag
    return torch.from_numpy(cost_map).float().cuda()


def compute_distance_map(ref_mask):
    ref_mask_np = ref_mask.detach().cpu().numpy().astype(np.float32)
    distance_map = skfmm.distance(ref_mask_np == 0)
    distance_map[ref_mask_np == 1] = 0
    return torch.from_numpy(distance_map).float().to(ref_mask.device)


def mask_to_edge(mask):
    """Convert binary mask to edge in PyTorch"""
    max_pool = F.max_pool2d(mask, 3, stride=1, padding=1)
    min_pool = -F.max_pool2d(-mask, 3, stride=1, padding=1)
    return (max_pool - min_pool).clamp(0, 1)


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
    mask = 1 -  torch.from_numpy(img_batch).cuda().float().unsqueeze(0) / 255 # [1, B, H, W]
    # mask = mask_to_edge(mask)
    weighting_mask = compute_edge_aware_cost(img_batch[0]).unsqueeze(0).unsqueeze(0).cuda()

    print("Mask shape:", mask.shape)

    v, lamb, iterations = 1e10, 0.0, 1

    # warm up the GPU
    ret = FastGeodis.generalised_geodesic2d(
        torch.zeros_like(mask),
        mask,  
        v, 
        lamb,
        iterations
    )

    # B = 70

    # # Make multiple copies of the mask and concatenate along batch dimension
    # mask = mask.repeat(1, B, 1, 1)

    # start_time = time.time()
    # ret = FastGeodis.generalised_geodesic2d(
    #     torch.zeros_like(mask),
    #     mask,
    #     v, 
    #     lamb,
    #     iterations
    # )
    # end_time = time.time()
    # print(f"Time taken for generalized geodesic transform: {end_time - start_time:.4f} seconds")

    # print("Input shape:", mask.shape)
    # print("Output shape:", ret.shape)

    # mask = mask.reshape(B, 1, mask.shape[2], mask.shape[3])[0].unsqueeze(0)
    # ret = ret.reshape(B, 1, mask.shape[2], mask.shape[3])[0].unsqueeze(0)

    # # Also scale the distance values accordingly
    # scale_y = 10
    # scale_x = 10
    # ret = ret * torch.sqrt(torch.tensor(scale_y**2 + scale_x**2)).cuda()

    # Compare with skfmm
    # cv_img = cv2.imread(img_dir_lst[0])
    # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    # img_fmm = cv2.inRange(cv_img, np.ones(3) * 128, np.ones(3) * 255) / 255.0
    # mask_fmm = torch.Tensor(img_fmm).cuda()
    # ret_fmm = compute_distance_map(mask_fmm)

    # start_time_fmm = time.time()
    # ret_fmm = compute_distance_map(mask_fmm)
    # end_time_fmm = time.time()
    # print(f"Time taken for skfmm distance transform: {end_time_fmm - start_time_fmm:.4f} seconds")
    # print("Speedup:", (end_time_fmm - start_time_fmm) / (end_time - start_time))

    # # Display the original mask and the transformed image using plt subfigures
    # fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    # ax[0].imshow(mask[0,0].squeeze().cpu().numpy(), cmap='gray')
    # ax[0].set_title('Original Mask')
    # ax[0].axis('off')
    # ax[1].imshow(weighting_mask.squeeze().cpu().numpy())
    # ax[1].set_title('Weighting Mask')
    # ax[1].axis('off')
    # ax[2].imshow(ret[0,0].squeeze().cpu().numpy())
    # ax[2].set_title('Euclidean Distance Transform')
    # ax[2].axis('off')
    # ax[3].imshow(ret_fmm.squeeze().cpu().numpy())
    # ax[3].set_title('Fast Marching Method')
    # ax[3].axis('off')
    # plt.tight_layout()
    # plt.show()

    # Display the original mask and EDT for the 5 samples
    fig, ax = plt.subplots(2, 5, figsize=(20, 12))
    for i in range(5):
        ax[0, i].imshow(mask[0, i].squeeze().cpu().numpy(), cmap='gray')
        ax[0, i].set_title(f'Original Mask {i}')
        ax[0, i].axis('off')

        ax[1, i].imshow(ret[0, i].squeeze().cpu().numpy())
        ax[1, i].set_title(f'Geodesic Transform {i}')
        ax[1, i].axis('off')

    plt.tight_layout()
    plt.show()


