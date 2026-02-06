import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

def mask_to_edge(mask):
    """Convert binary mask to edge map"""
    max_pool = F.max_pool2d(mask, 3, stride=1, padding=1)
    min_pool = -F.max_pool2d(-mask, 3, stride=1, padding=1)
    return (max_pool - min_pool).clamp(0, 1)

def feydy_blurred_squared_distance(mask, kernel_size=101):
    """
    Compute Feydy-style blurred squared distances in a single convolution.

    mask: [B,1,H,W] binary mask (or edge map)
    kernel_size: size of the kernel (should be large for wide support)
    """
    B, C, H, W = mask.shape
    device = mask.device

    # --- create a pointy, wide-support kernel ---
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    r2 = xx**2 + yy**2  # squared distance from center

    # Normalize and make pointy: inverse + optional Gaussian weighting
    kernel = 1/ torch.sqrt(r2 + 1e-6)  # avoid div by zero
    # kernel = kernel / kernel.abs().sum()  # normalize
    kernel = kernel[None, None, :, :]     # [1,1,K,K]

    # Pad mask to keep output same size
    pad = kernel_size // 2
    mask_padded = F.pad(mask, (pad, pad, pad, pad))

    # Single convolution
    out = F.conv2d(mask_padded, kernel)

    return out

if __name__ == "__main__":
    # Load binary mask
    img_path = "data/consecutive_prediction/0617/0/00000.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = torch.tensor(img).float().cuda() / 255.0
    mask = mask[None, None, :, :]  # [1,1,H,W]

    edge = mask_to_edge(mask)
    bsd = feydy_blurred_squared_distance(edge, kernel_size=101)

    # Display
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].imshow(mask.squeeze().cpu().numpy(), cmap='gray')
    ax[0].set_title('Original Mask')
    ax[0].axis('off')
    ax[1].imshow(bsd.squeeze().cpu().numpy(), cmap='jet')
    ax[1].set_title('Feydy Blurred Squared Distance')
    ax[1].axis('off')
    plt.show()
