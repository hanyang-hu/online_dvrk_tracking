import torch
import torch.nn.functional as F

def morphological_convex_hull(batch_masks, max_iters=200):
    """
    Morphological convex hull for a batch of binary masks.
    Args:
        batch_masks: torch.Tensor of shape [B, H, W], dtype {bool, uint8, float}
                     Nonzero/True values are treated as foreground (1).
        max_iters: maximum iterations per directional growth (safety).
    Returns:
        hulls: torch.Tensor of shape [B, H, W], dtype same as input (0/1).
    """
    assert batch_masks.dim() == 3, "Expected [B, H, W]"
    device = batch_masks.device
    B, H, W = batch_masks.shape

    # normalize to uint8 (0/1)
    A = batch_masks.clone()
    if A.dtype != torch.uint8 and A.dtype != torch.bool and A.dtype != torch.float:
        A = A.to(torch.uint8)
    A = (A != 0).to(torch.uint8)  # [B,H,W] 0/1 uint8

    # If empty input => return zeros
    nonzero_any = A.view(B, -1).any(dim=1)
    if not nonzero_any.any():
        return A.to(batch_masks.dtype)

    # Define the 3x3 structuring elements from the lecture
    # Interpreting columns from the slides:
    # B1 column: rows "100", "100", "100"  -> [[1,0,0],[1,0,0],[1,0,0]]
    # B2 column: rows "111", "000", "000"  -> [[1,1,1],[0,0,0],[0,0,0]]
    # B3 column: rows "001", "001", "001"  -> [[0,0,1],[0,0,1],[0,0,1]]
    # B4 column: rows "000", "000", "111"  -> [[0,0,0],[0,0,0],[1,1,1]]
    # C column : rows "000", "010", "000"  -> [[0,0,0],[0,1,0],[0,0,0]]
    B1 = torch.tensor([[1,0,0],[1,0,0],[1,0,0]], dtype=torch.uint8, device=device)
    B2 = torch.tensor([[1,1,1],[0,0,0],[0,0,0]], dtype=torch.uint8, device=device)
    B3 = torch.tensor([[0,0,1],[0,0,1],[0,0,1]], dtype=torch.uint8, device=device)
    B4 = torch.tensor([[0,0,0],[0,0,0],[1,1,1]], dtype=torch.uint8, device=device)
    C  = torch.tensor([[0,0,0],[0,1,0],[0,0,0]], dtype=torch.uint8, device=device)

    kernels = [B1, B2, B3, B4]  # list of 3x3 uint8 tensors

    # helper to compute erosion(X, kernel) in batch using unfold
    # X: [B,H,W] uint8 (0/1)
    # kernel: [3,3] uint8 (0/1)
    def erosion(X, kernel):
        # pad with zeros (so kernel fits at edges)
        # unfold will give patches of size 3*3
        Xf = X.unsqueeze(1).float()  # [B,1,H,W] -> float for conv/unfold ops
        # use F.unfold: requires [B, C, H, W]
        patches = F.unfold(Xf, kernel_size=3, padding=1)  # [B, 9, H*W]
        # kernel mask
        km = kernel.view(-1).to(X.device).float()  # [9]
        required = int(km.sum().item())
        if required == 0:
            # kernel has no '1' -> erosion is all ones (since no constraints)
            return torch.ones_like(X, dtype=torch.uint8)
        # compute per-location sum of matched ones
        # patches shape [B,9,H*W]; multiply by km and sum
        sums = (patches * km.view(1, -1, 1)).sum(dim=1)  # [B, H*W]
        out = (sums == required).view(B, H, W)
        return out.to(torch.uint8)

    # hit-or-miss: (X ~ (k, C)) = erosion(X, k) & erosion(~X, C)
    def hit_or_miss(X, k, Cmask):
        e1 = erosion(X, k)              # [B,H,W] 0/1
        e2 = erosion(1 - X, Cmask)      # erosion on complement
        return (e1 & e2).to(torch.uint8)

    # For each of 4 directions, iterate until convergence:
    Xi_conv_list = []
    for k in kernels:
        Xi = A.clone()  # Xi_0 = A
        prev = torch.zeros_like(Xi)
        count = 0
        # iterate Xi_k = (Xi_{k-1} ~ (k, C)) ∪ A until Xi_k == Xi_{k-1}
        for _ in range(max_iters):
            hm = hit_or_miss(Xi, k, C)      # points to be added
            Xi_new = (hm | A).to(torch.uint8)
            count += 1
            if torch.equal(Xi_new, Xi) or count >= max_iters:
                Xi = Xi_new
                break
            Xi = Xi_new
        Xi_conv_list.append(Xi)

    # union of the four converged arrays
    union = torch.zeros_like(A)
    for Xi in Xi_conv_list:
        union = union | Xi

    # Trim union to bounding box of original component per sample
    # For each sample, find bbox of A (original). If empty -> keep zeros.
    hull = torch.zeros_like(union)
    for i in range(B):
        ai = A[i]
        if ai.any():
            rows = torch.any(ai, dim=1).nonzero(as_tuple=False).view(-1)
            cols = torch.any(ai, dim=0).nonzero(as_tuple=False).view(-1)
            rmin, rmax = rows.min().item(), rows.max().item()
            cmin, cmax = cols.min().item(), cols.max().item()
            # clip union to bbox
            hull[i, rmin:rmax+1, cmin:cmax+1] = union[i, rmin:rmax+1, cmin:cmax+1]
        else:
            # leave as zero
            pass

    # Return same dtype as input (but keep 0/1)
    out_dtype = batch_masks.dtype
    if out_dtype == torch.bool:
        return (hull != 0)
    elif out_dtype == torch.uint8:
        return hull
    else:
        return hull.to(out_dtype)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    mask_file = "data/consecutive_prediction/rw1/3/00003.png"
    mask = torch.tensor([plt.imread(mask_file)])
    mask = mask > 0.5
    mask = mask.repeat(10, 1, 1)  # Repeat the mask 10 times along the batch dimension
    print(mask.shape)  # Should print: torch.Size([10, 256, 256])
    h = morphological_convex_hull(mask)

    # Display the first mask and its convex hull
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Mask')
    plt.imshow(mask[0].cpu().numpy(), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Convex Hull')
    plt.imshow(h[0].cpu().numpy(), cmap='gray')
    plt.show()