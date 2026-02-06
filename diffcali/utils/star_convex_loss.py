import torch
import torch.nn.functional as F

def mask_to_edge(mask):
    """Convert binary mask to edge in PyTorch"""
    max_pool = F.max_pool2d(mask, 3, stride=1, padding=1)
    min_pool = -F.max_pool2d(-mask, 3, stride=1, padding=1)
    return (max_pool - min_pool).clamp(0, 1)


def precompute_angle_bins(n_bins, device):
    """
    Returns:
        dirs: (n_bins, 2) unit direction vectors
    """
    theta = torch.linspace(0, 2 * torch.pi, n_bins + 1, device=device)[:-1]
    dirs = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    return dirs


def normalize_radial_profile(radial_profile):
    """Normalize radial profile to sum to 1."""
    # return f_radial_profile / (f_radial_profile.sum(dim=1, keepdim=True) + 1e-9)
    return radial_profile / (radial_profile.sum(dim=1, keepdim=True) + 1e-9)


def batch_circular_emd_loss(F, G):
    """See this paper https://arxiv.org/pdf/0906.5499"""
    mu = (F - G).median(dim=-1, keepdim=True).values
    return torch.sum(
        torch.abs(
            F - G - mu
        ),
        dim=1
    )  # [B]


@torch.compile()
def batch_star_convex_radial_profiles(cx, cy, edges, dirs):
    B, _, H, W = edges.shape
    K = dirs.shape[0]
    device = edges.device

    # Pixel coordinates of (B, 1, H, W)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    xs = xs.float().unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1) # (B,1,H,W)
    ys = ys.float().unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1) # (B,1,H,W)

    # Center coordinates
    xs_c = xs - cx  # (B,1,H,W)
    ys_c = ys - cy  # (B,1,H,W)

    # Flatten edges and keep batch indices
    edge_idx = edges.bool().nonzero(as_tuple=False)  # (N_total,4) -> b,1,y,x
    b_idx = edge_idx[:,0]
    y_idx = edge_idx[:,2]
    x_idx = edge_idx[:,3]

    xs_flat = xs_c[b_idx,0,y_idx,x_idx]  # (N_total,)
    ys_flat = ys_c[b_idx,0,y_idx,x_idx]

    # Stack positions and normalize for dot product
    pos = torch.stack([xs_flat, ys_flat], dim=1)  # (N_total,2)
    r = torch.norm(pos, dim=1)                     # distances

    # Normalize positions for direction assignment
    pos_unit = pos / (r[:,None] + 1e-8)           # (N_total,2)

    # Compute dot products with precomputed directions
    # Assign each edge pixel to the closest direction
    dots = pos_unit @ dirs.T                       # (N_total, K)
    bin_idx = dots.argmax(dim=1)                  # (N_total,)

    # Combine batch and bin for scatter
    scatter_idx = b_idx * K + bin_idx

    # Scatter max per batch-direction
    profile = torch.zeros(B*K, device=device)
    profile.scatter_reduce_(0, scatter_idx, r, reduce='amax')
    profile = profile.view(B, K)

    return profile
