import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffcali.eval_dvrk.LND_fk import batch_lndFK
from diffcali.utils.projection_utils import *
from diffcali.utils.angle_transform_utils import mix_angle_to_rotmat


class EnergyEMDLoss(nn.Module):
    """
    1D EMD loss where total mass encodes number of keypoints
    """
    def __init__(self, lambda_cumsum=0.):
        super().__init__()
        self.lambda_cumsum = lambda_cumsum

    def forward(self, pred, target):
        """
        pred: [B, N] predicted heatmap / energy (>=0)
        target: [B, N] target heatmap (sum = #keypoints)
        """
        # Compute CDFs along sequence
        cdf_pred = torch.cumsum(pred, dim=-1)
        cdf_target = torch.cumsum(target, dim=-1)

        # 1D EMD
        emd = torch.mean(torch.abs(cdf_pred - cdf_target)) 

        # Add cumsum regularization to encourage mass conservation
        cumsum_diff = torch.mean(torch.abs(torch.sum(pred, dim=-1) - torch.sum(target, dim=-1)))

        return emd + self.lambda_cumsum * cumsum_diff


class MSELoss(nn.Module):
    def forward(self, pred, target):
        """
        pred:   [B, N] predicted heatmap
        target: [B, N] target heatmap
        """
        return torch.mean(
            (target - pred) ** 2
        )
    

class GaussianNLLLoss(nn.Module):
    """
    Diagonal Gaussian negative log likelihood
    """
    def forward(self, mu, logvar, target):
        """
        mu:      [B, P]
        logvar:  [B, P]
        target: [B, P]
        """
        var = torch.exp(logvar)
        return torch.mean(
            0.5 * (logvar + (target - mu) ** 2 / var)
        )
    

class MSELoss(nn.Module):
    """
    Mean Squared Error Loss
    """
    def forward(self, mu, target):
        """
        mu:      [B, P]
        target: [B, P]
        """
        return torch.mean(
            (target - mu) ** 2
        )
    

@torch.compile()
def keypoint_projection(pose, p_local1, p_local2, intr):
    cTr = pose[:, :6]  # [B, 6]
    joint_angles = pose[:, 6:]  # [B, 3]
    joint_angles_full = torch.cat([joint_angles, joint_angles[:, -1:].clone()], dim=1)  # [B, 4]
    
    # Compute 4x4 pose matrix
    R_cTr = mix_angle_to_rotmat(cTr[:, :3])  # [B, 3, 3]
    t_cTr = cTr[:, 3:].unsqueeze(-1)  # [B, 3, 1]
    pose_matrix_b = torch.cat(
        [torch.cat([R_cTr, t_cTr], dim=-1),
         torch.tensor([0.0, 0.0, 0.0, 1.0])
         .to(joint_angles.dtype)
         .to(joint_angles.device)
         .unsqueeze(0)
         .repeat(cTr.size(0), 1, 1)],
        dim=1
    )  # [B, 4, 4]

    R_list, t_list = batch_lndFK(joint_angles_full)
    R_list = R_list.to(joint_angles.device)
    t_list = t_list.to(joint_angles.device)
    p_img1 = get_img_coords_batch(
        p_local1,
        R_list[:,2,...],
        t_list[:,2,...],
        pose_matrix_b.to(joint_angles.dtype),
        intr,
        ret_cam_coords=False
    )
    p_img2 = get_img_coords_batch(
        p_local2,
        R_list[:,3,...],
        t_list[:,3,...],
        pose_matrix_b.to(joint_angles.dtype),
        intr,
        ret_cam_coords=False
    )

    keypoints = torch.stack((p_img1, p_img2), dim=1)  # [B, 2, 2]
    return keypoints


@torch.compile()
def keypoint_loss_batch(keypoints_a, keypoints_b, p=2, sqrt=False):
    """
    Computes the Chamfer distance between two sets of keypoints.

    Args:
        keypoints_a (torch.Tensor): Tensor of keypoints (shape: [B, 2, 2]).
        keypoints_b (torch.Tensor): Tensor of keypoints (shape: [B, 2, 2]).

    Returns:
        torch.Tensor: The computed Chamfer distance.
    """
    if keypoints_a.size(1) != 2 or keypoints_b.size(1) != 2:
        raise ValueError("This function assumes two keypoints per set in each batch.")

    # Permutation 1: A0->B0 and A1->B1
    dist_1 = torch.norm(keypoints_a[:, 0] - keypoints_b[:, 0], dim=1, p=p) + torch.norm(
        keypoints_a[:, 1] - keypoints_b[:, 1], dim=1, p=p
    )

    # Permutation 2: A0->B1 and A1->B0
    dist_2 = torch.norm(keypoints_a[:, 0] - keypoints_b[:, 1], dim=1, p=p) + torch.norm(
        keypoints_a[:, 1] - keypoints_b[:, 0], dim=1, p=p
    )

    # Choose the pairing that results in minimal distance for each batch
    min_dist = torch.min(dist_1, dist_2)  # [B]

    # Align the centerline for each batch
    centerline_loss = torch.norm(
        torch.mean(keypoints_a, dim=1) - torch.mean(keypoints_b, dim=1), dim=1, p=p
    )  # [B]

    return min_dist + centerline_loss if not sqrt else torch.sqrt(min_dist + centerline_loss)



class KeypointLoss(nn.Module):
    """
    L-1 loss for keypoint projections
    """
    def __init__(self):
        super().__init__()

        self.p_local1 = (
            torch.tensor([0.0, 0.0004, 0.0096])
        ).cuda()
        self.p_local2 = (
            torch.tensor([0.0, -0.0004, 0.0096])
        ).cuda()

        height = 480
        width = 640
        fx, fy, px, py = 1025.88223, 1025.88223, 167.919017, 234.152707
        scale = 1.0

        # scale the camera parameters
        width = int(width * scale)
        height = int(height * scale)
        fx = fx * scale
        fy = fy * scale
        px = px * scale
        py = py * scale

        self.intr = torch.tensor(
            [
                [fx, 0, px], 
                [0, fy, py], 
                [0, 0, 1]
            ],
        ).cuda()


    def forward(self, pose_pred, pose_target):
        # Project keypoints
        keypoints_pred = keypoint_projection(
            pose_pred,
            self.p_local1,
            self.p_local2,
            self.intr
        )  # [B, 2, 2]

        keypoints_target = keypoint_projection(
            pose_target,
            self.p_local1,
            self.p_local2,
            self.intr
        )  # [B, 2, 2]

        # Compute keypoint loss
        loss = keypoint_loss_batch(
            keypoints_pred,
            keypoints_target
        )

        return torch.mean(loss)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, N, d_model]
        # return x + self.pe[:, :x.size(1)]
        return x + self.pe # ensure input sequence length is always max_len


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        x: [B, N, d_model]
        """
        B, N, _ = x.shape
        positions = torch.arange(N, device=x.device).unsqueeze(0)  # [1, N]
        pos = self.pos_embed(positions)  # [1, N, d_model]
        return x + pos
        

class ContourTransformer(nn.Module):
    def __init__(
        self,
        feature_dim,
        d_model=16,
        num_heads=2,
        num_layers=2,
        max_len=300,
        dropout=0.1,
        conv_kernel_size=5  # local conv kernel size
    ):
        super().__init__()

        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoding = LearnedPositionalEncoding(d_model=d_model, max_len=max_len)

        # Local 1D convolution along contour
        # Conv1d expects [B, C, N]
        self.local_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2
        )
        self.conv_norm = nn.LayerNorm(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,   # IMPORTANT: [B, N, C]
            activation="gelu",
            norm_first=True     # Pre-LN = more stable
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Per-point prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        """
        x: [B, N, F]
        return: [B, N] heatmap (unnormalized logits)
        """
        x = self.input_proj(x)       # [B, N, d_model]
        x = self.pos_encoding(x)     # inject positional info

        # Local convolution with residual
        # Conv1d expects [B, C, N], output back to [B, N, C]
        x_conv = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.conv_norm(x + x_conv)  # residual + norm

        # Transformer encoder for global context
        x = self.encoder(x)

        # Output heatmap per point
        heatmap = self.head(x).squeeze(-1)  # [B, N]
        return heatmap


class SmoothBias(nn.Module):
    def __init__(self, N, kernel_size):
        super().__init__()
        self.raw_bias = nn.Parameter(torch.zeros(N))

        # # fixed smoothing kernel
        # kernel = torch.ones(1, 1, kernel_size) / kernel_size
        # self.register_buffer("kernel", kernel)
        # self.padding = kernel_size // 2

    def forward(self):
        b = self.raw_bias[None, None, :]  # [1,1,N]
        # # Pad for 'same' convolution
        # b = torch.nn.functional.pad(b, (self.padding, self.padding), mode='replicate')
        # # Apply 1D convolution to smooth the bias
        # b = torch.nn.functional.conv1d(b, self.kernel)
        b = b.squeeze()

        # Ensure symmetric bias
        N = b.size(0)
        b = 0.5 * (b + b[torch.arange(N-1, -1, -1)])  # symmetric

        return b


class ContourTipNet(nn.Module):
    """Main model for contour tip keypoint detection."""
    def __init__(self, feature_dim=6, max_len=300):
        super().__init__()
        # print("Initializing ContourTipNet model...")

        self.model = ContourTransformer(
            feature_dim=feature_dim,
            d_model=16,
            num_heads=2,
            max_len=max_len,
        )

        self.heatmap_bias = SmoothBias(N=max_len, kernel_size=5)
    
    def forward(self, x):
        """
        x: [B, N, F]
        returns: [B, N] heatmap
        """
        # Ensure flip equivariance by processing both original and flipped inputs
        x_flipped = x[ :, torch.arange(x.size(1)-1, -1, -1), :]  # Flip along N dimension
        x_cat = torch.cat([x, x_flipped], dim=0)  # [2B, N, F]

        # Process concatenated inputs through the model
        heatmap_cat = self.model(x_cat)

        # Split outputs back
        B = x.size(0)
        heatmap = heatmap_cat[:B]  # [B, N]
        heatmap_flipped = heatmap_cat[B:]  # [B, N]
        heatmap_unflipped = heatmap_flipped[ :, torch.arange(heatmap_flipped.size(1)-1, -1, -1)]  # Un-flip

        # Average the heatmaps
        heatmap_final = 0.5 * (heatmap + heatmap_unflipped)

        # Ensure heatmap output is positive by applying softplus

        # Adding learnable bias before sigmoid
        heatmap_final = torch.sigmoid(heatmap_final + self.heatmap_bias().unsqueeze(0))

        return heatmap_final
        

class ContourPoseNet(nn.Module):
    def __init__(
        self,
        feature_dim=12,
        pose_dim=9,
        pose_aux_dim=6, # centroid (y,x) and scale for current and previous frames
        d_model=32,
        num_heads=4,
        num_layers=3,
        max_len=300,
        conv_kernel=5
    ):
        super().__init__()

        self.pose_dim = pose_dim

        # Countour token embedding
        self.contour_mlp = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.contour_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2
        )
        self.contour_conv_norm = nn.LayerNorm(d_model)

        # Pose token embedding
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim+pose_aux_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Pose correction head
        self.pose_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        self.pose_mu = nn.Linear(d_model, pose_dim)

        # Keypoints heatmap head
        self.kpt_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

        self.pos_encoding = LearnedPositionalEncoding(d_model=d_model, max_len=max_len+1) # +1 for pose token

    def forward(self, contour_feats, pose_prev):
        """
        contour_feats: [B, N, F]
        pose_prev:     [B, P+6] (including centroids and scales)

        returns:
            heatmap: [B, N]
            mu:      [B, P]
            # logvar:  [B, P]
        """

        B, N, _ = contour_feats.shape

        # Contour token embedding
        h = self.contour_mlp(contour_feats)           # [B, N, D]

        h_conv = self.contour_conv(
            h.transpose(1, 2)
        ).transpose(1, 2)

        h = self.contour_conv_norm(h + h_conv) # residual + norm

        # Pose token embedding
        z = self.pose_mlp(pose_prev).unsqueeze(1)     # [B, 1, D]

        # Fusion (add positional encoding)
        tokens = torch.cat([z, h], dim=1)             # [B, 1+N, D]
        tokens = self.pos_encoding(tokens)  # inject positional info
        tokens_out = self.transformer(tokens)
        contour_tokens = tokens_out[:, 1:]                # [B, N, D]
        pose_token_out = tokens_out[:, 0]                 # [B, D]

        # Keypoint heatmap prediction
        heatmap = self.kpt_head(contour_tokens).squeeze(-1)
        heatmap = torch.sigmoid(heatmap)

        # Pose correction head prediction
        pose_latent = self.pose_head(pose_token_out)
        mu = self.pose_mu(pose_latent) + pose_prev[:, :self.pose_dim] # residual connection

        return heatmap, mu