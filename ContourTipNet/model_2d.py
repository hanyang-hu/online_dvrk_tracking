import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models import resnet18
from torchvision.ops import FeaturePyramidNetwork



class Heatmap2DLoss(nn.Module):
    """
    L2 loss for 2D heatmaps
    """
    def forward(self, pred_heatmap, target_heatmap):
        """
        pred_heatmap:   [B, H, W]
        target_heatmap: [B, H, W]
        """
        mse_loss = torch.mean(
            (pred_heatmap - target_heatmap) ** 2 # [B, H, W]
        ) # average over batch, height, width
        mass_loss = torch.mean(
            torch.abs(pred_heatmap.sum(dim=(1,2)) - target_heatmap.sum(dim=(1,2))) # [B]
        ) # average over batch
        return mse_loss + 0.0 * mass_loss


# class Heatmap2DLoss(nn.Module):
#     """
#     BCE loss with logits for 2D heatmaps
#     """
#     def forward(self, pred_heatmap, target_heatmap):
#         """
#         pred_heatmap:   [B, 1, H, W]
#         target_heatmap: [B, 1, H, W]
#         """
#         loss = F.binary_cross_entropy_with_logits(
#             pred_heatmap,
#             target_heatmap
#         )
#         return loss

class KeypointMatchingLoss(nn.Module):
    """
    Keypoint matching loss based on predicted and target heatmaps.
    """
    def forward(self, pred_heatmap, target_distmap):
        """
        pred_heatmap:   [B, H, W]
        target_distmap:       [B, H, W] (L_1 distance to nearest keypoint)
        """
        return torch.mean(
            pred_heatmap * target_distmap
        )


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


class DeconvHead(nn.Module):
    def __init__(self, in_channels=128, out_channels=1):
        super().__init__()

        self.layers = nn.Sequential(
            # 56 → 112
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),

            # 112 → 224
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),

            # smoothing before output
            nn.Conv2d(32, 32, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.layers(x)


class Tip2DNet(nn.Module):
    def __init__(self, mask_size=224, pretrained=False, fpn_dim=64, use_attention=False):
        super().__init__()
        assert mask_size == 224

        backbone = resnet18(pretrained=pretrained)
        backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

        # ResNet layers
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # C2: 56×56
        self.layer2 = backbone.layer2  # C3: 28×28
        self.layer3 = backbone.layer3  # C4: 14×14
        self.layer4 = backbone.layer4  # C5: 7×7

        # Whether to apply attention to C4: 14×14
        self.use_attention = use_attention
        if use_attention:
            # self.attention_layer = nn.TransformerEncoderLayer(
            #     d_model=512,
            #     nhead=8,
            #     dim_feedforward=2048,
            #     dropout=0.1,
            #     activation="relu",
            # )
            # self.transformer_encoder = nn.TransformerEncoder(
            #     self.attention_layer,
            #     num_layers=1
            # )
            # self.positional_encoding = LearnedPositionalEncoding(d_model=512, max_len=49)
            self.attention_layer = nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation="relu",
            )
            self.transformer_encoder = nn.TransformerEncoder(
                self.attention_layer,
                num_layers=1
            )
            self.positional_encoding = LearnedPositionalEncoding(d_model=256, max_len=196)

        # FPN using torchvision FPN
        in_channels_list = [64, 128, 256, 512]  # C2–C5
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=fpn_dim
        )

        # Fusion conv after concatenation of all FPN levels
        self.fpn_fuse = nn.Conv2d(fpn_dim * 4, fpn_dim, 3, padding=1)

        self.head = DeconvHead(in_channels=fpn_dim, out_channels=1)

    def forward(self, x):
        heatmap = self.raw_predict(x)
        return torch.sigmoid(heatmap)

    def raw_predict(self, x):
        """
        Forward pass without sigmoid activation.
        """
        x = self.stem(x)
        c2 = self.layer1(x)  # 56×56
        c3 = self.layer2(c2) # 28×28
        c4 = self.layer3(c3) # 14×14

        # Apply attention to C4 if enabled
        if self.use_attention:
            B, C, H, W = c4.shape
            c4_flat = c4.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C]
            c4_flat = self.positional_encoding(c4_flat)
            c4_flat = self.transformer_encoder(c4_flat)
            c4 = c4_flat.permute(0, 2, 1).view(B, C, H, W)

        c5 = self.layer4(c4) # 7×7

        # Pass features through FPN
        features = {"0": c2, "1": c3, "2": c4, "3": c5}
        fpn_out = self.fpn(features)

        # Upsample all levels to P2 resolution (56×56) and concatenate
        p2 = fpn_out["0"]
        p3 = F.interpolate(fpn_out["1"], size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p4 = F.interpolate(fpn_out["2"], size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p5 = F.interpolate(fpn_out["3"], size=p2.shape[-2:], mode="bilinear", align_corners=False)

        fpn_cat = torch.cat([p2, p3, p4, p5], dim=1)
        fpn_feat = self.fpn_fuse(fpn_cat)

        heatmap = self.head(fpn_feat)
        return heatmap


if __name__ == "__main__":
    model = Tip2DNet(mask_size=224, pretrained=False, use_attention=False)
    dummy_input = torch.randn(2, 1, 224, 224)
    output = model(dummy_input)
    print(output.shape)  # Expected: [2, 1, 224, 224]

