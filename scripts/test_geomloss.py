import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from sklearn.neighbors import KernelDensity
from torch.nn.functional import avg_pool2d

import torch
from geomloss import SamplesLoss

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def grid(W):
    x, y = torch.meshgrid([torch.arange(0.0, W).type(dtype) / W] * 2, indexing="xy")
    return torch.stack((x, y), dim=2).view(-1, 2)


def load_image(fname):
    img = imread(fname)
    img = (img[:, :]) / 255.0
    return 1 - img  # black = 1, white = 0


def as_measure(fname, size):
    weights = torch.from_numpy(load_image(fname)).type(dtype)
    # scale the image to 128 x 128
    weights = torch.nn.functional.interpolate(
        weights.unsqueeze(0).unsqueeze(0), size=(16, 16), mode="bilinear", align_corners=False
    ).squeeze(0).squeeze(0)
    sampling = weights.shape[0] // size
    weights = (
        avg_pool2d(weights.unsqueeze(0).unsqueeze(0), sampling).squeeze(0).squeeze(0)
    )
    weights = weights / weights.sum()

    samples = grid(size)
    return weights.view(-1), samples


N, M = (8, 8) if not use_cuda else (16, 16)

A, B = as_measure("data/consecutive_prediction/0617/0/00000.png", M), as_measure("data/synthetic_data/syn_rw15/0/00000.png", M)
C, D = as_measure("data/synthetic_data/syn_rw15/477/00477.png", M), as_measure("data/synthetic_data/syn_rw9/261/00261.png", M)
print(A[0].shape, A[1].shape)

x_i = grid(N).view(-1, 2)
a_i = (torch.ones(N * N) / (N * N)).type_as(x_i)

print(x_i.shape, a_i.shape, N)


x_i.requires_grad = True

import matplotlib

matplotlib.rc("image", cmap="gray")

grid_plot = grid(M).view(-1, 2).cpu().numpy()


def display_samples(ax, x, weights=None):
    """Displays samples on the unit square using a simple binning algorithm."""
    x = x.clamp(0, 1 - 0.1 / M)
    bins = (x[:, 0] * M).floor() + M * (x[:, 1] * M).floor()
    count = bins.int().bincount(weights=weights, minlength=M * M)
    ax.imshow(
        count.detach().float().view(M, M).cpu().numpy(),
        vmin=0,
        vmax=0.5 * count.max().item(),
    )

Loss = SamplesLoss("sinkhorn", blur=0.01, scaling=0.9)
models = []
import time
time_lst = []

for b_j, y_j in [A, B, C, D]:
    L_ab = Loss(a_i, x_i, b_j, y_j)
    [g_i] = torch.autograd.grad(L_ab, [x_i])
    models.append(x_i - g_i / a_i.view(-1, 1))

with torch.no_grad():
    for b_j, y_j in [A, B, C, D]:
        st = time.time()
        L_ab = Loss(a_i, x_i, b_j, y_j)
        et = time.time()
        time_lst.append(et - st)

print(f"Average time for loss computation per step: {np.mean(time_lst):.4f} seconds")

a, b, c, d = models

plt.figure(figsize=(14, 14))

# Display the target measures in the corners of our Figure
ax = plt.subplot(7, 7, 1)
ax.imshow(A[0].reshape(M, M).cpu())
ax.set_xticks([], [])
ax.set_yticks([], [])
ax = plt.subplot(7, 7, 7)
ax.imshow(B[0].reshape(M, M).cpu())
ax.set_xticks([], [])
ax.set_yticks([], [])
ax = plt.subplot(7, 7, 43)
ax.imshow(C[0].reshape(M, M).cpu())
ax.set_xticks([], [])
ax.set_yticks([], [])
ax = plt.subplot(7, 7, 49)
ax.imshow(D[0].reshape(M, M).cpu())
ax.set_xticks([], [])
ax.set_yticks([], [])

# Display the interpolating densities as a 5x5 waffle plot
for i in range(5):
    for j in range(5):
        x, y = j / 4, i / 4
        barycenter = (
            (1 - x) * (1 - y) * a + x * (1 - y) * b + (1 - x) * y * c + x * y * d
        )

        ax = plt.subplot(7, 7, 7 * (i + 1) + j + 2)
        display_samples(ax, barycenter)
        ax.set_xticks([], [])
        ax.set_yticks([], [])

plt.tight_layout()
plt.show()

