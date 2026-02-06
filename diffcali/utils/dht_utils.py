import torch
from torchvision import transforms
from skimage.measure import label, regionprops
import cv2

import sys
import os
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
submodule_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'deep_hough_transform'))
sys.path.insert(0, submodule_dir)

from model.network import Net 


class DeepCylinderLoss:
    """
    Use the DHT model to generate a heatmap and compute the loss.
    """
    def __init__(
        self, model_dir="./deep_hough_transform/dht_r50_nkl_d97b97138.pth", mask=None, 
        numAngle=100, numRho=100, img_size=(480, 640), input_size=(400, 400)
    ):

        self.model = Net(numAngle=100, numRho=100, backbone='resnet50').cuda()
        self.mask = mask

        if os.path.isfile(model_dir):
            checkpoint = torch.load(model_dir)
            if 'state_dict' in checkpoint.keys():
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print("=> no pretrained model found at '{}'".format(model_dir))

        self.model.eval()

        self.numAngle, self.numRho = numAngle, numRho
        self.H, self.W = img_size
        self.input_size = input_size
        self.D = (input_size[0] ** 2 + input_size[1] ** 2) ** 0.5  # Diagonal of the resized image
        self.dtheta = torch.pi / self.numAngle
        self.drho = (self.D + 1) / (self.numRho - 1)

        self.transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    def update_heatmap(self, mask=None):
        mask = self.mask if mask is None else mask
        if mask is None:
            raise ValueError("Mask must be provided to update the heatmap.")

        self.img_size = mask.shape[2:]  # Get the original image size
        self.heatmap = self.model(self.transform(mask)).squeeze()
        self.heatmap = self.heatmap.sigmoid()
        
        # # Manipulate the loss surface
        # binary_mask = torch.sigmoid(self.heatmap) > 1e-9 # thresholding
        # hmin, hmax = self.heatmap[~binary_mask].max(), self.heatmap.max()
        # self.heatmap[~binary_mask] = hmin
        # self.heatmap = (self.heatmap - hmin) / (hmax - hmin) + 1e-9 # Normalize to (0, \infty)
        # self.heatmap = -torch.log(self.heatmap)

    def line2ends(self, lines, to_int=True):
        """
        Convert a line (ax + by = 1) to its endpoints in the image.
        Input:
            line: torch.Tensor of shape (B, 2) representing a batch of line coefficients (a, b).
        Output:
            endpoints: torch.Tensor of shape (B, 2, 2) representing the endpoints of the lines.
        """
        def trunc_toward_zero(x: torch.Tensor) -> torch.Tensor:
            return torch.where(x >= 0, x.floor(), x.ceil()).to(torch.int32)

        a = lines[:, 0]
        b = lines[:, 1]
        H, W = self.H, self.W

        x1 = torch.zeros_like(a)
        x2 = torch.full_like(a, fill_value=W)
        y1 = torch.zeros_like(b)
        y2 = torch.full_like(b, fill_value=H)

        mask_vert = (b == 0.)
        if mask_vert.any():
            xv = 1 / a[mask_vert]
            x1[mask_vert] = xv
            x2[mask_vert] = xv
        
        y1[~mask_vert] = 1 / b[~mask_vert]
        y2[~mask_vert] = (1 - a[~mask_vert] * x2[~mask_vert]) / b[~mask_vert]

        if to_int:
            x1, y1, x2, y2 = [torch.where(t < 0, torch.ceil(t), torch.floor(t)).to(torch.int32) for t in (x1, y1, x2, y2)]
        
        return x1, y1, x2, y2

    def line2hough(self, line, to_int=True):
        """
        Convert a line (ax + by = 1) to Hough space coordinates.
        Input:
            line: torch.Tensor of shape (B, 2) representing a batchc of line coefficients (a, b).
        Output:
            hough_coords: torch.Tensor of shape (B, 2) representing the Hough space coordinates (theta, r).
        Note. The range of theta is [0, pi) and r is [-sqrt(H^2 + W^2), sqrt(H^2 + W^2)] where H = W = 400.
        """
        # Convert line coefficients to end points and scale them to the input size
        x1, y1, x2, y2 = self.line2ends(line, to_int=to_int)
        x1, x2 = x1 * self.input_size[1] / self.W, x2 * self.input_size[1] / self.W
        y1, y2 = y1 * self.input_size[0] / self.H, y2 * self.input_size[0] / self.H
        if to_int:
            x1, y1, x2, y2 = [t.to(torch.int32) for t in (x1, y1, x2, y2)]

        # Compute alpha
        theta = torch.atan2(y2 - y1, x2 - x1) 
        alpha = theta + torch.pi / 2 # alpha = theta + pi/2 in [0, pi)

        # Compute r
        r = torch.zeros_like(theta)
        x1c, y1c = x1 - self.input_size[1] / 2, y1 - self.input_size[0] / 2 # center the coordinates
        mask_vert = (theta == -torch.pi / 2) 
        if mask_vert.any():
            r[mask_vert] = x1c[mask_vert]  # For vertical lines, r = x - W/2
        k = torch.tan(theta[~mask_vert])  # slope of the line
        r[~mask_vert] = (y1c[~mask_vert] - k * x1c[~mask_vert]) / torch.sqrt(1 + k**2)  # For non-vertical lines, r = (y - k*x) / sqrt(1 + k^2)

        return torch.stack([alpha, r], dim=1)

    def hough2idx(self, hough_coords):
        """
        Convert Hough space coordinates to indices in the heatmap.
        Input:
            hough_coords: torch.Tensor of shape (B, 2) representing Hough space coordinates (theta, r).
        Output:
            idx: torch.Tensor of shape (B, 2) representing indices in the heatmap.
        """
        theta, r = hough_coords[:, 0], hough_coords[:, 1]
    
        theta_idx = (theta / self.dtheta).round()
        r_idx = (r / self.drho + self.numRho // 2).round()

        # theta_idx = torch.clamp(theta_idx, min=0, max=self.numAngle - 1)
        # r_idx = torch.clamp(r_idx, min=0, max=self.numRho - 1)

        return torch.stack([theta_idx, r_idx], dim=1)

    def __call__(self, projected_lines):
        """
        Evaluate a batch of projected line pairs and compute the loss.
        Input:
            projected_lines: torch.Tensor of shape (B, 2， 2) representing a batch of line pairs
        Output:
            loss: torch.Tensor of shape (B,) representing the computed loss.
        """
        # Concatenate the two batches of lines
        B = projected_lines.shape[0]
        projected_lines_1 = projected_lines[:, 0, :]
        projected_lines_2 = projected_lines[:, 1, :]
        lines = torch.cat([projected_lines_1, projected_lines_2], dim=0)

        # Convert lines to Hough coordinates
        hough_coords = self.line2hough(lines)
        idx = self.hough2idx(hough_coords)

        # # Compute the loss based on the heatmap
        # loss = torch.full(fill_value=-float('inf'), size=(2*B,), dtype=torch.float32, device=lines.device)
        # mask_in_bounds = (idx[:, 0] >= 0) & (idx[:, 0] < self.numAngle) & (idx[:, 1] >= 0) & (idx[:, 1] < self.numRho)
        # if mask_in_bounds.any():
        #     loss[mask_in_bounds] = self.heatmap[idx[mask_in_bounds, 0].long(), idx[mask_in_bounds, 1].long()]

        # Create one-hot maps [B, 1, numAngle, numRho]
        one_hot = torch.zeros((2 * B, 1, self.numAngle, self.numRho)).cuda()
        y = idx[:, 0].long()
        x = idx[:, 1].long()
        one_hot[torch.arange(2 * B), 0, y, x] = 1.0
        one_hot = one_hot[:B] + one_hot[B:]  # Combine the two batches

        # Apply Gaussian blur
        blurred = transforms.functional.gaussian_blur(one_hot, kernel_size=5, sigma=1) # [2*B, numAngle, numRho]
        
        # # Visualize the blurred heatmap
        # import matplotlib.pyplot as plt
        # plt.imshow(blurred[0].squeeze().cpu().numpy(), cmap='jet')
        # plt.show()

        # Compute cross entropy loss
        cross_entropy = torch.nn.functional.binary_cross_entropy(
            input=self.heatmap.expand(B, -1, -1),
            target=blurred.squeeze(1),
            reduction='none'
        )
        loss = cross_entropy.view(B, -1).sum(dim=1)  # Sum over the spatial dimensions

        # # Plot the heatmap and all evaluated poitns in bound
        # import matplotlib.pyplot as plt
        # plt.imshow(-self.heatmap.squeeze().cpu().numpy(), cmap="jet")
        # plt.scatter(idx[:,1].cpu().numpy(), idx[:,0].cpu().numpy(), s=1, c='white')
        # plt.show()
        
        return loss


class SmoothDeepCylinderLoss(DeepCylinderLoss):
    """
    A smooth version of the DeepCylinderLoss that uses a Gaussian kernel to smooth the heatmap.
    """
    def __init__(
        self, model_dir="./deep_hough_transform/dht_r50_nkl_d97b97138.pth", mask=None, numAngle=100, numRho=100, 
        img_size=(480, 640), input_size=(400, 400), sigma=None, beta=1e-2):
        super().__init__(model_dir=model_dir, mask=mask, numAngle=numAngle, numRho=numRho, img_size=img_size, input_size=input_size)

        if sigma is None:
            sigma = [torch.pi / 2, self.D / 2]
        self.sigma = sigma if isinstance(sigma, torch.Tensor) else torch.tensor(sigma, dtype=torch.float32)
        self.sigma = self.sigma.cuda()
        self.beta = beta # temperature parameter

    @torch.no_grad()
    def update_heatmap(self, mask=None):
        mask = self.mask if mask is None else mask
        if mask is None:
            raise ValueError("Mask must be provided to update the heatmap.")

        self.img_size = mask.shape[2:]  # Get the original image size
        self.heatmap = self.model(self.transform(mask)).squeeze()
        
        # Select the two most significant lines in the heatmap
        binary_kmap = self.heatmap.sigmoid().squeeze() > 1e-4
        kmap_label = label(binary_kmap.cpu().numpy(), connectivity=1)
        props = regionprops(kmap_label)
        self.has_two_lines = len(props) >= 2
        if self.has_two_lines:
            props = sorted(props, key=lambda x: x.area, reverse=True)[:2]
            p1, p2 = props[0].centroid, props[1].centroid

            # Convert back to tensors in Hough coordinates
            theta1, rho1 = p1[0] * self.dtheta, (p1[1] - self.numRho // 2) * self.drho
            theta2, rho2 = p2[0] * self.dtheta, (p2[1] - self.numRho // 2) * self.drho
            self.p1 = torch.tensor([theta1, rho1], device=self.heatmap.device)
            self.p2 = torch.tensor([theta2, rho2], device=self.heatmap.device)
            # print(f"Detected line centroids in Hough space: {self.p1}, {self.p2}")
        else:
            print(f"[At least two lines should be detected in the heatmap to use the smooth DHT loss. Only {len(props)} lines are detected, hence the smooth DHT loss is disabled.]")

    def eval_prob(self, hough_coords):
        """
        Compute the probability of a line in Hough space using a Gaussian kernel.
        Input:
            hough_coords: torch.Tensor of shape (B, 2) representing a batch of Hough coordinates (alpha, r).
        Output:
            probs: torch.Tensor of shape (B,) representing the probability of each line.
        """
        # Compute the Gaussian kernel
        d1 = torch.norm((hough_coords - self.p1) / self.sigma, dim=1)
        d2 = torch.norm((hough_coords - self.p2) / self.sigma, dim=1)
        
        probs = torch.exp(-d1**2 / (2 * self.beta)) + torch.exp(-d2**2 / (2 * self.beta))
        
        return probs

    def __call__(self, projected_lines):
        if not self.has_two_lines:
            return 0.0

        # Concatenate the two batches of lines
        B = projected_lines.shape[0]
        projected_lines_1 = projected_lines[:, 0, :]
        projected_lines_2 = projected_lines[:, 1, :]
        lines = torch.cat([projected_lines_1, projected_lines_2], dim=0)

        # Convert lines to Hough coordinates and evaluate the density
        hough_coords = self.line2hough(lines, to_int=False) # does not round the coordinates
        probs = self.eval_prob(hough_coords)

        # # Visualize the heatmap and  the population
        # import matplotlib.pyplot as plt
        # thetas = torch.linspace(0, torch.pi, steps=300)
        # rhos = torch.linspace(-self.D / 2, self.D / 2, steps=300)
        # grid = torch.cartesian_prod(thetas, rhos).cuda()  # Shape: (10000, 2)
        # print("Grid shape:", grid.shape)
        # probs_g = self.eval_prob(grid)
        # print("Probabilities shape:", probs_g.shape)
        # heatmap = probs_g.view(300, 300).cpu().numpy()  # Reshape to (100, 100) for visualization
        # extent = [-self.D / 2, self.D / 2, 0, torch.pi]  # Set the extent for the heatmap
        # aspect = (self.D) / torch.pi  # Aspect ratio for the heatmap
        # plt.figure(figsize=(10, 5))
        # plt.imshow(heatmap, extent=extent, aspect=aspect, cmap='jet', origin='lower')
        # plt.xlabel('Rho')
        # plt.ylabel('Theta')
        # plt.title('Probability of Lines in Hough Space')
        # p1, p2 = self.p1, self.p2
        # p1_idx = self.hough2idx(p1.unsqueeze(0)).squeeze().cpu().numpy()
        # p2_idx = self.hough2idx(p2.unsqueeze(0)).squeeze().cpu().numpy()
        # print("p1 idx:", p1_idx, "p2 idx:", p2_idx)
        # p1, p2 = p1.cpu().numpy(), p2.cpu().numpy()
        # plt.plot(p1[1], p1[0], color="yellow", marker="*", markersize=7, label='p1')
        # plt.plot(p2[1], p2[0], color="yellow", marker="*", markersize=7, label='p2')
        # plt.scatter(hough_coords[:,1].cpu().numpy(), hough_coords[:,0].cpu().numpy(), s=1, c='white')
        # plt.show()

        return -1 * (probs[:B] + probs[B:])


def main1():
    model_dir = "./deep_hough_transform/dht_r50_nkl_d97b97138.pth"
    mask_dir = "./deep_hough_transform/data/DVRK/7.jpg"
    
    # img = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
    # mask = transforms.ToTensor()(img).cuda() 
    # mask = mask.repeat(3, 1, 1).unsqueeze(0)
    # img = mask.squeeze().permute(1, 2, 0).cpu().numpy() * 255

    img = cv2.imread(mask_dir, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask = transforms.ToTensor()(img).cuda()
    mask = mask.unsqueeze(0)  # Add batch dimension

    # print(torch.abs(mask - mask2).sum().item())

    with torch.no_grad():
        DHT_loss = DeepCylinderLoss(model_dir=model_dir, mask=mask, img_size=mask.shape[2:])
        DHT_loss.update_heatmap()

        start_time = time.time()
        DHT_loss.update_heatmap(mask)
        end_time = time.time()

    print(f"Time taken to update heatmap: {end_time - start_time:.4f} seconds")

    # Coupute the hough coordinates for the largest line in the heatmap
    heatmap = DHT_loss.heatmap.squeeze().cpu().numpy()
    
    # Use Nelder-Mead optimization to find the line parameters
    from scipy.optimize import minimize
    def objective_function(params):
        a, b = params
        # # Convert (a, b) to Hough coordinates
        # hough_coords = DHT_loss.line2hough(torch.tensor([[a, b]]).cuda())
        # idx = DHT_loss.hough2idx(hough_coords)
        # if idx[0, 0] < 0 or idx[0, 0] >= DHT_loss.numAngle or idx[0, 1] < 0 or idx[0, 1] >= DHT_loss.numRho:
        #     return float('inf')
        # return -DHT_loss.heatmap[idx[0, 0].long(), idx[0, 1].long()].item()  # Negative for maximization
        lines_tensor = torch.tensor([[a, b], [a, b]]).cuda().unsqueeze(0)  # Create a batch of lines
        loss = DHT_loss(lines_tensor)
        return loss.item()  # Return the loss value for minimization

    # Correct coordinates: [(303, 0, 251, 399), (227, 0, 237, 399)]
    a = 0.00053764774973738
    b = 0.0055005500550055
    initial_guess = [0.0006, 0.005]  # Initial guess for (a, b)
    result = minimize(objective_function, initial_guess, method='Nelder-Mead', options={'maxiter': 1000, 'disp': True})
    a_opt, b_opt = result.x
    print(f"Optimized line parameters: a = {a_opt}, b = {b_opt}")
    print(f"Function evaluations: {result.nfev}, Function value: {result.fun}")

    # Plot the original image with the detected line and plot the corresponding points on the heatmap
    lines = torch.tensor([[a_opt, b_opt]]).cuda()  # Use the optimized line parameters
    hough_coords = DHT_loss.line2hough(lines)
    print("Hough coordinates:", hough_coords)
    print("Hough idx:", DHT_loss.hough2idx(hough_coords))
    idx = DHT_loss.hough2idx(hough_coords)
    print(lines, hough_coords, idx)

    # Plot the original image with the detected line
    x1, y1, x2, y2 = DHT_loss.line2ends(lines)
    x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
    print(f"Line endpoints: ({x1}, {y1}), ({x2}, {y2})")
    img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=2)

    # Plot the original image and the heatmap
    heatmap = DHT_loss.heatmap.squeeze().cpu().numpy()
    binary_kmap = DHT_loss.heatmap.squeeze().cpu().numpy() > 1e-4
    kmap_label = label(binary_kmap, connectivity=1)
    props = regionprops(kmap_label)
    # print(vars(props[0]))
    if len(props) < 2:
        print("No lines detected in the heatmap.")
    else:
        props = sorted(props, key=lambda x: x.area, reverse=True)[:2]
        print(props[0].area, props[1].area)
        p1, p2 = props[0].centroid, props[1].centroid
        p1_th = torch.tensor(p1, device=heatmap.device)
        print(p1_th)
        print(f"Detected line centroids: {p1}, {p2}")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Mask")
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap='jet')
    # draw the line on the heatmap
    plt.plot(idx[0, 1].item(), idx[0, 0].item(), 'ro', markersize=7)  # Mark the max point
    plt.plot(p1[1], p1[0], 'ro', markersize=3)  # Mark the first centroid
    plt.plot(p2[1], p2[0], 'ro', markersize=3)  # Mark the second centroid
    plt.title("Heatmap")
    plt.colorbar()
    plt.show()

def main2():
    model_dir = "./deep_hough_transform/dht_r50_nkl_d97b97138.pth"
    mask_dir = "./deep_hough_transform/data/DVRK/7.jpg"

    img = cv2.imread(mask_dir, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask = transforms.ToTensor()(img).cuda()
    mask = mask.unsqueeze(0)  # Add batch dimension

    # print(torch.abs(mask - mask2).sum().item())

    with torch.no_grad():
        DHT_loss = SmoothDeepCylinderLoss(model_dir=model_dir, mask=mask, img_size=mask.shape[2:])
        DHT_loss.update_heatmap()

        start_time = time.time()
        DHT_loss.update_heatmap(mask)
        end_time = time.time()

    print(f"Time taken to update heatmap: {end_time - start_time:.4f} seconds")

    # Generate a grid of theta and rho to evaluate DHT_loss.eval_prob
    thetas = torch.linspace(0, torch.pi, steps=300)
    rhos = torch.linspace(-DHT_loss.D / 2, DHT_loss.D / 2, steps=300)
    grid = torch.cartesian_prod(thetas, rhos).cuda()  # Shape: (10000, 2)
    print("Grid shape:", grid.shape)

    # Evaluate the probability of each line in Hough space
    probs = DHT_loss.eval_prob(grid)
    print("Probabilities shape:", probs.shape)
    # print(probs)

    # Convert the probs to heatmap
    heatmap = probs.view(300, 300).cpu().numpy()  # Reshape to (100, 100) for visualization

    # Plot the probabilities
    extent = [-DHT_loss.D / 2, DHT_loss.D / 2, 0, torch.pi]  # Set the extent for the heatmap
    aspect = (DHT_loss.D) / torch.pi  # Aspect ratio for the heatmap
    plt.figure(figsize=(10, 5))
    plt.imshow(heatmap, extent=extent, aspect=aspect, cmap='jet', origin='lower')
    plt.xlabel('Rho')
    plt.ylabel('Theta')
    plt.title('Probability of Lines in Hough Space')

    # Also plot p1 and p2
    p1, p2 = DHT_loss.p1, DHT_loss.p2
    p1_idx = DHT_loss.hough2idx(p1.unsqueeze(0)).squeeze().cpu().numpy()
    p2_idx = DHT_loss.hough2idx(p2.unsqueeze(0)).squeeze().cpu().numpy()
    print("p1 idx:", p1_idx, "p2 idx:", p2_idx)
    p1, p2 = p1.cpu().numpy(), p2.cpu().numpy()
    plt.plot(p1[1], p1[0], color="yellow", marker="*", markersize=7, label='p1')
    plt.plot(p2[1], p2[0], color="yellow", marker="*", markersize=7, label='p2')

    plt.show()

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # Example usage
    main1()
    # main2()