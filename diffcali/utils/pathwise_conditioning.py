import torch
from torch.quasirandom import SobolEngine
from pykeops.torch import Genred, KernelSolve
import math


def generate_qmc_normal(num_samples, dim):
    # Generate low-discrepancy points (Sobol sequence) in [0, 1]^dim
    sobol = SobolEngine(dimension=dim, scramble=True)
    u = sobol.draw(num_samples)
    # Transform to standard normal distribution by  Φ⁻¹(u) = sqrt(2) * erfinv(2u - 1)
    normal_samples = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2.0 * u - 1.0)
    return normal_samples


class PathwiseSampler():
    """
    Implementation of pathwise conditioning in KeOps.
    """
    def __init__(
        self, 
        kernel="RBF", 
        lengthscale=torch.Tensor([[5.,  1., 5.,  2., 2., 2., 10., 10., 10., 10.]]).cuda(),
        mean=0., # prior mean of the GP
        alpha=1e-3, # regularization term, equivalent to square of GP noise
        output_scale=1e-2, # output scale of the kernel function
        feature_size=1024, # dimension of Fourier features
        sample_size=70, # the sample size is fixed for defining KeOps aliases
        use_keops=True,
    ):
        self.kernel = kernel
        self.lengthscale = lengthscale
        self.mean = mean
        self.alpha = alpha
        self.output_scale = output_scale if isinstance(output_scale, torch.Tensor) else torch.Tensor([output_scale,]).cuda()
        self.feature_size = feature_size
        self.sample_size = sample_size
        self.use_keops = use_keops

        self.X, self.y = None, None
        self.w, self.v = None, None
        self.dim = lengthscale.shape[1]

        # Generate Fourier feature parameters
        self.theta = generate_qmc_normal(self.dim, self.feature_size).cuda() # shape [dim, feature_size]
        self.tau = 2 * math.pi * torch.rand(1, self.feature_size, device="cuda") # shape [1, feature_size]
        self.feature_scale = math.sqrt(2 * self.output_scale / self.feature_size)

        if self.kernel == "RBF":
            formula = "theta * Exp(- g * SqDist(x,y)) * b"
            aliases = [
                "x = Vi(" + str(self.dim) + ")",  # First arg:  i-variable of size self.dim
                "y = Vj(" + str(self.dim) + ")",  # Second arg: j-variable of size self.dim
                "b = Vj(" + str(self.sample_size) + ")",  # Third arg:  j-variable of size self.sample_size
                "theta = Pm(1)", # Fourth arg: output scale
                "g = Pm(1)",
            ]  
            self.g = torch.Tensor([0.5,]).cuda()
            # K = Genred(formula, aliases, axis = 1)
            self.Kinv = KernelSolve(formula, aliases, "b", axis=1)

            self.kernel_fn = lambda x, y: self.output_scale * torch.exp(-torch.cdist(x, y) ** 2 / 2)

        else:
            raise NotImplementedError(f"Kernel {self.kernel} is not implemented.")

    def update_data(self, X, y, cat=True):
        X = X / self.lengthscale
        
        if cat and self.X is not None:
            X = torch.cat([X, self.X], dim=0)
            y = torch.cat([y, self.y], dim=0)

        self.X = X.cuda().contiguous()
        self.y = y.cuda().contiguous()

    def draw_posterior_samples(self):
        if self.X is None or self.y is None:
            raise RuntimeError("Data is not set. Call update_data() first.")

        with torch.no_grad():
            # Compute Fourier features
            z_X = self.feature_scale * torch.cos(self.X @ self.theta + self.tau)

            # Generate weights for the prior term
            self.w = generate_qmc_normal(self.feature_size, self.sample_size).cuda() # shape [feature_size, sample_size]
            # self.w = torch.randn(self.feature_size, self.sample_size).cuda()  # shape [feature_size, sample_size]

            # Compute weights for the correction term
            prior_preds = self.mean + torch.matmul(z_X, self.w)  # shape [N, sample_size]
            corrections = (self.y[:, None] - prior_preds).squeeze().contiguous() # shape [N, sample_size]

            if self.use_keops:
                self.v = self.Kinv(self.X, self.X, corrections, self.output_scale, self.g, alpha=self.alpha) # shape [N, sample_size]
            else:
                self.v = torch.linalg.solve(
                    self.kernel_fn(self.X, self.X) + self.alpha * torch.eye(self.X.shape[0], device=self.X.device),
                    corrections
                )

    def evaluate(self, x):
        if self.w is None or self.v is None:
            raise RuntimeError("Posterior function samples are not drawn yet. Call draw_posterior_samples() first.")

        x = x / self.lengthscale

        # Compute weight-space prior
        z_x = self.feature_scale * torch.cos(
            torch.matmul(x, self.theta) + self.tau
        ) # shape [n, feature_size]
        prior_preds = self.mean + torch.matmul(z_x, self.w).T # shape [sample_size, n]

        # Compute function space updates
        k_Xx = self.kernel_fn(self.X, x)  # shape [n, N]
        corrections = torch.matmul(self.v.T, k_Xx)  # shape [sample_size, n]

        return prior_preds + corrections  # shape [sample_size, n]

    def evaluate_optim(self, x):
        if self.w is None or self.v is None:
            raise RuntimeError("Posterior function samples are not drawn yet. Call draw_posterior_samples() first.")

        assert x.shape[0] == self.sample_size and x.shape[1] == self.dim

        x = x / self.lengthscale

        z_x = self.feature_scale * torch.cos(
            torch.einsum("sd,df->sf", x, self.theta) + self.tau  # [S, F]
        )  # shape [sample_size, feature_size]
        prior_preds = self.mean + torch.sum(z_x * self.w.T, dim=1)  # [sample_size]

        k_Xx = self.kernel_fn(self.X, x)  # [N, sample_size]
        corrections = torch.einsum("sn,nm->sm", self.v.T, k_Xx)  # [sample_size, sample_size]
        corrections = torch.diagonal(corrections, dim1=0, dim2=1)  # take diagonal only → [sample_size]

        return prior_preds + corrections  # shape [sample_size]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ground_truth_fn = lambda x : torch.sin(3 * x) + 0.5 * x**2

    # Generate training data
    X_train = torch.Tensor([-2.0, -1.0, -0.5, 0.0, 0.1, 0.2, 0.35, 0.3, 2.0]).unsqueeze(-1).cuda()  # shape [9, 1]
    y_train = ground_truth_fn(X_train).squeeze()

    # Example usage
    sampler = PathwiseSampler(
        lengthscale=torch.Tensor([[0.7,]]).cuda(),
        output_scale=1.,
        mean=y_train.mean().item(),
        sample_size=2,
    )

    sampler.update_data(X_train, y_train)
    sampler.draw_posterior_samples()

    # Evaluation points
    x_test = torch.linspace(-3, 3, 200).unsqueeze(-1).cuda()  # shape [200, 1]
    with torch.no_grad():
        samples = sampler.evaluate(x_test).cpu()  # shape [sample_size, 200]
        x_test_cpu = x_test.squeeze().cpu()

    # Plotting
    plt.figure(figsize=(10, 6))
    for i in range(samples.shape[0]):
        plt.plot(x_test_cpu, samples[i], alpha=0.6, label=f"Sample {i+1}")
    plt.plot(x_test_cpu, ground_truth_fn(x_test_cpu), 'k--', label="Ground Truth (sin)")
    plt.scatter(X_train.cpu(), y_train.cpu(), color='red', label="Training Data")
    plt.title("Posterior Function Samples from GP")
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    