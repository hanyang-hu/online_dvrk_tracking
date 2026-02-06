import torch
from torch.quasirandom import SobolEngine
import numpy as np

from evotorch.core import Problem, Solution, SolutionBatch
from evotorch.tools.misc import Real, Vector
from evotorch.algorithms import CMAES
from evotorch.algorithms.searchalgorithm import SearchAlgorithm, SinglePopulationAlgorithmMixin

from typing import Optional, Tuple, Union


def _safe_divide(a: Union[Real, torch.Tensor], b: Union[Real, torch.Tensor]) -> Union[torch.Tensor]:
    tolerance = 1e-8
    if abs(b) < tolerance:
        b = (-tolerance) if b < 0 else tolerance
    return a / b


# @torch.compile()
def _h_sig(p_sigma: torch.Tensor, c_sigma: float, iter: int) -> torch.Tensor:
    """Boolean flag for stalling the update to the evolution path for rank-1 updates
    Args:
        p_sigma (torch.Tensor): The evolution path for step-size updates
        c_sigma (float): The learning rate for step-size updates
        iter (int): The current iteration (generation)
    Returns:
        stall (torch.Tensor): Whether to stall the update to p_c, expressed as a single torch float with 0 = continue, 1 = stall
    """
    # infer dimension from p_sigma
    d = p_sigma.shape[-1]
    # Compute the discounted squared sum
    squared_sum = torch.norm(p_sigma).pow(2.0) / (1 - (1 - c_sigma) ** (2 * iter + 1))
    # Check boolean flag and return
    stall = (squared_sum / d) - 1 < 1 + 4.0 / (d + 1)
    return stall.any().to(p_sigma.dtype)


def generate_student_t_distribution(num_samples, dim, v=10.):
    """
    Generate a batch of samples from a standard Student's t-distribution
    with v degrees of freedom. Output shape: [num_samples, dim].
    """
    # Standard normal samples
    normal_samples = torch.randn(num_samples, dim)

    # Chi-squared samples
    chi_squared_samples = torch.distributions.Chi2(v).sample((num_samples,))

    # Scale: ensure proper broadcasting
    t_samples = normal_samples / torch.sqrt(chi_squared_samples[:, None] / v)

    return t_samples


def farthest_point_sampling(data, subsample_size):
    """
    Use farthest point sampling (FPS) to subsample a batch of data in parallel.
    Input:
        data: torch.tensor, shape (N, D)
        subsample_size: int
    Output:
        subsample_idx: torch.tensor, shape (subsample_size)
    """
    N, D = data.shape
    device = data.device

    if N <= subsample_size:
        return torch.arange(N, device=device)

    dist = torch.full((N,), float("inf"), device=device)
    subsample_idx = torch.zeros(subsample_size, dtype=torch.long, device=device)

    for i in range(subsample_size):
        if i == 0:
            selected_idx = torch.randint(0, N, (1,), device=device).item()
        else:
            selected_idx = torch.argmax(dist).item()
        subsample_idx[i] = selected_idx
        selected_data = data[selected_idx].unsqueeze(0)  # shape (1, D)

        # Compute distances from selected_data to all points
        new_dist = torch.norm(data - selected_data, dim=1)
        dist = torch.minimum(dist, new_dist)

    return subsample_idx


def generate_qmc_normal(sobol, num_samples, dim):
    # Generate low-discrepancy points (Sobol sequence) in [0, 1]^dim
    u = sobol.draw(num_samples)[ :, :dim]
    # Transform to standard normal distribution by  Φ⁻¹(u) = sqrt(2) * erfinv(2u - 1)
    normal_samples = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2.0 * u - 1.0)
    return normal_samples


def generate_low_discrepancy_normal(num_samples, dim, ratio=0.3):
    # Determine number of samples to generate
    num_ld_samples = int(num_samples * ratio)
    num_normal_samples = num_samples - num_ld_samples

    # Generate low-discrepancy samples
    qmc_normal_samples = torch.randn(2*num_samples, dim).cuda()
    subsample_idx = farthest_point_sampling(qmc_normal_samples, num_ld_samples)
    ld_samples = qmc_normal_samples[subsample_idx]

    # Generate normal samples
    normal_samples = torch.randn(num_normal_samples, dim).cuda()

    return torch.cat((ld_samples, normal_samples), dim=0)


def generate_sigma_normal(num_samples, dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # assert num_samples >= 2 * dim + 1, "Number of samples must be at least 2 * dim + 1 for sigma sampling."
    
    if num_samples < 2 * dim + 1:
        return torch.randn(num_samples, dim, device=device)

    # Generate 2 * dim canonical sigma points excluding the mean
    c = torch.sqrt(torch.tensor(float(dim), device=device))
    zero = torch.zeros(dim, device=device)
    eye = torch.eye(dim, device=device)
    sigma_points = torch.stack([
        c * eye,  # positive directions
        -c * eye  # negative directions
    ], dim=0).view(-1, dim)  # shape (2*dim + 1, dim)

    # Generate normal samples
    normal_samples = torch.randn(num_samples - 2 * dim, dim).cuda()

    return torch.cat((sigma_points, normal_samples), dim=0)



class CMAES_cus(CMAES):
    """
    CMA-ES algorithm for optimization.
    
    This class extends the CMA-ES algorithm from evotorch to allow custom elite size (mu).
    """
    def __init__(
        self,
        problem: Problem,
        *,
        stdev_init: Real = 1.0,
        popsize: Optional[int] = None, 
        center_init: Optional[Vector] = None,
        c_m: Real = 2., # NOTE. Modified to 2.0
        c_sigma: Optional[Real] = None,
        c_sigma_ratio: Real = 1.0,
        damp_sigma: Optional[Real] = None,
        damp_sigma_ratio: Real = 1.0,
        c_c: Optional[Real] = None,
        c_c_ratio: Real = 1.0,
        c_1: Optional[Real] = None,
        c_1_ratio: Real = 1.0,
        c_mu: Optional[Real] = None,
        c_mu_ratio: Real = 4.0, # NOTE. Modified to 4.0
        active: bool = True,
        csa_squared: bool = False,
        stdev_min: Optional[Real] = None,
        stdev_max: Optional[Real] = None,
        limit_C_decomposition: bool = True,
        obj_index: Optional[int] = None,
        mu_size: Optional[int] = 15, # NOTE. Modified to 15
        sobol: Optional[SobolEngine] = None
    ):
        """
        `__init__(...)`: Initialize the CMAES solver.

        Args:
            problem (Problem): The problem object which is being worked on.
            stdev_init (Real): Initial step-size
            popsize: Population size. Can be specified as an int,
                or can be left as None in which case the CMA-ES rule of thumb is applied:
                popsize = 4 + floor(3 log d) where d is the dimension
            center_init: Initial center point of the search distribution.
                Can be given as a Solution or as a 1-D array.
                If left as None, an initial center point is generated
                with the help of the problem object's `generate_values(...)`
                method.
            c_m (Real): Learning rate for updating the mean
                of the search distribution. By default the value is 1.

            c_sigma (Optional[Real]): Learning rate for updating the step size. If None,
                then the CMA-ES rules of thumb will be applied.
            c_sigma_ratio (Real): Multiplier on the learning rate for the step size.
                if c_sigma has been left as None, can be used to rescale the default c_sigma value.

            damp_sigma (Optional[Real]): Damping factor for updating the step size. If None,
                then the CMA-ES rules of thumb will be applied.
            damp_sigma_ratio (Real): Multiplier on the damping factor for the step size.
                if damp_sigma has been left as None, can be used to rescale the default damp_sigma value.

            c_c (Optional[Real]): Learning rate for updating the rank-1 evolution path.
                If None, then the CMA-ES rules of thumb will be applied.
            c_c_ratio (Real): Multiplier on the learning rate for the rank-1 evolution path.
                if c_c has been left as None, can be used to rescale the default c_c value.

            c_1 (Optional[Real]): Learning rate for the rank-1 update to the covariance matrix.
                If None, then the CMA-ES rules of thumb will be applied.
            c_1_ratio (Real): Multiplier on the learning rate for the rank-1 update to the covariance matrix.
                if c_1 has been left as None, can be used to rescale the default c_1 value.

            c_mu (Optional[Real]): Learning rate for the rank-mu update to the covariance matrix.
                If None, then the CMA-ES rules of thumb will be applied.
            c_mu_ratio (Real): Multiplier on the learning rate for the rank-mu update to the covariance matrix.
                if c_mu has been left as None, can be used to rescale the default c_mu value.

            active (bool): Whether to use Active CMA-ES. Defaults to True, consistent with the tutorial paper and pycma.
            csa_squared (bool): Whether to use the squared rule ("CSA_squared" in pycma) for the step-size adapation.
                This effectively corresponds to taking the natural gradient for the evolution path on the step size,
                rather than the default CMA-ES rule of thumb.

            stdev_min (Optional[Real]): Minimum allowed standard deviation of the search
                distribution. Leaving this as None means that no such
                boundary is to be used.
                Can be given as None or as a scalar.
            stdev_max (Optional[Real]): Maximum allowed standard deviation of the search
                distribution. Leaving this as None means that no such
                boundary is to be used.
                Can be given as None or as a scalar.

            separable (bool): Provide this as True if you would like the problem
                to be treated as a separable one. Treating a problem
                as separable means to adapt only the diagonal parts
                of the covariance matrix and to keep the non-diagonal
                parts 0. High dimensional problems result in large
                covariance matrices on which operating is computationally
                expensive. Therefore, for such high dimensional problems,
                setting `separable` as True might be useful.

            limit_C_decomposition (bool): Whether to limit the frequency of decomposition of the shape matrix C
                Setting this to True (default) means that C will not be decomposed every generation
                This degrades the quality of the sampling and updates, but provides a guarantee of O(d^2) time complexity.
                This option can be used with separable=True (e.g. for experimental reasons) but the performance will only degrade
                without time-complexity benefits.


            obj_index (Optional[int]): Objective index according to which evaluation
                of the solution will be done.
        """

        # Initialize the base class
        SearchAlgorithm.__init__(
            self,
            problem, 
            center=self._get_center, 
            stepsize=self._get_sigma,
            pop_best_eval_left=self._get_pop_best_eval_left,
            pop_best_eval_right=self._get_pop_best_eval_right,
            pop_best_left=self._get_pop_best_left,
            pop_best_right=self._get_pop_best_right,
        )

        # Ensure that the problem is numeric
        problem.ensure_numeric()

        # CMAES can't handle problem bounds. Ensure that it is unbounded
        problem.ensure_unbounded()

        # Store the objective index
        if not isinstance(self, CMAES_bi_manual_cus) and not isinstance(self, CMAES_bi_manual_bd_cus):
            self._obj_index = problem.normalize_obj_index(obj_index)

        # Track d = solution length for reference in initialization of hyperparameters
        d = self._problem.solution_length
        if isinstance(self, CMAES_bd_cus) or isinstance(self, CMAES_bi_manual_cus) or isinstance(self, CMAES_bi_manual_bd_cus):
            d = d // 2

        # === Initialize population ===
        self.popsize = popsize
        if not popsize:
            # Default value used in CMA-ES literature 4 + floor(3 log n)
            popsize = 4 + int(np.floor(3 * np.log(d)))
        self.popsize = int(popsize)
        # Half popsize, referred to as mu in CMA-ES literature
        self.mu = int(np.floor(popsize / 2)) 
        self._population = problem.generate_batch(popsize=popsize)

        # === Initialize search distribution ===

        separable = False
        self.separable = separable

        # If `center_init` is not given, generate an initial solution
        # with the help of the problem object.
        # If it is given as a Solution, then clone the solution's values
        # as a PyTorch tensor.
        # Otherwise, use the given initial solution as the starting
        # point in the search space.
        if center_init is None:
            center_init = self._problem.generate_values(1)
        elif isinstance(center_init, Solution):
            center_init = center_init.values.clone()

        # Store the center
        self.m = self._problem.make_tensor(center_init).squeeze()
        valid_shaped_m = (self.m.ndim == 1) and (len(self.m) == self._problem.solution_length)
        if not valid_shaped_m:
            raise ValueError(
                f"The initial center point was expected as a vector of length {self._problem.solution_length}."
                " However, the provided `center_init` has (or implies) a different shape."
            )

        # Store the initial step size
        self.sigma = self._problem.make_tensor(stdev_init)

        if separable:
            # Initialize C as the diagonal vector. Note that when separable, the eigendecomposition is not needed
            self.C = self._problem.make_ones(d)
            # In this case A is simply the square root of elements of C
            self.A = self._problem.make_ones(d)
        else:
            # Initialize C = AA^T all diagonal.
            self.C = self._problem.make_I(d)
            self.A = self.C.clone()

        # === Initialize raw weights ===
        # Conditioned on popsize

        # w_i = log((lambda + 1) / 2) - log(i) for i = 1 ... lambda
        self.raw_weights = self.problem.make_tensor(np.log((popsize + 1) / 2) - torch.log(torch.arange(popsize) + 1))
        # positive valued weights are the first mu
        positive_weights = self.raw_weights[: self.mu]
        negative_weights = self.raw_weights[self.mu :]

        # Variance effective selection mass of positive weights
        # Not affected by future updates to raw_weights
        self.mu_eff = torch.sum(positive_weights).pow(2.0) / torch.sum(positive_weights.pow(2.0))

        # === Initialize search parameters ===
        # Conditioned on weights

        # Store fixed information
        self.c_m = c_m
        self.active = active
        self.csa_squared = csa_squared
        self.stdev_min = stdev_min
        self.stdev_max = stdev_max

        # Learning rate for step-size adaption
        if c_sigma is None:
            c_sigma = (self.mu_eff + 2.0) / (d + self.mu_eff + 3)
        self.c_sigma = c_sigma_ratio * c_sigma

        # Damping factor for step-size adapation
        if damp_sigma is None:
            damp_sigma = 1 + 2 * max(0, torch.sqrt((self.mu_eff - 1) / (d + 1)) - 1) + self.c_sigma
        self.damp_sigma = damp_sigma_ratio * damp_sigma

        # Learning rate for evolution path for rank-1 update
        if c_c is None:
            # Branches on separability
            if separable:
                c_c = (1 + (1 / d) + (self.mu_eff / d)) / (d**0.5 + (1 / d) + 2 * (self.mu_eff / d))
            else:
                c_c = (4 + self.mu_eff / d) / (d + (4 + 2 * self.mu_eff / d))
        self.c_c = c_c_ratio * c_c

        # Learning rate for rank-1 update to covariance matrix
        if c_1 is None:
            # Branches on separability
            if separable:
                c_1 = 1.0 / (d + 2.0 * np.sqrt(d) + self.mu_eff / d)
            else:
                c_1 = min(1, popsize / 6) * 2 / ((d + 1.3) ** 2.0 + self.mu_eff)
        self.c_1 = c_1_ratio * c_1

        # Learning rate for rank-mu update to covariance matrix
        if c_mu is None:
            # Branches on separability
            if separable:
                c_mu = (0.25 + self.mu_eff + (1.0 / self.mu_eff) - 2) / (d + 4 * np.sqrt(d) + (self.mu_eff / 2.0))
            else:
                c_mu = min(
                    1 - self.c_1, 2 * ((0.25 + self.mu_eff - 2 + (1 / self.mu_eff)) / ((d + 2) ** 2.0 + self.mu_eff))
                )
        self.c_mu = c_mu_ratio * c_mu

        # The 'variance aware' coefficient used for the additive component of the evolution path for sigma
        self.variance_discount_sigma = torch.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff)
        # The 'variance aware' coefficient used for the additive component of the evolution path for rank-1 updates
        self.variance_discount_c = torch.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff)

        # === Finalize weights ===
        # Conditioned on search parameters and raw weights

        # Positive weights always sum to 1
        positive_weights = positive_weights / torch.sum(positive_weights)

        if self.active:
            # Active CMA-ES: negative weights sum to alpha

            # Get the variance effective selection mass of negative weights
            mu_eff_neg = torch.sum(negative_weights).pow(2.0) / torch.sum(negative_weights.pow(2.0))

            # Alpha is the minimum of the following 3 terms
            alpha_mu = 1 + self.c_1 / self.c_mu
            alpha_mu_eff = 1 + 2 * mu_eff_neg / (self.mu_eff + 2)
            alpha_pos_def = (1 - self.c_mu - self.c_1) / (d * self.c_mu)
            alpha = min([alpha_mu, alpha_mu_eff, alpha_pos_def])

            # Rescale negative weights
            negative_weights = alpha * negative_weights / torch.sum(torch.abs(negative_weights))
        else:
            # Negative weights are simply zero
            negative_weights = torch.zeros_like(negative_weights)

        # Concatenate final weights
        self.weights = torch.cat([positive_weights, negative_weights], dim=-1)

        # === Some final setup ===

        # Initialize the evolution paths
        self.p_sigma = 0.0
        self.p_c = 0.0

        # Hansen's approximation to the expectation of ||x|| x ~ N(0, I_d).
        # Note that we could use the exact formulation with Gamma functions, but we'll retain this form for consistency
        self.unbiased_expectation = np.sqrt(d) * (1 - (1 / (4 * d)) + 1 / (21 * d**2))

        self.last_ex = None

        # How often to decompose C
        self.limit_C_decomposition = limit_C_decomposition
        if limit_C_decomposition:
            self.decompose_C_freq = max(1, int(np.floor(_safe_divide(1, 10 * d * (self.c_1.cpu() + self.c_mu.cpu())))))
        else:
            self.decompose_C_freq = 1

        # Use the SinglePopulationAlgorithmMixin to enable additional status reports regarding the population.
        SinglePopulationAlgorithmMixin.__init__(self)

        self.mu = mu_size  # Set the mu size for CMA-ES

        self.sobol = sobol

        c = torch.sqrt(torch.tensor(float(self._problem.solution_length), device=self._problem.device))
        zero = torch.zeros(self._problem.solution_length, device=self._problem.device)
        eye = torch.eye(self._problem.solution_length, device=self._problem.device)
        self.sigma_points = torch.stack([
            c * eye,  # positive directions
            -c * eye  # negative directions
        ], dim=0).view(-1, self._problem.solution_length)  # shape (2*dim + 1, dim)

        # Dual objective bookkeeping
        self._pop_best_eval_left = float('inf')
        self._pop_best_eval_right = float('inf')
        self._pop_best_left = None
        self._pop_best_right = None

    def _get_pop_best_eval_left(self):
        return self._pop_best_eval_left

    def _get_pop_best_eval_right(self):
        return self._pop_best_eval_right

    def _get_pop_best_left(self):
        return self._pop_best_left

    def _get_pop_best_right(self):
        return self._pop_best_right

    # @torch.compile()
    def sample_distribution(self, num_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample the population. All 3 representations of solutions are returned for easy calculations of updates.
        Note that the computation time of this operation of O(d^2 num_samples) unless separable, in which case O(d num_samples)
        Args:
            num_samples (Optional[int]): The number of samples to draw. If None, then the population size is used
        Returns:
            zs (torch.Tensor): A tensor of shape [num_samples, d] of samples from the local coordinate space e.g. z_i ~ N(0, I_d)
            ys (torch.Tensor): A tensor of shape [num_samples, d] of samples from the shaped coordinate space e.g. y_i ~ N(0, C)
            xs (torch.Tensor): A tensor of shape [num_samples, d] of samples from the search space e.g. x_i ~ N(m, sigma^2 C)
        """
        if num_samples is None:
            num_samples = self.popsize

        # Generate z values (num_samples, d)
        # zs = generate_qmc_normal(self.sobol, num_samples=num_samples - 2 * self._problem.solution_length, dim=self._problem.solution_length).to(self._problem.device)
        # zs = torch.cat([self.sigma_points, zs], dim=0)
        zs = generate_qmc_normal(self.sobol, num_samples=num_samples, dim=self._problem.solution_length).to(self._problem.device)

        # Construct ys = A zs
        ys = (self.A @ zs.T).T

        # Construct xs = m + sigma ys
        xs = self.m.unsqueeze(0) + self.sigma * ys

        return zs, ys, xs

    # @torch.compile()
    def decompose_C(self) -> None:
        self.A = torch.linalg.cholesky_ex(self.C).L # avoid synchronization with the CPU
        # self.A = torch.linalg.cholesky(self.C)

        # if self.A.isnan().any():
        #     raise ValueError("NaN detected in matrix A during sampling.")

    # @torch.compile()
    def update_m(self, zs: torch.Tensor, ys: torch.Tensor, assigned_weights: torch.Tensor) -> torch.Tensor:
        """Update the center of the search distribution m
        With zs and ys retained from sampling, this operation is O(popsize d), as it involves summing across popsize d-dimensional vectors.
        Args:
            zs (torch.Tensor): A tensor of shape [popsize, d] of samples from the local coordinate space e.g. z_i ~ N(0, I_d)
            ys (torch.Tensor): A tensor of shape [popsize, d] of samples from the shaped coordinate space e.g. y_i ~ N(0, C)
            assigned_weights (torch.Tensor): A [popsize, ] dimensional tensor of ordered weights
        Returns:
            local_m_displacement (torch.Tensor): A tensor of shape [d], corresponding to the local transformation of m,
                (1/sigma) (C^-1/2) (m' - m) where m' is the updated m
            shaped_m_displacement (torch.Tensor): A tensor of shape [d], corresponding to the shaped transformation of m,
                (1/sigma) (m' - m) where m' is the updated m
        """
        # Get the top-mu weights
        top_mu = torch.topk(assigned_weights, k=self.mu)
        top_mu_weights = top_mu.values
        top_mu_indices = top_mu.indices

        # Compute the weighted recombination in local coordinate space
        local_m_displacement = torch.sum(top_mu_weights.unsqueeze(-1) * zs[top_mu_indices], dim=0)
        # Compute the weighted recombination in shaped coordinate space
        shaped_m_displacement = torch.sum(top_mu_weights.unsqueeze(-1) * ys[top_mu_indices], dim=0)

        # Update m
        self.m = self.m + self.c_m * self.sigma * shaped_m_displacement

        # Return the weighted recombinations
        return local_m_displacement, shaped_m_displacement

    # @torch.compile()
    def update_p_sigma(self, local_m_displacement: torch.Tensor) -> None:
        """Update the evolution path for sigma, p_sigma
        This operation is bounded O(d), as is simply the sum of vectors
        Args:
            local_m_displacement (torch.Tensor): The weighted recombination of local samples zs, corresponding to
                (1/sigma) (C^-1/2) (m' - m) where m' is the updated m
        """
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + self.variance_discount_sigma * local_m_displacement

    # @torch.compile()
    def update_sigma(self) -> None:
        """Update the step size sigma according to its evolution path p_sigma
        This operation is bounded O(d), with the most expensive component being the norm of the evolution path, a d-dimensional vector.
        """
        d = self._problem.solution_length
        if isinstance(self, CMAES_bi_manual_cus) or isinstance(self, CMAES_bi_manual_bd_cus):
            d = d // 2

        # Compute the exponential update
        if self.csa_squared:
            # Exponential update based on natural gradient maximizing squared norm of p_sigma
            exponential_update = (torch.norm(self.p_sigma).pow(2.0) / d - 1) / 2
        else:
            # Exponential update increasing likelihood p_sigma having expected norm
            exponential_update = torch.norm(self.p_sigma) / self.unbiased_expectation - 1
        # Rescale exponential update based on learning rate + damping factor
        exponential_update = (self.c_sigma / self.damp_sigma) * exponential_update
        # Multiplicative update to sigma
        self.sigma = self.sigma * torch.exp(exponential_update)

    # @torch.compile()
    def update_p_c(self, shaped_m_displacement: torch.Tensor, h_sig: torch.Tensor) -> None:
        """Update the evolution path for rank-1 update, p_c
        This operation is bounded O(d), as is simply the sum of vectors
        Args:
            local_m_displacement (torch.Tensor): The weighted recombination of shaped samples ys, corresponding to
                (1/sigma) (m' - m) where m' is the updated m
            h_sig (torch.Tensor): Whether to stall the update based on the evolution path on sigma, p_sigma, expressed as a torch float
        """
        self.p_c = (1 - self.c_c) * self.p_c + h_sig * self.variance_discount_c * shaped_m_displacement

    # @torch.compile()
    def update_C(self, zs: torch.Tensor, ys: torch.Tensor, assigned_weights: torch.Tensor, h_sig: torch.Tensor) -> None:
        """Update the covariance shape matrix C based on rank-1 and rank-mu updates
        This operation is bounded O(d^2 popsize), which is associated with computing the rank-mu update (summing across popsize d*d matrices)
        Args:
            zs (torch.Tensor): A tensor of shape [popsize, d] of samples from the local coordinate space e.g. z_i ~ N(0, I_d)
            ys (torch.Tensor): A tensor of shape [popsize, d] of samples from the shaped coordinate space e.g. y_i ~ N(0, C)
            assigned_weights (torch.Tensor): A [popsize, ] dimensional tensor of ordered weights
            h_sig (torch.Tensor): Whether to stall the update based on the evolution path on sigma, p_sigma, expressed as a torch float
        """
        d = self._problem.solution_length
        # If using Active CMA-ES, reweight negative weights
        if self.active:
            assigned_weights = torch.where(
                assigned_weights > 0, assigned_weights, d * assigned_weights / torch.norm(zs, dim=-1).pow(2.0)
            )
        c1a = self.c_1 * (1 - (1 - h_sig**2) * self.c_c * (2 - self.c_c))  # adjust for variance loss
        weighted_pc = (self.c_1 / (c1a + 1e-23)) ** 0.5

        # Rank-1 update
        r1_update = c1a * (torch.outer(weighted_pc * self.p_c, weighted_pc * self.p_c) - self.C)
        # Rank-mu update
        rmu_update = self.c_mu * (
            torch.sum(assigned_weights.unsqueeze(-1).unsqueeze(-1) * (ys.unsqueeze(1) * ys.unsqueeze(2)), dim=0)
            - torch.sum(self.weights) * self.C
        )

        if r1_update.isnan().any():
            r1_update = 0.

        if rmu_update.isnan().any():
            rmu_update = 0.

        # Update C
        self.C = self.C + r1_update + rmu_update

        # if self.C.isnan().any():
        #     print("C matrix:", self.C)
        #     print("r1_update:", r1_update)
        #     print("rmu_update:", rmu_update)
        #     raise ValueError("NaN detected in matrix C during sampling.")

    # def _step(self):
    #     import time
        
    #     """Perform a step of the CMA-ES solver"""

    #     # === Sampling, evaluation and ranking ===

    #     # Sample the search distribution
    #     torch.cuda.synchronize()
    #     t_1 = time.time()
    #     zs, ys, xs = self.sample_distribution()
    #     torch.cuda.synchronize()
    #     t_2 = time.time()
    #     print(f"Sampling time: {t_2 - t_1} seconds")
    #     # Get the weights assigned to each solution
    #     assigned_weights = self.get_population_weights(xs)
    #     torch.cuda.synchronize()
    #     t_3 = time.time()
    #     print(f"Weight assignment (evaluation) time: {t_3 - t_2} seconds")

    #     # === Center adaption ===

    #     local_m_displacement, shaped_m_displacement = self.update_m(zs, ys, assigned_weights)
    #     torch.cuda.synchronize()
    #     t_4 = time.time()
    #     print(f"Center update time: {t_4 - t_3} seconds")

    #     # === Step size adaption ===

    #     # Update evolution path p_sigma
    #     self.update_p_sigma(local_m_displacement)
    #     # Update sigma
    #     self.update_sigma()

    #     # Compute h_sig, a boolean flag for stalling the update to p_c
    #     h_sig = _h_sig(self.p_sigma, self.c_sigma, self._steps_count)

    #     # === Unscaled covariance adapation ===

    #     # Update evolution path p_c
    #     self.update_p_c(shaped_m_displacement, h_sig)
    #     torch.cuda.synchronize()
    #     t_5 = time.time()
    #     print(f"Step size update time: {t_5 - t_4} seconds")
    #     # Update the covariance shape C
    #     self.update_C(zs, ys, assigned_weights, h_sig)
    #     torch.cuda.synchronize()
    #     t_6 = time.time()
    #     print(f"Covariance update time: {t_6 - t_5} seconds")

    #     # === Post-step corrections ===

    #     # Limit element-wise standard deviation of sigma^2 C
    #     if self.stdev_min is not None or self.stdev_max is not None:
    #         self.C = _limit_stdev(self.sigma, self.C, self.stdev_min, self.stdev_max)

    #     # Decompose C
    #     if (self._steps_count + 1) % self.decompose_C_freq == 0:
    #         self.decompose_C()
    #         torch.cuda.synchronize()

    #         t_7 = time.time()
    #         print(f"Decompose C time: {t_7 - t_6} seconds")

    #     # print("Total time: ", t_7 - t_1, "seconds")


class CMAES_bd_cus(CMAES_cus):
    """
    Block-diagonal CMA-ES with two independent covariance blocks.
    """

    def __init__(self, *args, split_dim: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)

        d = self._problem.solution_length
        device = self._problem.device

        # ---- block split ----
        if split_dim is None:
            assert d % 2 == 0, "solution_length must be even if split_dim is not provided"
            split_dim = d // 2

        self.blocks = [
            (0, split_dim),
            (split_dim, d),
        ]

        # ---- blockwise covariance ----
        self.C_blocks = [
            torch.eye(split_dim, device=device),
            torch.eye(d - split_dim, device=device),
        ]

        self.A_blocks = [
            torch.eye(split_dim, device=device),
            torch.eye(d - split_dim, device=device),
        ]

        # ---- blockwise evolution paths ----
        self.p_c_blocks = [
            torch.zeros(split_dim, device=device),
            torch.zeros(d - split_dim, device=device),
        ]

        # disable full C/A from parent
        self.C = None
        self.A = None

    def sample_distribution(self, num_samples: Optional[int] = None):
        if num_samples is None:
            num_samples = self.popsize

        zs = generate_qmc_normal(
            self.sobol,
            num_samples=num_samples,
            dim=self._problem.solution_length,
        ).to(self._problem.device)

        ys = torch.zeros_like(zs)

        for (i0, i1), A in zip(self.blocks, self.A_blocks):
            ys[:, i0:i1] = (A @ zs[:, i0:i1].T).T

        xs = self.m.unsqueeze(0) + self.sigma * ys
        return zs, ys, xs

    def decompose_C(self) -> None:
        for i in range(len(self.blocks)):
            self.A_blocks[i] = torch.linalg.cholesky_ex(
                self.C_blocks[i]
            ).L

    def update_p_c(self, shaped_m_displacement: torch.Tensor, h_sig: torch.Tensor) -> None:
        for bi, (i0, i1) in enumerate(self.blocks):
            self.p_c_blocks[bi] = (
                (1 - self.c_c) * self.p_c_blocks[bi]
                + h_sig * self.variance_discount_c * shaped_m_displacement[i0:i1]
            )

    def update_C(
        self,
        zs: torch.Tensor,
        ys: torch.Tensor,
        assigned_weights: torch.Tensor,
        h_sig: torch.Tensor,
    ) -> None:

        for bi, (i0, i1) in enumerate(self.blocks):
            zs_b = zs[:, i0:i1]
            ys_b = ys[:, i0:i1]
            C_b = self.C_blocks[bi]
            p_c_b = self.p_c_blocks[bi]

            d_b = i1 - i0

            # Active CMA-ES correction
            if self.active:
                weights_b = torch.where(
                    assigned_weights > 0,
                    assigned_weights,
                    d_b * assigned_weights / torch.norm(zs_b, dim=-1).pow(2.0),
                )
            else:
                weights_b = assigned_weights

            c1a = self.c_1 * (1 - (1 - h_sig**2) * self.c_c * (2 - self.c_c))
            weighted_pc = (self.c_1 / (c1a + 1e-23)) ** 0.5

            # Rank-1
            r1 = c1a * (
                torch.outer(weighted_pc * p_c_b, weighted_pc * p_c_b) - C_b
            )

            # Rank-mu
            rmu = self.c_mu * (
                torch.sum(
                    weights_b[:, None, None]
                    * (ys_b[:, :, None] * ys_b[:, None, :]),
                    dim=0,
                )
                - torch.sum(self.weights) * C_b
            )

            if torch.isnan(r1).any():
                r1 = 0.0
            if torch.isnan(rmu).any():
                rmu = 0.0

            self.C_blocks[bi] = C_b + r1 + rmu


class CMAES_bi_manual_cus(CMAES_cus):
    """
    Bi-manual CMA-ES with a SINGLE search distribution.
    Equivalent to CMAES_cus in dynamics, but tracks left/right
    objectives separately while optimizing their sum.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Dual-objective bookkeeping
        self._pop_best_eval_left = float("inf")
        self._pop_best_eval_right = float("inf")
        self._pop_best_left = None
        self._pop_best_right = None

        d = self._problem.solution_length
        device = self._problem.device

        # ---- split dimensions ----
        assert d % 2 == 0, "solution_length must be even if split_dim not provided"
        split_dim = d // 2
        self.blocks = [(0, split_dim), (split_dim, d)]

    # --------------------------------------------------
    # Accessors (for logging / diagnostics)
    # --------------------------------------------------
    def _get_pop_best_eval_left(self):
        return self._pop_best_eval_left

    def _get_pop_best_eval_right(self):
        return self._pop_best_eval_right

    def _get_pop_best_left(self):
        return self._pop_best_left

    def _get_pop_best_right(self):
        return self._pop_best_right

    # --------------------------------------------------
    # Weight computation (KEY MODIFICATION)
    # --------------------------------------------------
    def _get_population_weights(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate population, track left/right best separately,
        but rank by total loss = left + right.
        """

        # Evaluate population
        self._population.set_values(xs)
        self.problem.evaluate(self._population)

        # Fetch objectives
        eval_left = self._population.access_evals(obj_index=0).squeeze()
        eval_right = self._population.access_evals(obj_index=1).squeeze()

        # ---- track best LEFT ----
        best_left_idx = torch.argmin(eval_left)
        if eval_left[best_left_idx] < self._pop_best_eval_left:
            self._pop_best_eval_left = eval_left[best_left_idx]
            self._pop_best_left = xs[best_left_idx][:self.blocks[0][1]].detach().clone()

        # ---- track best RIGHT ----
        best_right_idx = torch.argmin(eval_right)
        if eval_right[best_right_idx] < self._pop_best_eval_right:
            self._pop_best_eval_right = eval_right[best_right_idx]
            self._pop_best_right = xs[best_right_idx][self.blocks[1][0]:self.blocks[1][1]].detach().clone()

        # ---- total loss for CMA-ES update ----
        eval_total = eval_left + eval_right

        # ---- rank by total loss ----
        ranks = torch.argsort(eval_total)
        assigned_weights = self.weights[ranks]

        return assigned_weights

    # --------------------------------------------------
    # Step (same as CMAES_cus, kept explicit for clarity)
    # --------------------------------------------------
    def _step(self):
        # --- sampling ---
        zs, ys, xs = self.sample_distribution()

        # --- evaluation + ranking ---
        assigned_weights = self._get_population_weights(xs)

        # --- center update ---
        local_disp, shaped_disp = self.update_m(zs, ys, assigned_weights)

        # --- sigma update ---
        self.update_p_sigma(local_disp)
        self.update_sigma()

        # --- covariance update ---
        h_sig = _h_sig(self.p_sigma, self.c_sigma, self._steps_count)
        self.update_p_c(shaped_disp, h_sig)
        self.update_C(zs, ys, assigned_weights, h_sig)

        # --- constraints ---
        if self.stdev_min is not None or self.stdev_max is not None:
            self.C = _limit_stdev(self.sigma, self.C, self.stdev_min, self.stdev_max)

        # --- decomposition ---
        if (self._steps_count + 1) % self.decompose_C_freq == 0:
            self.decompose_C()

        self._steps_count += 1


"""
class CMAES_bi_manual_bd_cus(CMAES_cus):
    def __init__(self, *args, split_dim: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)

        d = self._problem.solution_length
        device = self._problem.device

        # ---- split dimensions ----
        if split_dim is None:
            assert d % 2 == 0, "solution_length must be even if split_dim not provided"
            split_dim = d // 2
        self.blocks = [(0, split_dim), (split_dim, d)]

        # ---- blockwise covariance and Cholesky factors ----
        self.C_blocks = [torch.eye(split_dim, device=device),
                         torch.eye(d - split_dim, device=device)]
        self.A_blocks = [torch.eye(split_dim, device=device),
                         torch.eye(d - split_dim, device=device)]

        # ---- blockwise evolution paths ----
        self.p_c_blocks = [torch.zeros(split_dim, device=device),
                           torch.zeros(d - split_dim, device=device)]
        self.p_sigma_blocks = [torch.zeros(split_dim, device=device),
                               torch.zeros(d - split_dim, device=device)]

        # ---- blockwise step sizes ----
        self.sigma_blocks = [self.sigma.clone(), self.sigma.clone()]

        # Clear global sigma/C/A to avoid misuse
        self.sigma = None
        self.C = None
        self.A = None

        self.sigma = torch.tensor(0.0)  # dummy to avoid errors in base class

    def sample_distribution(self, num_samples: Optional[int] = None):
        if num_samples is None:
            num_samples = self.popsize

        zs = generate_qmc_normal(
            self.sobol, num_samples=num_samples, dim=self._problem.solution_length
        ).to(self._problem.device)

        ys = torch.zeros_like(zs)
        xs = torch.zeros_like(zs)

        for i, (i0, i1) in enumerate(self.blocks):
            ys[:, i0:i1] = (self.A_blocks[i] @ zs[:, i0:i1].T).T
            xs[:, i0:i1] = self.m[i0:i1].unsqueeze(0) + self.sigma_blocks[i] * ys[:, i0:i1]

        return zs, ys, xs

    def _get_population_weights_per_block(self, xs: torch.Tensor):
        self._population.set_values(xs)
        self.problem.evaluate(self._population)

        eval_left = self._population.access_evals(obj_index=0).squeeze()
        eval_right = self._population.access_evals(obj_index=1).squeeze()

        self.pop_eval_left = eval_left.detach()
        self.pop_eval_right = eval_right.detach()

        # Track best individuals
        best_idx_left = torch.argmin(eval_left)
        best_idx_right = torch.argmin(eval_right)

        if self._pop_best_eval_left is None or eval_left[best_idx_left] < self._pop_best_eval_left:
            self._pop_best_eval_left = eval_left[best_idx_left]
            self._pop_best_left = xs[best_idx_left][:self.blocks[0][1]].clone()

        if self._pop_best_eval_right is None or eval_right[best_idx_right] < self._pop_best_eval_right:
            self._pop_best_eval_right = eval_right[best_idx_right]
            self._pop_best_right = xs[best_idx_right][self.blocks[1][0]:self.blocks[1][1]].clone()

        # Compute weights per block
        ranks_left = torch.argsort(eval_left)
        ranks_right = torch.argsort(eval_right)
        weights_left = self.weights[ranks_left]
        weights_right = self.weights[ranks_right]

        return weights_left, weights_right

    def update_m_block(self, block_idx, zs_block, ys_block, assigned_weights):
        top_mu = torch.topk(assigned_weights, k=self.mu)
        w = top_mu.values
        idx = top_mu.indices
        local_disp = torch.sum(w.unsqueeze(-1) * zs_block[idx], dim=0)
        shaped_disp = torch.sum(w.unsqueeze(-1) * ys_block[idx], dim=0)
        self.m[self.blocks[block_idx][0]:self.blocks[block_idx][1]] += self.c_m * self.sigma_blocks[block_idx] * shaped_disp
        return local_disp, shaped_disp

    def update_p_sigma_block(self, block_idx, local_disp):
        self.p_sigma_blocks[block_idx] = (1 - self.c_sigma) * self.p_sigma_blocks[block_idx] + self.variance_discount_sigma * local_disp

    def update_sigma_block(self, block_idx):
        d_b = self.p_sigma_blocks[block_idx].shape[0]
        p = self.p_sigma_blocks[block_idx]
        if self.csa_squared:
            exponential_update = (torch.norm(p).pow(2.0) / d_b - 1) / 2
        else:
            exponential_update = torch.norm(p) / self.unbiased_expectation - 1
        exponential_update = (self.c_sigma / self.damp_sigma) * exponential_update
        self.sigma_blocks[block_idx] = self.sigma_blocks[block_idx] * torch.exp(exponential_update)

    def update_p_c_block(self, block_idx, shaped_disp, h_sig):
        p_c_b = self.p_c_blocks[block_idx]
        self.p_c_blocks[block_idx] = (1 - self.c_c) * p_c_b + h_sig * self.variance_discount_c * shaped_disp

    def update_C_block(self, block_idx, zs_block, ys_block, assigned_weights, h_sig):
        C_b = self.C_blocks[block_idx]
        p_c_b = self.p_c_blocks[block_idx]
        d_b = C_b.shape[0]

        if self.active:
            weights_b = torch.where(
                assigned_weights > 0,
                assigned_weights,
                d_b * assigned_weights / torch.norm(zs_block, dim=-1).pow(2.0),
            )
        else:
            weights_b = assigned_weights

        c1a = self.c_1 * (1 - (1 - h_sig ** 2) * self.c_c * (2 - self.c_c))
        weighted_pc = (self.c_1 / (c1a + 1e-23)) ** 0.5

        r1 = c1a * (torch.outer(weighted_pc * p_c_b, weighted_pc * p_c_b) - C_b)
        rmu = self.c_mu * (torch.sum(weights_b[:, None, None] * (ys_block[:, :, None] * ys_block[:, None, :]), dim=0)
                           - torch.sum(self.weights) * C_b)

        self.C_blocks[block_idx] = C_b + r1 + rmu

    def decompose_C(self):
        for i, C_b in enumerate(self.C_blocks):
            self.A_blocks[i] = torch.linalg.cholesky_ex(C_b).L

    def _step(self):
        zs, ys, xs = self.sample_distribution()
        weights_left, weights_right = self._get_population_weights_per_block(xs)

        # --- block-wise updates ---
        for block_idx, (i0, i1) in enumerate(self.blocks):
            zs_block = zs[:, i0:i1]
            ys_block = ys[:, i0:i1]
            assigned_weights = weights_left if block_idx == 0 else weights_right

            local_disp, shaped_disp = self.update_m_block(block_idx, zs_block, ys_block, assigned_weights)
            self.update_p_sigma_block(block_idx, local_disp)
            self.update_sigma_block(block_idx)

            h_sig = _h_sig(self.p_sigma_blocks[block_idx], self.c_sigma, self._steps_count)
            self.update_p_c_block(block_idx, shaped_disp, h_sig)
            self.update_C_block(block_idx, zs_block, ys_block, assigned_weights, h_sig)

        if (self._steps_count + 1) % self.decompose_C_freq == 0:
            self.decompose_C()

        self._steps_count += 1
"""


from evotorch.tools import set_default_logger_config
import logging

set_default_logger_config(
    logger_level=logging.WARNING,  # only print the message if it is at least a WARNING
    override=True,  # override the previous logging settings of EvoTorch
)


class DummyProblem(Problem):
    def __init__(self, solution_length, device):
        super().__init__(
            objective_sense="min",
            solution_length=solution_length, 
            device=device,
            initial_bounds=([-float('inf')] * solution_length, [float('inf')] * solution_length)
        )
    def _evaluate(self, solutions):
        return torch.zeros(solutions.shape[0], 1, device=solutions.device)


class CMAES_bi_manual_bd_cus(SearchAlgorithm):
    """
    Wrap two CMAES_cus instances to create a bi-manual CMA-ES with block-diagonal covariance.
    """
    def __init__(
        self,
        problem: Problem,
        *,
        stdev_init: Real = 1.0,
        popsize: Optional[int] = None, 
        center_init: Optional[Vector] = None,
        c_m: Real = 2., # NOTE. Modified to 2.0
        c_sigma: Optional[Real] = None,
        c_sigma_ratio: Real = 1.0,
        damp_sigma: Optional[Real] = None,
        damp_sigma_ratio: Real = 1.0,
        c_c: Optional[Real] = None,
        c_c_ratio: Real = 1.0,
        c_1: Optional[Real] = None,
        c_1_ratio: Real = 1.0,
        c_mu: Optional[Real] = None,
        c_mu_ratio: Real = 4.0, # NOTE. Modified to 4.0
        active: bool = True,
        csa_squared: bool = False,
        stdev_min: Optional[Real] = None,
        stdev_max: Optional[Real] = None,
        limit_C_decomposition: bool = True,
        obj_index: Optional[int] = None,
        mu_size: Optional[int] = 15, # NOTE. Modified to 15
        sobol: Optional[SobolEngine] = None
    ):
        # Initialize the base class
        SearchAlgorithm.__init__(
            self, 
            problem, 
            pop_best_eval_left=self._get_pop_best_eval_left,
            pop_best_eval_right=self._get_pop_best_eval_right,
            pop_best_left=self._get_pop_best_left,
            pop_best_right=self._get_pop_best_right,
        )

        self._problem = problem
        d = problem.solution_length
        device = problem.device
        assert d % 2 == 0, "solution_length must be divisible by 2"
        split_dim = d // 2
        self.blocks = [(0, split_dim), (split_dim, d)]

        # Create two CMAES_cus instances for each block
        # Each CMAES_cus sees only its block
        # Use same hyperparameters for both
        center0 = center_init[:split_dim] if center_init is not None else None
        center1 = center_init[split_dim:] if center_init is not None else None

        dummy_problem = DummyProblem(split_dim, device)

        self.cma_blocks = [
            CMAES_cus(dummy_problem, stdev_init=stdev_init, popsize=popsize, center_init=center0, c_m=c_m, c_sigma=c_sigma, c_sigma_ratio=c_sigma_ratio, damp_sigma=damp_sigma, damp_sigma_ratio=damp_sigma_ratio, c_c=c_c, c_c_ratio=c_c_ratio, c_1=c_1, c_1_ratio=c_1_ratio, c_mu=c_mu, c_mu_ratio=c_mu_ratio, active=active, csa_squared=csa_squared, stdev_min=stdev_min, stdev_max=stdev_max, limit_C_decomposition=limit_C_decomposition, obj_index=obj_index, mu_size=mu_size, sobol=sobol),
            CMAES_cus(dummy_problem, stdev_init=stdev_init, popsize=popsize, center_init=center1, c_m=c_m, c_sigma=c_sigma, c_sigma_ratio=c_sigma_ratio, damp_sigma=damp_sigma, damp_sigma_ratio=damp_sigma_ratio, c_c=c_c, c_c_ratio=c_c_ratio, c_1=c_1, c_1_ratio=c_1_ratio, c_mu=c_mu, c_mu_ratio=c_mu_ratio, active=active, csa_squared=csa_squared, stdev_min=stdev_min, stdev_max=stdev_max, limit_C_decomposition=limit_C_decomposition, obj_index=obj_index, mu_size=mu_size, sobol=sobol)
        ]

        # Track global m (combined)
        if center_init is None:
            self.m = torch.zeros(d, device=device)
            self.m[:split_dim] = self.cma_blocks[0].m
            self.m[split_dim:] = self.cma_blocks[1].m
        else:
            self.m = center_init.clone()

        # Track best per block
        self._pop_best_eval_left = float("inf")
        self._pop_best_eval_right = float("inf")
        self._pop_best_left = None
        self._pop_best_right = None

        # Steps count
        self._steps_count = 0
        # Decomposition frequency (use max of both blocks)
        self.decompose_C_freq = max(self.cma_blocks[0].decompose_C_freq,
                                    self.cma_blocks[1].decompose_C_freq)
    
        self._population = problem.generate_batch(popsize=popsize)

    def _get_pop_best_eval_left(self):
        return self._pop_best_eval_left

    def _get_pop_best_eval_right(self):
        return self._pop_best_eval_right

    def _get_pop_best_left(self):
        return self._pop_best_left

    def _get_pop_best_right(self):
        return self._pop_best_right

    def _step(self):
        """
        Perform one step of bi-manual optimization.
        Each CMAES_cus instance operates on its block, but evaluations
        are done jointly on the full population.
        """
        popsize = self.cma_blocks[0].popsize
        d = self._problem.solution_length
        split0, split1 = self.blocks[0][1], self.blocks[1][1]

        # --- Sampling ---
        zs_block0, ys_block0, xs_block0 = self.cma_blocks[0].sample_distribution(popsize)
        zs_block1, ys_block1, xs_block1 = self.cma_blocks[1].sample_distribution(popsize)

        # Merge blocks into full population
        xs_full = torch.zeros((popsize, d), device=self._problem.device)
        xs_full[:, :split0] = xs_block0
        xs_full[:, split0:] = xs_block1

        # --- Evaluate full population ---
        # We evaluate the full problem at once
        self._population.set_values(xs_full)
        self._problem.evaluate(self._population)

        # Extract objectives per block
        eval_left = self._population.access_evals(obj_index=0).squeeze()
        eval_right = self._population.access_evals(obj_index=1).squeeze()

        # --- Track bests ---
        best_idx_left = torch.argmin(eval_left)
        if eval_left[best_idx_left] < self._pop_best_eval_left:
            self._pop_best_eval_left = eval_left[best_idx_left]
            self._pop_best_left = xs_full[best_idx_left, :split0].clone()

        best_idx_right = torch.argmin(eval_right)
        if eval_right[best_idx_right] < self._pop_best_eval_right:
            self._pop_best_eval_right = eval_right[best_idx_right]
            self._pop_best_right = xs_full[best_idx_right, split0:].clone()

        # --- Compute weights per block ---
        ranks_left = torch.argsort(eval_left)
        ranks_right = torch.argsort(eval_right)
        weights_left = self.cma_blocks[0].weights[ranks_left]
        weights_right = self.cma_blocks[1].weights[ranks_right]

        # --- Update each CMAES_cus block ---
        for i, cma in enumerate(self.cma_blocks):
            if i == 0:
                zs_block = zs_block0
                ys_block = ys_block0
                assigned_weights = weights_left
            else:
                zs_block = zs_block1
                ys_block = ys_block1
                assigned_weights = weights_right

            local_disp, shaped_disp = cma.update_m(zs_block, ys_block, assigned_weights)
            cma.update_p_sigma(local_disp)
            cma.update_sigma()
            h_sig = _h_sig(cma.p_sigma, cma.c_sigma, self._steps_count)
            cma.update_p_c(shaped_disp, h_sig)
            cma.update_C(zs_block, ys_block, assigned_weights, h_sig)

            if (self._steps_count + 1) % cma.decompose_C_freq == 0:
                cma.decompose_C()

        # --- Update global m ---
        self.m[:split0] = self.cma_blocks[0].m
        self.m[split0:] = self.cma_blocks[1].m

        self._steps_count += 1
