import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gpytorch
import torch
import time
import tqdm

from diffcali.utils.pathwise_conditioning import PathwiseSampler


pt_dir = "./scripts/gp_data.pt"


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.keops.RBFKernel(
                ard_num_dims=train_x.shape[-1],
                # mu=5/2,
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


if __name__ == "__main__":
    data = torch.load(pt_dir)

    N = 700
    train_size = int(0.8 * N)

    # Randomly permute the data
    i = 0
    perm = torch.randperm(N)
    data["values"] = data["values"][i*N:(i+1)*N][perm]
    data["losses"] = data["losses"][i*N:(i+1)*N][perm]

    train_x = data["values"][:train_size].cuda()
    train_y = data["losses"][:train_size].cuda()
    test_x = data["values"][train_size:].cuda()
    test_y = data["losses"][train_size:].cuda()

    # normalize features
    mean_x = train_x.mean(dim=-2, keepdim=True)
    std_x = train_x.std(dim=-2, keepdim=True) + 1e-6 # prevent dividing by 0
    # train_x = (train_x - mean_x) / std_x
    # test_x = (test_x - mean_x) / std_x

    # # normalize labels
    mean_y, std_y = train_y.mean(),train_y.std()
    # train_y = (train_y - mean_y) / std_y
    # test_y = (test_y - mean_y) / std_y

    # print(train_y)

    # make continguous
    train_x, train_y = train_x.contiguous(), train_y.contiguous()
    test_x, test_y = test_x.contiguous(), test_y.contiguous()

    output_device = torch.device('cuda:0')

    train_x, train_y = train_x.to(output_device), train_y.to(output_device)
    test_x, test_y = test_x.to(output_device), test_y.to(output_device)

    print(
        f"Num train: {train_y.size(-1)}\n"
        f"Num test: {test_y.size(-1)}"
    )

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    model = ExactGPModel(train_x, train_y, likelihood).cuda()

    # Because we know some properties about this dataset,
    # we will initialize the lengthscale to be somewhat small
    # This step isn't necessary, but it will help the model converge faster.
    model.covar_module.base_kernel.lengthscale = 1.
    model.likelihood.noise = 1e-3
    # model.mean_module.constant = torch.Tensor([mean_y]).cuda()
    # model.covar_module.outputscale = 1 / std_xs

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    training_iter = 100
    iterator = tqdm.tqdm(range(training_iter), desc="Training")
    for i in iterator:
        start_time = time.time()
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        print_values = dict(
            loss=loss.item(),
            ls=model.covar_module.base_kernel.lengthscale.norm().item(),
            os=model.covar_module.outputscale.item(),
            noise=model.likelihood.noise.item(),
            mu=model.mean_module.constant.item(),
        )
        iterator.set_postfix(**print_values)
        loss.backward()
        optimizer.step()

    print("Learned hyperparameters: ")
    print(f"Lengthscale: {model.covar_module.base_kernel.lengthscale}")
    print(f"Outputscale: {model.covar_module.outputscale.item()}")
    print(f"Noise: {model.likelihood.noise.item()}")
    print(f"Mean: {model.mean_module.constant.item()}")

    model.covar_module.base_kernel.lengthscale = torch.Tensor([[5.,  1., 5.,  2., 2., 2., 10., 10., 10., 10.]]).cuda()  # [1, 10]
    model.likelihood.noise = 1e-4 + 1e-9
    # model.mean_module.constant = torch.Tensor([0.0]).cuda()

    # Get into evaluation (predictive posterior) mode
    model.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = model.likelihood(model(test_x))
        
    rmse = (observed_pred.mean - test_y).square().mean().sqrt().item()
    print(f"RMSE: {rmse:.3f}")
    print("Data: \n    ", test_y)
    # print("GP Prediction: ", observed_pred.mean)

    # # put the predictions back to the original scale
    # pred_mean = observed_pred.mean * std_y + mean_y
    # test_y = test_y * std_y + mean_y

    # rmse = (pred_mean - test_y).square().mean().sqrt().item()
    # print(f"RMSE (original scale): {rmse:.3f}")

    # test the result of pathwise conditioning
    sampler = PathwiseSampler(
        # mean=model.mean_module.constant.item(),
    )

    import time

    sync = torch.cuda.synchronize

    sampler.update_data(train_x, train_y)
    sampler.draw_posterior_samples()

    sync()
    start_time = time.time()
    sampler.draw_posterior_samples()
    sync()
    end_time = time.time()

    print(f"Pathwise conditioning took {end_time - start_time:.3f} seconds")

    # Evaluate the sampler on test points
    batch_preds = sampler.evaluate(test_x) # shape [sample_size, n]
    mean_preds = batch_preds.mean(dim=0)  # shape [n]

    rmse = (mean_preds - test_y).square().mean().sqrt().item()
    print(f"RMSE (pathwise conditioning): {rmse:.3f}")

    print("Pathwise Conditioning: \n    ", mean_preds)

