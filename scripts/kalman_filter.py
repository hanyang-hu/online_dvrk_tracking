from pykalman import KalmanFilter
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

measurements = [[ 0.4894, -0.2709,  0.1832, -0.8084,  0.0088,  0.0050,  0.1349, -0.2965,
         -0.0829,  0.5724,  0.5432],
        [ 0.4374, -0.2452,  0.1346, -0.8546,  0.0108,  0.0057,  0.1424, -0.3331,
         -0.1170,  0.5713,  0.5623],
        [ 0.3954, -0.2471,  0.1188, -0.8766,  0.0115,  0.0062,  0.1460, -0.3600,
         -0.1596,  0.5417,  0.5942],
        [ 0.3653, -0.2662,  0.1174, -0.8843,  0.0108,  0.0066,  0.1424, -0.3383,
         -0.2365,  0.5035,  0.6314],
        [ 0.2968, -0.2791,  0.0924, -0.9086,  0.0110,  0.0072,  0.1414, -0.3380,
         -0.0918,  0.6072,  0.5030],
        [ 0.2469, -0.2970,  0.0793, -0.9190,  0.0111,  0.0078,  0.1402, -0.3097,
         -0.1003,  0.5849,  0.5205],
        [ 0.2045, -0.2976,  0.0648, -0.9303,  0.0106,  0.0080,  0.1376, -0.3707,
         -0.1477,  0.5916,  0.5195],
        [ 0.1377, -0.2928,  0.0405, -0.9454,  0.0114,  0.0083,  0.1379, -0.3684,
         -0.0856,  0.5467,  0.5132],
        [ 0.1183, -0.2925,  0.0312, -0.9484,  0.0119,  0.0087,  0.1397, -0.4051,
         -0.1396,  0.5613,  0.5175],
        [ 0.0507, -0.3099,  0.0081, -0.9494,  0.0126,  0.0092,  0.1384, -0.3415,
         -0.1284,  0.5357,  0.5414],
        [ 0.0388, -0.3049,  0.0017, -0.9516,  0.0133,  0.0096,  0.1400, -0.3751,
         -0.1446,  0.5862,  0.5067],
        [ 0.0296,  0.3105,  0.0207,  0.9499,  0.0130,  0.0097,  0.1379, -0.4311,
         -0.1393,  0.5109,  0.5529],
        [ 0.0689,  0.3210,  0.0357,  0.9439,  0.0134,  0.0099,  0.1365, -0.4181,
         -0.1192,  0.5373,  0.5088],
        [ 0.1169,  0.3356,  0.0539,  0.9332,  0.0142,  0.0099,  0.1350, -0.3799,
         -0.1392,  0.5316,  0.5209],
        [ 0.1827,  0.3268,  0.0788,  0.9239,  0.0146,  0.0100,  0.1322, -0.3939,
         -0.1294,  0.5271,  0.5329],
        [ 0.2398,  0.3397,  0.1034,  0.9035,  0.0164,  0.0101,  0.1349, -0.3442,
         -0.1414,  0.5610,  0.5246],
        [ 0.3050,  0.3271,  0.1280,  0.8852,  0.0161,  0.0099,  0.1313, -0.3877,
         -0.1444,  0.4994,  0.5618],
        [ 0.3588,  0.3302,  0.1502,  0.8601,  0.0172,  0.0098,  0.1324, -0.3673,
         -0.1184,  0.5559,  0.5119],
        [ 0.4052,  0.3182,  0.1639,  0.8412,  0.0185,  0.0095,  0.1323, -0.3461,
         -0.1757,  0.5325,  0.5626],
        [ 0.4544,  0.3014,  0.1785,  0.8191,  0.0205,  0.0094,  0.1360, -0.3576,
         -0.1938,  0.5302,  0.5863]]

# Resolve sign ambiguity
measurements = np.array(measurements)
init_q = measurements[0, :4]
for i in range(len(measurements) - 1):
    q = measurements[i + 1, :4]
    if np.dot(init_q[1:], q[1:]) < 0:
        measurements[i + 1, :4] = -1 * measurements[i + 1, :4] # Flip the quaternion if the sign is different
    init_q = measurements[i + 1, :4]  # Update the initial quaternion for the next iteration

# Define the Kalman Filter (first-order model)
state_dim = 22
obs_dim = 11
dt = 1.0

F = np.eye(state_dim)
F[:11, 11:] = dt * np.eye(11)
Q = 1e-3 * np.eye(state_dim)
H = np.zeros((obs_dim, state_dim))
H[:, :obs_dim] = np.eye(obs_dim)
R = 1e-3 * np.eye(obs_dim) 

initial_state_mean = np.array(measurements.tolist()[0] + [0.0] * (state_dim - obs_dim))
initial_state_covariance = 1e-2 * np.eye(state_dim)

kf = KalmanFilter(
    transition_matrices=F,
    observation_matrices=H,
    transition_covariance=Q,
    observation_covariance=R,
    initial_state_mean=initial_state_mean,
    initial_state_covariance=initial_state_covariance,
    em_vars=['transition_covariance', 'observation_covariance']
)

# Check the filtering result
observations = measurements[1:]
n_timesteps = len(observations) + 1
filtered_state_means = np.zeros((n_timesteps, state_dim))
filtered_state_covariances = np.zeros((n_timesteps, state_dim, state_dim))
predicted_state_means = np.zeros((n_timesteps-1, state_dim))

start_time = time.time()

for t in range(n_timesteps - 1):
    if t == 0:
        filtered_state_means[t] = initial_state_mean
        filtered_state_covariances[t] = initial_state_covariance
    predicted_state_means[t] = kf.transition_matrices @ filtered_state_means[t]
    filtered_state_means[t + 1], filtered_state_covariances[t + 1] = (
        kf.filter_update(
            filtered_state_means[t],
            filtered_state_covariances[t],
            observations[t]
        )
    )
    # if t > 0:
    #     kf.em(observations[:t + 1], n_iter=1)  # EM step to update transition and observation covariance
    # print(kf.transition_covariance, kf.observation_covariance)

end_time = time.time()
print(f"Kalman Filter processing time: {(end_time - start_time)*1000:.2f} ms")

# Plot the results
idx = 6
# state_stds = np.sqrt([cov[idx, idx] for cov in filtered_state_covariances[1:]])
# upper_bound = predicted_state_means[1:, idx] + 0.1 * state_stds
# lower_bound = predicted_state_means[1:, idx] - 0.1 * state_stds
shifted_obs = measurements[0:n_timesteps - 1,:]
alpha = 1.
predicted_obs = alpha * predicted_state_means[:,:obs_dim] + (1 - alpha) * shifted_obs

plt.figure(figsize=(12, 6))
plt.plot(observations[:, idx], label='Observed X', marker='o')
plt.plot(predicted_obs[:,idx], label='Predicted X', linestyle='--')
plt.plot(measurements[:, idx], label='Shifted X', marker='x')
# plt.fill_between(
#     np.arange(n_timesteps - 1), lower_bound, upper_bound,
#     color='orange', alpha=0.3, label='Confidence Interval'
# )
plt.title('Kalman Filter: Observed vs Predicted State')
plt.xlabel('Time Step')
plt.ylabel('State Value')
plt.legend()
plt.grid()
plt.show()

# Calculate error of shift (use the previous timestep directly) and kalman filter
shift_error = np.linalg.norm(observations[:] - shifted_obs, axis=1).mean()
print(f"Shift Error: {shift_error}")
predict_error = np.linalg.norm(observations - predicted_obs, axis=1).mean()
print(f"Prediction Error: {predict_error}")
