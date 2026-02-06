import torch
import numpy as np
from pykalman import KalmanFilter
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DC_args = False  # Diagonalize covariances during EM

from diffcali.utils.angle_transform_utils import axis_angle_to_mix_angle

def make_cv_matrices(state_dim, dt, process_pos_std, process_vel_std, meas_std,
                     init_pos_std, init_vel_std, dtype=np.float64):
    # assert state_dim % 2 == 0
    pos_dim = state_dim

    # Transition matrix F (state_dim x state_dim)
    F = np.eye(state_dim, dtype=dtype)

    # Observation matrix H (pos_dim x state_dim) -- observe only position
    H = np.eye(state_dim, dtype=dtype)

    # Process covariance Q (state_dim x state_dim) -- initialize diagonal
    Q = np.eye(state_dim, dtype=dtype)

    # Observation covariance R (pos_dim x pos_dim)
    R = np.eye(pos_dim, dtype=dtype) * (meas_std ** 2)

    # Initial state covariance P0 (state_dim x state_dim)
    P0 = np.eye(state_dim, dtype=dtype)
    
    # Initial state mean (zero)
    x0 = np.zeros(state_dim, dtype=dtype)

    return F, H, Q, R, x0, P0

import numpy as np
from pykalman import KalmanFilter

def fit_pykalman_em_for_trajectory(obs, state_dim=10, dt=1.0,
                                   process_pos_std=20.0, process_vel_std=30.0, meas_std=5.0,
                                   init_pos_std=10.0, init_vel_std=500.0,
                                   em_iters=20, em_vars=('transition_covariance',
                                                        'observation_covariance',
                                                        'initial_state_covariance'),
                                   diagonalize_covariances=DC_args):

    obs = np.asarray(obs, dtype=np.float64)
    T, pos_dim = obs.shape
    assert state_dim == pos_dim

    F, H, Q_init, R_init, x0_init, P0_init = make_cv_matrices(
        state_dim, dt, process_pos_std, process_vel_std,
        meas_std, init_pos_std, init_vel_std
    )

    x0 = np.zeros(state_dim, dtype=np.float64)
    x0[:pos_dim] = obs[0]        # first frame's positions
    x0[pos_dim:] = 0.0           # zero initial velocity

    kf = KalmanFilter(
        transition_matrices=F,
        observation_matrices=H,
        transition_covariance=Q_init,
        observation_covariance=R_init,
        initial_state_mean=x0,
        initial_state_covariance=P0_init
    )

    # --- Custom EM loop enforcing diagonal covariances ---
    kf_em = kf
    for _ in range(em_iters):
        kf_em = kf_em.em(obs[1:], n_iter=1, em_vars=em_vars)

        if diagonalize_covariances:
            if 'transition_covariance' in em_vars:
                kf_em.transition_covariance = np.diag(np.diag(kf_em.transition_covariance))
            if 'observation_covariance' in em_vars:
                kf_em.observation_covariance = np.diag(np.diag(kf_em.observation_covariance))
            if 'initial_state_covariance' in em_vars:
                kf_em.initial_state_covariance = np.diag(np.diag(kf_em.initial_state_covariance))

    Q_learned = np.array(kf_em.transition_covariance, dtype=np.float64)
    R_learned = np.array(kf_em.observation_covariance, dtype=np.float64)
    P0_learned = np.array(kf_em.initial_state_covariance, dtype=np.float64)
    x0_learned = np.array(kf_em.initial_state_mean, dtype=np.float64)

    smoothed_means, smoothed_covs = kf_em.smooth(obs)

    return {
        'kf_em': kf_em,
        'smoothed_means': smoothed_means,
        'smoothed_covs': smoothed_covs,
        'Q': Q_learned,
        'R': R_learned,
        'P0': P0_learned,
        'x0': x0_learned
    }


def fit_all_trajectories_with_em(data_list, state_dim=10, dt=1.0,
                                 process_pos_std=20.0, process_vel_std=30.0, meas_std=5.0,
                                 init_pos_std=10.0, init_vel_std=500.0,
                                 em_iters=20):
    """
    data_list: list of numpy arrays, each shape (T, pos_dim). (This is your data_lst)
    Returns list of results (one dict per trajectory).
    """
    results = []
    for i, traj in enumerate(data_list):
        obs = np.asarray(traj, dtype=np.float64)
        # If your stored data contains more than pos_dim columns (e.g. full pose),
        # extract only the position dims that correspond to pos_dim:
        # here we assume obs already is (T, pos_dim).
        print(f"Fitting trajectory {i+1}/{len(data_list)} (T={obs.shape[0]}, dim={obs.shape[1]}) ...")
        res = fit_pykalman_em_for_trajectory(
            obs,
            state_dim=state_dim, dt=dt,
            process_pos_std=process_pos_std, process_vel_std=process_vel_std,
            meas_std=meas_std, init_pos_std=init_pos_std, init_vel_std=init_vel_std,
            em_iters=em_iters
        )
        results.append(res)
    return results


if __name__ == "__main__":
    idx_lst = [1, 5, 6, 7, 8, 9, 14, 15]

    data_lst = []
    for idx in idx_lst:
        file_path = f"./pose_results/rw{idx}_tracking_results.pth"
        data = torch.load(file_path)
        cTr, joint_angles = data['cTr'].cuda(), data['joint_angles'].cuda()
        axis_angles = cTr[:, :3]
        mix_angles = axis_angle_to_mix_angle(axis_angles)
        cTr[:, :3] = mix_angles
        data_lst.append(torch.cat([cTr, joint_angles], dim=1).detach().cpu().numpy())

    em_results = fit_all_trajectories_with_em(data_lst)

    # Compute full learned covariances across trajectories and save to ./data/kf_parameters/
    Qs = np.array([res['Q'] for res in em_results])   # (N, state_dim, state_dim)
    Rs = np.array([res['R'] for res in em_results])
    P0s = np.array([res['P0'] for res in em_results])

    mean_Q = np.mean(Qs, axis=0)
    mean_R = np.mean(Rs, axis=0)
    mean_P0 = np.mean(P0s, axis=0)

    std_Q = np.std(Qs, axis=0)
    std_R = np.std(Rs, axis=0)
    std_P0 = np.std(P0s, axis=0)

    if DC_args:
        print("Diagonalized covariances during EM fitting.")
        print("Mean Q diagonal:\n", np.diag(mean_Q))
        print("Mean R diagonal:\n", np.diag(mean_R))
        print("Mean P0 diagonal:\n", np.diag(mean_P0))
        print("")        
        print("Std Q diagonal:\n", np.diag(std_Q))
        print("Std R diagonal:\n", np.diag(std_R))
        print("Std P0 diagonal:\n", np.diag(std_P0))
    else:
        print("Mean Q:\n", mean_Q)
        print("Mean R:\n", mean_R)
        print("Mean P0:\n", mean_P0)
        print("")
        print("Std Q:\n", std_Q)
        print("Std R:\n", std_R)
        print("Std P0:\n", std_P0)

    out_dir = "./data/kf_parameters"
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "Q_mean.npy"), mean_Q)
    np.save(os.path.join(out_dir, "R_mean.npy"), mean_R)
    np.save(os.path.join(out_dir, "P0_mean.npy"), mean_P0)
