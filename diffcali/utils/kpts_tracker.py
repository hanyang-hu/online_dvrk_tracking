import torch
import numpy as np
from filterpy.kalman import KalmanFilter


def kpts_alignment_score(kpts_1, kpts_2):
    dist_1 = np.linalg.norm(kpts_1[0] - kpts_2[0]) + np.linalg.norm(kpts_1[1] - kpts_2[1])
    dist_2 = np.linalg.norm(kpts_1[0] - kpts_2[1]) + np.linalg.norm(kpts_1[1] - kpts_2[0])
    dist_c = np.linalg.norm((kpts_1[0] + kpts_1[1]) / 2 - (kpts_2[0] + kpts_2[1]) / 2)
    return min(dist_1, dist_2) + dist_c


class KeypointsTracker:
    def __init__(
        self, var=1e0, P0_pos=1e1, P0_vel=5e2, R0=1e1, 
        validation_gating=False,
        chi2_threshold=11.34, rej_tol=5, fast_alpha=0.3, slow_alpha=0.05, 
        tau=0., hard_rej_threshold=50. # tau=1.5, hard_rej_threshold=30.
    ):
        """
        A simple Kalman Filter based tracker for 3D keypoints.
        
        Args:
            var (float): Process noise variance.
            P0_pos (float): Initial position uncertainty.
            P0_vel (float): Initial velocity uncertainty.
            R0 (float): Measurement noise variance.
        """
        # State vector: [kpt1_x, kpt1_y, kpt1_vx, kpt1_vy, kpt2_x, kpt2_y, kpt2_vx, kpt2_vy]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # Constant velocity model
        self.kf.F = np.array(
            [
                [1, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
            ]
        )

        self.var = var
        self.P0_pos = P0_pos
        self.P0_vel = P0_vel
        self.R0 = R0

        self.validation_gating = validation_gating

        # Parameters for STA/LTA using exponential moving averages
        self.fast_alpha = fast_alpha
        self.slow_alpha = slow_alpha
        self.tau = tau
        self.hard_rej_threshold = hard_rej_threshold
        self.lta, self.sta = 10., 10.
        self.prev_kpts = None
        self.frame_cnt = 0

        self.chi2_threshold = chi2_threshold
        self.rej_tol = rej_tol
        self.rej_count = 0
        self.initialized = False
    
    def initialize(self, keypoints):
        keypoints_np = keypoints.cpu().numpy().reshape(-1) # [kpt1_x, kpt1_y, kpt2_x, kpt2_y]
        self.kf.x = np.array([keypoints_np[0], keypoints_np[1], 0, 0, keypoints_np[2], keypoints_np[3], 0, 0]) # Initial state

        self.kf.Q = np.eye(8) * self.var  # Process noise
        self.kf.P = np.diag([self.P0_pos, self.P0_pos, self.P0_vel, self.P0_vel, self.P0_pos, self.P0_pos, self.P0_vel, self.P0_vel]) # Initial uncertainty
        self.kf.R = np.eye(4) * self.R0 # Measurement noise

        self.lta, self.sta = 10., 10. # reset STA/LTA
        self.prev_kpts = None

        self.rej_count = 0

        self.initialized = True

    def preprocess(self, keypoints):
        """
        Given N detected keypoints, pick the pair (i,j) that maximizes
        the prior probability (i.e., minimizes the Mahalanobis distance)
        with respect to the predicted measurement from the Kalman filter.
        """

        self.initialized = False
        self.frame_cnt += 1

        keypoints_np = keypoints.cpu().numpy().reshape(-1)
        N = keypoints_np.shape[0] // 2
        kpts_obs = keypoints_np.reshape(N, 2)  # shape (N, 2)

        # Kalman filter prediction
        self.kf.predict()

        # Select the best keypoint pair
        if self.validation_gating:
            # Predicted measurement mean and covariance
            z_pred = self.kf.H @ self.kf.x
            S = self.kf.H @ self.kf.P @ self.kf.H.T + self.kf.R
            S_inv = np.linalg.inv(S)

            best_pair = None
            best_maha = np.inf
            best_z = None

            # Evaluate all ordered pairs (i ≠ j)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue

                    # Observation vector consistent with H order
                    z_try = np.array([
                        kpts_obs[i, 0],
                        kpts_obs[i, 1],
                        kpts_obs[j, 0],
                        kpts_obs[j, 1]
                    ], dtype=float)

                    y = z_try - z_pred
                    maha = float(y.T @ S_inv @ y)

                    if maha < best_maha:
                        best_maha = maha
                        best_pair = (i, j)
                        best_z = z_try

            # Use best pair
            z = best_z
            chi2 = best_maha

            ret = z if chi2 < self.chi2_threshold else None
        else:
            if self.prev_kpts is not None:
                best_pair = None
                best_score = np.inf
                best_z = None   

                # Evaluate all ordered pairs (i ≠ j)
                for i in range(N):
                    for j in range(N):
                        if i == j:
                            continue

                        # Observation vector consistent with H order
                        z_try = np.array([
                            kpts_obs[i, 0],
                            kpts_obs[i, 1],
                            kpts_obs[j, 0],
                            kpts_obs[j, 1]
                        ], dtype=float).reshape(2, 2)

                        score = kpts_alignment_score(z_try, self.prev_kpts)

                        if score < best_score:
                            best_score = score
                            best_pair = (i, j)
                            best_z = z_try

                ret = z = best_z.reshape(-1)
            else:
                # Just take the first two keypoints if no previous
                ret = z = np.array([
                    kpts_obs[0, 0],
                    kpts_obs[0, 1],
                    kpts_obs[1, 0],
                    kpts_obs[1, 1]
                ], dtype=float)

        # Update STA/LTA and perform additional rejection based on alignment score
        if self.prev_kpts is not None:
            score = kpts_alignment_score(z.reshape(2, 2), self.prev_kpts)
            self.sta = self.fast_alpha * score + (1 - self.fast_alpha) * self.sta
            self.lta = self.slow_alpha * score + (1 - self.slow_alpha) * self.lta

            ratio = (self.sta + 1e-9) / (self.lta + 1e-9)
            if ratio > self.tau and score > self.hard_rej_threshold:
                print(f"--Reject at frame {self.frame_cnt}: alignment score = {score:.4f}, STA = {self.sta:.4f}, LTA = {self.lta:.4f}, ratio = {ratio:.4f}")
                ret = None
            else:
                print(f"Accepted at frame {self.frame_cnt}: alignment score = {score:.4f}, STA = {self.sta:.4f}, LTA = {self.lta:.4f}, ratio = {ratio:.4f}")
                # self.prev_kpts = z.reshape(2, 2) # Update previous keypoints only if accepted

        # self.prev_kpts = z.reshape(2, 2) # Initialize previous keypoints

        if ret is None:
            self.rej_count += 1
        else:
            self.rej_count = 0

        if self.rej_count >= self.rej_tol:
            # Re-initialize the filter if too many consecutive rejections
            self.initialize(torch.from_numpy(z).float().view(2, 2).to(keypoints.device))
            # ret = z # Uncomment to use the current observation as the new reference, otherwise do not use ref_kpts for this frame

        return torch.from_numpy(ret).float().view(2, 2).to(keypoints.device) if ret is not None else None

    def postprocess(self, z, R_factor=None):# Update the measurements
        if R_factor is not None:
            self.kf.R *= R_factor # Adjust measurement noise if specified

        if not self.initialized:
            self.prev_kpts = z.cpu().numpy().reshape(2, 2) # Update previous keypoints by current OpenCV detection (if accepted) or optimized projections

        self.kf.update(z.cpu().numpy().reshape(-1) if z is not None else z)

        if R_factor is not None:
            self.kf.R /= R_factor # Restore original measurement noise

        # Return the filtered keypoints
        filtered_kpts = self.kf.x[[0, 1, 4, 5]]
        filtered_kpts_tensor = torch.from_numpy(filtered_kpts).float().view(2, 2).to(z.device)

        # self.prev_kpts = filtered_kpts_tensor.cpu().numpy().reshape(2, 2) # Update previous keypoints with filtered result

        return filtered_kpts_tensor

    def track(self, keypoints):
        z = self.preprocess(keypoints) 
        return self.postprocess(z) # update with measurements only if it has passed the validation gate

