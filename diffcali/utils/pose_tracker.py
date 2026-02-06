import numpy as np
import torch

from filterpy.kalman import KalmanFilter as FP_KalmanFilter


class OneEuroFilter:
    """
    One Euro Filter based pose tracker.
    """
    def __init__(
        self, 
        f_min=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), # minimum cutoff frequency for each dimension (the larger, the less smooth)
        alpha_d=0.3, # smoothing factor for the derivative
        beta=0.3, # speed coefficient
        dt=1/20,
        kappa=np.array([1.0 for _ in range(10)]) * 0.9, # more damping for the jaw angles
        joint_angles_lb=np.array([-1.5707, -1.3963, 0.,       0.]),
        joint_angles_ub=np.array([ 1.5707,  1.3963, 1.5707, 1.5707]),
        correction=True,
    ):
        self.f_min = f_min
        self.alpha_d = alpha_d
        self.beta = beta
        self.dt = dt

        self.kappa = kappa # factor for velocity correction

        self.joint_angles_lb = joint_angles_lb
        self.joint_angles_ub = joint_angles_ub
        
        self.correction = correction

    def reset(self, x_init):
        self.x_scaled_hat = x_init
        self.dx_hat = 0.

    def update(self, x):
        assert self.x_scaled_hat is not None

        x_scaled = x

        # Compute velocity and the smoothed velocity estimate
        dx = (x_scaled - self.x_scaled_hat) / self.dt
        self.dx_hat = self.alpha_d * dx + (1 - self.alpha_d) * self.dx_hat

        # Compute adaptive alpha
        fc = self.f_min + self.beta * np.abs(self.dx_hat)
        alpha = 1.0 / (1.0 + 1.0 / (2 * np.pi * fc * self.dt))

        # Compute smoothed state
        if self.correction:
            x_scaled_hat = alpha * x_scaled + (1 - alpha) * self.x_scaled_hat + (1 - alpha) * self.dx_hat * self.dt * self.kappa
        else:
            x_scaled_hat = alpha * x_scaled + (1 - alpha) * self.x_scaled_hat # original one euro filter
        x_hat = x_scaled_hat
        x_hat[-4:] = np.clip(x_hat[-4:], self.joint_angles_lb, self.joint_angles_ub)
        self.x_scaled_hat = x_hat

    def get_x_hat(self):
        return self.x_scaled_hat


class KalmanFilter:
    """
    FilterPy-based Kalman filter pose tracker.
    Same interface as OneEuroFilter.
    """
    def __init__(
        self,
        dt=1/20,
        process_noise_pos=1e-4,      # scalar or (D,)
        process_noise_vel=1e-3,      # scalar or (D,)
        measurement_noise=1e-2,      # scalar or (D,)
        joint_angles_lb=np.array([-1.5707, -1.3963, 0.,       0.]),
        joint_angles_ub=np.array([ 1.5707,  1.3963, 1.5707, 1.5707]),
        correction=True,
    ):
        self.dt = dt
        self.process_noise_pos = process_noise_pos
        self.process_noise_vel = process_noise_vel
        self.measurement_noise = measurement_noise

        self.joint_angles_lb = joint_angles_lb
        self.joint_angles_ub = joint_angles_ub
        self.correction = correction

        self.kf = None
        self.D = None

    def _diag(self, value, D):
        """Convert scalar or (D,) to diagonal matrix."""
        if np.isscalar(value):
            return np.eye(D) * value
        value = np.asarray(value)
        assert value.shape == (D,)
        return np.diag(value)

    def reset(self, x_init):
        x_init = np.asarray(x_init)
        self.D = x_init.shape[0]

        # FilterPy KalmanFilter
        self.kf = FP_KalmanFilter(dim_x=2 * self.D, dim_z=self.D)

        # Initial state
        self.kf.x = np.zeros(2 * self.D)
        self.kf.x[:self.D] = x_init

        # State transition
        self.kf.F = np.eye(2 * self.D)
        self.kf.F[:self.D, self.D:] = np.eye(self.D) * self.dt

        # Measurement model
        self.kf.H = np.zeros((self.D, 2 * self.D))
        self.kf.H[:, :self.D] = np.eye(self.D)

        # Covariances
        self.kf.P = np.eye(2 * self.D) * 1e-2

        Qp = self._diag(self.process_noise_pos, self.D)
        Qv = self._diag(self.process_noise_vel, self.D)

        self.kf.Q = np.zeros((2 * self.D, 2 * self.D))
        self.kf.Q[:self.D, :self.D] = Qp
        self.kf.Q[self.D:, self.D:] = Qv

        self.kf.R = self._diag(self.measurement_noise, self.D)

    def update(self, x):
        assert self.kf is not None

        x = np.asarray(x)

        # --- Predict ---
        self.kf.predict()

        # --- Update ---
        if self.correction:
            self.kf.update(x)

        # Joint-angle clipping (same semantics as OneEuro)
        self.kf.x[:self.D][-4:] = np.clip(
            self.kf.x[:self.D][-4:],
            self.joint_angles_lb,
            self.joint_angles_ub,
        )

        # # Print covariance diagonal for debugging
        # print("Covariance diag:", np.diag(self.kf.P))

    def get_x_hat(self):
        return self.kf.x[:self.D]