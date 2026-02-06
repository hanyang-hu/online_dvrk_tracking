
import os
import cv2
import glob
import numpy as np
from scipy.stats import chi2
import time


from datetime import datetime
from datetime import timedelta
from ordered_set import OrderedSet
import numpy as np
from scipy.stats import uniform

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.mfa import MFADataAssociator
from stonesoup.hypothesiser.mfa import MFAHypothesiser
from stonesoup.dataassociator.probability import JPDAwithLBP
from stonesoup.dataassociator.probability import JPDA
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.types.array import StateVectors
from stonesoup.functions import gm_reduce_single
from stonesoup.types.update import GaussianStateUpdate, GaussianMixtureUpdate
from stonesoup.types.state import TaggedWeightedGaussianState
from stonesoup.types.track import Track
from stonesoup.types.mixture import GaussianMixture
from stonesoup.types.numeric import Probability

import imageio


def get_reference_keypoints_auto(
        ref_img_path, num_keypoints=2, ref_img=None,
        quality_level=0.1, min_distance=5, block_size=15
    ):
    # Read data
    if ref_img_path is None and ref_img is None:
        raise ValueError("Either ref_img_path or ref_mask must be provided.")
    cv_img = ref_img if ref_img_path is None else cv2.imread(ref_img_path)

    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    ref_img = cv2.inRange(cv_img, np.ones(3) * 128, np.ones(3) * 255) 
    binary_mask = ref_img.astype(np.uint8)
    region_mask = (binary_mask > 0).astype(np.uint8)
    max_corners = num_keypoints        # Maximum number of corners to find
    
    corners = cv2.goodFeaturesToTrack(
        binary_mask,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
        mask=region_mask
    )

    output_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Draw the corners
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(output_image, (int(x), int(y)), radius=4, color=(0, 0, 255), thickness=-1)  # Red circles for corners

    ref_keypoints = corners
     
    return ref_keypoints.squeeze(1).tolist() 


if __name__ == "__main__":
    # Read the frames and relevant data from the data directory.
    import argparse

    parser = argparse.ArgumentParser(description="Multi-object Tracking Script")
    parser.add_argument('--task_name', type=str, default="rw15", help='Name of the task/data folder')
    args = parser.parse_args()
    task_name = args.task_name

    data_dir = os.path.join("./data/consecutive_prediction", task_name)
    
    frame_start = 0
    frame_end = len(os.listdir(data_dir))

    kpts_lst = []

    for i in range(frame_start, frame_end):
        frame_dir = os.path.join(data_dir, f"{i}")

        # Find the mask
        mask_lst = glob.glob(os.path.join(frame_dir, "*.png"))
        if len(mask_lst) == 0:
            raise ValueError(f"No mask found in {frame_dir}")
        if len(mask_lst) > 1:
            raise ValueError(f"Multiple masks found in {frame_dir}")

        mask_path = mask_lst[0]
        # frame = cv2.imread(frame_path)
        XXXX = mask_path.split("/")[-1].split(".")[0][1:]

        # Get mask filename
        ref_mask_path = os.path.join(frame_dir, "0" + XXXX + ".png")

        if i == frame_start:
            # Get the two keypoints for initialization
            keypoints = get_reference_keypoints_auto(ref_img_path=ref_mask_path)
        else:
            # Get more keypoints with lower quality
            keypoints = get_reference_keypoints_auto(ref_img_path=ref_mask_path, num_keypoints=4) 
        kpts_lst.append(keypoints)

    # Filter parameters
    var = 3e0 # process noise
    R0 = 3e1 # measurement noise
    P0_pos = 5e0 # initial position covariance
    P0_vel = 1e1 # initial velocity covariance

    # JPDA Filter parameters;
    prob_detect = 0.9 # probability of detection
    clutter_spatial_density = None # clutter density

    # # MFA Filter parameters
    # prob_detect = 0.9  # Prob. of detection
    # gate_level = 8  # Gate level
    # prob_gate = chi2.cdf(gate_level, 2)  # Prob. of gating, computed from gate level for hypothesiser
    # v_bounds = np.array([[-5, 30], [-5, 30]])  # Surveillance area bounds
    # lambdaV = 5  # Mean number of clutter points over the entire surveillance area
    # lambda_ = lambdaV/(np.prod(v_bounds[:, 0] - v_bounds[:, 1]))  # Clutter density per unit volume
    # slide_window = 3  # Slide window; used by MFA data associator

    # Use JPDA to track the keypoints (the initialization is obtained by the best two keypoints in the first frame)
    start_time = datetime.now().replace(microsecond=0)

    timesteps = [start_time]
    
    # Generate measurements.
    all_measurements = []

    # State space: [x, vx, y, vy]
    # Measurement space: [x1, y1]
    transition_model = CombinedLinearGaussianTransitionModel(
        [
            ConstantVelocity(var),
            ConstantVelocity(var),
        ]
    )
    measurement_model = LinearGaussian(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.eye(2) * R0, # measurement noise covariance
    )
    
    # We want to convert the keypoint detections into measurements for the JPDA tracker.
    for t_idx, kpts in enumerate(kpts_lst[1:], start=1):
        current_time = start_time + timedelta(seconds=t_idx)
        timesteps.append(current_time)

        measurement_set = set()
        for kp in kpts:
            measurement_set.add(
                Detection(
                    state_vector=np.array(kp),
                    timestamp=current_time,
                    measurement_model=measurement_model
                )
            )

        all_measurements.append(measurement_set)

    # Define the KF predictor and updator
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)

    hypothesiser = PDAHypothesiser(
        predictor=predictor,
        updater=updater,
        clutter_spatial_density=clutter_spatial_density,
        prob_detect=prob_detect
    )
    # data_associator = JPDA(hypothesiser=hypothesiser)

    # hypothesiser = PDAHypothesiser(predictor, updater, lambda_, prob_gate=prob_gate, prob_detect=prob_detect)
    # hypothesiser = MFAHypothesiser(hypothesiser)
    # data_associator = MFADataAssociator(hypothesiser, slide_window=slide_window)

    data_associator = JPDAwithLBP(hypothesiser=hypothesiser)

    # Defiine the prior for the two tracks based on initial frame
    P0 = np.diag([P0_pos, P0_vel, P0_pos, P0_vel])
    prior1 = GaussianState(
        state_vector= [
            [kpts_lst[0][0][0]], [0.], [kpts_lst[0][0][1]], [0.],
        ],
        covar=P0,
        timestamp=start_time,
    )
    prior2 = GaussianState(
        state_vector=[
            [kpts_lst[0][1][0]], [0.], [kpts_lst[0][1][1]], [0.],
        ],
        covar=P0,
        timestamp=start_time,
    )

    # prior1 = GaussianMixture(
    #     [
    #         TaggedWeightedGaussianState(
    #             state_vector= [
    #                 [kpts_lst[0][0][0]], [0.], [kpts_lst[0][0][1]], [0.],
    #             ],
    #             covar=P0,
    #             timestamp=start_time,
    #             weight=Probability(1), 
    #             tag=[]
    #         )
    #     ]
    # )
    # prior2 = GaussianMixture(
    #     [
    #         TaggedWeightedGaussianState(
    #             state_vector=[
    #                 [kpts_lst[0][1][0]], [0.], [kpts_lst[0][1][1]], [0.],
    #             ],
    #             covar=P0,
    #             timestamp=start_time,
    #             weight=Probability(1), 
    #             tag=[]
    #         )
    #     ]
    # )

    tracks = {Track([prior1]), Track([prior2])}

    # Run the JPDA tracker
    runtime = []
    for n, measurements in enumerate(all_measurements[1:], start=1):
        st = time.time()

        hypotheses = data_associator.associate(
            tracks,
            measurements,
            start_time + timedelta(seconds=n)
        )

        # Loop through each track, performing the association step
        for track in tracks:
            track_hypotheses = hypotheses[track]

            posterior_states = []
            posterior_state_weights = []
            for hypothesis in track_hypotheses:
                if not hypothesis:
                    posterior_states.append(hypothesis.prediction)
                else:
                    posterior_state = updater.update(hypothesis)
                    posterior_states.append(posterior_state)
                posterior_state_weights.append(hypothesis.probability)

            means = StateVectors([state.state_vector for state in posterior_states])
            covars = np.stack([state.covar for state in posterior_states], axis=2)
            weights = np.asarray(posterior_state_weights)

            # Reduce mixture of states to one posterior estimate Gaussian.
            post_mean, post_covar = gm_reduce_single(means, covars, weights)

            # Add a Gaussian state approximation to the track.
            track.append(
                GaussianStateUpdate(
                    post_mean, 
                    post_covar,
                    track_hypotheses,
                    track_hypotheses[0].measurement.timestamp
                )
            )

        # associations = data_associator.associate(tracks, measurements, start_time + timedelta(seconds=n))

        # for track, hypotheses in associations.items():
        #     components = []
        #     for hypothesis in hypotheses:
        #         if not hypothesis:
        #             components.append(hypothesis.prediction)
        #         else:
        #             update = updater.update(hypothesis)
        #             components.append(update)
        #     track.append(GaussianMixtureUpdate(components=components, hypothesis=hypotheses))

        et = time.time()
        runtime.append(et - st)

    print(f"Average runtime per frame: {np.mean(runtime):.4f} seconds")

    # Extract the filtered keypoints
    kpts_tracks = []
    for track in tracks:
        filtered_kpts = []
        for state in track:
            filtered_kpts.append(state.state_vector[[0, 2]].squeeze().tolist())
        kpts_tracks.append(filtered_kpts)
    
    # Plot the keypoints and detections, and save the results in "./kpts_tracking/", also save them as a GIF
    output_dir = "./kpts_tracking/"
    os.makedirs(output_dir, exist_ok=True)
    frames = []
    for t_idx in range(1, len(timesteps)):
        frame_dir = os.path.join(data_dir, f"{t_idx}")
        mask_lst = glob.glob(os.path.join(frame_dir, "*.png"))
        mask_path = mask_lst[0]
        cv_img = cv2.imread(mask_path)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # Add title on the left upper part
        bag_number = ''.join(filter(str.isdigit, task_name))
        title_text = f"Loopy Belief Propagation\nBag {bag_number}" if bag_number else f"Task {task_name}"
        # Draw multi-line title (cv2.putText doesn't support '\n' directly)
        x, y = 10, 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        color = (255, 255, 255)
        thickness = 2
        line_spacing = int(40 * scale)
        for i, line in enumerate(title_text.split('\n')):
            cv2.putText(
                cv_img,
                line,
                (x, y + i * line_spacing),
                font,
                scale,
                color,
                thickness,
                cv2.LINE_AA
            )

        # Plot the detections
        for kp in kpts_lst[t_idx]:
            cv2.circle(cv_img, (int(kp[0]), int(kp[1])), radius=4, color=(255, 0, 0), thickness=-1)  # red circles for detections

        # Plot the tracked keypoints: past frames as lines, current frame as green circles
        for track_kpts in kpts_tracks:
            # Draw lines for the previous 10 frames
            for i in range(max(0, t_idx - 10), t_idx):
                if i + 1 < len(track_kpts):
                    pt1 = tuple(map(int, track_kpts[i]))
                    pt2 = tuple(map(int, track_kpts[i + 1]))
                    cv2.line(cv_img, pt1, pt2, color=(0, 255, 0), thickness=2)  # green lines for trajectory
            # Draw green circle for the current frame
            if t_idx < len(track_kpts):
                kp = track_kpts[t_idx]
                cv2.circle(cv_img, (int(kp[0]), int(kp[1])), radius=4, color=(0, 255, 0), thickness=-1)  # green circle for tracked keypoint

        output_path = os.path.join(output_dir, f"tracked_{t_idx:04d}.png")
        cv2.imwrite(output_path, cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR))
        frames.append(cv_img)

    # Save as GIF in the current folder
    gif_path = f"./lbp_{task_name}.gif"
    frames_bgr = [frame for frame in frames]
    imageio.mimsave(gif_path, frames_bgr, duration=0.2)


"""
python scripts/jpda_tracking.py --task_name rw1
python scripts/jpda_tracking.py --task_name rw5
python scripts/jpda_tracking.py --task_name rw6
python scripts/jpda_tracking.py --task_name rw7
python scripts/jpda_tracking.py --task_name rw8
python scripts/jpda_tracking.py --task_name rw10
python scripts/jpda_tracking.py --task_name rw11
python scripts/jpda_tracking.py --task_name rw12
python scripts/jpda_tracking.py --task_name rw14
python scripts/jpda_tracking.py --task_name rw15
"""