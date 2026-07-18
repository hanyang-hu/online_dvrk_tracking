import yaml

from gui_live_tracking.config import LiveTrackingConfig
from gui_live_tracking.path_utils import validate_config


def make_common_files(tmp_path):
    cam = tmp_path / "camera.yaml"
    handeye = tmp_path / "handeye.yaml"
    lnd = tmp_path / "LND.json"
    weights = tmp_path / "weights.pth"
    mesh_dir = tmp_path / "meshes"
    cam.write_text("fx: 1\nfy: 1\npx: 0\npy: 0\n", encoding="utf-8")
    with open(handeye, "w", encoding="utf-8") as f:
        yaml.safe_dump({"PSM1_tvec": [0, 0, 0], "PSM1_rvec": [0, 0, 0]}, f)
    lnd.write_text("{}", encoding="utf-8")
    weights.write_text("x", encoding="utf-8")
    mesh_dir.mkdir()
    return cam, handeye, lnd, weights, mesh_dir


def test_ros2_mode_does_not_require_video_or_joint_yaml(tmp_path):
    cam, handeye, lnd, weights, mesh_dir = make_common_files(tmp_path)
    cfg = LiveTrackingConfig(
        input_mode="ros2",
        video_path=tmp_path / "missing.mp4",
        joint_angles_path=tmp_path / "missing.yaml",
        camera_calibration_path=cam,
        handeye_path=handeye,
        lnd_json_path=lnd,
        contour_tip_net_path=weights,
        mesh_dir=mesh_dir,
    )

    assert validate_config(cfg) == []


def test_offline_mode_requires_video_and_joint_yaml(tmp_path):
    cam, handeye, lnd, weights, mesh_dir = make_common_files(tmp_path)
    cfg = LiveTrackingConfig(
        input_mode="offline",
        video_path=tmp_path / "missing.mp4",
        joint_angles_path=tmp_path / "missing.yaml",
        camera_calibration_path=cam,
        handeye_path=handeye,
        lnd_json_path=lnd,
        contour_tip_net_path=weights,
        mesh_dir=mesh_dir,
    )

    errors = validate_config(cfg)
    assert any("Video does not exist" in err for err in errors)
    assert any("Joint angles yaml does not exist" in err for err in errors)


def test_ros2_topic_and_rate_validation(tmp_path):
    cam, handeye, lnd, weights, mesh_dir = make_common_files(tmp_path)
    cfg = LiveTrackingConfig(
        input_mode="ros2",
        camera_calibration_path=cam,
        handeye_path=handeye,
        lnd_json_path=lnd,
        contour_tip_net_path=weights,
        mesh_dir=mesh_dir,
        ros_image_topic="",
        ros_sync_queue_size=0,
        ros_sync_slop_sec=-1.0,
        mock_rate_hz=0,
        sample_timeout_sec=0,
    )

    errors = validate_config(cfg)
    assert any("ROS image topic" in err for err in errors)
    assert any("queue size" in err for err in errors)
    assert any("slop" in err for err in errors)
    assert any("Mock replay rate" in err for err in errors)
    assert any("Sample timeout" in err for err in errors)


def test_bridge_endpoint_configuration_removed():
    cfg = LiveTrackingConfig()

    assert not any(name.startswith("bridge_") for name in vars(cfg))


def test_turbo_handeye_init_validates_batch_settings(tmp_path):
    cam, handeye, lnd, weights, mesh_dir = make_common_files(tmp_path)
    cfg = LiveTrackingConfig(
        input_mode="ros2",
        camera_calibration_path=cam,
        handeye_path=handeye,
        lnd_json_path=lnd,
        contour_tip_net_path=weights,
        mesh_dir=mesh_dir,
        use_turbo_handeye_init=True,
        batch_size=50,
        sample_number=75,
        batch_iters=0,
    )

    errors = validate_config(cfg)
    assert any("sample number must be divisible" in err for err in errors)
    assert any("batch iterations" in err for err in errors)
