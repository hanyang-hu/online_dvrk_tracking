from diffcali.utils.ui_utils import get_reference_keypoints


if __name__ == "__main__":
    ref_img_path = "data/extra_set1/00075.jpg"

    keypoints = get_reference_keypoints(ref_img_path)
    print(keypoints)
