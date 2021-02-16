from typing import Optional, List

import cv2
import numpy as np

from .detection import get_detection_mask


def detect_compute_sift(
    img: np.ndarray, mask: Optional[np.ndarray] = None, nfeatures: Optional[int] = 500
):
    """Detect and compute SIFT features. Optionally mask them.

    Args:
        img (np.ndarray): Image to compute features on.
        mask (Optional[np.ndarray]): Binary mask.
        nfeatures (Optional[int]): Max number of features.

    Returns:
        kp: List of keypoints
        des: List of keypoint desciptors
    """
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)
    kp, des = sift.detectAndCompute(img, None)
    if nfeatures > 0 and len(kp) > nfeatures:
        kp = kp[:nfeatures]
        des = des[:nfeatures]
    if mask is not None:
        masked_index = mask_keypoints(kp, mask)
        kp = np.array(kp)[masked_index].tolist()
        des = des[masked_index]
    return kp, des


def mask_keypoints(kp: List, mask: np.ndarray):
    """Mask keypoints

    Args:
        kp (List): List of keypoints.
        mask (np.ndarray): Binary mask.

    Returns:
        inner_index: List of valid indices
    """
    inner_index = np.ndarray([0], dtype=np.int32)
    for i in range(len(kp)):

        if isinstance(kp, np.ndarray):
            x, y = int(kp[i, 0]), int(kp[i, 1])
        else:
            x, y = int(kp[i].pt[0]), int(kp[i].pt[1])

        if mask[y, x] == 1:
            inner_index = np.append(inner_index, i)
    return inner_index


def compute_ransac_homography(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    reprojection_threshold: Optional[float] = 0.5,
):
    """Compute RANSAC-based homography

    Args:
        src_points (np.ndarray): Array of source points.
        dst_points (np.ndarray): Array of destination points.
        reprojection_threshold (Optional[float]): Max allowed reprojection threshold.

    Returns:
        index: RANSAC mask.
        homography: Homography matrix.

    """
    assert src_points.shape[0] == dst_points.shape[0]
    assert src_points.shape[0] >= 4
    ransac_mask = np.ndarray([len(src_points)])
    homography, ransac_mask = cv2.findHomography(
        srcPoints=src_points,
        dstPoints=dst_points,
        ransacReprojThreshold=reprojection_threshold,
        method=cv2.FM_RANSAC,
        mask=ransac_mask,
    )

    index = [i for i in range(len(ransac_mask)) if ransac_mask[i] == 1]

    return index, homography


def detection_filtered_homography(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    src_det: Optional[dict] = None,
    dst_det: Optional[dict] = None,
    nfeatures: Optional[int] = 500,
    ratio: Optional[float] = 0.85,
):
    """Short summary.

    Args:
        src_img (np.ndarray): Source image.
        dst_img (np.ndarray): Destination image.
        src_det (Optional[dict]): Source detections.
        dst_det (Optional[dict]): Destination detections.
        nfeatures (Optional[int]): Max number of features.
        ratio (Optional[float]): Lowe's ratio, used for filtering keypoints.

    Returns:
        homography: Homography matrix.
        src_kp: Array of src keypoints.
        dst_kp: Array of dst keypoints.
        good_matches: List of good matches.
    """
    src_mask = get_detection_mask(src_det, src_img.shape) if src_det else None
    dst_mask = get_detection_mask(dst_det, dst_img.shape) if dst_det else None

    # SIFT + KNN Matching
    src_kp, src_des = detect_compute_sift(src_img, src_mask, nfeatures)
    dst_kp, dst_des = detect_compute_sift(dst_img, dst_mask, nfeatures)

    raw_matches = cv2.BFMatcher().knnMatch(src_des, dst_des, k=2)
    good_points = []
    good_matches = []
    for m1, m2 in raw_matches:
        if m1.distance < ratio * m2.distance:
            good_points.append((m1.trainIdx, m1.queryIdx))
            good_matches.append(m1)
    _src_kp = np.float32([src_kp[i].pt for (_, i) in good_points])
    _dst_kp = np.float32([dst_kp[i].pt for (i, _) in good_points])

    ransac_mask, homography = compute_ransac_homography(_src_kp, _dst_kp)
    return homography, _src_kp, _dst_kp, good_matches
