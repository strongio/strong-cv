import os
import glob
from typing import Optional

import cv2
import numpy as np
from tqdm.auto import tqdm

from ..io.video import Video
from ..utils import load_json
from ..utils.homography import detection_filtered_homography


class MOTDataGenerator:
    """Generate MOT-like data from static video detections"""

    def __init__(self, video_file: str, detections_file: str, output_path: str):
        self.video = Video(video_file)
        self.video.extract_frames(os.path.join(output_path, "frames"))
        self.detections = load_json(detections_file)
        self.output_path = output_path

        self.frame_paths = sorted(glob.glob(os.path.join(output_path, "frames/*.jpg")))

    def _load_img_and_detection(self, fid: int):
        """Load image and corresponding detection from fid

        Args:
            fid (int): Frame number.

        Returns:
            img: Image array.
            det: Detection dict
        """
        img = cv2.imread(self.frame_paths[fid])
        det = self.detections[str(fid)]
        return img, det

    def _project_detections(self, homography_matrix: np.ndarray, detections: dict):
        """Project bounding box (for now) using homography

        Args:
            homography_matrix (np.ndarray): Computed homography matrix.
            detections (dict): Detections we want to project.

        Returns:
            projected_detections: Dictionary of projected detections.
        """
        # Gather all points and project
        projected_detections = dict()
        points = np.ones((3, 2))
        for det_id, d in detections.items():
            points[:2, 0] = d["bbox"][:2]
            points[:2, 1] = d["bbox"][2:]
            projected_points = homography_matrix @ points
            projected_points = (projected_points[:2] / projected_points[-1]).astype(int)
            projected_detections[det_id] = {
                "bbox": projected_points.flatten(order="F").tolist(),
                "score": d["score"],
            }

        return projected_detections

    def generate_mot_sequences(
        self,
        start_frame: Optional[int] = 0,
        downsample: Optional[int] = 100,
        num_frames: Optional[int] = 5,
        num_features: Optional[int] = 1000,
        path_prefix: Optional[str] = "MOT",
    ):
        """Generate synthetic MOT sequences from a set of video frames.

        Args:
            start_frame (Optional[int]): Frame to start sampling.
            downsample (Optional[int]): Frames to downsample.
            num_frames (Optional[int]): Number of frames per sequence.
            num_features (Optional[int]): Number of features to compute homography.
            path_prefix (Optional[str]): MOT sequence output path prefix.
        """
        for frame_id in tqdm(range(start_frame, len(self.frame_paths), downsample)):
            out_path = os.path.join(self.output_path, f"{path_prefix}_{frame_id}")
            if not os.path.exists(out_path):
                os.makedirs(out_path)
                os.makedirs(os.path.join(out_path, "img1"))
                os.makedirs(os.path.join(out_path, "gt"))

            self.generate_mot_sequence(out_path, frame_id, num_frames, num_features)

    def generate_mot_sequence(
        self,
        path: str,
        frame_id: int,
        num_frames: Optional[int] = 5,
        num_features: Optional[int] = 1000,
    ):
        """Generate a synthetic MOT sequence using homography.

        Args:
            frame_id (int): Base frame id to sample.
            num_frames (Optional[int]): Number of frames to use in the sequence.
            num_features (Optional[int]): Number of features to compute homography.
        """
        base_img, base_det = self._load_img_and_detection(frame_id)
        mot_sequence = dict()
        for det_id, d in base_det.items():
            mot_sequence[det_id] = {"0": {"bbox": d["bbox"], "score": d["score"]}}
        images = [base_img]
        for i in range(1, num_frames):
            dst_img, dst_det = self._load_img_and_detection(frame_id + i)
            homography = detection_filtered_homography(
                dst_img, base_img, dst_det, base_det, nfeatures=num_features
            )
            projected_detections = self._project_detections(homography, base_det)
            for det_id, d in projected_detections.items():
                mot_sequence[det_id][str(i)] = d
            images.append(
                cv2.warpPerspective(base_img, homography, base_img.shape[:2][::-1])
            )

        self.write_mot_sequence(path, images, mot_sequence)

    def write_mot_sequence(self, path: str, images: list, mot_sequence: dict):
        """Write MOT sequence to file following MOTChallenge conventions

        Args:
            path (str): Output path for sequence.
            images (list): List of images.
            mot_sequence (dict): Sequence of track detections and ids
        """
        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(os.path.join(path, "img1"))
            os.makedirs(os.path.join(path, "gt"))

        # Write images
        for i, image in enumerate(images):
            cv2.imwrite(os.path.join(path, f"img1/{i:05d}.jpg"), image)

        # Write gt text
        with open(os.path.join(path, "gt/gt.txt"), "w") as f:
            for track_id, frame_detections in mot_sequence.items():
                for frame_id, detection in frame_detections.items():
                    x0, y0, x1, y1 = detection["bbox"]
                    w = x1 - x0
                    h = y1 - y0
                    score = detection["score"]
                    f.write(
                        f"{int(frame_id)+1},{int(track_id)+1},{x0},{y0},{w},{h},1,1,1\n"
                    )
