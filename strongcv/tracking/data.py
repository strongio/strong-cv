import os
import numpy as np

from ..io.video import Video
from ..utils import load_json
from ..utils.homography import detection_filtered_homography


class MOTDataGenerator:
    """Generate MOT-like data from static video detections"""

    def __init__(self, video_file: str, detections_file: str, output_path: str):
        self.video = Video(video_file)
        self.video.extract_frames(output_path)
        self.detections = load_json(detections_file)
        self.output_path = output_path

        self.frame_paths = sorted(glob.glob(os.path.join(output_path, "*.jpg")))

    def _load_img_and_detection(self, fid: int):
        """Load image and corresponding detection from fid

        Args:
            fid (int): Frame number.

        Returns:
            img: Image array.
            det: Detection dict
        """
        img = cv2.imread(self.frame_paths[fid])
        det = detections[str(fid)]
        return img, det

    def _project_detections(self, homography_matrix: np.ndarray, detections: dict):
        """Project bounding box (for now) using homography

        Args:
            homography_matrix (np.ndarray): Computed homography matrix.
            detections (dict): Detections we want to project.

        Returns:
            updated_bboxes: Dictionary of projected bounding boxes.
        """
        # Gather all points and project
        updated_bboxes = dict()
        points = np.ones((3, 2))
        for det_id, d in detections.items():
            points[0] = [0, 0]  # d["bbox"][:2]
            points[1] = [1080, 1920]  # d["bbox"][2:]
            updated_points = homography_matrix @ points
            updated_points = (updated_points[:2] / updated_points[-1]).flatten()
            updated_bboxes[det_id] = {"bbox": updated_points.astype(int).tolist()}

        return updated_bboxes
