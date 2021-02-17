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
            points[:2, 0] = d["bbox"][:2]
            points[:2, 1] = d["bbox"][2:]
            updated_points = homography_matrix @ points
            updated_points = (updated_points[:2] / updated_points[-1]).astype(int)
            updated_bboxes[det_id] = {
                "bbox": updated_points.flatten(order="F").tolist()
            }

        return updated_bboxes

    def generate_mot_sequence(
        self,
        frame_id: int,
        num_frames: Optional[int] = 5,
        num_features: Optional[int] = 1000,
    ):
        """Generate a synthetic MOT sequence using homography

        Args:
            frame_id (int): Base frame id to sample.
            num_frames (Optional[int]): Number of frames to use in the sequence.
            num_features (Optional[int]): Number of features to compute homography.

        Returns:
            updated_bboxes: Dict of synthetic detections with track ids
        """
        base_img, base_det = self._load_img_and_detection(fid)
        updated_bboxes = {
            "1": {
                det_id: {"bbox": d["bbox"], "score": d["score"]}
                for det_id, d in base_det.items()
            }
        }
        for i in range(2, num_frames + 1):
            dst_img, dst_det = self._load_img_and_detection(fid + i)
            homography = detection_filtered_homography(
                dst_img, base_img, dst_det, base_det, nfeatures=num_features
            )
            updated_bboxes[i] = self._project_detections(homography, base_det)

        return updated_bboxes
