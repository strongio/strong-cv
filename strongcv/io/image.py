import os
import time
import glob
from typing import Optional

import cv2
import numpy as np


class ImageDirectory:
    def __init__(
        self,
        input_path: str,
        ext: Optional[str] = "",
        sort: Optional[bool] = True,
        output_path: Optional[str] = ".",
        label: Optional[str] = "",
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.label = label

        # TODO: add support for more extension types
        self.ext = set([".png", ".jpg"])

        self.frame_paths = self._collect_frames(sort=sort, ext=ext)

    def _collect_frames(self, sort: Optional[bool] = True, ext: Optional[str] = ""):
        paths = glob.glob(f"{self.input_path}/*{ext}")
        if not ext:
            paths = [p for p in paths if os.path.splitext(p)[-1] in self.ext]
        if sort:
            paths = sorted(paths)
        return paths

    # TODO:
    #   1. Write video
    #   2. Sample with index
    #   3. Encode/decode
