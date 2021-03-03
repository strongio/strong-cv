import os
import time
import glob
from typing import Optional

import cv2
from PIL import Image
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
        self.ext = set([".png", ".jpg", ".jpeg"])

        self.frame_paths = self._collect_frames(sort=sort, ext=ext)
        self.num_frames = len(self.frame_paths)
        self.frame_counter = 0

    def _collect_frames(self, sort: Optional[bool] = True, ext: Optional[str] = ""):
        paths = glob.glob(f"{self.input_path}/*{ext}")
        if not ext:
            paths = [p for p in paths if os.path.splitext(p)[-1] in self.ext]
        if sort:
            paths = sorted(paths)
        return paths

    def __len__(self):
        return self.num_frames

    def __iter__(self, format="numpy"):
        assert format in ["numpy", "pil"]

        while True:
            self.frame_counter += 1
            if self.frame_counter == self.num_frames:
                break
            img = Image.open(self.frame_paths[self.frame_counter])
            if format == "numpy":
                yield np.array(img)
            elif format == "pil":
                yield img

        self._reset()

    def _reset(self):
        self.frame_counter = 0
