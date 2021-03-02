# Strong CV 💪 📷

## Introduction

`strongcv` is a small library containing some useful computer vision utilities. If we've
had to write it more than a handful of times, then it'll make its way here.

## Examples

### Video IO

Easily iterate through frames in videos without having to deal with OpenCV.
```python
from strongcv.io.video import Video

video = Video("clip.mp4")

for frame in video:
  # Do something, like pass it to a model
  out = model(frame)
```

Extract frames from a video
```python
from strongcv.io.video import Video

video = Video("clip.mp4")

video.extract_frames(output_path="frames")
```

### Image IO

Collect frames in a particular directory
```python
from strongcv.io.image import ImageDirectory

img_dir = ImageDirectory("frames")

for img_path in img_dir.frame_paths:

  # Do something with the image path
  print(img_path)
```
