# find-mtg-pentagon

Simple OpenCV-based detector for candidate pentagon landmarks on an MTG card back.

## Pi camera usage

The main entry point is `detect_mtg_back_center(frame)`, where `frame` is the NumPy array returned by a Raspberry Pi camera API such as `Picamera2.capture_array()`.

```python
from picamera2 import Picamera2

from mtg_pentagon_detector import detect_mtg_back_center

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888"}))
picam2.start()

frame = picam2.capture_array()
data = detect_mtg_back_center(frame)

print(data["result"])
```

The detector returns a dictionary with:

- `result`: the best pentagon match or `None`
- `candidates`: candidate blob/contour centers
- `mask`: threshold mask used for contour extraction
- `annotated`: BGR output image with debug overlays

## Local testing without a Pi camera

You can still test the script with a normal image file by loading it into a NumPy array first:

```bash
python mtg_pentagon_detector.py path/to/card_photo.jpg
```

That compatibility path writes:

- `mask.jpg`
- `pentagon_result.jpg`
