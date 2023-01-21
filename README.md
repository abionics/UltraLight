# Ultra Light

Ultra Light Fast Generic Face Detector 👨‍👩‍👧‍👦🖼

## Installation

```bash
pip install ultralight
```

## Usage sample

```python
import cv2
from ultralight import UltraLightDetector
from ultralight.utils import draw_faces

image = cv2.imread('sample.jpg')

detector = UltraLightDetector()
faces, scores = detector.detect(image)
print(f'Found {len(faces)} face(s)')
# >>> Found 5 face(s)

draw_faces(image, faces, scores)
cv2.imshow('result', image)
cv2.waitKey(0)
```

This sample can be found [here](samples/sample.py)

## More

PyPI: https://pypi.org/project/ultralight

Repository: https://github.com/abionics/UltraLight

Developer: Alex Ermolaev (Abionics)

Email: abionics.dev@gmail.com

License: MIT (see LICENSE.txt)
