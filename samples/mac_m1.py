import cv2

from ultralight import UltraLightDetector
from ultralight.utils import draw_faces


def main():
    # pip install onnxruntime-silicon
    image = cv2.imread('sample.jpg')

    detector = UltraLightDetector(providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    faces, scores = detector.detect(image)
    print(f'Found {len(faces)} face(s)')

    draw_faces(image, faces, scores)
    cv2.imshow('result', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
