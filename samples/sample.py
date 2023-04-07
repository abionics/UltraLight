import cv2

from ultralight import UltraLightDetector
from ultralight.utils import draw_faces


def main():
    image = cv2.imread('sample.jpg')

    detector = UltraLightDetector()
    boxes, scores = detector.detect_one(image)
    print(f'Found {len(boxes)} face(s)')

    draw_faces(image, boxes, scores)
    cv2.imshow('result', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
