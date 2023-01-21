import time
from multiprocessing import Pool

import cv2

from ultralight import UltraLightDetector

PROCESSES = 1
BATCH_SIZE = 100


def process(filenames: list[str]) -> tuple[int, float]:
    count = 0
    detector = UltraLightDetector()
    start_time = time.time()
    for filename in filenames:
        image = cv2.imread(filename)
        _, scores = detector.detect(image)
        count += len(scores)
    return count, time.time() - start_time


def main():
    batches = [
        ['sample.jpg' for _ in range(BATCH_SIZE)]
        for _ in range(PROCESSES)
    ]
    print(f'Processes: {PROCESSES}, batch size: {BATCH_SIZE}')
    start_time = time.time()
    if PROCESSES == 1:
        result = process(batches[0])
        results = [result]
    else:
        with Pool(PROCESSES) as pool:
            results = pool.map(process, batches)
    finish_time = time.time()
    print('Total faces:', sum(result[0] for result in results))
    total_time = finish_time - start_time
    total_speed = PROCESSES * BATCH_SIZE / total_time
    print(f'Total time: {total_time:.2f} seconds, total speed: {total_speed:.2f} images/second')
    detection_time = sum(result[1] for result in results) / PROCESSES
    detection_speed = PROCESSES * BATCH_SIZE / detection_time
    print(f'Detection time: {detection_time:.2f} seconds, detection speed: {detection_speed:.2f} images/second')
    difference = 100 * (1 - detection_time / total_time)
    print(f'Difference: {difference:.1f}%')


if __name__ == '__main__':
    main()
