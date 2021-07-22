from modules.yolo import YoloCV
import cv2 as cv

import time
import sys
import json
import os

yolo = YoloCV(os.path.abspath(os.path.dirname(__file__)) + '/modules')
class_dimensions = json.load(open(os.path.abspath(os.path.dirname(__file__)) + '/modules/class_averages.json', 'r'))
f = open("logs.txt", "a")

image_folder = "data/images/"

video = (len(sys.argv) > 1 and sys.argv[1] == '--video')

if video:
    image_folder = "data/video_images/"

image_path = os.path.abspath(os.path.dirname(__file__)) + "/" + image_folder

ids = [x.split('.')[0] for x in sorted(os.listdir(image_path))]

if not ids:
    print("\nError: No images in" + image_path)

for image_id in ids:

    start_time = time.time()

    image_file = image_path + image_id + ".png"
    image = cv.imread(image_file)
    detections = yolo.detect(image)

    for detection in detections:

        if not detection[1] in class_dimensions:
            continue

        box = detection[0]
        cv.rectangle(image, box[0], box[1], (255, 0, 0), 2)

    cv.imshow('Press SPACE to go to next image, ESC to EXIT', image)

    print('------------------------------------')
    output = '\nImage: %s   |  Objects detected: %s   |   Time taken: %.2fs   |   FPS: %.2f' % (
        image_id + '.png', len(detections), time.time() - start_time, 1.0 / (time.time() - start_time))

    f.write(output + '\n')
    print(output)

    if video:
        if cv.waitKey(1) == 27:
            exit()
    else:
        if cv.waitKey(0) != 32:
            exit()
