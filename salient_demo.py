import cv2
import os
import numpy as np
from find_objects import crop_image, etime, process_image

def show_results(img, center, radius, zoom_level):
    """
    This demo process images and tries to find salient objects in them in very
    naive way. The algorithm is the following:
    Phase 1.
        Try to find any faces (as they're important for ads) by using classifier
        (namely a cascade of boosted classifiers working with haar-like features)
    Phase 2.
        If no faces were found, try to find the most "distinctable" object by using
        SURF features. The demo is looking for a single object only and will
        group set of features into an object shape by using k-means algorithm.
    Use Space to navigate to next result and Esc to exit.
    """
    green_color = (0, 255, 0)
    # outlines found object
    cv2.circle(img, center, radius, green_color, 1, 8, 0)
    cv2.imshow('Naive Salient Object Detection', img)


if __name__ == '__main__':
    i = 0
    type = 'all'
    image_template = 'data/%s/image%s.jpg'
    image = image_template % (type, i)

    while True:
        image = image_template % (type, i)
        while not (os.path.exists(image)) and i < 1000:
            i += 1
            image = image_template % (type, i)
            gravity = 'data/%s/gravity%s.png' % (type, i)
        if i < 1000:
            start = etime()

            img, center, radius, zoom_level = process_image(image, None)
            show_results(img, center, radius, zoom_level)

            end = etime()
            print
            print end - start
        else:
            break
        k = cv2.waitKey(0)
        if k == 27:
            break
        i += 1

    cv2.destroyAllWindows()
