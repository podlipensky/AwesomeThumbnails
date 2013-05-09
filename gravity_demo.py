import cv2
import os
import numpy as np
from find_objects import crop_image, etime, process_image

def show_results(gravity_src, img, center, radius, zoom_level):
    """
    This demo process images and tries to find salient objects in them. Once
    object is found, it perform masking (cropping) of the most "interesting" part.
    All results are compared to gravity screenshots to show improvements.
    Use Space to navigate to next result and Esc to exit.
    """
    gravity = cv2.imread(gravity_src)
    gravity_width = 500
    gravity = cv2.resize(gravity, (gravity_width, gravity.shape[0]*gravity_width/gravity.shape[1]))

    img = crop_image(img, center, zoom_level, radius)

    # output results
    screen_size = (max(gravity.shape[0], img.shape[0]), gravity.shape[1] + img.shape[1], 3)
    screen = np.zeros(screen_size,np.uint8)
    for c in range(3):
        screen[0:gravity.shape[0], 0:gravity.shape[1]] = gravity[:,:]
        screen[0:img.shape[0], gravity.shape[1]:gravity.shape[1]+img.shape[1]] = img[:,:]

    cv2.imshow('Gravity Demo', screen)


if __name__ == '__main__':
    i = 0
    type = 'all'
    image_template = 'data/%s/image%s.jpg'
    image = image_template % (type, i)
    gravity = 'data/%s/gravity%s.png' % (type, i)

    while True:
        image = image_template % (type, i)
        gravity = 'data/%s/gravity%s.png' % (type, i)
        while not (os.path.exists(image)) and i < 1000: # and os.path.exists(gravity))
            i += 1
            image = image_template % (type, i)
            gravity = 'data/%s/gravity%s.png' % (type, i)
        if i < 1000:
            start = etime()

            img, center, radius, zoom_level = process_image(image, 160)
            show_results(gravity, img, center, radius, zoom_level)

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
