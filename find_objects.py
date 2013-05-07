#/usr/bin/env python
from math import sqrt
import os

import numpy as np
import cv2
from scipy.cluster import vq
from numpy import array
from weighted_kmeans import kmeans

green_color = (0, 255, 0)
blue_color = (255, 0, 0)
red_color = (0, 0, 255)

def etime():
    """See how much user and system time this process has used so far and return the sum."""
    user, sys, chuser, chsys, real = os.times()
    return user+sys

def canny_threshold(img, lowThreshold):
    ratio = 3
    kernel_size = 3
    detected_edges = cv2.GaussianBlur(img,(3,3),0)
    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
    return detected_edges

def display_canny_threshold(img, lowThreshold):
    detected_edges = canny_threshold(img, lowThreshold)
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('canny demo',dst)

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def find_face(img):
    cascade_fn = "trained/haarcascade_frontalface_alt.xml"
    nested_fn  = "trained/haarcascade_eye.xml"

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)

    hist = cv2.equalizeHist(img)

    rects = detect(hist, cascade)
    if not len(rects):
        rects = detect(hist, nested)

    return rects

def get_mask(img, center, radius):
    mask = np.zeros(img.shape, dtype=np.uint8)
    white = (255, 255, 255)
    cv2.circle(mask, center, radius, white, -1, 8, 0)
    return mask

def draw_keypoints(img, keypoints):
    # draw keypoints
    for p in keypoints:
        color = (0, 255, 0)
        coords = (int(p.pt[0]), int(p.pt[1]))
        radius = int(p.size*1.2/9.*2)
        cv2.circle(img, coords, radius, color, 1, 8, 0)


def get_sides(points):
    top = (0, 99999)
    right = (0, 0)
    bottom = (0, 0)
    left = (99999, 0)
    for p in points:
        pt = np.array(p.pt)
        if top[1] > pt[1]:
            top = pt
        if right[0] < pt[0]:
            right = pt
        if bottom[1] < pt[1]:
            bottom = pt
        if left[0] > pt[0]:
            left = pt
    return top, right, bottom, left

def add_points(keypoints, points, thres):
    sides = get_sides(points)
    for p in keypoints:
        pt = np.array(p.pt)
        for side in sides:
            if np.linalg.norm(pt - side) <= thres:
                points.append(p)

def point_to_tuple(p):
    return (int(p[0]), int(p[1]))

def keypoints(img, img_gray, radius):
    detector = cv2.SURF(800)
    norm = cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)

    # todo: dillute imag_gray before processing
    keypoints = detector.detect(img_gray)

    # filter out small features
    keypoints = [p for p in keypoints if p.size > 30]
#    draw_keypoints(img, keypoints)

    # k-means clustering
    points = array([[int(p.pt[0]), int(p.pt[1])] for p in keypoints])
#    center,dist = vq.kmeans(points,1, thresh=10)
    weights = [p.size for p in keypoints]
    center,dist = kmeans(points,1, weights=weights)
    # resize and crop the image
    x,y = center[0]
    c = array(center[0])

    iter = 30
    points = [p for p in keypoints if (p.pt[0]-x)**2 + (p.pt[1]-y)**2 <= dist**2]
    while iter:
        add_points(keypoints, points, 70)
        iter -= 1

    max_dist = 0
    for p in get_sides(points):
#        cv2.circle(img, point_to_tuple(p), 10, red_color, -1, 8, 0)
        if max_dist < np.linalg.norm(p - c):
            max_dist = np.linalg.norm(p - c)

    d = radius / max_dist

    return (x,y), d

def crop_image(img, center, d, radius):
    x,y = center
    x = int(x*d)
    y = int(y*d)
    img = cv2.resize(img, (0,0), img, d, d)
    radius = min(x,y,radius,int(img.shape[1]-x), int(img.shape[0]-y))
    new_center = (x,y)
#    cv2.circle(img, new_center, 10, red_color, 1, 8, 0)
    # crop the image
    mask = get_mask(img, new_center, radius)
    masked_image = cv2.bitwise_and(img, mask)
    cv2.imshow('cropped', masked_image)
    return masked_image


def get_skeleton(orig):
    img = orig.copy()
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    img = 255 - img
    img = cv2.dilate(img, element, iterations=3)

    done = False

    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    return skel


def detect_countours(img, img_gray):
    global thresh, contours, hierarchy, cnt, rect, box
    #   ret, thresh = cv2.threshold(img_gray,125,100,0)
    thresh = canny_threshold(img_gray, 100)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, green_color, 1)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, blue_color, 1)


def process_image(src, gravity_src):
    gravity = cv2.imread(gravity_src)
    gravity_width = 500
    gravity = cv2.resize(gravity, (gravity_width, gravity.shape[0]*gravity_width/gravity.shape[1]))

    img = cv2.imread(src)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    #    img_gray = cv2.erode(img_gray,element)
    img_gray = cv2.dilate(img_gray, element, iterations=3)
    # detect and draw contours, wrap them in blue rectangles
    #    detect_countours()
    radius = 160
    # try to find any faces
    faces = find_face(img_gray)
    #    draw_rects(img, faces, red_color)
    if len(faces):
        for x1, y1, x2, y2 in faces:
            center = (int(x1 + (x2 -x1) / 2), int(y1 + (y2-y1) / 2))
            face_radius = np.linalg.norm(np.array(center) - np.array([x1, y1]))
            d = 1
    else:
        center, d = keypoints(img, img_gray, radius)
    img = crop_image(img, center, d, radius)
    #    lowThreshold = 0
    #    max_lowThreshold = 100
    #    cv2.namedWindow('canny demo')
    #    fun = lambda x: display_canny_threshold(img, x)
    #    cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, fun)

    # output results
    screen_size = (max(gravity.shape[0], img.shape[0]), gravity.shape[1] + img.shape[1], 3)
    screen = np.zeros(screen_size,np.uint8)
    for c in range(3):
        screen[0:gravity.shape[0], 0:gravity.shape[1]] = gravity[:,:]
        screen[0:img.shape[0], gravity.shape[1]:gravity.shape[1]+img.shape[1]] = img[:,:]
    cv2.imshow('Awesome Thumbnails', screen)


if __name__ == '__main__':
    print __doc__


    i = 16
    type = 'bad'
    image = 'data/%s/image%s.jpg' % (type, i)
    gravity = 'data/%s/gravity%s.png' % (type, i)

    while True:
        image = 'data/%s/image%s.jpg' % (type, i)
        gravity = 'data/%s/gravity%s.png' % (type, i)
        while not (os.path.exists(image) and os.path.exists(gravity)) and i < 50:
            i += 1
            image = 'data/%s/image%s.jpg' % (type, i)
            gravity = 'data/%s/gravity%s.png' % (type, i)
        if i < 50:
            start = etime()

            process_image(image, gravity)

            end = etime()
            print
            print end - start
        else:
            break
        k = cv2.waitKey(0)
        if k == 27:
            break
        i += 1

#    cv2.imshow('original', get_skeleton(img))
    cv2.destroyAllWindows()

