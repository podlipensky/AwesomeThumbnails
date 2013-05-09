#/usr/bin/env python
import os

import numpy as np
import cv2
from numpy import array
from weighted_kmeans import kmeans

green_color = (0, 255, 0)
blue_color = (255, 0, 0)
red_color = (0, 0, 255)

def etime():
    """See how much user and system time this process has used so far and return the sum."""
    user, sys, chuser, chsys, real = os.times()
    return user+sys

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
#    if not len(rects):
#        rects = detect(hist, nested)

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

def keypoints(img, img_gray, radius=None):
    detector = cv2.SURF(800)
    keypoints = detector.detect(img_gray)

    # filter out small features
    keypoints = [p for p in keypoints if p.size > 30]
#    draw_keypoints(img, keypoints)

    # k-means clustering
    points = array([[int(p.pt[0]), int(p.pt[1])] for p in keypoints])
    weights = [p.size for p in keypoints]
    center,dist = kmeans(points,1, weights=weights)

    # resize and crop the image
    x,y = center[0]
    c = array(center[0])
    d = 0

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

    if not radius:
        radius = normalize_radius(img, x, y, int(max_dist))
    else:
        d = radius / max_dist

    return (x,y), d, radius

def normalize_radius(img, x, y, radius):
    return min(x,y,radius,int(img.shape[1]-x), int(img.shape[0]-y))

def crop_image(img, center, d, radius):
    x,y = center
    x = int(x*d)
    y = int(y*d)
    img = cv2.resize(img, (0,0), img, d, d)
    radius = normalize_radius(img, x, y, radius)
    new_center = (x,y)
#    cv2.circle(img, new_center, 10, red_color, 1, 8, 0)
    # crop the image
    mask = get_mask(img, new_center, radius)
    masked_image = cv2.bitwise_and(img, mask)
    cv2.imshow('cropped', masked_image)
    return masked_image

def process_image(src, radius):
    img = cv2.imread(src)
    # prepare image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    #    img_gray = cv2.erode(img_gray,element)
    img_gray = cv2.dilate(img_gray, element, iterations=3)

    faces = find_face(img_gray)

    # outline found faces
    #    draw_rects(img, faces, red_color)

    if len(faces):
        for x1, y1, x2, y2 in faces:
            center = (int(x1 + (x2 -x1) / 2), int(y1 + (y2-y1) / 2))
            if not radius:
                radius = int(np.linalg.norm(np.array(center) - np.array([x1, y1])))
            zoom_level = 1
    else:
        center, zoom_level, radius = keypoints(img, img_gray, radius)
    return img, center, radius, zoom_level
