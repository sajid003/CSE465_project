import cv2 as cv
import numpy as np


def horizontal_flip(img, mask):
    return cv.flip(img, 0), cv.flip(mask, 0)

def vertical_flip(img, mask):
    return cv.flip(img, 1), cv.flip(mask, 1)

def horizontal_and_vertical_flip(img, mask):
    return cv.flip(img, -1), cv.flip(mask, -1)


def zoom_in(img, mask):
    return cv.resize(img, None, fx=1.2, fy=1.2), cv.resize(mask, None, fx=1.2, fy=1.2)

def zoom_out(img, mask):
    return cv.resize(img, None, fx=0.9, fy=0.9), cv.resize(mask, None, fx=1.2, fy=1.2)


def translation(img, mask):
    rows, cols = img.shape
    tx = np.random.randint(-50, 50)
    ty = np.random.randint(-50, 50)
    M = np.float32([[1,0,tx],[0,1,ty]])
    return cv.warpAffine(img, M, (cols, rows)), cv.warpAffine(mask, M, (cols, rows))



