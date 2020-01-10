#!/usr/bin/env python3

import cv2
import numpy as np
from scipy import stats
from collections import defaultdict
import time
import csv
import glob
import random

target = cv2.imread('screen.png')
target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
target_hist = cv2.calcHist([target_hsv],
            [0, 1], None, [180, 256],
            [0, 180, 0, 256])
cv2.normalize(target_hist, target_hist, 0, 255, cv2.NORM_MINMAX)

def get_screen_area(frame):
    dst = backproject_screen(frame)
    dst = smooth_backprojection(dst)
    _, dst = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)
    dst = fill_screen_holes(dst)
    dst2 = remove_spots(dst)
    #return np.hstack((dst, dst2))
    return dst2

def backproject_screen(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.calcBackProject(\
                [hsv], [0, 1], target_hist,\
                [0, 180, 0, 256], 1)

def fill_screen_holes(dst):
    h, w = dst.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(dst, mask, (0, 0), 127)
    dst = np.where(dst == 0, 255, dst)
    dst = np.where(dst == 127, 0, dst)
    return dst

def smooth_backprojection(projection):
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    projection = cv2.filter2D(projection, -1, disc)
    for i in range(2):
        projection = cv2.filter2D(projection, -1, disc)
    return projection

def remove_spots(dst):
    for i in range(5, 90, 10):
        kernel = np.ones((i, i),np.uint8)
        dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    return dst

def find_digits(frame):

    roi = binarize_screen(frame)
    roi = remove_islands(roi)

    #return 0, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    last = find_solid_edge(roi)
    digits_end = last[1]
    digits_ratio = digits_end / roi.shape[1]
    second_digit_start_x = digits_ratio - .2557

    d0 = get_digit(roi, second_digit_start_x, .08)
    d1 = get_digit(roi, second_digit_start_x - .3836, .08)
    #draw_label(d1, d0, gray)
    #cv2.circle(gray, (last[1], last[0]), 10, 127)
    #cv2.rectangle(gray, (p0[0], p0[1]), (p1[0], p1[1]), 128, thickness = 2)

    #return d1 * 10 + d0, cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    return d1 * 10 + d0, None#cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def get_digits_bounds(frame):
    screen_area = get_screen_area(frame)
    moments = cv2.moments(screen_area)
    x_avg = int(moments['m10'] / moments['m00'])
    y_avg = int(moments['m01'] / moments['m00'])
    tl = (x_avg - 7 + 10, y_avg - 44)
    br = (x_avg + 144 + 10, y_avg + 48)
    return tl, br

def binarize_screen(frame):
    p0, p1 = get_digits_bounds(frame)
    roi = frame[p0[1]:p1[1], p0[0]:p1[0]]
    roi = clarify_digits(roi)
    return roi

def clarify_digits(screen):
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)[..., 2]

    screen = remove_glare(screen)
    screen = increase_contrast(screen)
    screen = remove_gradient(screen)
    screen = remove_specks_from_screen(screen)
    screen = threshold_screen(screen)

    return screen

def remove_glare(img):
    scores = stats.zscore(img, axis = None)
    mask = np.zeros(img.shape, dtype = np.uint8)
    mask[scores > 2.5] = 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel)
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)
    return img

def increase_contrast(img):
    return cv2.equalizeHist(img)

def remove_gradient(img):
    return img - cv2.GaussianBlur(img, (451, 3), 0)

def remove_specks_from_screen(img):
    for i in range(3, 7, 2):
        kernel = np.ones((i, i),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

def threshold_screen(img):
    return cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def remove_islands(roi):
    return roi
    num_components, labels = cv2.connectedComponents(roi)
    occurrences = np.bincount(labels.flatten())
    roi[occurrences[labels] < 80] = 0
    return roi

def find_solid_edge(gray):
    horizontal_pixel_counts = np.sum(gray, axis = 0) // 255
    non_empty_cols = np.flip(horizontal_pixel_counts) > 35
    col_from_right = np.argmax(non_empty_cols)
    col = horizontal_pixel_counts.shape[0] -  col_from_right

    where = np.argwhere(gray == 255)
    return np.max(where, axis = 0)[0], col

def draw_label(d1, d0, screen):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (500, 1400)
    fontScale = 4
    color = (255, 0, 0)
    thickness = 2
    cv2.putText(screen, str(d1 * 10 + d0), org, font,
                       fontScale, color, thickness, cv2.LINE_AA)

def detect_white_rect(roi, x, y, is_vertical = True):
    px, py = int(x * roi.shape[1]), int(y * roi.shape[0])
    height = 10
    width = 6
    if not is_vertical:
        height, width = width, height

    amount_filled = np.average(roi[py - height: py + height, px - width: px + width])
    rect_tl = (px - width, py - height)
    rect_br = (px + width, py + height)
    cv2.rectangle(roi, rect_tl, rect_br, 128, thickness = 2)
    if amount_filled > 60:
        return True
    return False

def get_digit(img, x, y):
    A = detect_white_rect(img, x + 0.05*1.96726, y, is_vertical = False)
    B = detect_white_rect(img, x + 0.1*1.96726, y + 0.15*1.23636)
    C = detect_white_rect(img, x + 0.1*1.96726, y + 0.56*1.23636)
    D = detect_white_rect(img, x + 0.05*1.96726, y + 0.7*1.23636, is_vertical = False)
    E = detect_white_rect(img, x, y + 0.56*1.23636)
    F = detect_white_rect(img, x, y + 0.15*1.23636)
    G = detect_white_rect(img, x + 0.05*1.96726, y + 0.33*1.23636, is_vertical = False)
    return segments_to_digit(A, B, C, D, E, F, G)

def segments_to_digit(A, B, C, D, E, F, G):
    num = (int(A) << 6) + (int(B) << 5) + \
          (int(C) << 4) + (int(D) << 3) + \
          (int(E) << 2) + (int(F) << 1) + int(G)
    nums = {0b0110000: 1, 0b1101101: 2, 0b1111001: 3,
            0b0110011: 4, 0b1011011: 5, 0b1011111: 6,
            0b1110000: 7, 0b1111111: 8, 0b1111011: 9}
    return defaultdict(lambda: 0, nums)[num]

def main():
    #process_image("trained_with/1578022928.093898.png")
    #process_video()
    print(get_accuracy())
    #process_and_save_images()

def process_and_save_images():
    train_x, train_y, test_x, test_y = get_datasets()
    for i in range(len(train_x)):
        _, screen = find_digits(cv2.imread(train_x[i]))
        #print(train_x[i])
        cv2.imwrite(str(i)+".png", screen)

def get_accuracy():
    train_x, train_y, test_x, test_y = get_datasets()
    return calc_accuracy(test_x + train_x, test_y + train_y)

def get_datasets():
    data = get_data()
    return split_data(data)

def get_data():
    datafile = open("data.csv", "r")
    data = []
    for row in datafile:
        filename, number = row.replace("\n", "").split(", ")
        filename = 'trained_with/' + filename
        data.append((filename, number))
    return data

def split_data(data):
    random.seed(a = 2342353)
    random.shuffle(data)
    train_x, train_y = zip(*data[0: len(data) // 2])
    test_x, test_y = zip(*data[len(data) // 2:])
    return train_x, train_y, test_x, test_y

def calc_accuracy(filenames, labels):
    correct = 0
    for filename, truth in zip(filenames, labels):
        number, _ = find_digits(cv2.imread(filename))
        if number == int(truth):
            correct += 1
        else:
            print(filename)
    return correct / len(filenames)


def process_image(filename):
    cv2.namedWindow('comp')
    frame = cv2.imread(filename)
    _, screen = find_digits(frame)
    comparison = np.hstack((screen, frame))
    show_indefinitely(comparison)

def show_indefinitely(image):
    cv2.imshow('', image)
    while True:
        cv2.waitKey(1)


def process_video():
    cv2.namedWindow('comp')
    cap = cv2.VideoCapture('20191127_161824.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 930)

    while True:
        for i in range(2):
            _, frame = cap.read()
        _, screen = find_digits(frame)
        comparison = np.hstack((screen, frame))
        cv2.imshow('comp', comparison)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
