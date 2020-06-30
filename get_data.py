#!/usr/bin/env python3

import cv2
import numpy as np
import time
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from get_numbers import read_digits
from parallel_flow import calc_flow

def draw_flow(frame, mag, ang):
    frame = frame.copy()
    color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    hsv = np.zeros_like(color)
    hsv[...,1] = 255
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,2] = np.minimum(4 * mag, 255)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    stacked = np.hstack((color, hsv))

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (100, 100)
    fontScale = 4
    color = (255, 0, 0)
    thickness = 2
    cv2.putText(stacked, str(np.average(mag)), org, font,
                       fontScale, color, thickness, cv2.LINE_AA)

    return stacked

def getNextNthFrame(cap, n):
    for i in range(n - 1):
        cap.read()
    _, frame = cap.read()
    return frame

def get_new_datapoint(frames):
    return frames[len(frames) * 2 // 3]


def write_number(img, num):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (100, 100)
    fontScale = 4
    color = (255, 0, 0)
    thickness = 2
    cv2.putText(img, str(num), org, font,
                fontScale, color, thickness, cv2.LINE_AA)

def create_histogram():
    num_bins = 40
    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    axs[0].hist(less, num_bins, facecolor='blue', alpha=0.5)
    axs[1].hist(more, num_bins, facecolor='red', alpha=0.5)
    plt.show()

class SustainedMovementDetector():
    def __init__(self, initial_frame, fps, skip, min_time):
        self.last_frame = initial_frame
        self.fps = fps
        self.skip = skip
        self.min_time = min_time
        self.moving_count = 0
        self.was_moving = None
        self.is_moving = False
        self.flow = None

    def detect(self, frame):
        self.calc_is_moving(frame)

        if self.is_moving:
            self.moving_count += 1
            if self.moving_significantly():
                return True
        elif was_moving:
            self.moving_count = 0
        return False

    def calc_is_moving(self, frame):
        self.last_frame = frame
        self.was_moving = self.is_moving

        if np.average(np.abs(frame - self.last_frame)) > 40:
            self.flow = None
            self.is_moving = True
        else:
            self.calc_is_moving_optical_flow()

    def calc_is_moving_optical_flow(self):
        flow = calc_flow(self.last_frame, frame)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        self.flow = mag, ang
        self.is_moving = np.average(mag) > 0.09

    def moving_significantly(self):
        return self.moving_count * self.skip > int(self.fps * self.min_time)

def capture_data(frames):
    name = str(time.time()) + ".png"
    img = frames[len(frames) * 2 // 3]
    num, img = read_digits(img)
    write_number(img, num)
    #cv2.imwrite(name, img)

def print_progress(fps, i):
    t = (3 * i) / fps
    if int(t) == t:
        print(t)

def main():
    min_time = 0.5
    skip = 3
    cap = cv2.VideoCapture('20191127_161824.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    last = None
    frames = []
    i = 0

    initial_img = getNextNthFrame(cap, skip)
    initial_gray = cv2.cvtColor(initial_img, cv2.COLOR_BGR2GRAY)
    detector = SustainedMovementDetector(initial_gray, fps, skip, min_time)
    while True:
        i += 1
        print_progress(fps, i)

        img = getNextNthFrame(cap, skip)
        frames.append(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if detector.detect(gray):
            capture_data(frames)
            frames = []

        if i > 20 and detector.flow is not None:
            gray = draw_flow(gray, detector.flow[0], detector.flow[1])

        cv2.imshow('window', gray)
        cv2.waitKey(1)

        while i > 20 and detector.flow is not None:
            print(i)
            cv2.waitKey(1)

if __name__ == "__main__":
    main()
