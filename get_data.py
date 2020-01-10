#!/usr/bin/env python3

import cv2
import numpy as np
import time
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from get_numbers import find_digits
from parallel_flow import calc_flow

def draw_flow(frame, mag, ang, is_moving):
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
    if is_moving:
        cv2.putText(stacked, str(np.average(mag)), (300, 300), font,
                           fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('', stacked)

def calc_is_moving(mag):
    return np.average(mag) > 0.09

def getNextNthFrame(cap, n):
    for i in range(n - 1):
        cap.read()
    _, frame = cap.read()
    return cap.read()

def get_new_datapoint(frames):
    return frames[len(frames) * 2 // 3]

less = []
more = []
def main():
    cap = cv2.VideoCapture('20191127_161824.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    last = None
    is_moving = False
    frames = []
    skip = 3
    minTime = 0.5
    moving_count = 0
    i = 0
    a = time.time()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    while True:
        i += 1

        t = (3 * i) / fps
        if int(t) == t:
            print(t)

        _, color = getNextNthFrame(cap, skip)
        frame = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        if last is not None:
            was_moving = is_moving

            if np.average(np.abs(frame - last)) > 40:
                is_moving = True
            else:
                flow = calc_flow(last, frame)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                is_moving = calc_is_moving(mag)

            """
            if is_moving:
               less.append(np.average(np.abs(fBlur - lBlur)))
            else:
               more.append(np.average(np.abs(fBlur - lBlur)))
            """

            frames.append(color)
            if is_moving:
                #print("m")
                moving_count += 1
                # actually is moving
                if moving_count * skip > int(fps * minTime):
                    #print("M")
                    frames = frames[:len(frames) - moving_count]
                    # enough frames
                    if len(frames) * skip > int(fps * minTime):
                        name = str(time.time()) + ".png"
                        img = frames[len(frames) * 2 // 3]
                        num, img = find_digits(img)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        org = (100, 100)
                        fontScale = 4
                        color = (255, 0, 0)
                        thickness = 2
                        cv2.putText(img, str(num), org, font,
                                    fontScale, color, thickness, cv2.LINE_AA)

                        cv2.imwrite(name, img)
                        #print("Yay")
                    frames = []
            if was_moving and not is_moving:
                moving_count = 0

            #draw_flow(frame, mag, ang, is_moving)

        last = frame
        #cv2.waitKey(1)

    num_bins = 40
    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    axs[0].hist(less, num_bins, facecolor='blue', alpha=0.5)
    axs[1].hist(more, num_bins, facecolor='red', alpha=0.5)
    plt.show()
    print(time.time() - a)

if __name__ == "__main__":
    main()
