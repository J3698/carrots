#!/usr/bin/env python3

from threading import Thread, Lock
import numpy as np
import cv2

def calc_flow(last, frame):
    if frame.shape[0] * frame.shape[1] <= 480 * 270:
        return calc_flow_normal(last, frame)
    else:
        return calc_flow_parallel(last, frame)

def calc_flow_normal(last, frame):
    return cv2.calcOpticalFlowFarneback(\
            last, frame, None, 0.5, 5, 15, 3, 7, 1.5, 0)

def calc_flow_parallel(last, frame):
    last_split = split(last)
    frame_split = split(frame)
    workers = [FlowWorker(*i) for i in zip(last_split, frame_split)]
    startAndJoinThreads(workers)
    return merge(*(w.flow for w in workers))

def startAndJoinThreads(threads):
    for t in threads:
        t.start()
    for t in threads:
        t.join()

class FlowWorker(Thread):
    def __init__(self, last, frame):
        super().__init__()
        self.last = last
        self.frame = frame
        self.flow = None

    def run(self):
        self.flow = calc_flow(self.last, self.frame)

def split(img):
    top, bottom = np.array_split(img, 2, axis = 0)
    top_left, top_right = np.array_split(top, 2, axis = 1)
    bottom_left, bottom_right = np.array_split(bottom, 2, axis = 1)
    return top_left, top_right, bottom_left, bottom_right

def merge(tl, tr, bl, br):
    bottom = np.concatenate((bl, br), axis = 1)
    top = np.concatenate((tl, tr), axis = 1)
    return np.concatenate((top, bottom))
