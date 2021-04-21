#! /use/bin/env python

import cv2 as cv
import numpy as np


def point(pt1, pt2, t):
    return np.array(pt1 + (pt2 - pt1) * t, dtype=pt1.dtype)

def make_points(pt1, pt2, t = 0.25):
    return (point(pt1, pt2, t), point(pt1, pt2, 1-t))

def draw_rectangle(
    img, pt1, pt2, color, thickness=1, lineType=8, shift=0, corners=True, sides=True
):
    """
    All parameters except the following are taken from OpenCV's implemntation of rectangle

    @param corners should the corners be drawn (25% from the corners)
    @param sides should the sides be drawn (25% from the center of the side)
    """
    if corners and sides:
        return cv.rectangle(img, pt1, pt2, color, thickness, lineType)
    points = np.array([pt1, pt2, [pt1[0], pt2[1]], [pt2[0], pt1[1]]], dtype=np.uint16)
    bottom_left = np.min(points, axis=0)
    top_right = np.max(points, axis=0)
    top_left = np.array([bottom_left[0], top_right[1]], dtype=np.uint16)
    bottom_right = np.array([top_right[0], bottom_left[1]], dtype=np.uint16)

    # There might be a better way but this is shorter
    top_ends = make_points(top_left, top_right)
    bottom_ends = make_points(bottom_left, bottom_right)

    right_ends = make_points(bottom_right, top_right)
    left_ends = make_points(bottom_left, top_left)

    if corners:
        line_points = [
            (top_right, top_ends[1]),
            (top_right, right_ends[1]),
            (top_left, top_ends[0]),
            (top_left, left_ends[1]),
            (bottom_left, bottom_ends[0]),
            (bottom_left, left_ends[0]),
            (bottom_right, bottom_ends[1]),
            (bottom_right, right_ends[0]),
        ]
        for p1, p2 in line_points:
            cv.line(img, tuple(p1), tuple(p2), color, thickness, lineType, shift)
        return img

    if sides:
        line_points = [
            (top_ends[0], top_ends[1]),
            (bottom_ends[0], bottom_ends[1]),
            (left_ends[0], left_ends[1]),
            (right_ends[0], right_ends[1]),
        ]
        for p1, p2 in line_points:
            cv.line(img, tuple(p1), tuple(p2), color, thickness, lineType, shift)
        return img


if __name__ == "__main__":
    image = np.zeros([500, 500, 3], dtype=np.uint8)  # black image
    draw_rectangle(image, (10, 10), (50, 50), (255, 255, 255))
    draw_rectangle(image, (170, 120), (120, 170), (255, 255, 0), corners=False)
    draw_rectangle(image, (30, 330), (80, 180), (255, 0, 255), sides=False)
    cv.imshow("3 rectangles", image)
    cv.waitKey(2000)
