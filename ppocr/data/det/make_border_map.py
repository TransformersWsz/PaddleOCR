# -*- coding:utf-8 -*- 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
np.seterr(divide='ignore', invalid='ignore')
import pyclipper
from shapely.geometry import Polygon
import sys
import warnings
warnings.simplefilter("ignore")


def draw_border_map(polygon, canvas, mask, shrink_ratio):
    polygon = np.array(polygon)
    assert polygon.ndim == 2
    assert polygon.shape[1] == 2

    polygon_shape = Polygon(polygon)
    if polygon_shape.area <= 0:
        return
    distance = polygon_shape.area * (
        1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

    padded_polygon = np.array(padding.Execute(distance)[0])
    cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

    xmin = padded_polygon[:, 0].min()
    xmax = padded_polygon[:, 0].max()
    ymin = padded_polygon[:, 1].min()
    ymax = padded_polygon[:, 1].max()
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    polygon[:, 0] = polygon[:, 0] - xmin
    polygon[:, 1] = polygon[:, 1] - ymin
    # a = np.linspace(0, 4 - 1, num=6)
    # a = [0.  0.6 1.2 1.8 2.4 3. ]
    xs = np.broadcast_to(
        np.linspace(
            0, width - 1, num=width).reshape(1, width), (height, width))
    ys = np.broadcast_to(
        np.linspace(
            0, height - 1, num=height).reshape(height, 1), (height, width))

    distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
    for i in range(polygon.shape[0]):
        j = (i + 1) % polygon.shape[0]
        absolute_distance = _distance(xs, ys, polygon[i], polygon[j])
        distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
    distance_map = distance_map.min(axis=0)

    xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
    xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
    ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
    ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
    canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
        1 - distance_map[ymin_valid - ymin:ymax_valid - ymax + height,
                         xmin_valid - xmin:xmax_valid - xmax + width],
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])


def _distance(xs, ys, point_1, point_2):
    '''
    compute the distance from point to a line
    ys: coordinates in the first axis
    xs: coordinates in the second axis
    point_1, point_2: (x, y), the end of the line
    '''
    height, width = xs.shape[:2]
    square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[
        1] - point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) / (
        2 * np.sqrt(square_distance_1 * square_distance_2))
    square_sin = 1 - np.square(cosin)
    # numpy.nan_to_num(x):
    # 使用0代替数组x中的nan元素，使用有限的数字代替inf元素
    square_sin = np.nan_to_num(square_sin)
    result = np.sqrt(square_distance_1 * square_distance_2 * square_sin /
                     square_distance)

    result[cosin <0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin <0]
    # self.extend_line(point_1, point_2, result)
    return result


def extend_line(point_1, point_2, result, shrink_ratio):
    ex_point_1 = (
        int(
            round(point_1[0] + (point_1[0] - point_2[0]) * (1 + shrink_ratio))),
        int(
            round(point_1[1] + (point_1[1] - point_2[1]) * (1 + shrink_ratio))))
    cv2.line(
        result,
        tuple(ex_point_1),
        tuple(point_1),
        4096.0,
        1,
        lineType=cv2.LINE_AA,
        shift=0)
    ex_point_2 = (
        int(
            round(point_2[0] + (point_2[0] - point_1[0]) * (1 + shrink_ratio))),
        int(
            round(point_2[1] + (point_2[1] - point_1[1]) * (1 + shrink_ratio))))
    cv2.line(
        result,
        tuple(ex_point_2),
        tuple(point_2),
        4096.0,
        1,
        lineType=cv2.LINE_AA,
        shift=0)
    return ex_point_1, ex_point_2


def MakeBorderMap(data):
    shrink_ratio = 0.4
    thresh_min = 0.3
    thresh_max = 0.7

    im = data['image']
    text_polys = data['polys']
    ignore_tags = data['ignore_tags']

    canvas = np.zeros(im.shape[:2], dtype=np.float32)
    mask = np.zeros(im.shape[:2], dtype=np.float32)

    for i in range(len(text_polys)):
        if ignore_tags[i]:
            continue
        draw_border_map(
            text_polys[i], canvas, mask=mask, shrink_ratio=shrink_ratio)
    canvas = canvas * (thresh_max - thresh_min) + thresh_min

    data['threshold_map'] = canvas
    data['threshold_mask'] = mask
    return data
