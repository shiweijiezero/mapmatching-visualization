import numpy as np
import math
import pandas as pd
import os
import pickle
import pyprind
from numpy import mat, sqrt
from utils.mygps import GPSDataset
from utils.mymee import MeeDataset
from utils.mymap import MapDataset
from utils.commands import *
from utils.myfilter import *
from shapely.geometry import *
from utils.myutils import *
import numpy as np
from utils.mybase import *
from config import DefaultConfig
import json
import copy
import os
from .mydb import dbsession
from rtree import index
from collections.abc import Iterable
from geopy.distance import geodesic
from haversine import haversine, Unit

x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
Rc = 6378245.0  # 长半轴, 赤道半径
Rj = 6356863.0  # 短半轴， 极半径   +-
ee = 0.00669342162296594323  # 偏心率平方
distance_per_degree = 111200


# def CMF(gps_matched_lines: list, mee_matched_lines: list):
#     """
#     :param matched_line1: [[(pos1),(pos2),...],[(pos2),(pos3),...]],MEE
#     :param matched_line2: [[(pos1),(pos2),...],[(pos2),(pos3),...]],GPS
#     :return: float
#     """

#     tree = index.Rtree()

#     segments1, segments2 = set(), set()
#     corridor_map = {}

#     for line in gps_matched_lines:
#         segments = zip(
#             [myround(pos) for pos in line[0:-1:1]],
#             [myround(pos) for pos in line[1::1]]
#         )  # [(p1,p2),(p2,p3),...]
#         # segments_ = zip(
#         #     [myround(pos) for pos in line[1::2]],
#         #     [myround(pos) for pos in line[0::2]]
#         # )  # [(p1,p2),(p2,p3),...]
#         segments1.update(segments)  # 取出所有线段并去重
#         # segments1.update(segments_)  # 反向的所有线段

#     for line in mee_matched_lines:
#         segments = zip(
#             [myround(pos) for pos in line[0:-1:1]],
#             [myround(pos) for pos in line[1::1]]
#         )  # [(p1,p2),(p2,p3),...]
#         # segments_ = zip(
#         #     [myround(pos) for pos in line[1::2]],
#         #     [myround(pos) for pos in line[0::2]]
#         # )  # [(p1,p2),(p2,p3),...]
#         segments2.update(segments)  # 取出所有线段并去重
#         # segments2.update(segments_)  # 反向的所有线段

#     for idx, x in enumerate(segments2):
#         corridor_map[idx] = Corridor(idx, x[0][0], x[0][1], x[1][0], x[1][1])
#         tree.insert(idx, corridor_map[idx].corridor_MBR)

#     denominator = sum([dis_between_pos(x[0], x[1]) for x in segments1])
#     if (denominator == 0):
#             return -1

#     molecule = 0.0

#     for x in segments1:
#         segment = Segment(0, x[0][0], x[0][1], x[1][0], x[1][1])
#         candidates = list(tree.intersection(segment.rectangle))
#         length = dis_between_pos(x[0], x[1])
#         for candidate in candidates:
#             length -= getLength(segment, corridor_map.get(candidate))
#         molecule += max(0.0, length)

#     return molecule / denominator

def CMF_Debug(gps_matched_lines: list, mee_matched_lines: list, mee_result_rids: list, gps_result_rids: list,
              corridor_width):
    """
    :param matched_line1: [[(pos1),(pos2),...],[(pos2),(pos3),...]],MEE
    :param matched_line2: [[(pos1),(pos2),...],[(pos2),(pos3),...]],GPS
    :return: float
    """

    p = index.Property()
    p.type = 0  # RT_RTree = 0; RT_MVRTree = 1; RT_TPRTree = 2
    tree = index.Rtree(properties=p)

    segments1, segments2 = [], []
    corridor_map = {}

    for line in gps_matched_lines:
        segments = zip(
            [myround(pos) for pos in line[0:-1:1]],
            [myround(pos) for pos in line[1::1]]
        )  # [(p1,p2),(p2,p3),...]
        # segments_ = zip(
        #     [myround(pos) for pos in line[1::2]],
        #     [myround(pos) for pos in line[0::2]]
        # )  # [(p1,p2),(p2,p3),...]
        segments1.extend(segments)  # 取出所有线段并去重
        # segments1.update(segments_)  # 反向的所有线段

    for line in mee_matched_lines:
        segments = zip(
            [myround(pos) for pos in line[0:-1:1]],
            [myround(pos) for pos in line[1::1]]
        )  # [(p1,p2),(p2,p3),...]
        # segments_ = zip(
        #     [myround(pos) for pos in line[1::2]],
        #     [myround(pos) for pos in line[0::2]]
        # )  # [(p1,p2),(p2,p3),...]
        segments2.extend(segments)  # 取出所有线段并去重
        # segments2.update(segments_)  # 反向的所有线段

    print("Length GPS lines: %s ids: %s" % (str(len(segments1)), str(len(gps_result_rids))))
    print("Length MEE lines: %s ids: %s" % (str(len(segments2)), str(len(mee_result_rids))))
    if len(segments2) <= 2:
        return -1

    start = segments2[0]
    end = segments2[-1]
    preLine1, preLine2 = getCorridorLines(corridor_width, start[0][0], start[0][1], start[1][0], start[1][1])
    preSegment = start
    segments2.append((end[1], (end[1][0] + end[1][0] - end[0][0], end[1][1] + end[1][1] - end[0][1])))

    for rid, x in zip(mee_result_rids, segments2[1:]):
        curLine1, curLine2 = getCorridorLines(corridor_width, x[0][0], x[0][1], x[1][0], x[1][1], x[0][0], x[0][1],
                                              preLine1, preLine2)
        corridor_map[rid] = Corridor(rid, preSegment[0][0], preSegment[0][1], preSegment[1][0], preSegment[1][1],
                                     corridor_width=corridor_width,
                                     points=[preLine1[0], preLine2[0], curLine1[0], curLine2[0]],
                                     )
        tree.insert(rid, corridor_map[rid].corridor_MBR)
        preLine1, preLine2, preSegment = curLine1, curLine2, x

    print("rtree objects == segments: ", tree.count(tree.get_bounds()) == len(mee_result_rids))
    denominator = sum([dis_between_pos(x[0], x[1]) for x in segments1])
    if (denominator == 0):
        return -1

    molecule = 0.0

    misMatchedLength = []
    matchdetail = []
    for rid, x in zip(gps_result_rids, segments1):
        segment = Segment(0, x[0][0], x[0][1], x[1][0], x[1][1])
        candidates = list(tree.intersection(segment.rectangle))

        length = dis_between_pos(x[0], x[1])
        detail = "Total Length: %0.3f. " % length
        for candidate in candidates:
            l = getLength(segment, corridor_map.get(candidate))
            detail += "%s--%0.3f; " % (str(candidate), l)
            length -= l
        molecule += max(0.0, length)
        misMatchedLength.append(max(0.0, length) / dis_between_pos(x[0], x[1]))
        matchdetail.append(detail)

    return molecule / denominator, misMatchedLength, matchdetail


def CMF(gps_matched_lines: list, mee_matched_lines: list, corridor_width):
    """
    :param matched_line1: [[(pos1),(pos2),...],[(pos2),(pos3),...]],MEE
    :param matched_line2: [[(pos1),(pos2),...],[(pos2),(pos3),...]],GPS
    :return: float
    """
    tree = index.Rtree()

    segments1, segments2 = [], []
    corridor_map = {}

    for line in gps_matched_lines:
        segments = zip(
            [myround(pos) for pos in line[0:-1:1]],
            [myround(pos) for pos in line[1::1]]
        )  # [(p1,p2),(p2,p3),...]
        # segments_ = zip(
        #     [myround(pos) for pos in line[1::2]],
        #     [myround(pos) for pos in line[0::2]]
        # )  # [(p1,p2),(p2,p3),...]
        segments1.extend(segments)  # 取出所有线段并去重
        # segments1.update(segments_)  # 反向的所有线段

    for line in mee_matched_lines:
        segments = zip(
            [myround(pos) for pos in line[0:-1:1]],
            [myround(pos) for pos in line[1::1]]
        )  # [(p1,p2),(p2,p3),...]
        # segments_ = zip(
        #     [myround(pos) for pos in line[1::2]],
        #     [myround(pos) for pos in line[0::2]]
        # )  # [(p1,p2),(p2,p3),...]
        segments2.extend(segments)  # 取出所有线段并去重
        # segments2.update(segments_)  # 反向的所有线段

    if len(segments2) <= 2:
        return -1

    start = segments2[0]
    end = segments2[-1]
    preLine1, preLine2 = getCorridorLines(corridor_width, start[0][0], start[0][1], start[1][0], start[1][1])
    preSegment = start
    segments2.append((end[1], (end[1][0] + end[1][0] - end[0][0], end[1][1] + end[1][1] - end[0][1])))

    for idx, x in enumerate(segments2[1:]):
        curLine1, curLine2 = getCorridorLines(corridor_width, x[0][0], x[0][1], x[1][0], x[1][1], x[0][0], x[0][1],
                                              preLine1, preLine2)
        corridor_map[idx] = Corridor(idx, preSegment[0][0], preSegment[0][1], preSegment[1][0], preSegment[1][1],
                                     corridor_width=corridor_width,
                                     points=[preLine1[0], preLine2[0], curLine1[0], curLine2[0]])
        tree.insert(idx, corridor_map[idx].corridor_MBR)
        preLine1, preLine2, preSegment = curLine1, curLine2, x

    denominator = sum([dis_between_pos(x[0], x[1]) for x in segments1])
    if (denominator == 0):
        return -1

    molecule = 0.0

    for x in segments1:
        segment = Segment(0, x[0][0], x[0][1], x[1][0], x[1][1])
        candidates = list(tree.intersection(segment.rectangle))
        length = dis_between_pos(x[0], x[1])
        for candidate in candidates:
            length -= getLength(segment, corridor_map.get(candidate))
        molecule += max(0.0, length)

    return molecule / denominator


def get_corridor_lines(lines, corridor_width):
    if len(lines) <= 1:
        return []
    upLine = []
    downLine = []
    k = []
    # print("Road Line: ", lines)
    pre_pre_point = [None, None]
    preLine1, preLine2, pre_point = None, None, lines[0]
    for i in range(1, len(lines)):
        cur_point = lines[i]
        curLine1, curLine2 = getCorridorLines(corridor_width, pre_point[0], pre_point[1], cur_point[0], cur_point[1],
                                              pre_point[0], pre_point[1], preLine1, preLine2)
        k.append(curLine1[1])
        # if preLine1 != None:
        #     x1, y1, x2, y2 = curLine1[0][0] - preLine1[0][0], curLine1[0][1] - preLine1[0][1], curLine2[0][0] - preLine2[0][0], curLine2[0][1] - preLine2[0][1]
        #     t = abs(x1*y2 - x2*y1)
        #     parallel.append(t)
        #     if t > 10e-5:
        #         curLine1, curLine2 = curLine2, curLine1
        upLine.append(curLine1[0])

        downLine.append(curLine2[0])
        preLine1, preLine2 = curLine1, curLine2
        pre_pre_point = pre_point
        pre_point = cur_point

    curLine1, curLine2 = getCorridorLines(corridor_width, pre_point[0], pre_point[1],
                                          pre_point[0] + (pre_point[0] - pre_pre_point[0]),
                                          pre_point[1] + (pre_point[1] - pre_pre_point[1]))
    upLine.append(curLine1[0])
    downLine.append(curLine2[0])
    # x1, y1, x2, y2 = curLine1[0][0] - preLine1[0][0], curLine1[0][1] - preLine1[0][1], curLine2[0][0] - preLine2[0][0], curLine2[0][1] - preLine2[0][1]
    # if abs(x1*y2 - x2*y1) < 10e-5:
    #     upLine.append(curLine1[0])
    #     downLine.append(curLine2[0])
    # else:
    #     upLine.append(curLine2[0])
    #     downLine.append(curLine1[0])

    upLine.extend(downLine[::-1])
    upLine.append(upLine[0])
    # print("Corridor Line: ", upLine)
    # print(k)
    return upLine


# def get_corridor_lines(lines):
#     if len(lines) <= 1:
#         return []
#     upLine = []  #走廊上边缘
#     downLine = []  #走廊下边缘
#     pre_point = lines[0]
#     [pre_down, pre_up, lat_up, lat_down]  = getExactCorridor(lines[0][0], lines[0][1], lines[1][0], lines[1][1])
#     upLine.append(pre_up)
#     downLine.append(pre_down)

#     for cur_point in lines[1:]:
#         [pre_down, pre_up, cur_up, cur_down]  = getExactCorridor(pre_point[0], pre_point[1], cur_point[0], cur_point[1])
#         upLine.append(cur_up)
#         downLine.append(cur_down)
#         pre_point = cur_point

#     upLine.extend(downLine[::-1])
#     upLine.append(upLine[0])

#     return upLine

def getApproximateCorridor(lon1, lat1, lon2, lat2, dis):
    lat_change = dis / distance_per_degree
    lon_change = math.abs(math.cos(lat_change * (math.PI / 180)))
    x, y = abs(lon2 - lon1), abs(lat2 - lat1)
    if x == 0 or y / x > 1:
        return [(lon1 - lon_change, lat1), (lon1 + lon_change, lat1),
                (lon2 - lon_change, lat2), (lon2 + lon_change, lat2)]
    else:
        return [(lon1, lat1 - lat_change), (lon1, lat1 + lat_change),
                (lon2, lat2 - lat_change), (lon2, lat2 + lat_change)]


def getExactCorridor(lon1, lat1, lon2, lat2, dis):
    x, y = abs(lon2 - lon1), abs(lat2 - lat1)
    lon1_min, lon1_max, lat1_min, lat1_max = get_bounding_box(dis, lat1, lon1)
    lon2_min, lon2_max, lat2_min, lat2_max = get_bounding_box(dis, lat2, lon2)

    if x == 0 or y / x > 1:
        return [(lon1_min, lat1), (lon1_max, lat1),
                (lon2_max, lat2), (lon2_min, lat2)]
    else:
        return [(lon1, lat1_min), (lon1, lat1_max),
                (lon2, lat2_max), (lon2, lat2_min)]


def get_bounding_box(distance, latittude, longitude):
    latittude = np.radians(latittude)
    longitude = np.radians(longitude)
    angular_distance = distance / Rc

    lat_min = latittude - angular_distance
    lat_max = latittude + angular_distance

    delta_longitude = np.arcsin(np.sin(angular_distance) / np.cos(latittude))
    lon_min = longitude - delta_longitude
    lon_max = longitude + delta_longitude
    lon_min = np.degrees(lon_min)
    lat_max = np.degrees(lat_max)
    lon_max = np.degrees(lon_max)
    lat_min = np.degrees(lat_min)
    return lon_min, lon_max, lat_min, lat_max


def angleFromCoordinate(lng1, lat1, lng2, lat2):
    # 线段方向与正北方向顺时针夹角
    dLon = (lng2 - lng1)
    x = math.sin(dLon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)

    brng = math.atan2(x, y)

    brng = math.degrees(brng)
    brng = (brng + 360) % 360
    return brng


def getPointWithBearing(distance, lng, lat, bearing):
    p = geodesic(kilometers=distance / 1000.0).destination((lat, lng), bearing)  # (lat, lng)
    return (p[1], p[0])  # 返回（lng, lat）


def getVerticalPoints(distance, lng1, lat1, lng2, lat2):
    x, y = lng2 - lng1, lat2 - lat1
    angular_distance = distance / Rc
    if x == 0:
        angle = 0
    else:
        angle = angleFromCoordinate(lng1, lat1, lng2, lat2)

    p1 = getPointWithBearing(distance, lng1, lat1, angle + 90)
    p2 = getPointWithBearing(distance, lng1, lat1, angle - 90)
    return [p1, p2]


def getParallelLines(distance, lng1, lat1, lng2, lat2):
    p1, p2 = getVerticalPoints(distance, lng1, lat1, lng2, lat2)
    k = (lat2 - lat1) / (lng2 - lng1) if (lng2 - lng1) != 0 else None  # 获取斜率，不存在为None
    return [(p1, k), (p2, k)]


def pointToPointDistance(lon1, lat1, lon2, lat2):
    return dis_between_pos((lon1,lat1),(lon2,lat2))
    # return haversine((lon1, lat1), (lon2, lat2), unit=Unit.KILOMETERS)


def getIntersectionPoint(line1, line2):
    # line format：（point, k）, point:(lng, lat)
    if line1[1] == None:
        line1, line2 = line2, line1
    k1 = line1[1]
    k2 = line2[1]

    # if k1 == k2 or (k2==None and abs(k1)>8) or (abs(k2 - k1) < 10e-4) or (abs(k1)>5 and abs(k2) > 5) or (abs(k1) < 0.2 and abs(k2) < 0.2):
    if k1 == k2:
        return line2[0]
    else:
        b1 = line1[0][1] - k1 * line1[0][0]  # b = y-kx

        if k2 == None:  # Line2直线斜率不存在操作
            b2 = 0
        else:
            b2 = line2[0][1] - k2 * line2[0][0]  # b = y-kx

        if k2 == None:
            x = line2[0][0]
        else:
            x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0

        return (x, y)


def getDiagonalIntersections(line1, line2, line3, line4):
    p1 = getIntersectionPoint(line1, line3)
    p2 = getIntersectionPoint(line2, line4)
    p3 = getIntersectionPoint(line1, line4)
    p4 = getIntersectionPoint(line2, line3)
    return [(p1, p2), (p3, p4)]


def getCorridorLines(distance, lng1, lat1, lng2, lat2, lng3=None, lat3=None, line1=None, line2=None):
    line3, line4 = getParallelLines(distance, lng1, lat1, lng2, lat2)
    if line1 == None or line2 == None:
        return line3, line4
    else:
        k = line3[1]
        points = getDiagonalIntersections(line1, line2, line3, line4)

        # triangle = [
        #     (Point(lng3, lat3), Point(lng1, lat1)),
        #     (Point(lng1, lat1), Point(lng2, lat2)),
        #     (Point(lng2, lat2), Point(lng3, lat3))
        # ]

        # 两组点的选择  方式： 判断角
        pre = (lng3, lat3)
        center = (lng1, lat1)
        latter = (lng2, lat2)
        if isTargetPoint(pre, center, latter, points[0][0]) or isTargetPoint(pre, center, latter, points[0][1]):
            return (points[0][0], k), (points[0][1], k)
        else:
            return (points[1][0], k), (points[1][1], k)


def isTargetPoint(pre, center, latter, target):
    v_pre = (pre[0] - center[0], pre[1] - center[1])
    v_target = (target[0] - center[0], target[1] - center[1])
    v_latter = (latter[0] - center[0], latter[1] - center[1])
    return isAcuteAngle(v_pre, v_target) and isAcuteAngle(v_latter, v_target)


def isAcuteAngle(vector1, vector2):
    return vector1[0] * vector2[0] + vector1[1] * vector2[1] >= 0


def cross(p1, p2, p3):  # 跨立实验
    lng1 = p2.lng - p1.lng
    lat1 = p2.lat - p1.lat
    lng2 = p3.lng - p1.lng
    lat2 = p3.lat - p1.lat
    return lng1 * lat2 - lng2 * lat1


def findIntersectionPoint(line1, line2):
    if line1[0].lng - line1[1].lng == 0:
        line1, line2 = line2, line1  # 保证line存在斜率
    x1 = line1[0].lng  # 取四点坐标
    y1 = line1[0].lat
    x2 = line1[1].lng
    y2 = line1[1].lat

    x3 = line2[0].lng
    y3 = line2[0].lat
    x4 = line2[1].lng
    y4 = line2[1].lat

    k1 = (y2 - y1) * 1.0 / (x2 - x1)
    b1 = y1 * 1.0 - x1 * k1 * 1.0

    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return (x, y)


def isRayIntersectsSegment(point, line):  # [x,y] [lng,lat]
    # 判断点point发出的射线是否与line相交：判断点，边
    if line[0].lat == line[1].lat:  # 排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if line[0].lat > point.lat and line[1].lat > point.lat:  # 线段在射线上边
        return False
    if line[0].lat < point.lat and line[1].lat < point.lat:  # 线段在射线下边
        return False
    if line[0].lng < point.lng and line[1].lng < point.lng:  # 线段在射线左边
        return False

    xseg = line[1].lng - (line[1].lng - line[0].lng) * (line[1].lat - point.lat) / (line[1].lat - line[0].lat)  # 求交
    if xseg < point.lng:  # 交点在射线起点的左侧
        return False
    return True


def isOnLine(point, line):
    x1 = line[0].lng - point.lng
    y1 = line[0].lat - point.lat
    x2 = line[1].lng - point.lng
    y2 = line[1].lat - point.lat
    return (x1 * y2 - x2 * y1) == 0 and x1 * x2 <= 0 and y1 * y2 <= 0


def isInside(point, QuardAngle):
    # 判断点是否在四边形内, 判断边上 和 判断内部
    # 判断边上
    for line in QuardAngle:
        if isOnLine(point, line):
            return True
    # 判断内部
    num = 0
    for line in QuardAngle:
        if isRayIntersectsSegment(point, line):
            num += 1
    return num % 2 == 1


def getLength(segment, corridor):
    # 计算线段在走廊中的长度
    targetLine = (segment.start, segment.end)
    parallelQuardAngle = corridor.corridor
    QuardAngle = [(parallelQuardAngle[0], parallelQuardAngle[1]),
                  (parallelQuardAngle[1], parallelQuardAngle[2]),
                  (parallelQuardAngle[2], parallelQuardAngle[3]),
                  (parallelQuardAngle[3], parallelQuardAngle[0])]
    # 判断线段与四条边的相交情况
    intersection = [isIntersection(targetLine, _) for _ in QuardAngle]
    # 计算交点，考虑交点在四边形顶点处（对交点进行去重）
    intersectionPoints = set()
    for flag, line in zip(intersection, QuardAngle):
        if flag:
            intersectionPoints.add(findIntersectionPoint(targetLine, line))
    # 计算交点数
    num = len(intersectionPoints)
    if num == 0:  # 交点为0， 线段在四边形内部或者在四边形外
        if isInside(targetLine[0], QuardAngle) and isInside(targetLine[1], QuardAngle):
            return pointToPointDistance(targetLine[0].lng, targetLine[0].lat, targetLine[1].lng, targetLine[1].lat)
        else:
            return 0.0
    if num == 1:  # 交点为1， 线段一点在四边形内部，另一点在四边形外
        p = intersectionPoints.pop()
        if isInside(targetLine[0], QuardAngle):
            return pointToPointDistance(targetLine[0].lng, targetLine[0].lat, p[0], p[1])
        else:
            return pointToPointDistance(targetLine[1].lng, targetLine[1].lat, p[0], p[1])
    if num == 2:  # 交点为2， 线段跨越四边形
        p1 = intersectionPoints.pop()
        p2 = intersectionPoints.pop()
        return pointToPointDistance(p1[0], p1[1], p2[0], p2[1])
    return 0.0


def isIntersection(line1, line2):
    # 判断两线段是否相交， 这里两条线段重合时判定为不相交
    if max(line1[0].lng, line1[1].lng) <= min(line2[0].lng, line2[1].lng) \
            or max(line1[0].lat, line1[1].lat) <= min(line2[0].lat, line2[1].lat) \
            or max(line2[0].lng, line2[1].lng) <= min(line1[0].lng, line1[1].lng) \
            or max(line2[0].lat, line2[1].lat) <= min(line1[0].lat, line1[1].lat):
        return False
    if cross(line1[0], line1[1], line2[0]) * cross(line1[0], line1[1], line2[1]) <= 0 \
            and cross(line2[0], line2[1], line1[0]) * cross(line2[0], line2[1], line1[1]) <= 0:
        return True
    return False


class Point():
    def __init__(self, lng, lat):
        self.lng = float(lng)
        self.lat = float(lat)


class Segment():
    def __init__(self, id, lng1, lat1, lng2, lat2):
        self.id = id
        self.start = Point(lng1, lat1)
        self.end = Point(lng2, lat2)
        # 线段外接矩形形式 [minlng, minlat, maxlng, maxlat]
        self.rectangle = (min(lng1, lng2), min(lat1, lat2),
                          max(lng1, lng2), max(lat1, lat2))


class Corridor(Segment, ):

    def __init__(self, id, lng1, lat1, lng2, lat2, corridor_width, points=None):
        super(Corridor, self).__init__(id, lng1, lat1, lng2, lat2)
        if points == None:
            p1, p2, p3, p4 = getExactCorridor(lng1, lat1, lng2, lat2, corridor_width)
            self.corridor = [Point(p1[0], p1[1]),
                             Point(p2[0], p2[1]),
                             Point(p3[0], p3[1]),
                             Point(p4[0], p4[1])]
        else:
            p1, p2, p3, p4 = points
            x1, x2, y1, y2 = p3[0] - p1[0], p4[0] - p2[0], p3[1] - p1[1], p4[1] - p2[1]
            if abs(x1 * y2 - x2 * y1) < 10e-5:
                self.corridor = [Point(*p1), Point(*p3), Point(*p4), Point(*p2)]
            else:
                self.corridor = [Point(*p1), Point(*p4), Point(*p3), Point(*p2)]

        # corridor外接矩形形式 [minlng, minlat, maxlng, maxlat]
        self.corridor_MBR = (
            min(self.corridor[0].lng, self.corridor[1].lng, self.corridor[2].lng, self.corridor[3].lng),
            min(self.corridor[0].lat, self.corridor[1].lat, self.corridor[2].lat, self.corridor[3].lat),
            max(self.corridor[0].lng, self.corridor[1].lng, self.corridor[2].lng, self.corridor[3].lng),
            max(self.corridor[0].lat, self.corridor[1].lat, self.corridor[2].lat, self.corridor[3].lat))



# def shapely_cmf(gps_points, polygon_lines):
#    polygon_points = [x[0] for x in polygon_lines]
#    polygon_points.append(polygon_lines[-1][1])
#
#    intersection_length = 0
#    line_length = LineString(gps_points).length
#
#    zone = Polygon(LineString(polygon_points))
#    for i in range(len(gps_points) - 1):
#        one_line = LineString(gps_points[i:i + 2])
#        one_intersection = one_line.intersection(zone)
#        intersection_length += one_intersection.length
#
#    return (line_length - intersection_length) / line_length


def get_level(hierarchy, curr_index):
    """
    返回当前节点的层级
    """
    level = 0
    while hierarchy[curr_index][3] != -1:
        curr_index = hierarchy[curr_index][3]
        level += 1
    return level

#处理树信息数组的函数
def operate_tree(contours, hierarchy):
    #hierarchy中的元素形如[next_sibling, previous_sibling, first_child, father]

    hole_polygons = {}

    for i in range(len(hierarchy)):
        curr_level = get_level(hierarchy, i)
        #对第一层和第二层的轮廓进行操作
        if curr_level == 1:
            if i not in hole_polygons:
                hole_polygons[i] = dict()
            hole_polygons[i]["outer"] = contours[i]
        elif curr_level == 2:
            father_index = hierarchy[i][3]
            if "holes" not in hole_polygons[father_index]:
                hole_polygons[father_index]["holes"] = list()
            hole_polygons[father_index]["holes"].append(contours[i])

    return hole_polygons



def shapely_cmf(gps_points, contours):
    """
    计算CMF, 如果有错误发生, 返回-1
    gps_points: gps经纬度坐标点的列表
    contours: operate_tree的返回结果, 处理过的轮廓的字典
    """
    hole_polygons = []
    for hole_polygon in contours.values():
        if "holes" in hole_polygon:
            hole_polygons.append(Polygon(hole_polygon["outer"], hole_polygon["holes"]))
        else:
            hole_polygons.append(Polygon(hole_polygon["outer"]))

    intersection_length = 0
    line_length = LineString(gps_points).length

    try:
        for j in range(len(hole_polygons)):
            zone = hole_polygons[j]
            for i in range(len(gps_points)-1):
                one_line = LineString(gps_points[i:i+2])
                one_intersection = one_line.intersection(zone)
                intersection_length += one_intersection.length
    except BaseException as data:
        print("================================================================")
        print("Error!")
        print(data)
        print("Function exited. Will return -1.")
        print("================================================================")
        return -1

    return (line_length - intersection_length) / line_length
