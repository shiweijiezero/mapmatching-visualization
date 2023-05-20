from typing import List, Tuple, Set, Dict
from .edge import Edge

#多边形
class Polygon:
    def __init__(self, *edges: Edge):
        self.edges = []
        for edge in edges:
            self.edges.append(edge)

    def __str__(self):
        ans = ""
        count = 0
        for edge in self.edges:
            ans += "{: >2}. ".format(count) + edge.__str__() + "\n"
            count += 1
        return ans[:-1]

    def get_vertices(self):
        ans = set()
        for edge in self.edges:
            ans.add(edge.start_vertex)
        return ans


import geopy
import geopy.distance


def get_point(lon, lat, distance, direction):
    """
    根据经纬度，距离，方向获得一个地点
    :param lon: 经度
    :param lat: 纬度
    :param distance: 距离（米）
    :param direction: 方向, 正北方向为0的顺时针旋转角度（北：0，东：90，南：180，西：270）
    :return: 一个列表: 经度, 纬度
    """
    start = geopy.Point(lat, lon)
    d = geopy.distance.distance(kilometers=distance/1000.0)
    end = d.destination(point=start, bearing=direction)
    return [round(end.longitude,10), round(end.latitude,10)]