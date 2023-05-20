from typing import List, Tuple, Set, Dict
from .edge import Edge
from .vertex import Vertex
from .graph import Graph
from .polygon import *
from .vector import D2Vector, D3Vector
import math

#计算一条线段的斜率
def cal_slope(v1_s: Vertex, v1_e: Vertex) -> float:
    slope = None
    #计算斜率
    #防止除零错误
    if v1_s.x == v1_e.x:
        slope = float('inf')
    else:
        slope = (v1_s.y - v1_e.y) / (v1_s.x - v1_e.x)
    return slope

#判断两条线段是否平行
def is_parallel(v1_s: Vertex, v1_e: Vertex, v2_s: Vertex, v2_e: Vertex) -> bool:
    return cal_slope(v1_s, v1_e) == cal_slope(v2_s, v2_e)

#计算两个二维向量的叉积
def vector_product_D2Vector(v1: D2Vector, v2: D2Vector) -> D3Vector:
    return D3Vector(v1.y * v2.z - v2.y * v1.z,
                 -1 * (v1.x * v2.z - v2.x * v1.z),
                 v1.x * v2.y - v2.x * v1.y)

#通过两个点生成二维向量
def make_D2Vector(v1_s: Vertex, v1_e: Vertex):
    return D2Vector(v1_e.x - v1_s.x,
                 v1_e.y - v1_s.y)

#判断两条线段是否相交于一点, 返回判断结果和交点
def is_intersected(v1_s: Vertex, v1_e: Vertex, v2_s: Vertex, v2_e: Vertex) -> Tuple[bool, Vertex]:
    def is_intersected_helper(v1_s: Vertex, v1_e: Vertex, v2_s: Vertex, v2_e: Vertex) -> [bool, Vertex]:
        #剪枝
        #TODO
        #先看下是否平行
        #如果是平行, 那么两线段只有两种可能:
        #	1.他们无交点
        #	2.他们重合或部分重合, 即无数个交点
        #无论是哪种情况, 都不把他作为普通的相交与一点考虑

        if is_parallel(v1_s, v1_e, v2_s, v2_e):
            return [False, None]

        #两线不平行
        #先考虑边界情况:
        #	1.是否可能两线段共用了一个交点, 那个点就是交点
        if v1_s == v2_s or v1_s == v2_e:
            return [True, v1_s.copy()]
        elif v1_e == v2_s or v1_e == v2_e:
            return [True, v1_e.copy()]

        #通过向量叉积判断是否相交
        v_seg1 = make_D2Vector(v1_s, v1_e)
        v_seg2 = make_D2Vector(v2_s, v2_e)

        v1 = make_D2Vector(v1_s, v2_s)
        v2 = make_D2Vector(v1_s, v2_e)
        v3 = make_D2Vector(v2_s, v1_s)
        v4 = make_D2Vector(v2_s, v1_e)

        #计算判断线段2的两个点是否分别在线段1的两边
        D3V_1 = vector_product_D2Vector(v1, v_seg1)
        D3V_2 = vector_product_D2Vector(v2, v_seg1)

        maybe = None
        #如果两次算出来的三维向量有一个的z值为0, 说明可能有交点,
        #且可以确定如果真的有交点, 那它就是线段2的一个端点
        if D3V_1.z == 0:
            maybe = v2_s.copy()
            #如果v2起点疑似, 就要换俩向量去考察线段1 否则永远结果是0
            v3 = make_D2Vector(v2_e, v1_s)
            v4 = make_D2Vector(v2_e, v1_e)
        elif D3V_2.z == 0:
            maybe = v2_e.copy()
        #如果两者算出来异号 那么说明线段2的两个端点分列线段1所在的直线的两侧
        #那么需要继续考察线段1的两个端点是否在线段2所在直线的两次
        #换句话说, 如果这里同号就肯定不行了
        if (D3V_1.z > 0 and D3V_2.z > 0) or (D3V_1.z < 0 and D3V_2.z < 0):
            return [False, None]

        #继续考察
        D3V_3 = vector_product_D2Vector(v3, v_seg2)
        D3V_4 = vector_product_D2Vector(v4, v_seg2)
        #如果前面出现了疑似点 那这里必不会再出现 否则线段1就和2平行了
        #所以这里的v3必定是make_D2Vector(v2_s, v1_s)的结果
        #同理v4必定是make_D2Vector(v2_s, v1_e)的结果
        #而如果这里出现疑似点, 前面已经证明了线段2的两个端点分局线段1所在直线两侧
        #说明这就是交点了
        if D3V_3.z == 0:
            return [True, v1_s.copy()]
        elif D3V_4.z == 0:
            return [True, v1_e.copy()]
        if (D3V_3.z > 0 and D3V_4.z > 0) or (D3V_3.z < 0 and D3V_4.z < 0):
            return [False, None]

        #如果成功到这.  说明必有一个交点
        #如果有过一个可能点 那么就是它 否则重新计算交点
        #直接计算两条线段分别所在的两条直线的交点即可
        if maybe:
            return [True, maybe]
        else:
            return [True, cal_intersection(v1_s, v1_e, v2_s, v2_e)]

    ans = is_intersected_helper(v1_s,v1_e,v2_s,v2_e)
    if ans[0]:
        ratio1 = round(((ans[1].x - v1_s.x) ** 2 + (ans[1].y - v1_s.y) ** 2) ** 0.5 / ((v1_e.x - v1_s.x) ** 2 + (v1_e.y - v1_s.y) ** 2) ** 0.5,3)
        ratio2 = round(((ans[1].x - v2_s.x) ** 2 + (ans[1].y - v2_s.y) ** 2) ** 0.5 / ((v2_e.x - v2_s.x) ** 2 + (v2_e.y - v2_s.y) ** 2) ** 0.5,3)
        ans[1].s_num = [v1_s.s_num + ratio1,v2_s.s_num + ratio2]
    return ans

#计算两条不平行线段的交点
def cal_intersection(v1_s: Vertex, v1_e: Vertex, v2_s: Vertex, v2_e: Vertex) -> Vertex:
    ans = Vertex(0, 0)
    slope1 = cal_slope(v1_s, v1_e)
    slope2 = cal_slope(v2_s, v2_e)
    if slope1 == 0:
        ans.y = v1_s.y
        ans.x = (v1_s.y - v2_s.y) / slope2 + v2_s.x
    elif slope2 == 0:
        ans.y = v2_s.y
        ans.x = (v2_s.y - v1_s.y) / slope1 + v1_s.x
    elif slope1 == float("inf"):
        ans.x = v1_s.x
        ans.y = slope2 * (ans.x - v2_s.x) + v2_s.y
    elif slope2 == float("inf"):
        ans.x = v2_s.x
        ans.y = slope1 * (ans.x - v1_s.x) + v1_s.y
    else:
        ans.x = (v2_s.y - v1_s.y + slope1 * v1_s.x - slope2 * v2_s.x) / (slope1 - slope2)
        ans.y = slope1 * (v2_s.y - v1_s.y - slope2 * (v2_s.x - v1_s.x))
    return ans

#找到所有交点
def find_intersections(p1: Polygon, p2: Polygon) -> Set[Vertex]:
    ans = []
    for edge1 in p1.edges:
        for edge2 in p2.edges:
            #剪枝
            #TODO
            result = is_intersected(edge1.start_vertex, edge1.end_vertex, edge2.start_vertex, edge2.end_vertex)
            if result[0]:
                ans.append(result[1])
    return ans

#通过点依次描边构造一个多边形
def make_polygon(s_num:int,*points:[float,float]):
    vertices = []
    edges = []
    for point in points:
        vertices.append(Vertex(point[0], point[1],s_num))
        s_num += 1
    length = len(vertices)
    for i in range(length):
        edges.append(Edge(vertices[i],vertices[(i + 1) % length]))
    return [Polygon(*edges),s_num]

#连接边
def connect_vertices(graph:Graph,vertices:List[Vertex]):
    length = len(vertices)
    for i in range(length):
        graph.add_one_edge(vertices[i],vertices[(i + 1) % length])

#找到最左的顶点, 原算法找的最左下, 不是一个好选择
def find_most_left_vertex(vertices:List[Vertex]):
    min_x = float("inf")
    min = None
    for vertex in vertices:
        if vertex.x < min_x:
            min_x = vertex.x
            min = vertex
    return min

#通过二维向量的点乘反推夹角, 夹角大小为[0,180]
def get_angle(v1: D2Vector, v2: D2Vector) -> float:
    #如果有零向量, 规定它和另一个的夹角是180度, 即无论如何最后选它
    if v1.get_module() == 0 or v2.get_module() == 0:
        return 180.0
    #这里要round一下防止精度错误
    return math.acos(round((v1.x * v2.x + v1.y * v2.y) / v1.get_module() / v2.get_module(),6)) / math.pi * 180.0

#计算从向量1到向量2要旋转的逆时针角度, 夹角大小为[0,360)
def cal_CCW_angle(v1:D2Vector,v2:D2Vector):
    if vector_product_D2Vector(v1,v2).z >= 0:
        return get_angle(v1,v2)
    elif vector_product_D2Vector(v1,v2).z < 0:
        return 360 - get_angle(v1,v2)

#生成公共轮廓
def make_contour(graph:Graph,most_left_vertex:Vertex):
    #上一个点
    last_vertex = most_left_vertex
    #当前的点
    curr_vertex = None
    #答案
    
    for vertex in graph.matrix[most_left_vertex]:
        if graph.matrix[most_left_vertex][vertex] == "unvisited":
            print(vertex)
            curr_vertex = vertex
            graph.matrix[most_left_vertex][vertex] = "visited"
            break
    #当前边, 注意, 永远是以新的点向外辐射向量
    curr_edge = make_D2Vector(curr_vertex, last_vertex)
    ans = [[last_vertex.x,last_vertex.y], [curr_vertex.x,curr_vertex.y]]

    while curr_vertex != most_left_vertex:
        #下一个候选点
        candidate = []
        for vertex in graph.matrix[curr_vertex]:
            if graph.matrix[curr_vertex][vertex] == "unvisited" and vertex != last_vertex:
                one_edge = make_D2Vector(curr_vertex,vertex)
                candidate.append([vertex,cal_CCW_angle(curr_edge, one_edge)])
        #选择逆时针角度最**小**的, 这和原算法描述的不同
        candidate.sort(key=lambda x:x[1])
        last_vertex = curr_vertex
        curr_vertex = candidate[0][0]
        graph.matrix[last_vertex][curr_vertex] = "visited"
        curr_edge = make_D2Vector(curr_vertex,last_vertex)
        ans.append([curr_vertex.x,curr_vertex.y])
    return ans[:-1]


def get_rectangle(segement, half_width):
    """
    由线段得到矩形

    segement: 一条线段 两个端点
    half_width: 指要生成的矩形的半宽(即两边扩开的半径), 传入的应该是米为单位

    return: 一个列表, 矩形的逆时针排序的四个顶点
    """
    v_1 = Vertex(segement[0][0], segement[0][1])
    v_2 = Vertex(segement[1][0], segement[1][1])

    vec_segment = make_D2Vector(v_1, v_2)

    # 得到当前向量的角
    # 注意, 后面方法要求的是顺时针(CW)旋转角度, 所以要360减一下
    basis_angle = 360 - cal_CCW_angle(D2Vector(0, 1), vec_segment)

    # 旋转得到四个矩形的顶点, 基础角 + 新角即可
    p1 = get_point(segement[0][0], segement[0][1], half_width, (basis_angle + 90) % 360)
    p2 = get_point(segement[0][0], segement[0][1], half_width, (basis_angle + 270) % 360)
    p3 = get_point(segement[1][0], segement[1][1], half_width, (basis_angle + 270) % 360)
    p4 = get_point(segement[1][0], segement[1][1], half_width, (basis_angle + 90) % 360)

    return [p1, p2, p3, p4]