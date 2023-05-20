import traj_dist.distance as tdist
import numpy as np
import pandas
from utils.myutils import dis_between_pos, getCrossPoint
from scipy.spatial import ConvexHull, convex_hull_plot_2d

# import matplotlib.pyplot as plt


import math


# def LIP(poly1, poly2):
#
#     pset = poly2+poly1
#     hull = ConvexHull(np.array(pset))
#
#     return hull.vertices


def poly_intersection(poly1, poly2):
    intsecs = []

    area = []  # 每个凸包区域

    add_head1 = 0
    add_head2 = 0
    add_tail1 = 0
    add_tail2 = 0

    for i, p1_first_point in enumerate(poly1[:-1]):
        p1_second_point = poly1[i + 1]
        # area.append(p1_first_point)

        add_tail1 += 1
        add_tail2 = add_head2  # 每次内循环需要重置二号标志

        for j, p2_first_point in enumerate(poly2[:-1]):
            # print(j)
            if (j < add_head2):
                continue

            # 利用1号的某个线段逐个扫描2号的每个线段

            p2_second_point = poly2[j + 1]
            add_tail2 += 1

            intersection_result = getCrossPoint((p1_first_point, p1_second_point),
                                                (p2_first_point, p2_second_point))
            if (intersection_result):
                # 发现交点则将扫描过的线段加入此区域
                area += poly1[add_head1:add_tail1]

                area.append(intersection_result)

                area += poly2[add_head2:add_tail2][::-1]

                # print(f"H1:{add_head1},T1:{add_tail1},H2:{add_head2},T2:{add_tail2}")

                length = sum([dis_between_pos(*x) for x in zip(area[:-1], area[1:])])

                intsecs.append({'area': area, 'length': length})
                area = [intersection_result]  # 从交点开始构造新的区域

                add_head1 = add_tail1
                add_head2 = add_tail2
                break

    area += poly1[add_head1:add_tail1]

    area.append(poly1[-1])

    area.append(poly2[-1])

    area += poly2[add_head2:add_tail2][::-1]

    length = sum([dis_between_pos(*x) for x in zip(area[:-1], area[1:])])
    intsecs.append({'area': area, 'length': length})
    return intsecs

def traj_lines_to_numpy(traj_lines):
    x = [p[0] for p in traj_lines]
    y = [p[1] for p in traj_lines]

    traj_np = np.stack([np.array(x), np.array(y)], axis=1)
    # print(traj_np.shape)
    return traj_np


def get_area_of_points(points):
    cop = {"type": "Polygon", "coordinates": [points]}
    from shapely.geometry import shape
    return shape(cop).area

    # try:
    #     hull = ConvexHull(np.array(points))
    #     return hull.area
    # except Exception as e:
    #     # print(e)
    #     # print(points)
    #     return 0


def LIP(poly1, poly2):

    LQ = 0
    LS = 0
    for i, p1_first_point in enumerate(poly1[:-1]):
        p1_second_point = poly1[i + 1]
        LQ += dis_between_pos(p1_first_point, p1_second_point)
    for j, p2_first_point in enumerate(poly2[:-1]):
        p2_second_point = poly2[j + 1]
        LS += dis_between_pos(p2_first_point, p2_second_point)

    LIP = 0
    intsecs = poly_intersection(poly1, poly2)

    for area in intsecs:
        weight = (area['length']) / (LS + LQ)
        LIP += get_area_of_points(area['area'])

    return LIP, intsecs


if (__name__ == '__main__'):
    import matplotlib.pyplot as plt

    plt.figure()

    poly1 = [(0.5, 1.5), (1.5, 3), (2.9, 1.5), (4.1, 3.5)]

    poly2 = [(1, 2.5), (1.8, 1.5), (3, 3), (4, 1.5)]

    x1, y1 = zip(*poly1)
    x2, y2 = zip(*poly2)
    plt.plot(x1, y1)
    plt.plot(x2, y2)

    plt.show()

    LIP_value, intsecs = LIP(poly1, poly2)

    # print(intsecs)
    print(f"LIP:{LIP_value}")
    for p in intsecs:
        pg = [list(x) for x in p['area']]
        # print(pg)

        #     plt.figure()
        coord = pg
        coord.append(coord[0])  # repeat the first point to create a 'closed loop'

        xs, ys = zip(*coord)  # create lists of x and y values

        plt.plot(xs, ys)

    plt.show()
