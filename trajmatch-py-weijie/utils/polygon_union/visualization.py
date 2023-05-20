import numpy as np
from matplotlib import pyplot as plt
import matplotlib


# 设置可视范围
def set_bounds(x_low, x_up, y_low, y_up, *axes):
    for axis in axes:
        axis.set_xbound(x_low, x_up)
        axis.set_ybound(y_low, y_up)


# 初始化
def init(num, figsize_x, figsize_y):
    plt.figure(num, figsize=(figsize_x, figsize_y))


def draw_before_union(p1_points, p2_points, vertices):
    figure = plt.figure(1)

    axes1 = figure.add_subplot(2, 1, 1)
    axes1.set_title("Before union")

    pgon1 = plt.Polygon(p1_points, color="y", fill=False)
    pgon2 = plt.Polygon(p2_points, color="b", fill=False)

    axes1.add_patch(pgon1)
    axes1.add_patch(pgon2)
    for vertex in vertices:
        axes1.plot(vertex.x, vertex.y, "ro")
        axes1.annotate(vertex.s_num, xy=(vertex.x, vertex.y))

    set_bounds(-4, 8, -4, 8, axes1)

    figure.canvas.draw()


def draw_result(points):
    figure = plt.figure(1)

    axes1 = figure.add_subplot(2, 1, 2)
    axes1.set_title("Union result")

    x = []
    y = []
    for i in range(len(points)):
        x.append(points[i][0])
        y.append(points[i][1])
    x.append(points[0][0])
    y.append(points[0][1])

    axes1.plot(x, y, 'go-', linewidth=2)

    set_bounds(-4, 8, -4, 8, axes1)

    figure.canvas.draw()


def draw_segments(segments):
    figure = plt.figure(2)

    axes1 = figure.add_subplot(1, 1, 1)
    axes1.set_title("Union result")

    for i in range(len(segments)):
        x = []
        y = []
        x.append(segments[i][0][0])
        y.append(segments[i][0][1])
        x.append(segments[i][1][0])
        y.append(segments[i][1][1])
        axes1.plot(x, y, 'go-', linewidth=2)

    set_bounds(-4, 8, -4, 8, axes1)

    figure.canvas.draw()
    plt.axis('off')


# def draw_rectangles(points, x_low, x_up, y_low, y_up, face_color="black", rect_color="white"):
#     plt.style.use('dark_background')
#     figure = plt.figure(2)
#     # 规定大小
#     plt.figure(2, figsize=(9, 9))
#
#     axes1 = figure.add_subplot(1, 1, 1)
#
#     pgon1 = plt.Polygon(points, color=rect_color, fill=True)
#
#     axes1.add_patch(pgon1)
#
#     set_bounds(x_low, x_up, y_low, y_up, axes1)
#
#     figure.canvas.draw()


def draw_rectangles(points, x_low, x_up, y_low, y_up):
    # 规定大小
    figure =plt.figure(2, figsize=(9, 9))

    axes1 = figure.add_subplot(1, 1, 1)
    axes1.set_title("Before union")

    pgon1 = plt.Polygon(points, color="black", fill=True)

    axes1.add_patch(pgon1)

    set_bounds(x_low, x_up, y_low, y_up, axes1)

    figure.canvas.draw()



def save(filename, dpi):
    figure = plt.figure(2)
    plt.axis('off')
    figure.subplots_adjust(top=1, bottom=0, left=0, right=1)
    figure.savefig(filename + ".png", pad_inches=0, dpi=dpi)


def show():
    plt.show()
