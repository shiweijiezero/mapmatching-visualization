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
import numpy as np
from utils.mybase import *
from config import DefaultConfig
import json
import copy
import os
from .mydb import dbsession

from collections.abc import Iterable

PRECISION = 7


def load_or_create_gps_obj(opt: DefaultConfig) -> GPSDataset:
    obj_fname = opt.get_gps_obj_name()
    if os.path.exists(obj_fname):
        with open(obj_fname, mode='rb') as f:
            dataset = pickle.load(f)
            print("Loaded gps obj from [%s]" % obj_fname)
    else:
        dataset = GPSDataset(opt)
        dataset.save()
    return dataset


def load_graph_obj(opt: DefaultConfig):
    obj_fname = opt.get_graph_object_file_name()
    with open(obj_fname, mode='rb') as f:
        obj = pickle.load(f)
        print("Loaded graph obj from [%s]" % obj_fname)
    return obj


def load_or_create_mee_obj(opt: DefaultConfig) -> MeeDataset:
    obj_fname = opt.get_mee_obj_name()
    if os.path.exists(obj_fname):
        with open(obj_fname, mode='rb') as f:
            bs_dataset = pickle.load(f)
            print("Loaded mee obj from [%s]" % obj_fname)
    else:
        bs_dataset = MeeDataset(opt)
        bs_dataset.save()
    return bs_dataset


def strToTrajectory(s):
    res = []
    s = s.split('|')[1]
    for p in s.split(','):
        t = p.split(' ')
        res.append((float(t[0]), float(t[1])))

    return res

    #
    # pass
    # if (type(s) is str):
    #     res = []
    #     s = s.split('|')[1]
    #     for p in s.split(','):
    #         t = p.split(' ')
    #         res.append((float(t[0]), float(t[1])))
    #
    #     return res
    # elif (type(s) is list):
    #     for p in s:
    #         p = p.split(' ')
    #         yield (p[0], p[1])
    # else:
    #     raise Exception("Invalid traj str:" + str(s))


def load_or_create_map_obj(opt: DefaultConfig) -> MapDataset:
    obj_fname = opt.get_map_obj_name()
    if os.path.exists(obj_fname):
        with open(obj_fname, mode='rb') as f:
            dataset = pickle.load(f)
            print("Loaded map obj from [%s]" % obj_fname)
    else:
        dataset = MapDataset(opt)
        dataset.save()
    return dataset


def info_check(traj_list: list):
    for i in range(len(traj_list) - 1):
        if (interval_between(traj_list[i], traj_list[i + 1]) == 0):
            continue
        speed = speed_between(traj_list[i], traj_list[i + 1])
        if (i >= 1):
            angle = angle_between(traj_list[i - 1], traj_list[i], traj_list[i + 1])
        else:
            angle = 0
        print("%d  速度:%.2fkm/h  采样间隔:%f.2s  角度:%2.f°" % (
            i, speed, interval_between(traj_list[i], traj_list[i + 1]), angle))


def intercept_mee_traj_by_timestamp(gps_line: list, mee_trjlist: list):
    begin_ts = gps_line[0].ts
    end_ts = gps_line[-1].ts
    mee_line = []
    for traj in mee_trjlist:
        if (traj.ts >= begin_ts and traj.ts <= end_ts):
            mee_line.append(traj)
        if (traj.ts > end_ts):
            break
    return mee_line


def export_one_trajobj_list(traj_obj_list: list, fname, offset=(0, 0)):
    """
    write x,y,ts
    """
    from datetime import datetime
    trip = [(x.lng + offset[0], x.lat + offset[1], int(datetime.timestamp(x.ts))) for x in traj_obj_list]
    # fname = 'output/trip_%d.txt' % index
    with open(fname, mode='w', encoding='utf-8') as f:
        for pos in trip:
            f.write(' '.join([str(x) for x in pos]) + '\n')
    print("Saved %s." % fname)


def parse_match_result(fname, m, type=0):
    with open(fname, mode='r', encoding='utf-8') as f:
        rst = list(f.readlines())
    if (type == 0):
        lines = []

        for seg in rst[0].split(','):
            if ('null' in seg):
                continue
            poss = seg.split(' ')
            poss = list(zip(poss[::2], poss[1::2]))
            lines.append(poss)  # TODO:??
        return lines
    else:
        if (len(rst) == 1):
            # have no road result
            return []
        else:
            matched_road_ids = set([int(x) for x in rst[1].split(',')])

            return [m[rid] if rid >= 0 else m[-1 * rid][::-1] for rid in matched_road_ids]


def myround(pos: tuple):
    return (round(float(pos[0]), PRECISION), round(float(pos[1]), PRECISION))


def count_point_in_line(line):
    points = set()
    if ('lng' in dir(line[0])):
        line = [(pos.lng, pos.lat) for pos in line]
    for pos in line:
        points.add(myround(pos))
    if (myround(line[0]) == myround(line[-1])):
        print("Line with same begin and end!")
        return -1
    return len(points)


def RMF(gps_matched_lines: list, mee_matched_lines: list):
    """
    :param matched_line1: [[(pos1),(pos2),...],[(pos2),(pos3),...]],MEE
    :param matched_line2: [[(pos1),(pos2),...],[(pos2),(pos3),...]],GPS
    :param MAP: MapDataset
    :return: float
    """
    segments1, segments2 = set(), set()

    for line in gps_matched_lines:
        segments = zip(
            [myround(pos) for pos in line[0:-1:1]],
            [myround(pos) for pos in line[1::1]]
        )  # [(p1,p2),(p2,p3),...]
        # segments_ = zip(
        #     [myround(pos) for pos in line[1::2]],
        #     [myround(pos) for pos in line[0::2]]
        # )  # [(p1,p2),(p2,p3),...]
        segments1.update(segments)  # 取出所有线段并去重
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
        segments2.update(segments)  # 取出所有线段并去重
        # segments2.update(segments_)  # 反向的所有线段

    denominator = sum([dis_between_pos(x[0], x[1]) for x in segments1])

    molecule = sum(
        [dis_between_pos(x[0], x[1]) for x in segments2.union(segments1) - (segments2.intersection(segments1))])

    if (denominator == 0):
        return -1

    return molecule / denominator


def trajobjlist2gpsposline(trajobjlist):
    return [(x.lng, x.lat) for x in trajobjlist]


def save_match_target(opt, vid,
                      lines_after_interval_with_ts,
                      mees_after_weighted_mean):
    gps_num = len(lines_after_interval_with_ts)
    match_target_dirpath = os.path.join(
        opt.output_dir,
        "%d_%d" % (vid, gps_num)
    )

    resultpath = os.path.join(match_target_dirpath, 'result')
    trippath = os.path.join(match_target_dirpath, 'trip')

    # mkdir
    if (not os.path.exists(match_target_dirpath)):
        os.mkdir(match_target_dirpath)
        os.mkdir(resultpath)
        os.mkdir(trippath)

    # generate trip
    print("Gen GPS trips...")
    for i, line in enumerate(lines_after_interval_with_ts):
        fname = os.path.join(trippath, "trip_%s.txt" % str(i))
        export_one_trajobj_list(line, fname)

    print("Gen MEE trips...")
    for i, line in enumerate(mees_after_weighted_mean):
        fname = os.path.join(trippath, "trip_%s.txt" % str(i + gps_num))
        export_one_trajobj_list(line, fname)
    # cp_path(trippath, opt.java_match_target_dir_name)

    return resultpath, trippath


def get_match_result(opt, resultpath, trippath, gps_num, result_type, MAP, java_args='-mmOF-HMM -mc20'):
    # no exist result then do java match
    if (len(os.listdir(resultpath)) == 0):
        # copy trip to java input
        cp_path(trippath, opt.java_match_target_dir_name)
        # do the match
        exec_match_command(opt=opt, java_args=java_args)
        # copy match result to result path
        cp_path(opt.java_match_result_dir_name, resultpath)

        print("Got match result: %s" % str(os.listdir(resultpath)))

    # parse result from result path
    result_filenames = list(os.listdir(resultpath))

    gps_matched_lines = []
    mee_matched_lines = []

    for i in range(0, len(result_filenames)):
        fname = os.path.join(
            resultpath,
            "matchresult_%d.txt" % i
        )
        if (i < gps_num):
            gps_matched_lines += parse_match_result(fname, MAP, result_type)
        else:
            mee_matched_lines += parse_match_result(fname, MAP, result_type)

    return gps_matched_lines, mee_matched_lines


def remove_duplicated_points(trajobjs_or_tuples, dup_range=0.000):
    """
    can use in traj or tuple list
    """
    trajobjs_or_tuples = copy.deepcopy(trajobjs_or_tuples)

    i = 1
    if (len(trajobjs_or_tuples) == 0):
        return []
    else:
        temp = trajobjs_or_tuples[0]

    if (type(trajobjs_or_tuples[0]) is TrajPoint):
        dis_func = dis_between
        ts_dis_func = interval_between
    else:
        dis_func = dis_between_pos
        ts_dis_func = lambda x, y: 1

    while (i < len(trajobjs_or_tuples)):
        if (dis_func(temp, trajobjs_or_tuples[i]) <= dup_range or ts_dis_func(temp, trajobjs_or_tuples[i]) == 0):
            del trajobjs_or_tuples[i]
        else:
            temp = trajobjs_or_tuples[i]
            i += 1
    # if (len(trajobjs_or_tuples) == 1):
    #     return []
    return trajobjs_or_tuples


def process_traj(vid, opt=None, use_rnn=False, interval=65, speed=100, angle=120, roadmapK=0,
                 weighted_mean_windowsize=10,
                 sigmaM=20,
                 density=0, ignore_sparse=True, order_filter_windowsize=5, order_filter_step=5, cutguest=0,
                 dbsession_=dbsession, enhance_mee=True):
    # list of Traj
    raw_gps_line_with_ts = dbsession_.get_gps_line_with_ts(vid)
    # raw_gps_line_with_ts = remove_duplicated_points(raw_gps_line_with_ts)

    raw_gps_line_without_ts = [(x.lng, x.lat) for x in raw_gps_line_with_ts]

    lines_after_interval_with_ts = list(map(
        remove_duplicated_points,
        [raw_gps_line_with_ts]
    ))

    lines_after_interval = [[x.pos for x in lines_after_interval_with_ts[0]]]

    # # list of tuple
    # lines_after_interval = list(map(
    #     lambda x: x.gps_pos_line(),
    #     list(lines_after_interval_with_ts)
    # ))

    # list of Traj
    # lines_after_interval_with_ts = [x.gps_line for x in lines_after_interval_with_ts]

    # list of Traj
    mee_lines_with_ts = list(map(
        partial(intercept_mee_traj_by_timestamp, mee_trjlist=dbsession_.get_mee_line_with_ts(vid)),
        lines_after_interval_with_ts
    ))

    # assert len(lines_after_interval)==len(mee_lines_with_ts)

    # =================================================
    mee_lines_with_ts = list(map(
        remove_duplicated_points,
        mee_lines_with_ts
    ))

    # list of Traj
    mee_lines_without_ts = list(map(
        lambda line: [(p.lng, p.lat) for p in line],
        list(mee_lines_with_ts)
    ))

    # list of Traj
    mees_after_speed_filter = list(map(
        partial(mee_speed_filter, speed_threshold=speed, ignore_sparse=ignore_sparse),
        list(mee_lines_with_ts)
    ))

    # # list of Traj
    # mees_after_order_filter = list(map(
    #     partial(order_filter, windowsize=order_filter_windowsize, step=order_filter_step),
    #     list(mees_after_speed_filter)
    #
    # ))
    mees_after_order_filter = None

    last_len_after_angle = -1

    # list of Traj
    mees_after_angle_filter = mees_after_speed_filter
    c_angle = 0
    while (len(mees_after_angle_filter) > 0 and last_len_after_angle != len(mees_after_angle_filter[0])):
        c_angle += 1
        last_len_after_angle = len(mees_after_angle_filter[0])
        mees_after_angle_filter = list(map(
            partial(mee_angle_filter, angle_threshold=angle, ignore_sparse=ignore_sparse),
            list(mees_after_angle_filter)
        ))
    #     print(f"AngleFilterTime:{c_angle}")

    # mees_fakesampled = list(map(
    #     partial(improve_sample_density, density=density),
    #     list(mees_after_angle_filter)
    #
    # ))

    mees_after_weighted_mean = list(map(
        partial(mean_filter, window_size=weighted_mean_windowsize, sigmaM=sigmaM),
        mees_after_angle_filter
    ))

    # if (len(lines_after_interval) != len(mees_after_weighted_mean)):
    #     raise Exception('lengths of gps and mee mismatch!')

    def enhance_mee_line(gpsline, meeline):
        # meeline = [gps_line[0]] + meeline + [gps_line[-1]]
        meeline = [gps_line[0]] + meeline[1:-1] + [gps_line[-1]]
        return meeline

    if (enhance_mee):
        for i, gps_line in enumerate(lines_after_interval_with_ts):
            mees_after_weighted_mean[i] = enhance_mee_line(gps_line, mees_after_weighted_mean[i])

    mees_after_rnn = []

    if (roadmapK != 0):
        mees_after_weighted_mean_without_ts = list(map(
            trajobjlist2gpsposline,
            mees_after_weighted_mean
        ))
        roadmap = dbsession.query_ball_roads(mees_after_weighted_mean_without_ts, k=int(roadmapK))
    else:
        roadmap = []

    result = [
        roadmap,
        [raw_gps_line_without_ts],
        lines_after_interval_with_ts,
        lines_after_interval,
        mee_lines_without_ts,
        mees_after_speed_filter,
        mees_after_order_filter,
        mees_after_angle_filter,
        mees_after_weighted_mean,
        mees_after_rnn]

    # assert len(lines_after_interval) == len(mees_after_weighted_mean)

    # for i, v in enumerate(result):
    #     result[i] = list(filter(lambda x: len(x) >= 2, result[i]))

    return result


# def process_traj(vid, opt=None, use_rnn=False, interval=65, speed=100, angle=120, roadmapK=0,
#                  weighted_mean_windowsize=10,
#                  sigmaM=20,
#                  density=0, ignore_sparse=True, order_filter_windowsize=5, order_filter_step=5, cutguest=0,
#                  dbsession_=dbsession, enhance_mee=True):
#     # list of Traj
#     raw_gps_line_with_ts = dbsession_.get_gps_line_with_ts(vid)
#     # raw_gps_line_with_ts = remove_duplicated_points(raw_gps_line_with_ts)
#
#     raw_gps_line_without_ts = [(x.lng, x.lat) for x in raw_gps_line_with_ts]
#
#     # list of TrajLine
#     lines_after_interval_with_ts = gps_interval_filter(vid, raw_gps_line_with_ts, min_interval=interval,
#                                                        cutguest=cutguest)
#
#     # list of tuple
#     lines_after_interval = list(map(
#         lambda x: x.gps_pos_line(),
#         list(lines_after_interval_with_ts)
#     ))
#
#     # list of Traj
#     lines_after_interval_with_ts = [x.gps_line for x in lines_after_interval_with_ts]
#
#     # list of Traj
#     mee_lines_with_ts = list(map(
#         partial(intercept_mee_traj_by_timestamp, mee_trjlist=dbsession_.get_mee_line_with_ts(vid)),
#         lines_after_interval_with_ts
#     ))
#
#     # assert len(lines_after_interval)==len(mee_lines_with_ts)
#
#     mee_lines_with_ts = list(map(
#         remove_duplicated_points,
#         mee_lines_with_ts
#     ))
#
#     # list of Traj
#     mee_lines_without_ts = list(map(
#         lambda line: [(p.lng, p.lat) for p in line],
#         list(mee_lines_with_ts)
#     ))
#
#     # list of Traj
#     mees_after_speed_filter = list(map(
#         partial(mee_speed_filter, speed_threshold=speed, ignore_sparse=ignore_sparse),
#         list(mee_lines_with_ts)
#     ))
#
#     # # list of Traj
#     # mees_after_order_filter = list(map(
#     #     partial(order_filter, windowsize=order_filter_windowsize, step=order_filter_step),
#     #     list(mees_after_speed_filter)
#     #
#     # ))
#     mees_after_order_filter = None
#
#
#     last_len_after_angle = -1
#
#     # list of Traj
#     mees_after_angle_filter = mees_after_speed_filter
#     c_angle = 0
#     while (len(mees_after_angle_filter) > 0 and last_len_after_angle != len(mees_after_angle_filter[0])):
#         c_angle += 1
#         last_len_after_angle = len(mees_after_angle_filter[0])
#         mees_after_angle_filter = list(map(
#             partial(mee_angle_filter, angle_threshold=angle, ignore_sparse=ignore_sparse),
#             list(mees_after_angle_filter)
#
#         ))
#     # print(f"AngleFilterTime:{c_angle}")
#
#     # mees_fakesampled = list(map(
#     #     partial(improve_sample_density, density=density),
#     #     list(mees_after_angle_filter)
#     #
#     # ))
#
#     mees_after_weighted_mean = list(map(
#         partial(mean_filter, window_size=weighted_mean_windowsize, sigmaM=sigmaM),
#         mees_after_angle_filter
#     ))
#
#     if (len(lines_after_interval) != len(mees_after_weighted_mean)):
#         raise Exception('lengths of gps and mee mismatch!')
#
#     def enhance_mee_line(gpsline, meeline):
#         # meeline = [gps_line[0]] + meeline + [gps_line[-1]]
#         meeline = [gps_line[0]] + meeline[1:-1] + [gps_line[-1]]
#         return meeline
#
#     if (enhance_mee):
#         for i, gps_line in enumerate(lines_after_interval_with_ts):
#             mees_after_weighted_mean[i] = enhance_mee_line(gps_line, mees_after_weighted_mean[i])
#
#     mees_after_rnn = []
#
#     if (roadmapK != 0):
#         mees_after_weighted_mean_without_ts = list(map(
#             trajobjlist2gpsposline,
#             mees_after_weighted_mean
#         ))
#         roadmap = dbsession.query_ball_roads(mees_after_weighted_mean_without_ts, k=int(roadmapK))
#     else:
#         roadmap = []
#
#     result = [
#         roadmap,
#         [raw_gps_line_without_ts],
#         lines_after_interval_with_ts,
#         lines_after_interval,
#         mee_lines_without_ts,
#         mees_after_speed_filter,
#         mees_after_order_filter,
#         mees_after_angle_filter,
#         mees_after_weighted_mean,
#         mees_after_rnn]
#
#     # assert len(lines_after_interval) == len(mees_after_weighted_mean)
#
#     # for i, v in enumerate(result):
#     #     result[i] = list(filter(lambda x: len(x) >= 2, result[i]))
#
#     return result
#

def inSegment(p, line, line2):
    '''
    检查某交点是否在线段line上（不含line的端点），在求交点时已经确认两条直线不平行
    所以，对于竖直的line，line2不可能竖直，却有可能水平，所以检查p是否在line2上，只能检查x值即p[0]
    '''
    if line[0][0] == line[1][0]:  # 如果line竖直
        if p[1] > min(line[0][1], line[1][1]) and p[1] < max(line[0][1], line[1][1]):
            # if p[1] >= min(line2[0][1],line2[1][1]) and p[1] <= max(line2[0][1],line2[1][1]):
            if p[0] >= min(line2[0][0], line2[1][0]) and p[0] <= max(line2[0][0], line2[1][0]):
                return True
    elif line[0][1] == line[1][1]:  # 如果line水平
        if p[0] > min(line[0][0], line[1][0]) and p[0] < max(line[0][0], line[1][0]):
            # if p[0] >= min(line2[0][0],line2[1][0]) and p[0] <= max(line2[0][0],line2[1][0]):
            if p[1] >= min(line2[0][1], line2[1][1]) and p[1] <= max(line2[0][1], line2[1][1]):
                return True
    else:
        if p[0] > min(line[0][0], line[1][0]) and p[0] < max(line[0][0], line[1][0]):
            # line为斜线时，line2有可能竖直也有可能水平，所以对x和y都需要检查
            if p[1] >= min(line2[0][1], line2[1][1]) and p[1] <= max(line2[0][1], line2[1][1]) and p[0] >= min(
                    line2[0][0], line2[1][0]) and p[0] <= max(line2[0][0], line2[1][0]):
                return True
    return False


def getLinePara(line):
    '''简化交点计算公式'''
    a = line[0][1] - line[1][1]
    b = line[1][0] - line[0][0]
    c = line[0][0] * line[1][1] - line[1][0] * line[0][1]
    return a, b, c


def getCrossPoint(line1, line2):
    '''计算交点坐标，此函数求的是line1中被line2所切割而得到的点，不含端点'''
    a1, b1, c1 = getLinePara(line1)
    a2, b2, c2 = getLinePara(line2)
    d = a1 * b2 - a2 * b1
    p = [0, 0]
    if d == 0:  # d为0即line1和line2平行
        return ()
    else:
        p[0] = round((b1 * c2 - b2 * c1) * 1.0 / d, 13)  # 工作中需要处理有效位数，实际可以去掉round()
        p[1] = round((c1 * a2 - c2 * a1) * 1.0 / d, 13)
    p = tuple(p)
    if inSegment(p, line1, line2):
        # print(p)
        return p
    else:
        return ()


from scipy import spatial


def furthest_points_length(pts):
    if (len(pts) < 5):
        return 0
    try:
        candidates = [pts[x] for x in spatial.ConvexHull(pts).vertices]
        # get distances between each pair of candidate points
        dist_mat = spatial.distance_matrix(candidates, candidates)

        # get indices of candidates that are furthest apart
        i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

        return dis_between_pos(candidates[i], candidates[j])
    except:
        return 0


def verify_mee(mee_line):
    if (len(mee_line) < 4 or len(mee_line) > 100):
        raise Exception(f"mee length invalid: {len(mee_line)}")
    fl = furthest_points_length(mee_line)
    if (fl > 50 or fl < 2):
        raise Exception(f"mee furthest length invalid:{fl:.2f}")

    return None
