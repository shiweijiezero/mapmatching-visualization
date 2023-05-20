from utils.mybase import interval_between, TrajPoint, TrajLine, speed_between, angle_between, dis_between
import copy
import math
from functools import partial


def guest_filter(vid, traj_list: list, min_interval=65, min_leng=2, cutguest=0):
    """
    :param vid:
    :param traj_list:
    :param min_interval:
    :param min_leng:
    :return: list of TrajLine
    """
    i = 0
    begin = i
    temp = [traj_list[i]]
    rst = []
    while (i < (len(traj_list) - 1)):
        flag = True

        if (interval_between(traj_list[i], traj_list[i + 1]) > min_interval):  # 间隔过大则切分
            flag = False
        if (cutguest == 1 and traj_list[i].has_guest != traj_list[i + 1].has_guest):  # 载客状态反转则切分
            flag = False
        if (interval_between(traj_list[i], traj_list[i + 1]) == 0):  # 时间间隔为0则跳过这个点
            i += 1
            continue

        if flag:  # flag为True则不切分
            temp.append(traj_list[i + 1])
        else:
            if (len(temp) >= min_leng):
                new_line = TrajLine()
                new_line.vid = vid
                new_line.slice_index = (begin, i - 1)
                new_line.gps_line = temp
                rst.append(new_line)
            begin = i + 1
            temp = [traj_list[i + 1]]
        i += 1
    if (len(temp) >= min_leng):
        new_line = TrajLine()
        new_line.vid = vid
        new_line.slice_index = (begin, i - 1)
        new_line.gps_line = temp
        rst.append(new_line)

    return rst


def gps_interval_filter(vid, traj_list: list, min_interval=65, min_leng=2, cutguest=0):
    """
    :param vid:
    :param traj_list:
    :param min_interval:
    :param min_leng:
    :return: list of TrajLine
    """
    i = 0
    begin = i
    temp = [traj_list[i]]
    rst = []
    while (i < (len(traj_list) - 1)):
        flag = True

        if (interval_between(traj_list[i], traj_list[i + 1]) > min_interval):  # 间隔过大则切分
            flag = False
        if (cutguest == 1 and traj_list[i].has_guest != traj_list[i + 1].has_guest):  # 载客状态反转则切分
            flag = False
        if (interval_between(traj_list[i], traj_list[i + 1]) == 0):  # 时间间隔为0则跳过这个点
            i += 1
            continue

        if flag:  # flag为True则不切分
            temp.append(traj_list[i + 1])
        else:
            if (len(temp) >= min_leng):
                new_line = TrajLine()
                new_line.vid = vid
                new_line.slice_index = (begin, i - 1)
                new_line.gps_line = temp
                rst.append(new_line)
            begin = i + 1
            temp = [traj_list[i + 1]]
        i += 1
    if (len(temp) >= min_leng):
        new_line = TrajLine()
        new_line.vid = vid
        new_line.slice_index = (begin, i - 1)
        new_line.gps_line = temp
        rst.append(new_line)

    return rst


def get_interval_threshold_of_trajs(traj_list_: list):
    if (len(traj_list_) <= 1):
        return 0
    total_interval = 0
    max_interval = 0
    for i in range(0, len(traj_list_) - 1):
        max_interval = max(max_interval, interval_between(traj_list_[i], traj_list_[i + 1]))
        total_interval += interval_between(traj_list_[i], traj_list_[i + 1])
    return ((total_interval / len(traj_list_) - 1) + 2 * max_interval) / 3


def mee_speed_filter(traj_list_: list, speed_threshold=100, ignore_sparse=True):
    traj_list = copy.deepcopy(traj_list_)
    interval_threshold = get_interval_threshold_of_trajs(traj_list)
    i = 0
    while (i < len(traj_list) - 1):
        this = traj_list[i]
        nex = traj_list[i + 1]
        if (speed_between(this, nex) > speed_threshold):
            # do not filter the sparse point
            if (ignore_sparse and interval_between(this, nex) > interval_threshold):
                i += 1
                continue
            del traj_list[i]
        else:
            i += 1
    return traj_list


def order_filter(traj_list_: list, windowsize=5, step=3):
    if (len(traj_list_) <= 2 or windowsize <= 1):
        return traj_list_

    # 旅行商问题
    L = len(traj_list_)
    distance_table = [[0] * L for _ in range(0, L)]
    for i in range(L - 1):
        for j in range(i + 1, L):
            distance_table[i][j] = distance_table[j][i] = dis_between(traj_list_[i], traj_list_[j])

    begin = 0
    end = min(len(traj_list_), begin + windowsize)
    rstids = list(range(len(traj_list_)))

    while (end <= len(traj_list_)):

        pathofid = [begin]
        leftids = list(range(begin + 1, end))
        while (len(leftids) != 0):
            now = pathofid[-1]
            # 贪心,每次找最近的点
            leftids.sort(key=lambda nextid: distance_table[nextid][now])
            pathofid.append(leftids[0])
            del leftids[0]
        # 将排序后的一段放入结果
        rstids[begin:end] = pathofid

        if (end == len(traj_list_)):
            break

        # 移动窗口
        begin += step
        end = min(len(traj_list_), begin + windowsize)

    rst = []

    for i, v in enumerate(rstids):
        new = traj_list_[v]
        old = traj_list_[i]
        rst.append(TrajPoint(lng=new.lng, lat=new.lat, ts=old.ts, has_guest=new.has_guest))

    return rst


def mee_angle_filter(traj_list_: list, angle_threshold=120, ignore_sparse=True):
    traj_list = copy.deepcopy(traj_list_)
    interval_threshold = get_interval_threshold_of_trajs(traj_list)
    i = 1
    while (i < len(traj_list) - 2):

        a1 = angle_between(traj_list[i - 1], traj_list[i], traj_list[i + 1])
        a2 = angle_between(traj_list[i], traj_list[i + 1], traj_list[i + 2])

        # print(f'a1:{a1},a2:{a2}')

        if (a1 < angle_threshold) and (a2 < angle_threshold) or a1 < 40:
            # do not filter the sparse point
            if (ignore_sparse and interval_between(traj_list[i], traj_list[i + 1]) > interval_threshold):
                i += 1
                continue
            if (a2 < a1):  # remove the sharper angle first
                traj_list[i], traj_list[i + 1] = traj_list[i + 1], traj_list[i]
            # print("remove")
            del traj_list[i]
        else:
            i += 1
    return traj_list


def improve_sample_density(traj_list_: list, density: 5):
    if (density == 0):
        return traj_list_

    def gen_some_points_between(t1: TrajPoint, t2: TrajPoint, density):
        density *= int((dis_between(t1, t2) / 0.05) ** 0.3) + 1
        xd = (t2.lng - t1.lng) / density
        yd = (t2.lat - t1.lat) / density
        td = (t2.ts - t1.ts) / density

        return [
            TrajPoint(
                # lng=t1.lng + 0.0001 + i * xd + 0.0001 * (random.random() - 0.5),
                # lat=t1.lat + 0.0001 + i * yd + 0.0001 * (random.random() - 0.5),
                lng=t1.lng + i * xd,
                lat=t1.lat + i * yd,
                ts=t1.ts + i * td,
                has_guest=int((t1.has_guest + t2.has_guest) / 2)
            ) \
            for i in range(0, density + 1)]

    final_rst = []
    if (len(traj_list_) < 1):
        return traj_list_
    for i in range(0, len(traj_list_) - 1):
        final_rst += gen_some_points_between(traj_list_[i], traj_list_[i + 1], density=density)

    return final_rst


def mymean(traj_list_: list, target_traj: TrajPoint):
    rst = copy.deepcopy(target_traj)
    rst.lng = sum(map(
        lambda x: x.lng,
        traj_list_
    )) / len(traj_list_)
    rst.lat = sum(map(
        lambda x: x.lat,
        traj_list_
    )) / len(traj_list_)
    return rst


def weight_between(t1: TrajPoint, t2: TrajPoint, sigmaM):
    up = math.exp(-1 * (interval_between(t1, t2) ** 2) / (2 * sigmaM ** 2))
    down = (2 * math.pi) ** 0.5 * sigmaM

    if (up == 0):
        up = 0.00001

    return up / down


def weighted_mean(traj_list_: list, target_traj: TrajPoint, sigmaM):
    rst = copy.deepcopy(target_traj)

    traj_list_x = copy.deepcopy(traj_list_)
    traj_list_y = copy.deepcopy(traj_list_)

    traj_list_x.sort(key=lambda x: float(x.lng))
    traj_list_y.sort(key=lambda x: float(x.lat))

    traj_list_x = traj_list_x[1:-1]
    traj_list_y = traj_list_y[1:-1]

    coef_x = sum(map(
        partial(weight_between, t2=target_traj, sigmaM=sigmaM),
        traj_list_x
    ))

    coef_y = sum(map(
        partial(weight_between, t2=target_traj, sigmaM=sigmaM),
        traj_list_y
    ))

    # print("COFX:", coef_x, "COFY", coef_y)

    sum_x = 0
    for t in traj_list_x:
        sum_x += weight_between(t, target_traj, sigmaM) * t.lng
    sum_y = 0
    for t in traj_list_y:
        sum_y += weight_between(t, target_traj, sigmaM) * t.lat

    if (coef_x * coef_y == 0):
        return 0
    rst.lng = sum_x / coef_x
    rst.lat = sum_y / coef_y
    return rst


def mean_filter(traj_list_: list, window_size=5, sigmaM=5):
    if (window_size < 3):
        return traj_list_

    # print("Traj length:", len(traj_list_))
    final_rst = copy.deepcopy(traj_list_)

    if (window_size % 2 == 0):
        window_size += 1
    radius = int((window_size - 1) / 2)

    for target_index in range(radius, len(traj_list_) - radius):
        target_traj = traj_list_[target_index]
        window_list = copy.deepcopy(traj_list_[target_index - radius: target_index + radius + 1])

        avg_dist = sum([dis_between(*x) for x in zip(window_list[:-1], window_list[1:])]) / window_size
        # print(f"avg window dist:{avg_dist:.3f}")

        if(avg_dist>1):
            continue

        # final_rst[target_index] = mymean(window_list, target_traj)
        final_rst[target_index] = weighted_mean(window_list, target_traj, sigmaM)

    return final_rst
