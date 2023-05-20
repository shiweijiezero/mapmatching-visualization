from utils import *
from config import DefaultConfig
import fire
import logging
import functools
import asyncio
import multiprocessing
import json
from datetime import datetime

# ==========================================================================


def extract_method_from_args(mee_args):
    MethodList = ['CRF', 'INT']
    method = 'HMM'
    for M in MethodList:
        if (M in mee_args):
            method = M
    return method


def match_one(fname, vid, gps_java_arg, mee_java_arg, opt, return_dict, result_type):
    # fname, vid, gps_java_arg, mee_java_arg, opt, return_dict = arg
    dbsession = MYDB()
    try:

        roadmap, \
        raw_gps_line_without_ts, \
        lines_after_interval_with_ts, \
        lines_after_interval, \
        mee_lines_without_ts, \
        mees_after_speed_filter, \
        mees_after_order_filter, \
        mees_after_angle_filter, \
        mees_after_weighted_mean, \
        mees_after_rnn = process_traj(
            vid=vid,
            interval=opt.interval,
            speed=opt.speed,
            angle=opt.angle,
            roadmapK=opt.roadmapK,
            weighted_mean_windowsize=opt.windowsize,
            sigmaM=opt.sigmaM,
            density=opt.density,
            ignore_sparse=opt.ignore_sparse,
            order_filter_windowsize=opt.orderwindowsize,
            order_filter_step=opt.orderstep,
            cutguest=opt.cutguest,
            dbsession_=dbsession
        )
        lines_after_interval_with_ts = list(filter(lambda x: len(x) > 2, lines_after_interval_with_ts))
        mees_after_weighted_mean = list(filter(lambda x: len(x) > 2, mees_after_weighted_mean))
        return_dict[f'{vid}_0'] = lines_after_interval_with_ts
        return_dict[f'{vid}_1'] = mees_after_weighted_mean

        # return rmf
    except Exception as e:
        info = '%d\t%s' % (vid, str(e))
        # logging.log(logging.ERROR, info)
        print(info)
        logging.exception(info)


def match_all(kwargs):
    opt = DefaultConfig()
    opt.parse(kwargs)
    opt.netmatch_url = 'http://127.0.0.1:8090/netmatch'

    logging.basicConfig(filename="%s.log" % kwargs['fname'], filemode="w",
                        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%M-%Y %H:%M:%S", level=logging.ERROR)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    fname = kwargs['fname']
    detail = kwargs['detail']
    result_type = kwargs['result_type']

    gps_java_arg, mee_java_arg = opt.java_args.split('|')
    # print(f"GPS Arg:{gps_java_arg}, MEE Arg:{mee_java_arg}")

    # task_vids = list(dbsession.get_gps_vids(1000))

    num = kwargs.get('num', 3)
    dbsession = MYDB()
    task_vids = list(dbsession.db['gps_trace'].find({"$where": "this.trace.length > 800"}, {'_id': 1}).limit(num))
    task_vids = [x['_id'] for x in task_vids]
    # task_vids = [
    #     25839, 1487, 1079, 1560, 1311, 1361, 1685, 1355, 1252, 18346, 7423, 3143, 12225, 15616, 22445, 2631, 20767,
    #     18488, 17357, 1531, 14961, 9990, 14563, 17192, 20363, 22252, 22023, 14449, 18264, 20702, 22428, 15123]
    # task_vids = [
    #     25839, 1487, 1079, 1560, 1311, 1361, 1685]

    # print(f"Start testing {len(task_vids)} targets...")
    # print(f"java args: {opt.java_args}")

    process_pool = []
    for vid in task_vids:
        p = multiprocessing.Process(target=match_one,
                                    args=[fname, vid, gps_java_arg, mee_java_arg, opt, return_dict, result_type])
        process_pool.append(p)
    worker_num = 20

    while (len(process_pool) > 0):
        for task in process_pool[:worker_num]:
            task.start()

        for task in process_pool[:worker_num]:
            task.join()

        process_pool = process_pool[worker_num:]

    return_values = list(return_dict.values())
    lines = []
    for item in return_values:
        lines += item
    c = 0
    for line in lines:
        with open(f'{fname}/trip_{c}.txt', mode='w') as f:
            for point in line:
                f.write(f'{point.lng} {point.lat} {int(datetime.timestamp(point.ts))}\n')
        c+=1


def export_offline_data(**kwargs):
    import time

    start = time.clock()

    # fname = 'output/2-16-2052.txt'
    fname = kwargs['fname']
    # method = "HCI"
    method = kwargs.get('method', 'HCI')
    # [0.6820037747374237, 'windowsize:3,sigmaM:50', '-mmOF-HMM -mc20 -sa3 -ms10 -tw110|-mmOF-CRF -mc180 -sa3 -ms200 -tw3500']

    detail = True if 'detail' in kwargs else False

    result_type = kwargs.get('result_type', 'W')  # 'S' for split seg line ,'W' for whole line mode
    print(f"Result type:{result_type}")

    num = kwargs.get('num', 3)
    for interval in [65]:
        for mee_mc in [180]:
            for gps_ms in [10]:
                for weighted_window_size in [3]:
                    for sigma in [50]:
                        for gps_tw in [110]:
                            for mee_tw in [3500]:
                                for mee_ms in [200]:
                                    for mee_sa in [3]:
                                        for ignore_sparse in [1]:
                                            for cutguest in [0]:
                                                if ('H' in method):
                                                    match_all(kwargs={
                                                        'fname': fname,
                                                        'java_args': f'-mmOF-HMM -mc20 -sa3 -ms{gps_ms} -tw{gps_tw}|'
                                                                     f'-mmOF-HMM -mc{mee_mc} -sa{mee_sa} -ms{mee_ms} -tw{mee_tw}',
                                                        'windowsize': weighted_window_size,
                                                        'sigmaM': sigma,
                                                        'ignore_sparse': ignore_sparse,
                                                        'cutguest': cutguest,
                                                        'detail': detail,
                                                        'result_type': result_type,
                                                        'num': num,
                                                        'interval': interval

                                                    })

    elapsed = (time.clock() - start)
    print("Time used:", elapsed)


if (__name__ == "__main__"):
    fire.Fire()
