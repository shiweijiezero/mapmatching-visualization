from utils import *
from config import DefaultConfig
import fire
import logging
import functools
import asyncio
import multiprocessing
import json


# ==========================================================================


def extract_method_from_args(mee_args):
    MethodList = ['CRF', 'INT']
    method = 'HMM'
    for M in MethodList:
        if (M in mee_args):
            method = M
    return method


def match_one(fname, vid, gps_java_arg, mee_java_arg, opt, return_dict):
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
            dbsession_=dbsession,
            enhance_mee=opt.enhance_mee
        )
        print(f"Start matching {vid}.")

        if (len(mees_after_weighted_mean[0]) < 3):
            print(f"Not a valid target {vid}")
            return
        gps_result_rids = netmatch(traj_str=str(lines_after_interval_with_ts), java_args=gps_java_arg, opt=opt)
        mee_result_rids = netmatch(traj_str=str(mees_after_weighted_mean), java_args=mee_java_arg, opt=opt)
        # print(gps_result_rids)
        # print(mee_result_rids)
        gps_matched_lines = [
            dbsession.get_road_poss(rid) if rid >= 0 else dbsession.get_road_poss(-1 * rid)[::-1]
            for rid in gps_result_rids]
        mee_matched_lines = [
            dbsession.get_road_poss(rid) if rid >= 0 else dbsession.get_road_poss(-1 * rid)[::-1]
            for rid in mee_result_rids]

        # rmf = RMF(gps_matched_lines, mee_matched_lines)
        with open(fname, mode='a', encoding='utf-8') as f:
            info = f'{vid}\tmee\t{str(mee_result_rids)}'
            f.write(info + '\n')
            info = f'{vid}\tgps\t{str(gps_result_rids)}'
            f.write(info + '\n')
        print(f"Finish {vid}")
        # print("[Result]", info)
        # return_dict[vid] = rmf
        # return rmf
    except Exception as e:
        info = '%s\t%s' % (vid, str(e))
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

    gps_java_arg, mee_java_arg = opt.java_args.split('|')
    # print(f"GPS Arg:{gps_java_arg}, MEE Arg:{mee_java_arg}")

    # task_vids = list(dbsession.get_gps_vids(1000))

    num = kwargs.get('num', 3)
    dbsession = MYDB()
    task_vids = list(
        dbsession.db['gps_trace_has_guest'].find({"$where": "this.trace.length > 5"}, {'_id': 1}).limit(num))
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
                                    args=[fname, vid, gps_java_arg, mee_java_arg, opt, return_dict])
        process_pool.append(p)

    worker_num = kwargs.get('worker_num',20)

    while (len(process_pool) > 0):
        for task in process_pool[:worker_num]:
            task.start()

        for task in process_pool[:worker_num]:
            task.join()

        process_pool = process_pool[worker_num:]

    # rmf_values = list(return_dict.values())
    # rmf_values = list(filter(lambda x: x != -1, rmf_values))
    # rmf = sum(rmf_values)
    #
    # print(f"RMF avg: {rmf / len(rmf_values)}")
    # with open(fname, mode='a', encoding='utf-8') as f:
    #     f.write(f"##RMF avg: {rmf / len(task_vids)}\n")
    #     f.write(f'##paras: {kwargs}\n')
    #
    # return rmf


def do_match(**kwargs):
    import time

    start = time.clock()

    # fname = 'output/2-16-2052.txt'
    fname = kwargs['fname']
    # method = "HCI"
    method = kwargs.get('method', 'HCI')
    # [0.6820037747374237, 'windowsize:3,sigmaM:50', '-mmOF-HMM -mc20 -sa3 -ms10 -tw110|-mmOF-CRF -mc180 -sa3 -ms200 -tw3500']

    num = kwargs.get('num', 3)
    for enhance_mee in [False,True]:
        for interval in [500]:
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
                                                    # if ('C' in method):
                                                    #     for ce in [15]:
                                                    #         match_all(kwargs={
                                                    #             'fname': fname,
                                                    #             'java_args': f'-mmOF-HMM -mc20 -sa3 -ms{gps_ms} -tw{gps_tw}|'
                                                    #                          f'-mmOF-CRF -mc{mee_mc} -sa{mee_sa} -ms{mee_ms} -tw{mee_tw} -ce{ce}',
                                                    #             'windowsize': weighted_window_size,
                                                    #             'sigmaM': sigma,
                                                    #             'ignore_sparse': ignore_sparse,
                                                    #             'cutguest': cutguest,
                                                    #             'detail': detail,
                                                    #             'result_type': result_type,
                                                    #             'num': num,
                                                    #             'interval': interval,
                                                    #             'enhance_mee': enhance_mee
                                                    #         })
                                                    if ('H' in method):
                                                        match_all(kwargs={
                                                            'fname': f'{fname}-enhanceMEE{enhance_mee}',
                                                            'java_args': f'-mmOF-HMM -mc20 -sa3 -ms{gps_ms} -tw{gps_tw}|'
                                                                         f'-mmOF-HMM -mc{mee_mc} -sa{mee_sa} -ms{mee_ms} -tw{mee_tw}',
                                                            'windowsize': weighted_window_size,
                                                            'sigmaM': sigma,
                                                            'ignore_sparse': ignore_sparse,
                                                            'cutguest': cutguest,
                                                            'num': num,
                                                            'interval': interval,
                                                            'enhance_mee': enhance_mee

                                                        })

                                                    # if ('I' in method):
                                                    #     match_all(kwargs={
                                                    #         'fname': fname,
                                                    #         'java_args': f'-mmOF-HMM -mc20 -sa3 -ms{gps_ms} -tw{gps_tw}|'
                                                    #                      f'-mmOF-INT -mc{mee_mc} -sa{mee_sa} -ms{mee_ms} -tw{mee_tw}',
                                                    #         'windowsize': weighted_window_size,
                                                    #         'sigmaM': sigma,
                                                    #         'ignore_sparse': ignore_sparse,
                                                    #         'cutguest': cutguest,
                                                    #         'detail': detail,
                                                    #         'result_type': result_type,
                                                    #         'num': num,
                                                    #         'interval': interval,
                                                    #         'enhance_mee': enhance_mee
                                                    #     })
                                    #
                                    #
                                    # t1 = Thread(target=match_all, kwargs={
                                    #     'fname': fname,
                                    #     'worker_num': 1,
                                    #     'java_args': f'-mmOF-HMM -mc20 -sa3 -ms{gps_ms} -tw{gps_tw}|-mmOF-CRF -mc{mee_mc} -sa{mee_sa} -ms{mee_ms} -tw{mee_tw}',
                                    #     'windowsize': weighted_window_size,
                                    #     'sigmaM': sigma
                                    # })
                                    # t2 = Thread(target=match_all, kwargs={
                                    #     'fname': fname,
                                    #     'worker_num': 1,
                                    #     'java_args': f'-mmOF-HMM -mc20 -sa3 -ms{gps_ms} -tw{gps_tw}|-mmOF-HMM -mc{mee_mc} -sa{mee_sa} -ms{mee_ms} -tw{mee_tw}',
                                    #     'windowsize': weighted_window_size,
                                    #     'sigmaM': sigma
                                    # })
                                    #
                                    # # thread_pool.append(t1)
                                    # # thread_pool.append(t2)
                                    # t1.start()
                                    # t2.start()
                                    # t1.join()
                                    # t2.join()

    elapsed = (time.clock() - start)
    print("Time used:", elapsed)


if (__name__ == "__main__"):
    fire.Fire()
    #run sample
    #PYTHONPATH=. python scripts/match_exp_multi_rid.py do_match --fname='output/4-1-100.txt' --method='H' --num=10