import datetime


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

        try:

            mees_after_weighted_mean_without_ts = list(map(
                trajobjlist2gpsposline,
                mees_after_weighted_mean
            ))
            verify_mee(mees_after_weighted_mean_without_ts[0])
        except Exception as e:
            raise e

        print(f"Start matching {vid} with mee {len(mees_after_weighted_mean[0])} points.")

        if (len(mees_after_weighted_mean[0]) < 3):
            print(f"Not a valid target {vid}")
            return
        gps_result_rids = netmatch(traj_str=str(lines_after_interval_with_ts), java_args=gps_java_arg, opt=opt,
                                   url='http://127.0.0.1:8090/netmatch')
        mee_result_rids = netmatch(traj_str=str(mees_after_weighted_mean), java_args=mee_java_arg, opt=opt,
                                   url='http://127.0.0.1:8090/netmatch')
        # print(gps_result_rids)

        # print(mee_result_rids)
        gps_matched_lines = [
            dbsession.get_road_poss(rid) if rid >= 0 else dbsession.get_road_poss(-1 * rid)[::-1]
            for rid in gps_result_rids]
        mee_matched_lines = [
            dbsession.get_road_poss(rid) if rid >= 0 else dbsession.get_road_poss(-1 * rid)[::-1]
            for rid in mee_result_rids]

        rmf = RMF(gps_matched_lines, mee_matched_lines)
        cmf = CMF(gps_matched_lines, mee_matched_lines, corridor_width=50)

        if (float(rmf) >= 2 or float(rmf) == -1):
            return None

        with open(fname + '_cmf', mode='a', encoding='utf-8') as  f:
            info = f'{vid}\t{"%.3f" % cmf}'
            f.write(info + '\n')
        with open(fname + '_rmf', mode='a', encoding='utf-8') as f:
            info = ""
            info += '\n'.join([str(x) for x in mee_result_rids])
            info += '='*10
            info += '\n'.join([str(x) for x in gps_result_rids])
            # info = f'{vid}\t{"%.3f" % rmf}\t{gps_result_rids}\t{mee_result_rids}\n{len(mees_after_weighted_mean[0])}\t{mees_after_weighted_mean}'
            f.write(info + '\n')
        print("[Result]", info)
        return_dict[vid] = [rmf, cmf]
        # return rmf
    except Exception as e:
        info = '%s\t%s' % (vid, str(e))
        # logging.log(logging.ERROR, info)
        print(info)
        logging.exception(info)


def match_all(kwargs):
    opt = DefaultConfig()
    opt.parse(kwargs)
    opt.windowsize  = 3
    opt.netmatch_url = 'http://127.0.0.1:8090/netmatch'
    opt.mee_url = 'http://127.0.0.1:8090/netmatch'
    opt.gps_url = 'http://127.0.0.1:8090/netmatch'

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
        dbsession.db['gps_trace_has_guest'].find({"$where": "this.trace.length > 50"}, {'_id': 1}).limit(num))
    # task_vids = [x['_id'] for x in task_vids]
    # task_vids = ['23113_0']

    task_vids = ['246341_0', '246341_1', '246341_2', '246341_3', '246341_4', '246341_5', '246341_6', '246341_7', '246341_8',
     '14189_0', '14189_1', '14189_2', '14189_3', '14189_4', '14189_5', '14189_6', '14189_7', '14189_8', '17348_0',
     '17348_1', '17348_2', '17348_3', '17348_4', '17348_5', '16819_0', '239971_0', '239971_1', '239971_2', '239971_3',
     '239971_4', '239971_5', '239971_6', '239971_7', '239971_8', '24533_0', '24533_1', '24533_2', '237624_0',
     '237624_1', '237624_2', '237624_3', '237624_4', '237624_5', '237624_6', '237624_7', '237624_8', '237624_9',
     '237624_10', '237624_11', '237624_12', '237624_13', '237624_14', '237624_15', '237624_16', '237624_17',
     '237624_18', '237624_19', '237624_20', '237624_21', '237624_22', '237624_23', '237624_24', '237624_25',
     '237624_26', '237624_27', '237624_28', '237624_29', '237624_30', '244366_0', '244366_1', '244366_2', '244366_3',
     '244366_4', '244366_5', '244366_6', '244366_7', '244366_8', '244366_9', '244366_10', '244366_11', '244366_12',
     '244366_13', '244366_14', '244366_15', '244366_16', '244366_17', '223522_0']

    # task_vids = [
    #     25839, 1487, 1079, 1560, 1311, 1361, 1685, 1355, 1252, 18346, 7423, 3143, 12225, 15616, 22445, 2631, 20767,
    #     18488, 17357, 1531, 14961, 9990, 14563, 17192, 20363, 22252, 22023, 14449, 18264, 20702, 22428, 15123]
    # task_vids = ['237008_6', '237008_6', '237008_6', '237008_6', '237008_6']

    # print(f"Start testing {len(task_vids)} targets...")
    # print(f"java args: {opt.java_args}")

    worker_num = kwargs.get('worker_num', 20)

    mypool = multiprocessing.Pool(processes=worker_num)
    
    for vid in task_vids:
        mypool.apply_async(match_one, args=[fname, vid, gps_java_arg, mee_java_arg, opt, return_dict])

    mypool.close()
    mypool.join()

    # process_pool = []
    # for vid in task_vids:
    #     p = multiprocessing.Process(target=match_one,
    #                                 args=[fname, vid, gps_java_arg, mee_java_arg, opt, return_dict])
    #     # process_pool.append(p)
    #     p.start()
    #     p.join()

    #
    # while (len(process_pool) > 0):
    #     for task in process_pool[:worker_num]:
    #         task.start()
    #
    #     for task in process_pool[:worker_num]:
    #         task.join()
    #
    #     process_pool = process_pool[worker_num:]

    return_values = list(return_dict.values())
    rmf_values = [x[0] for x in filter(lambda x: x[0] != -1 and x[0] < 2, return_values)]
    cmf_values = [x[1] for x in filter(lambda x: x[1] != -1, return_values)]
    rmf = sum(rmf_values)
    cmf = sum(cmf_values)

    print(f"RMF avg: {rmf / len(rmf_values)}")
    print(f"CMF avg: {cmf / len(cmf_values)}")
    with open('summary.txt', mode='a', encoding='utf-8') as f:
        f.write(f"##{str(datetime.datetime.now())[:10]}\n")
        f.write(f"##RMF avg: {rmf / len(rmf_values)}\n")
        f.write(f"##CMF avg: {cmf / len(cmf_values)}\n")
        f.write(f'##paras: {kwargs}\n\n')

    return rmf


def do_match(**kwargs):
    import time

    start = time.clock()

    # fname = 'output/2-16-2052.txt'
    fname = kwargs['fname']
    # method = "HCI"
    method = kwargs.get('method', 'HCI')
    # [0.6820037747374237, 'windowsize:3,sigmaM:50', '-mmOF-HMM -mc20 -sa3 -ms10 -tw110|-mmOF-CRF -mc180 -sa3 -ms200 -tw3500']

    with open('summary.txt', mode='a', encoding='utf-8') as f:
        f.write(f"==Start:{str(datetime.datetime.now())}\n")

    num = kwargs.get('num', 3)



    for angle in [100]:
        for enhance_mee in [False]:
            for interval in [999999]:
                for mee_mc in [65]:
                    for gps_tw in [110]:
                        for mee_tw in [0]:
                            for mee_ms in [200]:
                                for mee_sa in [1]:
                                    for ignore_sparse in [0]:
                                        for mapspeed in [40]:
                                            if ('C' in method):
                                                for ce in [1]:
                                                    match_all(kwargs={
                                                        'fname': f'{fname}-CRF-mapspeed{mapspeed}',
                                                        'java_args': f'-mmOF-HMM -mc20 -sa{3} -ms{10} -tw{gps_tw}|'
                                                                     f'-mmOF-CRF -mc{mee_mc} -sa{mee_sa} -ms{mee_ms} -tw{mee_tw} -ce{ce} -dc{mapspeed}',
                                                        'windowsize': 3,
                                                        'sigmaM': 50,
                                                        'ignore_sparse': False,
                                                        'num': num,
                                                        'interval': interval,
                                                        'enhance_mee': enhance_mee,
                                                        'angle': angle,
                                                    })

    with open('summary.txt', mode='a', encoding='utf-8') as f:

        f.write(f"=======================End:{str(datetime.datetime.now())}\n\n")


if (__name__ == "__main__"):
    fire.Fire()
