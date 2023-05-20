import traj_dist.distance as tdist
import numpy as np
import pandas

from scipy.spatial import ConvexHull, convex_hull_plot_2d

# import matplotlib.pyplot as plt

from utils.lip import LIP
import math

if (__name__ == '__main__'):
    from utils import *

    dbsession = MYDB()
    # task_vids = list(
    #     dbsession.db['gps_trace_has_guest'].find({"$where": "this.trace.length >50 "}, {'_id': 1}).limit(10000))

    task_vids = list(
        dbsession.db['gps_trace_has_guest'].find({}, {'_id': 1}).limit(10000))

    task_vids = [x['_id'] for x in task_vids]
    # task_vids = ['24398_5', '243484_5', '240961_0', '17048_2', '201145_2']
    opt = DefaultConfig()
    opt.interval = 9999999999999999999
    # metrics = ['hausdorff', 'frechet', 'dtw']
    metrics = ['LIP']

    res = pandas.DataFrame(columns=metrics)

    for metric in metrics:
        col = []

        for vid in task_vids:
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
                    enhance_mee=False
                )

                mees_after_weighted_mean_without_ts = list(map(
                    trajobjlist2gpsposline,
                    mees_after_weighted_mean
                ))
                try:
                    verify_mee(mees_after_weighted_mean_without_ts[0])
                except Exception as e:
                    raise e

                print(f'{vid}-{metric}')

                if (metric == 'LIP'):

                    origin_dist, _ = LIP(
                        raw_gps_line_without_ts[0],
                        mee_lines_without_ts[0])
                    filter_dist, _ = LIP(
                        raw_gps_line_without_ts[0],
                        mees_after_weighted_mean_without_ts[0])

                else:

                    origin_dist = eval(f'tdist.{metric}')(
                        traj_lines_to_numpy(raw_gps_line_without_ts[0]),
                        traj_lines_to_numpy(mee_lines_without_ts[0]))
                    filter_dist = eval(f'tdist.{metric}')(
                        traj_lines_to_numpy(raw_gps_line_without_ts[0]),
                        traj_lines_to_numpy(mees_after_weighted_mean_without_ts[0]))
                imp_p = (origin_dist - filter_dist) / origin_dist * 100

                print(f"Origin_dist:{origin_dist:.3f},Filter_dist:{filter_dist:.3f},Improve:{imp_p:.3f}%")
                with open('evallog.csv', mode='a') as f:
                    f.write(f'{vid},{imp_p:.3f}\n')
                col.append(imp_p)
            except Exception as e:
                print(str(e)+f"VID:{vid}")
                pass
                # with open('evallog.csv', mode='a') as f:
                #     f.write(f'{vid},error\n')
        res[metric] = col
        # res[metric] = pandas.DataFrame(res[metric])
        # res[metric].describe()
    print(res.describe())
    # res.to_csv('preprocess_result.csv')

    # pandas.DataFrame.hist(res)
