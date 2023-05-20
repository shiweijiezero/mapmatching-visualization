import pandas as pd
import os
import pickle
import pyprind

from utils.mybase import Dataset, TrajPoint, split_dataframe
from utils.coord_transform import wgs84_to_bd09

from multiprocessing import Pool, cpu_count
import datetime
from numpy import array


def dateparse(time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))


def parse_df(df):
    traj_dict = {}

    bar = pyprind.ProgBar(iterations=len(df))
    for _, line in df.iterrows():
        bar.update()
        vid, tstamp, operator, longti, lati = line

        longti, lati = wgs84_to_bd09(float(longti), float(lati))

        vid = int(vid)
        if (vid not in traj_dict):
            traj_dict[vid] = []
        traj_dict[vid].append(TrajPoint(longti, lati, tstamp))

    return traj_dict


# wgs84
class MeeDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.obj_fname = opt.get_mee_obj_name()
        print("Reading mees file from [%s]" % opt.get_mee_file_name())
        self.df = pd.read_csv(
            opt.get_mee_file_name(),
            sep=',', nrows=opt.nrows,
            header=None, parse_dates=[1],
            infer_datetime_format=True,
            date_parser=dateparse)
        self.df.columns = ['vehicle_id', 'tstamp', 'operator', 'lati', 'longti']

        self.df.sort_values("tstamp", inplace=True)

        self.traj_dict = {}

        n_core = cpu_count()

        dfs = split_dataframe(self.df, n_core)

        pool = Pool(n_core)

        results = pool.map_async(parse_df, iterable=dfs)

        pool.close()
        pool.join()

        print("Start combining...")
        for traj_dict in results.get():
            for k in traj_dict:
                if (k not in self.traj_dict):
                    self.traj_dict[k] = traj_dict[k]
                else:
                    self.traj_dict[k] += traj_dict[k]
        print("Start sorting...")
        for k in self.traj_dict:
            self.traj_dict[k].sort(key=lambda x: x.ts)

        self.df = None

    def get_gps_line_with_ts(self, vid):
        return self.traj_dict[vid]

    def get_gps_line_without_ts(self, vid):
        return [(traj_obj.lng, traj_obj.lat) for traj_obj in self.traj_dict[vid]]

    # def query_by_traj(self, traj_lines: list, k=10):
    #
    #     query_data = []
    #
    #     # if (type(traj_list[0][0]) is float):
    #     #     query_data = traj_list  # just one traj line
    #     # else:
    #     for line in traj_lines:
    #         query_data += line
    #
    #     data = list(self.nodepos2nid.keys())
    #     pidslist = self.kdtree.query(array(query_data), k=k)[1].tolist()
    #
    #     if (k > 1):
    #         rst = []
    #         for pids in pidslist:
    #             rst += pids
    #     else:
    #         rst = pidslist
    #
    #     poss = [data[x] for x in rst]
    #     nids = [self.nodepos2nid[pos] for pos in poss]
    #     rids = set()
    #     for nid in nids:
    #         rids = rids.union(self.nid2rids[nid])
    #     roads = [self[rid] for rid in rids]
    #     return roads
    #
