import pandas as pd
import os
import pickle
import pyprind
import numpy as np
from utils.mybase import Dataset, TrajPoint, split_dataframe
from utils.coord_transform import gcj02_to_bd09

from multiprocessing import Pool, cpu_count


def parse_df(df):
    traj_dict = {}
    bar = pyprind.ProgBar(iterations=len(df))
    for _, line in df.iterrows():
        bar.update()
        vid, guest_status, tstamp, longti, lati, speed, direction = line
        vid = int(vid)
        if (vid not in traj_dict):
            traj_dict[vid] = []
        longti, lati = gcj02_to_bd09(float(longti), float(lati))
        traj_dict[vid].append(TrajPoint(lng=longti, lat=lati, ts=tstamp, has_guest=int(guest_status)))
    return traj_dict


# gcj02
class GPSDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.obj_fname = opt.get_gps_obj_name()
        print("Reading GPS file from [%s]" % opt.get_gps_file_name())
        self.df = pd.read_csv(opt.get_gps_file_name(), sep=',', nrows=opt.nrows, header=None, parse_dates=[2])
        self.df.columns = ['vehicle_id', 'useless1', 'tstamp', 'lati', 'longti', 'useless2', 'useless3']
        self.df.sort_values('tstamp', inplace=True)

        self.traj_dict = dict()
        n_cores = cpu_count()
        print("Multi Threading[%d] parsing dataframe..." % n_cores)
        df_split = split_dataframe(self.df, n_cores)

        pool = Pool(n_cores)
        results = pool.map_async(parse_df, iterable=df_split)
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
        """
        :param vid:
        :return: list of Traj obj
        """
        return self.traj_dict[vid]

    def get_gps_line_without_ts(self, vid):
        """
        :param vid:
        :return: list of (x,y)
        """
        return [(traj_obj.lng, traj_obj.lat) for traj_obj in self.traj_dict[vid]]
