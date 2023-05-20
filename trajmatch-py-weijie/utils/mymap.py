from utils.mybase import Dataset, split_dataframe
from config import DefaultConfig
import pandas as pd
import json
import pyprind
from .coord_transform import wgs84_to_bd09
import os

from scipy.spatial import KDTree
from numpy import array

json.encoder.FLOAT_REPR = lambda x: format(x, '.15f')
PRECISION = 13


# wgs84
class MapDataset(Dataset):
    def __init__(self, opt: DefaultConfig):
        self.opt = opt
        self.obj_fname = opt.get_map_obj_name()
        self.nid2rids = {}
        self.pos2nid = {}
        self.rid2nids = {}
        self.rid2roadinfo = {}
        # read nodes
        print("Reading Nodes file from [%s]" % opt.get_node_file_name())
        self.node_df = pd.read_csv(opt.get_node_file_name(), sep=',', nrows=None, header=0)

        # save pos2id dict
        for nid in self.node_df.iloc[:, 0].tolist():
            pos = self.get_node_by_id(nid)
            self.pos2nid[self._roundpos(pos)] = nid
            # print("Indexing pos: ", str(pos), 'to', nid)

        print("Reading Edges file from [%s]" % opt.get_edge_file_name())
        self.df = pd.read_csv(opt.get_edge_file_name(), sep=',', nrows=opt.nrows, header=0)
        for _, line in self.df.iterrows():
            rid, beginid, endid, rleng, rname, rspeed, nodesinfo, forward, backward = line
            self.rid2roadinfo[rid] = line  # save all roda info

        # save rod2nids and nid2rid
        print("Indexing road info.")
        bar = pyprind.ProgBar(iterations=len(self.rid2roadinfo.keys()))
        for rid in self.rid2roadinfo.keys():
            bar.update()
            poss = self.get_all_nodes_pos(self.rid2roadinfo[rid])
            rid, beginid, endid, rleng, rname, rspeed, nodesinfo, type1, type2 = self.rid2roadinfo[rid]

            for nid in [beginid, endid]:
                if (nid not in self.nid2rids):
                    self.nid2rids[nid] = set()
                self.nid2rids[nid].add(rid)
                if (rid not in self.rid2nids):
                    self.rid2nids[rid] = []
                self.rid2nids[rid].append(nid)

            for pos in poss:
                if (self._roundpos(pos) not in self.pos2nid):
                    # print("New point : %s" % str(pos))
                    nid = len(self.pos2nid)
                    self.pos2nid[self._roundpos(pos)] = nid
                    self.node_df.loc[nid] = [nid] + list(pos)[::-1]
                nid = self.pos2nid[self._roundpos(pos)]
                if (nid not in self.nid2rids):
                    self.nid2rids[nid] = set()
                self.nid2rids[nid].add(rid)
                if (rid not in self.rid2nids):
                    self.rid2nids[rid] = []
                self.rid2nids[rid].append(nid)

        self.kdtree = KDTree(data=array(list(self.pos2nid.keys())))
        self.df = None

        print("Finish all parsing.")

    def _roundpos(self, pos: tuple):
        return (round(pos[0], PRECISION), round(pos[1], PRECISION))

    def __getitem__(self, rid):  # get all pos from road[rid]
        return self.get_all_nodes_pos(self.rid2roadinfo[rid])

    def get_node_by_id(self, nid):
        # node = self.node_df.iloc[int(nid), :]
        node = self.node_df[self.node_df['Tower Node ID'] == int(nid)]
        return self._roundpos(wgs84_to_bd09(node[2], node[1]))  # lng lat

    def get_nid_by_pos(self, pos):
        return self.pos2nid[self._roundpos(pos)]

    def get_all_nodes_pos(self, line):
        rid, beginid, endid, rleng, rname, rspeed, nodesinfo, type1, type2 = line
        nodesinfo = nodesinfo.split(';')
        i = 0  # !!
        othernodes = []
        while (i < (len(nodesinfo) / 2)):
            lat = float(nodesinfo[2 * i])
            lng = float(nodesinfo[2 * i + 1])
            i += 1
            othernodes.append(self._roundpos(wgs84_to_bd09(lng, lat)))
        # all_nodes = [self.get_node_by_id(beginid)] + othernodes + [self.get_node_by_id(endid)]
        return othernodes

    def query_by_pos_lines(self, pos_lines: list, k=10, offset=(0, 0), return_type='roadpos'):
        if (k <= 0):
            return []

        query_data = []

        # if (type(traj_list[0][0]) is float):
        #     query_data = traj_list  # just one traj line
        # else:
        for line in pos_lines:
            query_data += line

        data = list(self.pos2nid.keys())
        pidslist = self.kdtree.query(array(query_data) + array(offset), k=k)[1].tolist()

        if (k > 1):
            rst = []
            for pids in pidslist:
                rst += pids
        else:
            rst = pidslist

        poss = [data[x] for x in rst]

        nids = [self.pos2nid[self._roundpos(pos)] for pos in poss]
        if (return_type == 'nids'):
            return set(nids)
        if (return_type == 'roadpos'):
            rids = set()
            for nid in nids:
                rids = rids.union(self.nid2rids[nid])
            roads = [self[rid] for rid in rids]
            return roads
        if (return_type == 'allroadinfo'):
            rids = set()
            for nid in nids:
                rids = rids.union(self.nid2rids[nid])
            roads = [self.rid2roadinfo[rid] for rid in rids]
            return roads

    def get_road_by_id(self, rid):
        return self[rid]
